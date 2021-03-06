#!/usr/bin/env python
import os
import math
from glob import glob
import threading

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml


OUTPUT_GROUND_TRUTH_STATE = False
# Only applicable to simulator. If True, bypass the traffic light classifier and
# output the ground truth light state.

OUTPUT_ALL_STATES = False
# If True, the stop line waypoint of the upcoming light is published for
# green/unknown lights as well as red/yellow lights:
#   Green/Unknown: negative waypoint index
#   Red/Yellow: positive waypoint index
# If False, only the stop line waypoint of an upcomng red/yellow light is
# published.

# Detection parameters
NONE_IDX = -1
NONE_DIST = 1e6
STATE_COUNT_THRESHOLD = 3
DETECT_DIST_MAX = 80

# Image logging parameters (for training)
TRAIN_DIR = '/home/btamm/Documents/udacity/term3/CarND-Capstone_old/training_data/sim_train'
TRAIN_ANNOTATION_FILE = '/home/btamm/Documents/udacity/term3/CarND-Capstone_old/training_data/train.txt'
TRAIN_LOG_INTERVAL = 2
TRAIN_DIST_MAX = 80
TRAIN_DIST_MIN = 21
TRAIN_DO_LOG = False
TRAIN_STATES = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN]

STATE_TO_COLOR = {
    TrafficLight.RED: 'Red',
    TrafficLight.YELLOW: 'Yellow',
    TrafficLight.GREEN: 'Green',
    TrafficLight.UNKNOWN: 'Unknown',
}

# TODO - implement startup delay timer to avoid TF error message?


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.yaw = 0.0
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # .header (Header)
        #        .seq (uint32)
        #        .stamp (time?)
        #              .secs (int)
        #              .nsecs (int)
        #        .frame_id (string - 0 = no frame, 1 = global frame)
        # .pose (Pose)
        #      .position (Point)
        #               .x (float64)
        #               .y (float64)
        #               .z (float64)
        #      .orientation (Quaternion)
        #                  .x (float64)
        #                  .y (float64)
        #                  .z (float64)
        #                  .w (float64)

        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # .header (Header)
        # .waypoints (Waypoint[])
        #
        # Waypoint.pose (PoseStamped)
        #         .twist (TwistStamped)
        #               .header (Header)
        #               .twist (Twist)
        #                     .linear (Vector3)
        #                            .x (float64)
        #                            .y (float64)
        #                            .z (float64)
        #                     .angular (Vector3)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        # .header (Header)
        # .lights (TrafficLight[])
        #
        # TrafficLight.header (Header)
        #             .pose (PoseStamped)
        #             .state (uint8 - 0 = RED, 1 = YELLOW, 2 = GREEN, 3 = UNKNOWN)

        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        # .header (Header)
        # .height (uint32)
        # .width (uint32)
        # .encoding (string)
        # .is_bigendian (uint8)
        # .step (uint32 - full row length in bytes)
        # .data (uint8[] - size = step * rows)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # ['camera_info']
        #                ['focal_length_x'] = 1345.200806 (site only)
        #                ['focal_length_y'] = 1353.838257 (site only)
        #                ['image_width'] = 800
        #                ['image_height']= 600
        # ['stop_line_positions'][i] = (x, y) (float - x, y order tbc)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        # index into self.waypoints

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = NONE_IDX
        self.state_count = 0
        self.frame_count = 0
        self.log_count = 0

        self.light_classifier = TLClassifier(self.config['location'])

        if TRAIN_DO_LOG:
            if not os.path.exists(TRAIN_DIR):
                os.makedirs(TRAIN_DIR)
            log_files = sorted(glob(os.path.join(TRAIN_DIR, '*.jpg')))
            if len(log_files) > 0:
                split_fname = log_files[-1][:-4].split('_')
                self.log_count = int(split_fname[-1])

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        self.yaw = self.get_yaw_from_pose(self.pose.pose)

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        do_stop = (state == TrafficLight.RED) or (state == TrafficLight.YELLOW)
        if not OUTPUT_ALL_STATES:
            # original interface
            if self.state != state:
                self.state_count = 0
                self.state = state
                published_wp = None
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if do_stop else NONE_IDX
                self.last_wp = light_wp
                published_wp = light_wp
            else:
                published_wp = self.last_wp

        else:
            # publising all states
            if self.state != state:
                self.state_count = 0
                self.state = state
            if self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                if not do_stop:
                    light_wp = -light_wp
                self.last_wp = light_wp
                published_wp = light_wp
            else:
                published_wp = self.last_wp

        if published_wp is not None:
            self.upcoming_red_light_pub.publish(Int32(published_wp))

        self.state_count += 1

        # rospy.logwarn('Published WP = {}'.format(published_wp))

    def get_closest(self, pose, points, lookback_dist=1e6):
        """Get the point closest to the given pose.

        Args:
            pose (Pose): position around which to search
            points: list of (x, y) tuples
            lookback_dist: consider points as much as this distance behind the
                pose

        Returns:
            int: index of closest line in points
            float: distance to closest point
        """
        # assuming small z
        x_in = pose.position.x
        y_in = pose.position.y

        i_closest = NONE_IDX
        dist_closest = NONE_DIST # magnitude - always positive
        for i, pt in enumerate(points):
            x_pt = pt[0]
            y_pt = pt[1]
            dx = x_pt - x_in
            dy = y_pt - y_in
            dist = math.sqrt(dx**2 + dy**2)

            # determine if point is in front or behind pose
            # If point is behind, angle between yaw and delta vectors will be
            # > 90 deg -> yaw dot delta < 0.
            yaw_vec = (math.cos(self.yaw), math.sin(self.yaw))
            delta_vec = (dx, dy)
            yaw_dot_delta = yaw_vec[0]*delta_vec[0] + yaw_vec[1]*delta_vec[1]

            # Make forward distances positive and backwards distances negative.
            if yaw_dot_delta >= 0:
                dist = abs(dist)
            else:
                dist = -abs(dist)

            is_valid = (abs(dist) < dist_closest) and (dist >= -lookback_dist)

            if is_valid:
                i_closest = i
                dist_closest = abs(dist)

        return i_closest, dist_closest


    def get_closest_waypoint_index(self, pose):
        """Identifies the closest path waypoint to the given pose.
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position around which to search

        Returns:
            int: index of the closest waypoint in self.waypoints.waypoints
        """
        i_closest = NONE_IDX
        if self.waypoints:
            waypoints_tuples = [(wp.pose.pose.position.x , wp.pose.pose.position.y) \
                for wp in self.waypoints.waypoints]
            i_closest, _ = self.get_closest(pose, waypoints_tuples)

        return i_closest


    def get_closest_light(self, pose):
        """Returns the closest light to the given pose.

        Args:
            pose (Pose): position around which to search

        Returns:
            TrafficLight: closest light in self.lights
            float: distance to closest light (1e6 if no light found)
        """
        light = None
        i_closest = NONE_IDX
        dist_closest = NONE_DIST
        if self.lights:
            lights_tuples = [(lt.pose.pose.position.x , lt.pose.pose.position.y) \
                for lt in self.lights]
            i_closest, dist_closest = self.get_closest(pose, lights_tuples,
                                                       lookback_dist=0)
        if i_closest > NONE_IDX:
            light = self.lights[i_closest]

        return light, dist_closest


    def get_stop_waypoint(self, light):
        """Returns the stop waypoint for a given light.

        Args:
            light (TrafficLight): light around which to search

        Returns:
            int: index of the closest waypoint in self.waypoints.waypoints
        """
        # List of positions that correspond to the line to stop in front of for
        # a given intersection
        stop_line_positions = self.config['stop_line_positions']

        i_wp = NONE_IDX
        if light:
            i_line, _ = self.get_closest(light.pose.pose, stop_line_positions)
            line_pose = Pose()
            line_pose.position.x = stop_line_positions[i_line][0]
            line_pose.position.y = stop_line_positions[i_line][1]
            i_wp = self.get_closest_waypoint_index(line_pose)
            # rospy.logwarn('Stop Line WP, x, y = {}, {:.2f}, {:.2f}'.format(
            #     i_wp,
            #     line_pose.position.x,
            #     line_pose.position.y))

        return i_wp


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify (currently not used)

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        if TRAIN_DO_LOG or OUTPUT_GROUND_TRUTH_STATE:
            return light.state
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            classification = self.light_classifier.get_classification(cv_image)
            return classification


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        line_wp = NONE_IDX
        state = TrafficLight.UNKNOWN

        if self.has_image:

            # get closest light
            if (self.pose):
                light, light_dist = self.get_closest_light(self.pose.pose)
                # rospy.logwarn('Vehicle WP, x, y = {}, {:.2f}, {:.2f}'.format(
                #     self.get_closest_waypoint_index(self.pose.pose),
                #     self.pose.pose.position.x,
                #     self.pose.pose.position.y))
                # rospy.logwarn('Light WP, x, y = {}, {:.2f}, {:.2f}'.format(
                #     self.get_closest_waypoint_index(light.pose.pose),
                #     light.pose.pose.position.x,
                #     light.pose.pose.position.y))
            else:
                light = None
                light_dist = NONE_DIST

            # get waypoint near corresponding stop line
            line_wp = self.get_stop_waypoint(light)

            if TRAIN_DO_LOG:
                rospy.logwarn('True State = {}'.format(STATE_TO_COLOR[light.state]))

            if light_dist <= DETECT_DIST_MAX:
                # classify image
                state = self.get_light_state(light)

            rospy.logwarn('State = {}, Light at {:.2f}'.format(STATE_TO_COLOR[state], light_dist))

            # Log training data
            self.frame_count += 1
            if TRAIN_DO_LOG:
                do_log = (state in TRAIN_STATES) and \
                         (light_dist <= TRAIN_DIST_MAX) and \
                         (light_dist >= TRAIN_DIST_MIN) and \
                         ((self.frame_count % TRAIN_LOG_INTERVAL) == 0)
                if do_log:
                    cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                    image_file = self.save_training_image(cv_image)
                    self.save_annotation(image_file, light, light_dist)

        return line_wp, state


    def save_training_image(self, image):
        """Save the input image for offline detector/classifier training.

        Args:
            image: cv image

        Returns:
            string: image file path
        """
        self.log_count += 1
        fname = 'train_{0:0>5}.jpg'.format(self.log_count)
        image_file = os.path.join(TRAIN_DIR, fname)
        cv2.imwrite(image_file, image)
        return image_file


    def save_annotation(self, image_file, light, dist):
        """Append data to training annotation file.

        Args:
            image_file (str): path to training image
            light (TrafficLight): closest traffic light
            dist (float): distance to traffic light [m]
        """
        state = TrafficLight.UNKNOWN
        if light:
            state = light.state
        data = {
            'path': image_file,
            'state': STATE_TO_COLOR[state],
            'dist': dist,
            'x': light.pose.pose.position.x,
            'y': light.pose.pose.position.y
        }
        data_line = '{}, {}, {}, {}, {}'.format(data['path'],
                                                data['state'],
                                                data['dist'],
                                                data['x'],
                                                data['y'])
        with open(TRAIN_ANNOTATION_FILE, 'a+') as out_file:
            out_file.write(data_line + os.linesep)


    def get_yaw_from_pose(self, pose):
        orient_quat = (pose.orientation.x,
                       pose.orientation.y,
                       pose.orientation.z,
                       pose.orientation.w)
        orient_euler = tf.transformations.euler_from_quaternion(orient_quat)
        yaw = orient_euler[2]
        return yaw


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
