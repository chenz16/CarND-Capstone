#!/usr/bin/env python
import os
import math

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

# Image logging parameters (for training)
TRAIN_DIR = '/home/btamm/Documents/udacity/term3/CarND-Capstone_old/training_data/new'
TRAIN_LOG_INTERVAL = 3
TRAIN_DIST_MAX = 100
TRAIN_DO_LOG = False

# Detection parameters
STATE_COUNT_THRESHOLD = 3
DETECT_DIST_MAX = 80

# Other parameters
STATE_TO_COLOR = {
    TrafficLight.RED: 'Red',
    TrafficLight.YELLOW: 'Yellow',
    TrafficLight.GREEN: 'Green',
    TrafficLight.UNKNOWN: 'Unknown',
}

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
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
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.frame_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

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
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            do_stop = (state == TrafficLight.RED) or (state == TrafficLight.YELLOW)
            light_wp = light_wp if do_stop else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1


    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
	# assuming small z
        x_in = pose.position.x
        y_in = pose.position.y

	i_closest = -1
        dist_closest = 1e6
        if self.waypoints is not None:
            for i, wp in enumerate(self.waypoints.waypoints):
                x_wp = wp.pose.pose.position.x
                y_wp = wp.pose.pose.position.y
                dist = math.sqrt((x_in - x_wp)**2 + (y_in - y_wp)**2)
                if dist < dist_closest:
                    i_closest = i
                    dist_closest = dist

        return i_closest

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

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        # find the closest visible stop line
        if (self.pose):
            i_closest, dist_closest = self.get_closest_stop_line(self.pose.pose,
                                                                 stop_line_positions)
        else:
            i_closest = -1
            dist_closest = 1e6

        light_wp = -1
        state = TrafficLight.UNKNOWN
        if dist_closest <= DETECT_DIST_MAX:
            # classify image
            state = self.get_light_state(light) # currently does not use input arg
            # find waypoint closest to closest stop line
            line_pose = Pose()
            line_pose.position.x = stop_line_positions[i_closest][0]
            line_pose.position.y = stop_line_positions[i_closest][1]
            light_wp = self.get_closest_waypoint(line_pose)

        rospy.logwarn('State = {}, Stop line at {:.2f}'.format(STATE_TO_COLOR[state],
                                                               dist_closest))


        # Log training data
        if TRAIN_DO_LOG:
            do_log = (dist_closest <= TRAIN_DIST_MAX) and \
                ((self.frame_count % TRAIN_LOG_INTERVAL) == 0)
            if do_log:
                self.frame_count += 1
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
	        self.save_training_image(cv_image)

        return light_wp, state

    def get_closest_stop_line(self, pose, stop_line_positions):
        """Get the stop line closest to the given pose.

        Args:
            pose (Pose): position around which to search
            stop_line_positions: list of (x, y) tuples

        Returns:
            int: index of closest line in stop_line_positions
            float: distance to closest line
        """
        # assuming small z
        x_in = pose.position.x
        y_in = pose.position.y

	i_closest = -1
        dist_closest = 1e6
        for i, line in enumerate(stop_line_positions):
            x_line = line[0]
            y_line = line[1]
            dx = x_line - x_in
            dy = y_line - y_in
            dist = math.sqrt(dx**2 + dy**2)

            # determine if line is in front or behind pose
            orient_quat = (pose.orientation.x,
                           pose.orientation.y,
                           pose.orientation.z,
                           pose.orientation.w)
            orient_euler = tf.transformations.euler_from_quaternion(orient_quat)
            yaw = orient_euler[2]
            yaw_vec = (math.cos(yaw), math.sin(yaw))
            delta_vec = (dx, dy)
            # If line is behind, angle between yaw and delta vectors will be
            # > 90 deg -> yaw dot delta < 0.
            yaw_dot_delta = yaw_vec[0]*delta_vec[0] + yaw_vec[1]*delta_vec[1]

            if (dist < dist_closest) and (yaw_dot_delta > 0):
                i_closest = i
                dist_closest = dist

        return i_closest, dist_closest


    def save_training_image(self, image):
        """Save the input image for offline detector/classifier training.

        Args:
            image: cv image
        """
        fname = 'train_{0:0>5}.jpg'.format(self.frame_count)
        # rospy.logwarn('Saving {}'.format(fname))
        cv2.imwrite(os.path.join(TRAIN_DIR, fname), image)

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
