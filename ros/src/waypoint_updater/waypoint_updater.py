#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2



import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', PointCloud2, self.obstacle_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose_t = None # pose at current time t
        self.waypoints_base = None # base waypoints
        self.LenMapWP = 0 #map length
        self.tl_dis  = 0;
        self.loop()
        #rospy.spin()

    def loop(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if (self.pose_t is not None) and (self.waypoints_base is not None):
                self.get_final_waypoints()
                self.publish_final_waypoints()
                #pass
            rate.sleep()
        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose_t = msg
        # rospy.logwarn('base_waypoints:%s', self.pose_t )

        #pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        pass
        self.waypoints_base = waypoints.waypoints
        # ypoints
        self.LenMapWP = len(self.waypoints_base)
        #rospy.logwarn('base_waypoints received - size:%s', self.waypoints_base)


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.tl_dis = msg.data;

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance2(self, a, b):
        #return lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

    def find_closest(self, position):
        min_dis = 100000
        index_closest = 0
        for i in range(len(self.waypoints_base)):
            dist = self.distance2(position, self.waypoints_base[i].pose.pose.position)
            if dist < min_dis:
                min_dis = dist
                index_closest = i
        return index_closest


    def find_next(self, position, yaw_t):
        index_next= self.find_closest(position)
        map_x = self.waypoints_base[index_next].pose.pose.position.x
        map_y = self.waypoints_base[index_next].pose.pose.position.y
        heading = math.atan2(map_y - position.y, map_x - position.x)
        if math.fabs(yaw_t-heading)> math.pi/4:
            index_next += 1
        return index_next

    def get_yaw_t(self):
        orientation = [
            self.pose_t.pose.orientation.x,
            self.pose_t.pose.orientation.y,
            self.pose_t.pose.orientation.z,
            self.pose_t.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(orientation)
        return euler[2]   # z direction

    def get_final_waypoints(self):
        yaw_t = self.get_yaw_t()
        index_nxtw = self.find_next(self.pose_t.pose.position, yaw_t)
        #rospy.logwarn('WayPointNextIndex:%s', index_nxtw)


        self.final_waypoints = self.waypoints_base[index_nxtw:index_nxtw+LOOKAHEAD_WPS]
        if index_nxtw+LOOKAHEAD_WPS > self.LenMapWP:
            self.final_waypoints[index_nxtw:self.LenMapWP] = self.waypoints_base[index_nxtw:self.LenMapWP]
            self.final_waypoints[self.LenMapWP+1:index_nxtw+LOOKAHEAD_WPS]=self.waypoints_base[1:index_nxtw+LOOKAHEAD_WPS-self.LenMapWP]
        else:
            self.final_waypoints = self.waypoints_base[index_nxtw:index_nxtw+LOOKAHEAD_WPS]
        #rospy.logwarn('first point:%s', self.final_waypoints[0].pose.pose.position)
        #rospy.logwarn('last point:%s', self.final_waypoints[LOOKAHEAD_WPS-1].pose.pose.position)
        rospy.logwarn('first point:%s', self.final_waypoints[0].twist.twist.linear)
        rospy.logwarn('last point:%s', self.final_waypoints[LOOKAHEAD_WPS-1].twist.twist.linear)

    def publish_final_waypoints(self):
        fw = Lane()
        fw.header.stamp = rospy.Time(0)
        fw.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(fw)

    # def update_speed_waypoint(self, index_nxtw):
    #     spd_current = get_waypoint_velocity(self, pose_t)
    #     traff_dis   = self.tl_dis
    #     spd_trg = 10*1.6/3.6
    #     acc_max = 0.5
    #     dec_max = 0.5
    #     stop_gap = 3
    #     if traffic_dis > 0 and (traffic_dis-stop_gap)< spd_current**2/(2*dec_max):
    #         spd_trg = 0
    #


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
