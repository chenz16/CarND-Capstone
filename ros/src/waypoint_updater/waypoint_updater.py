#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistStamped

import math
import tf
import numpy as np

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

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number
STOP_DIS_TL  = 3.0  # Desired stop distance ahead of traffic light
DEC_SCHEDULE = 1.0 # pre-determined deceleration
ACC_SCHEDULE = 2.0 # pre-determined acceleration
TIME_DELY = 0.0  # time delay

# if distance_buget - desired_stop_distance < STOP_DISGAP: STOP
STOP_DISGAP = 0

# if distance_buget - desired_stop_distance > STOP_DISGAP: GO
GO_DISGAP   = 3.5

# if distance_buget/desired_stop_distance > GO_DISRATIO: STOP
GO_DISRATIO = 0.4


NODE_FRQ = 50 # node running req
kmph2mps = 1.0/3.6
TAU = 0.2 # low pass filter time constant for filtering speed
# MIN_DIS_SWMODE = 2 # min distance to switch mode from NON-STOPPING to STOPPING

MAX_SPD = rospy.get_param('/waypoint_loader/velocity')*kmph2mps

class WaypointUpdater(object):
    def __init__(self):
        self.passing_tl = False
        self.passing_tl_dis = 0.0
        self.tl_dis_last = 500.0
        self.tl_dis_fused = 500.0
        self.v_flt = 0
        self.v_last = 0
        rospy.init_node('waypoint_updater')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', PointCloud2, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose_t = None # pose at current time t
        self.waypoints_base = None # base waypoints
        self.LenMapWP = 0 #map length
        self.tl_dis  = 300 # consider tl
        self.loop()
        self.v_t = 0 # current vehicle speed
        self.dis2future = np.zeros(LOOKAHEAD_WPS)
        self.InStopping = None

        rospy.spin()

    def loop(self):
        rate = rospy.Rate(NODE_FRQ)
        self.InStopping = 0
        while not rospy.is_shutdown():
            if (self.pose_t is not None) and (self.waypoints_base is not None):
                self.get_final_waypoints()
                self.publish_final_waypoints()
            rate.sleep()
        rospy.spin()

    def pose_cb(self, msg):
        self.pose_t = msg

    def waypoints_cb(self, waypoints):
        self.waypoints_base = waypoints.waypoints
        self.LenMapWP = len(self.waypoints_base)
        # rospy.logwarn('base_waypoints received - size:%s', self.waypoints_base)

    def velocity_cb(self, velocity):
        self.v_t = velocity.twist.linear.x

        #rospy.logwarn('current speed:%s', self.v_t)
    def traffic_cb(self, msg):
        tl_index = msg.data
        if tl_index <=-1:
            self.tl_dis = 500
        else:
            pose_tl = self.waypoints_base[tl_index]
            self.tl_dis = self.distance2(pose_tl.pose.pose.position, self.pose_t.pose.position)
        #rospy.logwarn('tl_dis:%s', self.tl_dis)

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

# dis of current pose to future waypoints
    def distance_t2future(self):
        self.dis2future = np.zeros(LOOKAHEAD_WPS)
        self.dis2future[0] = self.distance2(self.final_waypoints[0].pose.pose.position, self.pose_t.pose.position)
        for i in range(1, LOOKAHEAD_WPS):
            self.dis2future[i]= self.dis2future[i-1]+ self.distance2(self.final_waypoints[i].pose.pose.position, self.final_waypoints[i-1].pose.pose.position)

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

    def fuse_dis_tl(self): #optional to use
        if abs(self.tl_dis- self.tl_dis_last)< 5 and self.tl_dis<50 and self.v_t>0.5:
            self.tl_dis_fused -= self.v_t/NODE_FRQ
        else:
            self.tl_dis_fused = self.tl_dis
        self.tl_dis_last = self.tl_dis


# Plan velocity
    def update_velocity(self):

        self.v_flt = TAU*NODE_FRQ*self.v_flt/(1.0+TAU*NODE_FRQ) + self.v_t/(1.0+TAU*NODE_FRQ)
        self.fuse_dis_tl()
        self.distance_t2future()
        dec_schedule = DEC_SCHEDULE  # desired dec rate in normal situation
        s = self.v_flt**2/(2*np.absolute(dec_schedule)) # desired stopping distance
        s = max(s,0.01)
        dis_stop_budget = self.tl_dis -STOP_DIS_TL-self.v_flt*TIME_DELY
        # rospy.logwarn('tf_dis %s, dis budget:%s, veh spd %s' %(self.tl_dis, dis_stop_budget, self.v_flt))
        dis_stop_budget = max(dis_stop_budget, 0.01)
        dec_target = self.v_flt**2/(2*dis_stop_budget) # plan dec

        #rospy.logwarn('tf_dis:%s, dec_target:%s', self.tl_dis, dec_target)

        # manage vehicle modes:
        if self.InStopping is 0 and dis_stop_budget-s< STOP_DISGAP and dis_stop_budget/s > GO_DISRATIO and not self.passing_tl : #and dis_stop_budget>MIN_DIS_SWMODE:
            self.InStopping=1
        elif self.InStopping is 1 and dis_stop_budget-s> GO_DISGAP: #and dis_stop_budget > 30:
            self.InStopping=0

        if self.InStopping is 1 and not self.passing_tl and self.v_flt<0.01 and  dis_stop_budget < 5:
            self.passing_tl = True
        if self.InStopping is 0 and self.passing_tl and self.v_flt> 1 and dis_stop_budget> 50 and self.passing_tl_dis>10:
            self.passing_tl = False
            self.passing_tl_dis = 0.0
        if  self.passing_tl:
            self.passing_tl_dis += self.v_t/NODE_FRQ
        else:
            self.passing_tl_dis = 0

        #rospy.logwarn('passing tl? %s, in stopping ? %s, self_passing_dis %s ' %(self.passing_tl,self.InStopping, self.passing_tl_dis))

        # plan vehicle speed during cruise/acc
        if self.InStopping is 0:
            vel_RateLimit = ACC_SCHEDULE # acc
            for i in range(0, LOOKAHEAD_WPS):
                spd_target = np.sqrt(2.0*vel_RateLimit*self.dis2future[i] + (self.v_t)**2) #spd_target
                if not self.passing_tl_dis:
                    dis_stop_budget_pred = dis_stop_budget - self.dis2future[i]
                    dis_stop_budget_pred = max(dis_stop_budget_pred,0.0)
                    max_spd = np.sqrt(2.0*dis_stop_budget_pred*dec_schedule*1.0)
                    max_spd= min(max_spd, MAX_SPD)
                else: # if during pass tl
                    max_spd = MAX_SPD
                if spd_target> max_spd: spd_target = max_spd
                self.final_waypoints[i].twist.twist.linear.x= spd_target #spd_target

        # plan speed during braking
        if self.InStopping is 1:
            acc_target = - dec_target
            for i in range(0, LOOKAHEAD_WPS):
                dis_stop_budget_pred = dis_stop_budget - self.dis2future[i]
                dis_stop_budget_pred = max(dis_stop_budget_pred,0.0)
                spd_square = 2*dec_target*dis_stop_budget_pred
                self.final_waypoints[i].twist.twist.linear.x = np.sqrt(spd_square)

        # update info
        spd_next1 = self.final_waypoints[0].twist.twist.linear.x
        spd_next2 = self.final_waypoints[1].twist.twist.linear.x
        dis_next = self.dis2future[1]-self.dis2future[0]
        acc_target = (spd_next2**2 - spd_next1**2)/(2.0*dis_next)
        acc_actual = (self.v_t - self.v_last)*NODE_FRQ

        rospy.logwarn('Mode:%2d, actl_acc:%5.2f, acc_trg:%5.2f, BugtStpD:%5.2f, DesStpD:%5.2f'
                        %(self.InStopping, acc_actual, acc_target,  dis_stop_budget, s))

        self.v_last = self.v_t


    def get_final_waypoints(self):
        yaw_t = self.get_yaw_t()
        index_nxtw = self.find_next(self.pose_t.pose.position, yaw_t)
        if index_nxtw+LOOKAHEAD_WPS > self.LenMapWP: # in case the index exceeds the maximum
            self.final_waypoints[index_nxtw:self.LenMapWP] = self.waypoints_base[index_nxtw:self.LenMapWP]
            self.final_waypoints[self.LenMapWP+1:index_nxtw+LOOKAHEAD_WPS]=self.waypoints_base[1:index_nxtw+LOOKAHEAD_WPS-self.LenMapWP]
        else:
            self.final_waypoints = self.waypoints_base[index_nxtw:index_nxtw+LOOKAHEAD_WPS]
        self.update_velocity()


    def publish_final_waypoints(self):
        fw = Lane()
        fw.header.stamp = rospy.Time(0)
        fw.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(fw)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
