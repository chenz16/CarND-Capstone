This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 127.0.0.1:4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Review code design 
#### NODE: waypoint_updater.py
1. This node outputs desired vehicle future moving waypoints (x, y) and planned speeds (v) at each of these waypoints. The ouputs are published to the topic final_waypoints. 

2. Details of subfunctions in the mode are explained as follows:

1).  read the current vehicle pose (xt, yt, yaw_t) 

    def pose_cb(self, msg):
        self.pose_t = msg
        
2).  identify the current location (xt, yt) in the map (base_waypoints) 
 
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
  
  3).  plan the vehicle speed based on traffic light position
  
      def update_velocity(self):
        self.distance_t2future()
        dec_schedule = -1.0
        s = self.v_t**2/(2*np.absolute(dec_schedule ))
        # set brake mode when approaching traffic light
        if self.InStopping is 0 and (self.tl_dis-2) < s:
            self.InStopping=1

        # plan vehicle speed during cruise/acc
        if self.InStopping is 0:
            vel_RateLimit = 1.5 # acc
            for i in range(0, LOOKAHEAD_WPS):
                spd_target = np.sqrt(2*vel_RateLimit*self.dis2future[i] + (1.1*self.v_t)**2) #spd_target
                if spd_target>11: spd_target = 11
                #spd_target = 11
                self.final_waypoints[i].twist.twist.linear.x= spd_target #spd_target

        # plan speed during braking
        if self.InStopping is 1:
            dec_target = dec_schedule
            for i in range(0, LOOKAHEAD_WPS):
                ds = self.tl_dis -2- self.dis2future[i]
                if ds<0: ds=0
                self.final_waypoints[i].twist.twist.linear.x = np.sqrt(2*np.absolute(dec_target)*ds)


4). finally publish all the information to final_waypoint topoic:

    def get_final_waypoints(self):
        yaw_t = self.get_yaw_t()
        index_nxtw = self.find_next(self.pose_t.pose.position, yaw_t)
        self.final_waypoints = self.waypoints_base[index_nxtw:index_nxtw+LOOKAHEAD_WPS]
        if index_nxtw+LOOKAHEAD_WPS > self.LenMapWP:
            self.final_waypoints[index_nxtw:self.LenMapWP] = self.waypoints_base[index_nxtw:self.LenMapWP]
            self.final_waypoints[self.LenMapWP+1:index_nxtw+LOOKAHEAD_WPS]=self.waypoints_base[1:index_nxtw+LOOKAHEAD_WPS-self.LenMapWP]
        else:
            self.final_waypoints = self.waypoints_base[index_nxtw:index_nxtw+LOOKAHEAD_WPS]

        self.update_velocity()
        self.update_tl_dis()

        # rospy.logwarn('traffic light distance:%s, spd actual:%s, InStopping:%s, spd target[0, 1, final]:%s, %s, %s',
        #                 self.tl_dis, self.v_t, self.InStopping,self.final_waypoints[0].twist.twist.linear.x,
        #                 self.final_waypoints[1].twist.twist.linear.x,self.final_waypoints[LOOKAHEAD_WPS-1].twist.twist.linear.x)

    def publish_final_waypoints(self):
        fw = Lane()
        fw.header.stamp = rospy.Time(0)
        fw.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(fw)
