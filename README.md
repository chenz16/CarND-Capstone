[//]: # (Image References)
[tl_detector_training_loss]: imgs/tl_detector_training_loss.png "training loss"
[tl_detector_sim_test_still]: imgs/sim_test_still.jpg "detection on simulated image"
[tl_detector_site_test_still]: imgs/site_test_still.jpg "detection on site image"
[sim_test_video]: imgs/sim_test_0p3.mp4
[site_test_video]: imgs/site_test_0p3.mp4


This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).


### Tensorflow Version
The frozen object detection inference graph requires tensorflow-gpu>=1.2.

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
3. In one terminal, play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. In a second terminal, launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. In a third terminal, view the recorded video and compare with the classifications output in the second terminal
```bash
rosrun image_view image_view image:=/image_raw
```


### Review code design
#### NODE: waypoint_updater.py

##### This node outputs planned future vehicle moving waypoints and speeds at each of these waypoints. The outputs are published to the topic ```final_waypoints```.

##### Details of subfunctions in the mode are explained as follows:

* read the current vehicle pose (Actually Only xt, yt, yaw_t are our interested signals)

```
    def pose_cb(self, msg):
        self.pose_t = msg
```

* identify the current location (xt, yt) in the map (base_waypoints)

```
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
```
 * plan the vehicle speed at each one of future waypoints based on traffic light prediction. This is the core function of the node.  A desired stopping distance is first calculated based on current vehicle speed and scheduled deceleration.  The desired stopping distance is then compared with the distance budget of stopping which is based on the distance to the next red light. If traffic light is green, the distance is defaulted to very large number. 
  
 We design two modes for vehicle operation: one is STOPPING  mode and the other is NON-STOPPING which tries to maintain the desired vehicle speed (~11m/s). A bunch of conditions are defined to transit in/out of the each mode. 
 
 In STOPPING mode, the deceleration target at each waypoint is calculated based on (actual or planned) vehicle speed and the distance of that waypoint to the target stopping position which is a few meters away from the traffic light position. 
 
 In NON-STOPPING mode, the acceleration is defaulted to 5 m/s^2. Low number was considered, which caused acceleration delay during the moment of quick change of traffic light colors. 
  
```  
    def update_velocity(self):
        self.distance_t2future()
        dec_schedule = DEC_SCHEDULE  # desired dec rate in normal situation
        s = self.v_t**2/(2*np.absolute(dec_schedule)) # desired stopping distance
        s = max(s,0.01)
        dis_stop_bugget = self.tl_dis-STOP_DIS_TL-self.v_t*TIME_DELY
        dis_stop_bugget = max(dis_stop_bugget, 0.01)
        dec_target = self.v_t**2/(2*dis_stop_bugget) # plan dec
        #rospy.logwarn('tf_dis:%s, dec_target:%s', self.tl_dis, dec_target)

        # manage vehicle modes:
        if self.InStopping is 0 and dis_stop_bugget-s< STOP_DISGAP  and dis_stop_bugget/s > GO_DISRATIO:
            self.InStopping=1
        elif self.InStopping is 1 and dis_stop_bugget-s> GO_DISGAP:
            self.InStopping=0

        # plan vehicle speed during cruise/acc
        if self.InStopping is 0:
            vel_RateLimit = ACC_SCHEDULE # acc
            for i in range(0, LOOKAHEAD_WPS):
                spd_target = np.sqrt(2*vel_RateLimit*self.dis2future[i] + (1.05*self.v_t)**2) #spd_target
                if spd_target>11: spd_target = 11
                self.final_waypoints[i].twist.twist.linear.x= spd_target #spd_target

        # plan speed during braking
        if self.InStopping is 1:
            dec_target = dec_target*1.01 # overcome signal delay
            for i in range(0, LOOKAHEAD_WPS):
                if i is 0:
                    spd_prev = self.v_t
                    ds = self.dis2future[0]
                else:
                    spd_prev = self.final_waypoints[i-1].twist.twist.linear.x
                    ds = self.dis2future[i]-self.dis2future[i-1]
                spd_square = spd_prev**2 - 2*dec_target*ds
                spd_square = max(spd_square,0)
                self.final_waypoints[i].twist.twist.linear.x = np.sqrt(spd_square)

```

* finally publish all the information to final_waypoint topic:
```
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
        
    def publish_final_waypoints(self):
        fw = Lane()
        fw.header.stamp = rospy.Time(0)
        fw.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(fw)
```

#### NODE: tl_detector.py
##### Architecture
The traffic light detector uses an SSD Mobilenet detector trained to detect and classify red, yellow, and green traffic lights.

##### Training
The [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) was used to finetune a pretrained detector from the [Tensorflow Model Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

The [pretrained checkpoint](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) was finetuned on a mix of simulated images, Udacity test site images, and public road images of traffic lights:

| Dataset      | Image Count |
|:------------ | -----------:|
| Simulator    |         409 |
| Test Site    |         412 |
| Public Road  |         372 |

All the images were hand annotated using [LabelImg](https://github.com/tzutalin/labelImg).

The public road images were selected from [Udacity Dataset 1](https://github.com/udacity/self-driving-car/tree/master/annotations).

A plot of the training loss over 2800 mini batches of 16 images is shown below. All selected hyperparameters can be seen in the [training configuration file](ros/src/tl_detector/light_classification/training/config/ssd_mobilenet_v1_udacity_combo.config).

![alt text][tl_detector_training_loss]

Sample annotated detection images from the simulator and site test data sets are shown below. Annotated test videos are available as well: [simulator video][sim_test_video], [site video][site_test_video].

![alt text][tl_detector_sim_test_still]
![alt text][tl_detector_site_test_still]
