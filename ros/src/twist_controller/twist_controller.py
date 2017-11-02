import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, WB, steer_ratio, min_speed, max_lat_accel, max_steer, VM, wr, brake_deadband, CONTROL_FREQ):
        # Store vehice and control parameters
        self.VM = VM
        self.wr = wr
        self.brake_deadband = brake_deadband
        self.CONTROL_FREQ = CONTROL_FREQ
        # init yaw controller
        self.yaw_controller = YawController(WB, steer_ratio, min_speed, max_lat_accel, max_steer)
        # init filters
        self.steer_filter = LowPassFilter(0.5,1/CONTROL_FREQ)
    	self.brake_filter = LowPassFilter(0.2,1/CONTROL_FREQ)
    	self.throttle_filter = LowPassFilter(0.3,1/CONTROL_FREQ)
        #init PID controllers
        self.brake_pid = PID(10,.05,0,mn=0)
    	self.throttle_pid = PID(8,.05,0,mn=0,mx=1)

        # init timestamp
        self.timestamp = rospy.get_time()

    def control(self, linear_vel, angular_vel, current_vel, dbw_enabled):
        # Get sample_time
        latest_timestamp = rospy.get_time()
        duration = latest_timestamp - self.timestamp
        sample_time = duration + 1e-6  # to avoid division by zero
        self.timestamp = latest_timestamp

        if not dbw_enabled: #Vehicle under manual control, reset all PID and filter values
            self.steer_filter.last_val = 0.0
            self.brake_filter.last_val = 0.0
            self.throttle_filter.last_val = 0.0
            self.throttle_pid.reset()
            self.brake_pid.reset()
            brake = 0
            throttle = 0
            steer = 0
        else:
            # Use supplied yaw_controller to get desired steering wheel angle, and then filter
            #steer = self.steer_filter.filt(self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel))
            steer = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
            # Find v_error and control throttle or brake using PID controller
            v_err = linear_vel - current_vel

            if linear_vel < 0.5: #Stop vehicle at very low speed command
                throttle = 0.0
                brake = 1e5
                self.brake_filter.last_val = 0.0
                self.throttle_filter.last_val = 0.0
                self.throttle_pid.reset()
                self.brake_pid.reset()
            elif v_err > 0: #Accelerate
                brake = 0.0
                self.brake_filter.last_val = 0.0
                self.brake_pid.reset()
                throttle = self.throttle_filter.filt(self.throttle_pid.step(v_err,sample_time))
            else: #decelerate
                throttle = 0.0
                self.throttle_filter.last_val = 0.0
                self.throttle_pid.reset()
                brake = self.brake_filter.filt(self.brake_deadband + self.brake_pid.step(-v_err,sample_time)*self.VM*self.wr)

        # Return commands to dbw
        # throttle = 1
        # brake = 0
        # steer = 5
        return throttle, brake, steer
