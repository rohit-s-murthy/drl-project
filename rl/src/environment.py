import rospy
from hector_uav_msgs.msg import Altimeter
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Quaternion, Point, Pose, Vector3, Vector3Stamped, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu, Range, Image
from hector_uav_msgs.msg import Altimeter
import message_filters
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import gazeboInterface as gazebo

class Environment():

	def __init__(self):

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

		self.gazebo = gazebo()
		self.running_step = 0.3  # Convert to ros param
        self.max_incl = 0.7
        self.max_altitude = 10.0
        self.vel_min = -2.0
        self.vel_max = 2.0
        self.goalPos = [0.0, -5.0, 2.0]

	def _step(self, action):
		# Input: action
		# Output: nextState, reward, isTerminal, [] (not sending any debug information)

		vel = Twist()

		vel.linear.x = action[0]
        vel.linear.y = action[1]
		vel.linear.z = action[2]

        self.vel_pub.publish(vel)
        time.sleep(self.running_step)
        poseData, imuData, velData = self.takeObservation()
        self.gazebo.pauseSim()

        reward, isTerminal = self.processData(poseData, imuData, velData)

        nextState = [poseData.position.x, poseData.position.y, poseData.position.z]

		return nextState, reward, isTerminal, []

	def _reset(self):

        # 1st: resets the simulation to initial values
        self.gazebo.resetSim()

        # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # 3rd: Don't want to start the agent from the ground
        self.takeoff()

        # 4th: Get init state
        # TODO: Should initial state have some randomness?
        initStateData, _, _ = self.takeObservation()

        initState = [initStateData.position.x, initStateData.position.y, initStateData.position.z]

		return initState

	def _sample(self):

        vel_x = random.uniform(self.vel_min, self.vel_max)
        vel_y = random.uniform(self.vel_min, self.vel_max)
        vel_z = random.uniform(self.vel_min, self.vel_max)

		return [vel_x, vel_y, vel_z]

	def takeObservation(self):
		# TODO: Using wait_for_message for now, might change to ApproxTimeSync later 

        poseData = None
        while poseData is None:
            try:
                poseData = rospy.wait_for_message('/ground_truth_to_tf/pose', Pose, timeout=5)
            except:
                rospy.loginfo("Current drone pose not ready yet, retrying to get robot pose")

        velData = None
        while velData is None:
            try:
                velData = rospy.wait_for_message('/fix_velocity', Pose, timeout=5)
            except:
                rospy.loginfo("Current drone velocity not ready yet, retrying to get robot velocity")

        imuData = None
        while imuData is None:
            try:
                imuData = rospy.wait_for_message('/raw_imu', Imu, timeout=5)
            except:
                rospy.loginfo("Current drone imu not ready yet, retrying to get robot imu")
        
        return poseData, imuData, velData

    def _distance(self, pose):

        currentPos = [pose.position.x, pose.position.y, pose.position.z]
        dist = np.linalg.norm(currentPos - self.goalPos)
        return dist
    
    def getReward(self, poseData, imuData, velData):
        # Input: poseData, imuData
        # Output: reward according to the defined reward function

        reward = 0

        error = self._distance(poseData)
        reward += -error

        angletoGoal = np.arctan2(np.abs(poseData.position.y - goalPos[1]), np.abs(poseData.position.x - goalPos[2]))
        currentAngle = np.arctan2(velData.vector.y, velData.vector.x)

        if(-np.pi/6 < currentAngle < np.pi/6):
            reward += 1
        else:
            reward -= 5

        return reward

    def processData(self, poseData, imuData, velData):

        done = False
        
        euler = tf.transformations.euler_from_quaternion([imuData.orientation.x, imuData.orientation.y, imuData.orientation.z, imuData.orientation.w])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        pitch_bad = not(-self.max_incl < pitch < self.max_incl)
        roll_bad = not(-self.max_incl < roll < self.max_incl)
        altitude_bad = data_position.position.z > self.max_altitude

        if altitude_bad or pitch_bad or roll_bad:
            rospy.loginfo ("(Terminating Episode: Unstable quad) >>> ("+str(altitude_bad)+","+str(pitch_bad)+","+str(roll_bad)+")")
            done = True
            reward = -200  # TODO: Scale this down?
        else:
            reward = self.getReward(poseData, imuData, velData)

        return reward,done

    def takeoff(self):

        rate = rospy.Rate(10)
        count = 0
        msg = Twist()

        # while not rospy.is_shutdown():
        while count < 10:
            msg.linear.z = 0.5
            rospy.loginfo('Lift off')

            pub_.publish(msg)
            count = count + 1
            rate.sleep()

        print('Take-off sequence completed')
        return


