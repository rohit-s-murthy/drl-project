# drl-project
A Deep Reinforcement Learning Project for the class CMU 10-703

0. Install(from source) the hector-gazebo simulator from:
http://wiki.ros.org/hector_quadrotor/Tutorials/Quadrotor%20outdoor%20flight%20demo

Use this command to install other dependencies if you don't have them:
rosdep install --from-paths src --ignore-src -r -y

1. For using pr2 teleop:
git clone https://github.com/PR2/pr2_apps.git

2. For using teleop twist keyboard:
sudo apt-get install ros-indigo-teleop-twist-keyboard

3. To launch the basic demo:
roslaunch hector_quadrotor_demo outdoor_flight_gazebo.launch

4. For teleop use either of the following (the second one is more complicated but is needed to get off the ground)
roslaunch pr2_teleop teleop_keyboard.launch
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

5. Execute the python publisher:
chmod +x pub.py (only the first time in the same folder as pub.py)
python pub.py
