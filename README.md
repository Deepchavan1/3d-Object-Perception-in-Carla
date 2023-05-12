# Integration of 3D Object Detection (RTM3D) in carla using carla-ros-bridge
## 3d Object Detection
For implementing 3D object detecion, [RTM3D](https://arxiv.org/abs/2001.03343) have been used. This method predicts the nine perspective keypoints of a 3D bounding box in image space, and then utilize
the geometric relationship of 3D and 2D perspectives to recover the dimension, location, and orientation in 3D space. The implementation of this paper is referred from [here](https://github.com/maudzung/RTM3D). The model has been trained and pretrained model has been used for implementing 3D object detection in carla.
## Carla-ros-bridge
Installation of carla can be referred from [here](https://carla.readthedocs.io/en/latest/start_quickstart/). Now, install ros noetic for ubuntu. Follow [this](http://wiki.ros.org/ROS/Installation) for installing ros depending upon your requirements. Further for integration carla-ros-bridge refer [here](https://carla.readthedocs.io/en/latest/ros_documentation/). The ROS bridge enables two-way communication between ROS and CARLA. The information from the CARLA server is translated to ROS topics. In the same way, the messages sent between nodes in ROS get translated to commands to be applied in CARLA.
## Integration of 3D object detection in carla
Create a catkin workspace and build this catkin workspace. Further create a package named object_perception which has rospy cv_bridge packages. Now, in scripts folder in src/object_perception put the code files in this repository except launch file. Build catkin workspace again.
Now for crearting launch folder and place launch file in this folder
