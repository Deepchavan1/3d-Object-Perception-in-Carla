#!/usr/bin/env python3

import queue
import cv2
import rospy


from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from rtm3d_model import RTM3D


class Detector:
    def __init__(self, ros_rate):
        self.loadParameters()
        self.bridge = CvBridge()
        self.rgb_image = None
        self.rtm3d = RTM3D()
        self.ros_rate = ros_rate

    
    def loadParameters(self):
        self.image_topicname = rospy.get_param(
            "image_topic_name", "/carla/ego_vehicle/rgb_front/image")
        self.detect_pub_topic_name = rospy.get_param(
            "detect_pub_image_topic_name", "/object/detected_image")



    def subscribeToTopics(self):
        rospy.loginfo("Subscribed to topics")
        rospy.Subscriber(self.image_topicname, Image,
                         self.storeImage, buff_size = 2**24, queue_size=1)

    
    def publishToTopics(self):
        rospy.loginfo("Published to topics")
        self.detectionPublisher = rospy.Publisher(self.detect_pub_topic_name, Image, queue_size=1)


    def sameImage(self, img):
        try:
            frame = self.bridge.imgmsg_to_cv2(img, 'bgr8')
        except CvBridgeError as e:
            rospy.loginfo(str(e))
        self.rgb_image = frame
        print("Published")
        self.callPublisher(self.rgb_image)


    
    def storeImage(self, img): # Copy for Obj Detection
        try:
            frame = self.bridge.imgmsg_to_cv2(img, 'bgr8')
            # rospy.loginfo("RGB Image Stored")
        except CvBridgeError as e:
            rospy.loginfo(str(e))
        self.rgb_image = frame
        self.detect3d()

    def detect3d(self):
        detected_img = self.rtm3d.detect(self.rgb_image)
        print("Published")
        self.callPublisher(detected_img)

    def callPublisher(self, image):
        detected_img = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        self.detectionPublisher.publish(detected_img)
