# simply publish mono left, optionally crop it (current default) and the disparity.
# there are two Scenario for the cropping which will give different sized images.

# Scenario 1 (current):crop the same section from both left and right, the depth map is computed from these cropped mono, so the disparity (or depth) will have the same size as the cropped left and right

# Scenario 2 : do as above but then crop the disparity or depth AFTER it's been computed, then we have a disparity smaller than the cropped left and right (like a 2x crop)
import cv2
import depthai as dai
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np

rclpy.init()
node = rclpy.create_node('oakd')
img_pub = node.create_publisher(Image, "mono_left", 1)
disparity_pub = node.create_publisher(Image, "disparity_oak", 1)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
manipLeft = pipeline.create(dai.node.ImageManip)
manipRight = pipeline.create(dai.node.ImageManip)
# manipDisparity = pipeline.create(dai.node.ImageManip)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutDisparity = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutDisparity.setStreamName('disparity')

# Properties
monoLeft.setCamera("left")
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

LEFT_SOCKET = dai.CameraBoardSocket.CAM_B # that's left, LEFT is deprecated
# StereoDepth properties
stereo.initialConfig.setConfidenceThreshold(255)
stereo.initialConfig.setLeftRightCheck(True)
stereo.setDepthAlign(LEFT_SOCKET)
stereo.initialConfig.setSubpixel(False)

# ImageManip nodes for cropping
crop_x_min = 0.2
crop_y_min = 0.2
crop_x_max = 0.8
crop_y_max = 0.8

manipLeft.initialConfig.setCropRect(crop_x_min, crop_y_min, crop_x_max, crop_y_max)
manipRight.initialConfig.setCropRect(crop_x_min, crop_y_min, crop_x_max, crop_y_max)
# manipDisparity.initialConfig.setCropRect(crop_x_min, crop_y_min, crop_x_max, crop_y_max)

# Linking
monoLeft.out.link(manipLeft.inputImage)
monoRight.out.link(manipRight.inputImage)

manipLeft.out.link(stereo.left)
manipRight.out.link(stereo.right)

# stereo.disparity.link(manipDisparity.inputImage)
stereo.disparity.link(xoutDisparity.input)
manipLeft.out.link(xoutLeft.input)
# manipDisparity.out.link(xoutDisparity.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qDisparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    while rclpy.ok():
        inLeft = qLeft.get()
        frameLeft = inLeft.getCvFrame()

        inDisparity = qDisparity.get()
        frameDisparity = inDisparity.getCvFrame()

        # Publish mono left image
        headerLeft = Header()
        headerLeft.frame_id = "body"
        headerLeft.stamp = node.get_clock().now().to_msg()
        imgLeft = Image()
        imgLeft.header = headerLeft
        imgLeft.height = inLeft.getHeight()
        imgLeft.width = inLeft.getWidth()
        imgLeft.is_bigendian = 0
        imgLeft.encoding = "mono8"
        imgLeft.step = inLeft.getWidth()
        imgLeft.data = frameLeft.tobytes()
        img_pub.publish(imgLeft)

        # Publish cropped disparity image
        headerDisparity = Header()
        headerDisparity.frame_id = "body"
        headerDisparity.stamp = node.get_clock().now().to_msg()
        imgDisparity = Image()
        imgDisparity.header = headerDisparity
        imgDisparity.height = inDisparity.getHeight()
        imgDisparity.width = inDisparity.getWidth()
        imgDisparity.is_bigendian = 0
        imgDisparity.encoding = "mono8"
        imgDisparity.step = inDisparity.getWidth()
        imgDisparity.data = frameDisparity.tobytes()
        disparity_pub.publish(imgDisparity)

rclpy.shutdown()


