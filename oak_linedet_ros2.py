#  error
# [ERROR] [1745389802.755317656] [image_subscriber]: Error in synchronized_callback: <function Duration.nanosec at 0x7fff157b8fe0> returned a result with an exception set
#  fixed
"""
    goal: detect vertical structure and publish its center line (tbd: ros2 polygone?) 
    tested on rapsberry pi in virtualenvironment envDepthAI
    Usage:
    with an oak camera connected to the pi, run the publisher (to get live mono preview)
        python mono_preview_pub.py
    Or play a rosbag:
        ros2 bag play rosbag2_2025_04_16-10_22_10/
    Then run this file

    Detection is done by contour, so there is no line detection (too many lines, not reliable)
    setup for disparity with subpixel set to on (from corresponding publisher)
"""
# 
import cv2 as cv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
from geometry_msgs.msg import Point, PointStamped
from geometry_msgs.msg import Polygon, Point32
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import os
import sys
from scipy import ndimage  # for max and min filter

from message_filters import ApproximateTimeSynchronizer, Subscriber

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)

        # Create subscribers for the mono and disparity images
        self.mono_sub = Subscriber(self, Image, '/mono_left', qos_profile=best_effort_qos)
        self.disparity_sub = Subscriber(self, Image, '/disparity_oak', qos_profile=best_effort_qos)

        # Create the TimeSynchronizer
        # The '10' here is the queue size, and '0.1' is the allowed delay (in seconds)
        # between messages to be considered synchronized. Adjust the delay as needed.
        self.time_synchronizer = ApproximateTimeSynchronizer([self.mono_sub, self.disparity_sub], 10, 0.1)
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.publisher = self.create_publisher(
            Image,
            '/vert_line_det_img',
            1)
        self.polygon_publisher_ = self.create_publisher(Polygon, 'vert_line_polygon', 10)
        self.marker_publisher_ = self.create_publisher(Marker, 'vert_line_marker', 10)

        if 'DISPLAY' in os.environ:
            self.cv_window_name = "lineDetImage"
            cv.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)  # Allow resizing
            cv.namedWindow('mono', cv.WINDOW_NORMAL)
            cv.namedWindow('disparity', cv.WINDOW_NORMAL)
        else:
            self.cv_window_name = None

    def publish_line(self, line_points):
        """ publish polygon msg made of 2 points, top and bottom of center line of vertical structure
        p1.x,p1.y: top x,y of line, in pixel, refereing to the received (subscribed) image, that image is currently cropped
        p2.x,p2.y: same but for the bottom point
        p1.z,p2.z: distance to the middle of the line
        """


        print(f'in publish_line:{line_points, line_points[0], line_points[0][0]}')
        polygon_msg_list = Polygon()
        if len(line_points) == 2:
            p1 = Point32(x=float(line_points[0][0]), y=float(line_points[0][1]), z=float(line_points[0][2]))
            p2 = Point32(x=float(line_points[1][0]), y=float(line_points[1][1]), z=float(line_points[1][2]))
            polygon_msg_list.points = [p1, p2]
        else:
            self.get_logger().warn("Need list of two points")
        self.get_logger().info(f"Publishing Polygon with points: {polygon_msg_list.points}")
        self.polygon_publisher_.publish(polygon_msg_list)

        # marker_msg_list = self.visualize_line_polygon(line_points)
        # self.get_logger().info(f"Publishing Marker with points: {[p.x for p in marker_msg_list.points]}")
        # self.marker_publisher_.publish(marker_msg_list)

    def visualize_line_polygon(self, points, header_stamp):
        """can't show polygon direcly in rviz, this does it.
           NOTE moved this elsewhere currently not called.
        """

        marker_msg = Marker()
        marker_msg.header.frame_id = "map"
        marker_msg.header.stamp = header_stamp  # Use the provided timestamp
        marker_msg.ns = "line_visualization"  # namespace
        marker_msg.id = 0
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD
        marker_msg.lifetime.sec = 0.1
        marker_msg.scale.x = 0.05  # Line width
        marker_msg.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green line
        pix_to_m = 0.005

        if len(points) == 2:
            # p1 = Point(x=float(1.0), y=float(points[0][0] * pix_to_m), z=float(points[0][1] * pix_to_m))
            # p2 = Point(x=float(1.0), y=float(points[1][0] * pix_to_m), z=float(points[1][1] * pix_to_m))
            p1 = Point(x=float(1.0), y=float(0.0), z=float(1.0))
            p2 = Point(x=float(1.0), y=float(0.0), z=float(-1.0))
            marker_msg.points = [p1, p2]
            self.get_logger().info(f"Publishing line marker with points: {[p1,p2]}")
        else:
            self.get_logger().warn("Need a list of two points.")

        return marker_msg

    def visualize_line_polygon_clocknow(self, points):
        """can't show polygon direcly in rviz, this does it."""
        marker_msg = Marker()
        marker_msg.header.frame_id = "body"
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.ns = "line_visualization"  # namespace
        marker_msg.id = 0
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD
        marker_msg.lifetime.sec = 0.1

        marker_msg.scale.x = 0.05  # Line width
        marker_msg.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green line
        pix_to_m = 0.01

        if len(points) == 2:
            p1 = Point(x=float(1.0), y=float(points[0][0] * pix_to_m), z=float(points[0][1] * pix_to_m))
            p2 = Point(x=float(1.0), y=float(points[1][0] * pix_to_m), z=float(points[1][1] * pix_to_m))
            marker_msg.points = [p1, p2]
        else:
            self.get_logger().warn("Need a list of two points.")

        return marker_msg

    def synchronized_callback(self, mono_msg: Image, disparity_msg: Image):
        """Callback executed when synchronized mono and disparity images are received."""
        self.get_logger().info(f"Mono timestamp: {mono_msg.header.stamp.sec}.{mono_msg.header.stamp.nanosec}")
        self.get_logger().info(f"Disparity timestamp: {disparity_msg.header.stamp.sec}.{disparity_msg.header.stamp.nanosec}")

        try:
            # Convert ROS Image messages to OpenCV images
            mono_img = self.bridge_image(mono_msg)
            disparity_img = self.bridge_image(disparity_msg)

            if mono_img is not None and disparity_img is not None:
                cv.imshow('mono', mono_img)
                cv.imshow('disparity', disparity_img)
                cv.waitKey(1)
                # self.line_det_callback(mono_img.copy(), mono_msg.header) # Pass a copy and the header
                # line_points = self.line_det_callback(mono_img.copy(), mono_msg.header) # Get line points
                line_points = self.line_det_callback(mono_img.copy(),disparity_img.copy(), mono_msg.header) # Get line points
                if line_points is not None:
                    # marker_msg = self.visualize_line_polygon(line_points, mono_msg.header.stamp) # Pass the timestamp
                    # self.get_logger().info(f"Publishing Marker with points: {[p.x for p in marker_msg.points]}")
                    # self.marker_publisher_.publish(marker_msg)
                    marker_msg = Marker()
                    marker_msg.header.frame_id = "body"
                    marker_msg.header.stamp = mono_msg.header.stamp
                    marker_msg.type = Marker.LINE_STRIP
                    marker_msg.action = Marker.ADD
                    marker_msg.ns = "line_visualization"  # namespace
                    marker_msg.id = 0
                    marker_msg.scale.x = 0.05
                    marker_msg.color.r = 1.0
                    marker_msg.color.a = 1.0
                    pix_to_m =0.005
                    p1 = Point(x=float(line_points[0][2]), y=float(line_points[0][0] * pix_to_m), z=float(line_points[0][1] * pix_to_m))
                    p2 = Point(x=float(line_points[1][2]), y=float(line_points[1][0] * pix_to_m), z=float(line_points[1][1] * pix_to_m))
                    # p1 = Point(x=float(1.0), y=float(line_points[0][0] * pix_to_m), z=float(line_points[0][1] * pix_to_m))
                    # p2 = Point(x=float(1.0), y=float(line_points[1][0] * pix_to_m), z=float(line_points[1][1] * pix_to_m))
                    # p1 = Point(x=float(1.0), y=float(0.0), z=float(1.0))
                    # p2 = Point(x=float(1.0), y=float(0.0), z=float(-1.0))
                    marker_msg.points = [p1, p2]
                    # marker_msg.points = [Point(x=0.0, y=1.0, z=0.0), Point(x=1.0, y=0.0, z=0.0)]
                    try:
                        self.marker_publisher_.publish(marker_msg)
                        self.get_logger().info(f"Published line Marker with points: {[p.x for p in marker_msg.points],[p.y for p in marker_msg.points]}")
                    except Exception as e:
                        self.get_logger().error(f"Error publishing simple marker: {e}")
            else:
                self.get_logger().warn("Could not convert one or both synchronized images to OpenCV format.")


        except Exception as e:
            self.get_logger().error(f"Error in synchronized_callback: {e}")

    def bridge_image(self, msg: Image) -> np.ndarray or None:
        """ converts ROS Image message to OpenCV image."""
        width = msg.width
        height = msg.height
        encoding = msg.encoding
        if encoding == "mono16":
            data = np.frombuffer(msg.data, dtype=np.uint16)
        else:
            data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == "bgr8":
            cv_image = data.reshape((height, width, 3))
        elif encoding == "rgb8":
            cv_image = data.reshape((height, width, 3))[:, :, ::-1]
        elif encoding == "mono16":
            cv_image = data.reshape((height, width))
        elif encoding == "mono8":
            cv_image = data.reshape((height, width))
        else:
            self.get_logger().warn(f"Unsupported encoding: {encoding}")
            return None
        return cv_image

    def line_det_callback(self, mono_img,disp_img, header):
    # def line_det_callback(self, mono_img, header):
        if mono_img is not None:
            self.get_logger().info('Processing mono image for line detection')
            do_denoise = False
            do_thresh = True
            do_minfilt = True
            do_maxfilt = True
            frameLeftColor = cv.cvtColor(mono_img, cv.COLOR_GRAY2BGR)
            temp_mono_img = mono_img.copy() 

            if do_maxfilt:
                temp_mono_img = ndimage.maximum_filter(temp_mono_img, size=10)
            if do_minfilt:
                temp_mono_img = ndimage.minimum_filter(temp_mono_img, size=5)
            # keep only dark Sectionss
            if do_thresh:
                # TODO here need match warehouse scene (vert structure grey level)
                # thresh = 30 # grey level
                thresh = 50 # grey level
                # thresh = 70 # grey level using rosbag so should be good for warehouse
                temp_mono_img[temp_mono_img >= thresh] = 255
                temp_mono_img[temp_mono_img < thresh] = 0
            if do_denoise:
                ksz = 5
                kernel = np.ones((ksz,ksz),np.uint8)
                temp_mono_img = cv.erode(temp_mono_img,kernel,iterations=2)
            do_contour_det = True

            f,B = 470.051, 0.0750492 # reuse Chobit's warehuse OAK param
            if do_contour_det:
                binary = temp_mono_img.copy()
                inverted_binary = ~binary
                contours, hierarchy = cv.findContours(inverted_binary,
                    cv.RETR_TREE,
                    cv.CHAIN_APPROX_SIMPLE)
                line_points = None
                dist2roi = 10000 # arb large value
                for c in contours:
                    x, y, w, h = cv.boundingRect(c)
                    # TODO here need match warehouse param
                    if h > 100 and (w > 10 and w < 90):
                    # if h > 150 and (w > 50 and w < 100):
                        # structure detected get its depth from disparity
                        lcx = x+w//2 # line center x
                        lcy = y+h//2 # line center y
                        roiw = w//6 # take 1/6 of the detected strcuture, can adjust
                        roih = roiw

                        dist_roi =  disp_img[lcx-roiw:lcx+roiw,lcy-roih:lcy+roih] 
                        dist_roi =  dist_roi[dist_roi > 140] # 140: disparity corresponding to roughly 2m
                        dist2roi = (f * B *8) / np.mean(dist_roi)  # i8 =2^3; 0.125 = 1/(2^3) for 3 bits / pixel, = 1 px split in 8 subpixels
                        # dist2roi = (f * B ) / (np.mean(dist_roi) * 0.125) # 0.125 = 1/(2^3) for 3 bits / pixel, = 1 px split in 8 subpixels
                        # dist2roi = (f * B) / np.mean(disp_img[lcx-roiw:lcx+roiw,lcy-roih:lcy+roih])
                        lspt1 = (lcx,y,dist2roi) # 3D point, z = 0
                        lspt2 = (lcx,y+h,dist2roi)
                        # lspt1 = (lcx,0,0) # 3D point, z = 0
                        # lspt2 = (lcx,h,0)
                        line_points = [lspt1,lspt2]
                        cv.line(frameLeftColor,lspt1[:-1],lspt2[:-1],(255,0,0),5)
                        # roi where depth is measured
                        tlpt = (lcx - roiw,lcy - roih)
                        brpt = (lcx + roiw,lcy + roih)
                        cv.rectangle(frameLeftColor,tlpt,brpt,(255,0,0),5)
                        # cv.line(frameLeftColor,(0,10),(20,10),(0,255,0),9)

            # for debug
            # cv.line(frameLeftColor,(100,0),(100,200),(0,255,0),5)
            cv.imshow(self.cv_window_name,frameLeftColor)
            if line_points is not None:
                self.get_logger().info(f" vertical line: x = {line_points}, \n dist: {dist2roi}")
                self.publish_line(line_points)
                new_msg = Image()
                new_msg.header = header # Use the header of the synchronized mono image
                new_msg.height = mono_img.shape[0]
                new_msg.width = mono_img.shape[1]
                new_msg.encoding = "bgr8"
                new_msg.step = new_msg.width * 3
                new_msg.data = frameLeftColor.tobytes()

                self.publisher.publish(new_msg)
                self.get_logger().info(f"published image with line = {line_points}")
                return line_points
            else:
                self.get_logger().info(f"no vertical line")


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    cv.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
