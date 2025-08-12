#!/usr/bin/env python3
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
"""

import cv2 as cv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
# for point cloud
from geometry_msgs.msg import Point,PointStamped
from geometry_msgs.msg import Polygon, Point32
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import os
import sys

from scipy import ndimage  # for max and min filter
sys.path.insert(0, '/home/pi/Programs/warehouse_nav/lines_and_template')
# sys.path.insert(0, '/home/pi/Programs/warehouse_nav/lines_and_template')

# subscriber then publisher
class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(
            Image,
            '/mono_left',
            self.image_callback,
            qos_profile=best_effort_qos)
            # 1)
        self.templatecog_pt = self.create_subscription(
            Point,
            '/templateCOG',
            self.tempcog_callback,
            1)
        self.publisher = self.create_publisher(
            Image,
            '/vert_line_det_img',  
            1)
        self.polygon_publisher_ = self.create_publisher(Polygon, 'vert_line_polygon', 10)
        self.marker_publisher_ = self.create_publisher(Marker, 'vert_line_marker', 10)
        # self.timer_ = self.create_timer(1.0,self.image_callback)
        self.seq = 0
        self.latesttempcog = None

        if 'DISPLAY' in os.environ:
            self.cv_window_name = "lineDetImage"
            cv.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)  # Allow resizing
        else:
            self.cv_window_name = None

    def tempcog_callback(self,msg):
        self.latesttempcog = msg
        self.get_logger().info(f"Latest template COG = {msg.x,msg.y}.")

    def publish_line(self,line_points):
        """get two points and show a line segment """
        # line_points= [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
        # line_points_numpy = np.array([[0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])

        print(f'in publish_line:{line_points, line_points[0], line_points[0][0]}')
        polygon_msg_list = Polygon()
        if len(line_points) == 2:
            p1 = Point32(x=float(line_points[0][0]), y=float(line_points[0][1]), z=float(line_points[0][2]))
            p2 = Point32(x=float(line_points[1][0]), y=float(line_points[1][1]), z=float(line_points[1][2]))
            polygon_msg_list.points = [p1, p2]
        else:
            self.get_logger().warn("Input should be a list of two points to define a line polygon.")
        # Publish the Polygon message (not directly visualizable in standard RViz2)
        self.get_logger().info(f"Publishing Polygon with points: {polygon_msg_list.points}")
        self.polygon_publisher_.publish(polygon_msg_list)

        # polygon_msg_numpy = self.define_line_polygon(line_points_numpy.tolist()) # Convert numpy array to list
        # self.get_logger().info(f"Publishing Polygon with points (from numpy): {polygon_msg_numpy.points}")
        # self.polygon_publisher_.publish(polygon_msg_numpy)

        # Publish the Marker message (visualizable as a line in RViz2)
        marker_msg_list = self.visualize_line_polygon(line_points)
        self.get_logger().info(f"Publishing Marker with points: {[p.x for p in marker_msg_list.points]}")
        # self.marker_publisher_.publish(marker_msg_list)

        # marker_msg_numpy = self.visualize_line_polygon(line_points_numpy)
        # self.get_logger().info(f"Publishing Marker with points (from numpy): {[p.x for p in marker_msg_numpy.points]}")
        # self.marker_publisher_.publish(marker_msg_numpy)

    def visualize_line_polygon(self, points):
        """can't show polygon direcly in rviz, this does it."""
        marker_msg = Marker()
        marker_msg.header.frame_id = "map"
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.ns = "line_visualization" # namespace
        marker_msg.id = 0
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD
        marker_msg.lifetime.sec = 0.1

        marker_msg.scale.x = 0.05  # Line width
        marker_msg.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green line
        # temprrary trick to scale pixels to world coordinate
        # TODO: ccould onvert to point cloud
        pix_to_m = 0.01

        if len(points) == 2:
            # reorder so that the line appears as a vertical line in rviz
            # pixel px,py -> rviz x,y,z = value,px,py
            # the polygon is unchanged (screen px,py)
            p1 = Point(x=float(1.0), y=float(points[0][0]*pix_to_m), z=float(points[0][1]*pix_to_m))
            p2 = Point(x=float(1.0), y=float(points[1][0]*pix_to_m), z=float(points[1][1]*pix_to_m))
            marker_msg.points = [p1, p2]
        else:
            self.get_logger().warn("Need a list of two points.")

        return marker_msg

    def image_callback(self, msg):
        try:
            width = msg.width
            height = msg.height
            encoding = msg.encoding
            data = np.frombuffer(msg.data, dtype=np.uint8)  # Convert to NumPy array

            if encoding == "bgr8":
                cv_image = data.reshape((height, width, 3))  # Reshape for BGR
            elif encoding == "rgb8":
                cv_image = data.reshape((height, width, 3))[:, :, ::-1] # Reshape and RGB to BGR
            elif encoding == "mono8":
                cv_image = data.reshape((height, width))  # Reshape for grayscale
            else:
                self.get_logger().warn(f"Unsupported encoding: {encoding}")
                return  # Don't try to display

            if cv_image is not None and not cv_image.size == 0: # Check if image is valid
                                                                # if so run the template detector
                # do_denoise = True
                do_denoise = False
                do_thresh = True
                # do_thresh = False
                do_minfilt = True
                do_maxfilt = True
                frameLeftColor= cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR)
                if do_maxfilt:
                    cv_image = ndimage.maximum_filter(cv_image, size=10)
                if do_minfilt:
                    cv_image = ndimage.minimum_filter(cv_image, size=5)
                # keep only dark Sectionss
                if do_thresh:
                    thresh = 70 # grey level 
                    cv_image [cv_image >= thresh] = 255
                    cv_image [cv_image <thresh] = 0
                    # ret, binary = cv.threshold(cv_image, 60, 255, 
                # denoise
                if do_denoise:
                    ksz = 5
                    kernel = np.ones((ksz,ksz),np.uint8)
                    # cv_image = cv.dilate(cv_image,kernel,iterations=1)
                    cv_image = cv.erode(cv_image,kernel,iterations=2)
                do_contour_det = True

                if do_contour_det:
                    # Convert the grayscale image to binary
                      # cv.THRESH_OTSU)
                    binary = cv_image.copy()
                    # need black bg to detect object contours
                    # so we invert the image
                    inverted_binary = ~binary
                    # get contours on the inverted binary image
                    # Contours are around WHITE blobs.
                    # hierarchy variable contains info on the relationship between the contours
                    contours, hierarchy = cv.findContours(inverted_binary,
                      cv.RETR_TREE,
                      cv.CHAIN_APPROX_SIMPLE)
                    # lspt1, lspt2 = None, None
                    line_points = None
                    search_x_vert = width //2
                    for c in contours:
                      x, y, w, h = cv.boundingRect(c)
                     
                    # select contour by area or height...
                      if latesttempcog is not None:
                          search_x_vert = latesttempcog.x
                      if h > 150 and (w > 50 and w < 100):
                      # if (cv.contourArea(c)) > 10:
                        lspt1 = (x+w//2,0,0) # 3D point, z = 0
                        lspt2 = (x+w//2,h,0)
                        line_points = [lspt1,lspt2]
                        cv.line(frameLeftColor,lspt1[:-1],lspt2[:-1],(255,0,0),5)
                # print(f'before: cv_image.shape={cv_image.shape}') 
                if line_points is not None:
                    self.get_logger().info(f" vertical line: x = {line_points}")
                    self.publish_line(line_points)
                else:
                    self.get_logger().info(f"no vertical line")
                    
                cv.imshow(self.cv_window_name,frameLeftColor)
                # new ROS Image message following Chobits' model
                new_msg = Image()
                new_msg.header = msg.header  
                new_msg.height = height
                new_msg.width = width
                new_msg.encoding = "bgr8"  # encoding: color to show the template ROI
                new_msg.step = width * 3  # BGR8: 3 bytes per pixel
                new_msg.data = frameLeftColor.tobytes() # Convert to bytes

                self.publisher.publish(new_msg)

                # publish the template COG as a geometry_msgs
                # add fake depth(for testing function) to point tempcog
                # publish only if a valid COG is found
                if self.cv_window_name is not None:
                    cv.waitKey(1)
            else:
                self.get_logger().error("Could not create OpenCV image. Check encoding and data.")

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)

    # Destroy the window when the node is stopped
    cv.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
