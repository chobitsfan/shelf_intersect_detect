#!/usr/bin/env python3
# goal: publish line (tbd: ros2 polygone?) of vertical structure
# tested on rapsberry pi in virtualenvironment envDepthAI
# with an oak camera connected to the pi, run the publisher:
# python mono_preview_pub.py
# in another terminal or tab in tmux, run this file:
# the published image with template is published and can be seen in rviz with topic
# usage 
# run either mono_preview_pub.py to get a live image from the oak,
# or play a relevant rosbag, for example:
# ros2 bag play first_good_test_16042025/rosbag2_2025_04_16-10_22_10/
# 
# TODO to adjust proper filter or threshold, 
# tbd

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

sys.path.insert(0, '/home/pi/Programs/warehouse_nav/lines_and_template')
# sys.path.insert(0, '/home/pi/Programs/warehouse_nav/lines_and_template')

import LSD_utils as mll

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
        self.publisher = self.create_publisher(
            Image,
            '/templatematch_image',  # Publish the image with template to a new topic
            1)
        self.point_publisher = self.create_publisher(
            Point,
            '/templateCOG',  # template Center of Gravity
            10)
        self.pointstamped_pub = self.create_publisher(
            PointStamped,
            '/templateCOG_stampd',  # template Center of Gravity
            10)
        self.polygon_publisher_ = self.create_publisher(Polygon, 'line_polygon', 10)
        self.marker_publisher_ = self.create_publisher(Marker, 'line_marker', 10)
        # self.timer_ = self.create_timer(1.0,self.image_callback)
        self.seq = 0
        if 'DISPLAY' in os.environ:
            self.cv_window_name = "lineDetImage"
            cv.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)  # Allow resizing
        else:
            self.cv_window_name = None
        #Create default parametrization LSD
        self.lsd = cv.createLineSegmentDetector(0)
        self.min_line_length = 150 # should deend on input image height, for now from warehouse inim shape 
                                    # is 288,38470
        init_h_angle = 10
        init_v_angle = 10 # delta from 90°
        self.h_cos_tol = np.cos(init_h_angle*np.pi/180.0)
        self.v_cos_tol0 = np.cos((90-init_v_angle)*np.pi/180.0)
        self.v_cos_tol1 = np.cos((90+init_v_angle)*np.pi/180.0)
        self.up = np.array([[0],[1]])

    def publish_line(self,line_points):
        """get two points and show a line segment """
        # line_points= [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
        # line_points_numpy = np.array([[0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])

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
        self.marker_publisher_.publish(marker_msg_list)

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
        marker_msg.lifetime.sec = 0

        marker_msg.scale.x = 0.05  # Line width
        marker_msg.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green line
        # temprrary trick to scale pixels to world coordinate
        # TODO: convert to point cloud!
        pix_to_m = 0.01

        if len(points) == 2:
            # p1 = Point(x=float(points[0][0]), y=float(points[0][1]), z=float(0))
            # p2 = Point(x=float(points[1][0]), y=float(points[1][1]), z=float(0))
            p1 = Point(x=float(points[0][0]*pix_to_m), y=float(points[0][1]*pix_to_m), z=float(0))
            p2 = Point(x=float(points[1][0]*pix_to_m), y=float(points[1][1]*pix_to_m), z=float(0))
            # p1 = Point(x=float(points[0][0]), y=float(points[0][1]), z=float(points[0][2]))
            # p2 = Point(x=float(points[1][0]), y=float(points[1][1]), z=float(points[1][2]))
            marker_msg.points = [p1, p2]
        else:
            self.get_logger().warn("Input should be a list of two points to visualize a line.")

        return marker_msg

    # compute the line origin at the image border
    # takes a line_mll structure, size of image
    # use a margin so that point is displayed (could be done outside)
    def line_orig(self,line_mll,W,H):
        marg_px = 5
    #    marg = marg_px # in pixels, attention to the equivalent in normlized points when setting the ROI
        a,b,c=line_mll[4],line_mll[5],line_mll[6]
        y=H - marg_px
        x = -1/a*(b*H+c)
        if x>W:
            # need to use the width
            x = W - marg_px
            y = -1/b*(a*W+c)
        if x<0:
            # need to set to 0 + margin
            x =  marg_px
            y = -1/b*(a*W+c)
        return np.asarray([int(x),int(y)])

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
                do_denoise = False
                do_thresh = False
                # keep only dark Sectionss
                if do_thresh:
                    cv_image [cv_image >= 60] = 255
                    cv_image [cv_image <60] = 0
                # denoise
                if do_denoise:
                    kernel = np.ones((5,5),np.uint8)
                    cv_image = cv.dilate(cv_image,kernel,iterations=1)
                    cv_image = cv.erode(cv_image,kernel,iterations=2)
                # cv_image = cv.morphologyEx(cv_image,cv.MORPH_OPEN,kernel)
                print(f'before: cv_image.shape={cv_image.shape}') 
                linesLeft = self.lsd.detect(cv_image)[0] 
                print(f'after: len(linesLeft) = {len(linesLeft)}') 
                linesLeft = mll.from_lsd(linesLeft)

                # filter out the short segments
                linesLeft = linesLeft[linesLeft[...,7] > self.min_line_length]
                # filter out lines within 90° +/- chosen angle
                ab = linesLeft[..., 4:6] * np.array([1, -1]) # coefficient a,b -> normal vector to line
                scal = np.matmul(ab, self.up).reshape(-1)
                #print(scal)
                #print(self.v_cos_tol0,self.v_cos_tol1)
                linesV = linesLeft[(np.abs(scal) < (self.v_cos_tol0)) & (np.abs(scal) > (self.v_cos_tol1))]
                print(f'after filter length: len(linesLeft) = {len(linesLeft)}') 

                frameLeftColor= cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR)
                # if len(linesLeft) is not None:
                if len(linesV) > 2: # is not None:
                    ptc_1 = self.line_orig(linesV[0],frameLeftColor.shape[1],frameLeftColor.shape[0])
                    ptc_2 = self.line_orig(linesV[2],frameLeftColor.shape[1],frameLeftColor.shape[0])
                    # print(f'ptc_1 = {ptc_1}, ptc_2 = {ptc_2}')
                    self.get_logger().info(f" vertical line: x = {ptc_1,ptc_2}")
                    # frameLeftColor= cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR)
                    mll.draw_lines(frameLeftColor, linesV, (200, 20, 20), 3)
                    line_points= [[ptc_1[1], ptc_1[0], 0.0], [ptc_2[1], ptc_2[0], 0.0]]
                    self.publish_line(line_points)
                    # self.publish_line([ptc_1,ptc_2])
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
