import cv2 as cv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
# for point cloud
from geometry_msgs.msg import Point, PointStamped
from geometry_msgs.msg import Polygon, Point32
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import os
import sys

from scipy import ndimage  # for max and min filter

# subscriber then publisher
class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)
        self.mono_subscription = self.create_subscription(
            Image,
            '/mono_left',
            self.mono_callback,
            qos_profile=best_effort_qos)
        self.disparity_subscription = self.create_subscription(
            Image,
            '/disparity_oak',
            self.disparity_callback,
            qos_profile=best_effort_qos)
        self.publisher = self.create_publisher(
            Image,
            '/vert_line_det_img',
            1)
        self.polygon_publisher_ = self.create_publisher(Polygon, 'vert_line_polygon', 10)
        self.marker_publisher_ = self.create_publisher(Marker, 'vert_line_marker', 10)
        self.seq = 0
        if 'DISPLAY' in os.environ:
            self.cv_window_name = "lineDetImage"
            cv.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)  # Allow resizing
            cv.namedWindow('mono', cv.WINDOW_NORMAL)
            cv.namedWindow('disparity', cv.WINDOW_NORMAL)
        else:
            self.cv_window_name = None
        self.mono_left_msg = None
        self.disparity_msg = None

    def publish_line(self, line_points):
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
        self.marker_publisher_.publish(marker_msg_list)

        # marker_msg_numpy = self.visualize_line_polygon(line_points_numpy)
        # self.get_logger().info(f"Publishing Marker with points (from numpy): {[p.x for p in marker_msg_numpy.points]}")
        # self.marker_publisher_.publish(marker_msg_numpy)

    def visualize_line_polygon(self, points):
        """can't show polygon direcly in rviz, this does it."""
        marker_msg = Marker()
        marker_msg.header.frame_id = "map"
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.ns = "line_visualization"  # namespace
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
            # p1 = Point(x=float(points[0][0]*pix_to_m), y=float(points[0][1]*pix_to_m), z=float(0))
            # p2 = Point(x=float(points[1][0]*pix_to_m), y=float(points[1][1]*pix_to_m), z=float(0))
            p1 = Point(x=float(1.0), y=float(points[0][0] * pix_to_m), z=float(points[0][1] * pix_to_m))
            p2 = Point(x=float(1.0), y=float(points[1][0] * pix_to_m), z=float(points[1][1] * pix_to_m))
            marker_msg.points = [p1, p2]
        else:
            self.get_logger().warn("Need a list of two points.")

        return marker_msg

    def disparity_callback(self, msg):
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
                return  

            if cv_image is not None and not cv_image.size == 0: # Check if image is valid
                self.disparity_img = cv_image.copy()
                print(f'disparity {width}x{height}')
                cv.imshow('disparity', self.disparity_img)
                self.disparity_msg = msg # Store the ROS message
                self.process_images() # Call the processing function
            else:
                self.get_logger().error("Didn't get disparity.")

        except Exception as e:
            self.get_logger().error(f"Error in disparity_callback: {e}")

    def mono_callback(self, msg):
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
                return  

            if cv_image is not None and not cv_image.size == 0: # Check if image is valid
                self.mono_left_img = cv_image.copy()
                print(f'mono {width}x{height}')
                cv.imshow('mono', self.mono_left_img)
                self.mono_left_msg = msg # Store the ROS message
                self.process_images() # Call the processing function
        except Exception as e:
            self.get_logger().error(f"Didnt get monoleft: {e}")

    def process_images(self):
        if self.disparity_msg is not None and self.mono_left_msg is not None:
            # Optional: Check if the timestamps are close enough to be considered synchronized
            mono_stamp = self.mono_left_msg.header.stamp
            disparity_stamp = self.disparity_msg.header.stamp
            time_diff_ns = abs((mono_stamp.sec - disparity_stamp.sec) * 1e9 + (mono_stamp.nanosec - disparity_stamp.nanosec))
            if time_diff_ns < 1e9: # 1 second tolerance (nanosec)
                self.get_logger().info('Got synchronized mono and disparity images')
                self.line_det_callback(self.mono_left_img.copy()) # Pass a copy to avoid modification issues
                # "Reset" images
                self.disparity_msg = None
                self.mono_left_msg = None
            else:
                self.get_logger().warn(f'Mono and disparity images are not synchronized. Time difference: {time_diff_ns / 1e9:.3f} seconds.')
        # Optionally, you could add logic to handle cases where one stream is consistently faster.


    def line_det_callback(self, mono_img):
        if mono_img is not None:
            self.get_logger().info('Processing mono image for line detection')
            do_denoise = False
            do_thresh = True
            do_minfilt = True
            do_maxfilt = True
            frameLeftColor = cv.cvtColor(mono_img, cv.COLOR_GRAY2BGR)
            temp_mono_img = mono_img.copy() # Work on a copy

            if do_maxfilt:
                temp_mono_img = ndimage.maximum_filter(temp_mono_img, size=10)
            if do_minfilt:
                temp_mono_img = ndimage.minimum_filter(temp_mono_img, size=5)
            # keep only dark Sectionss
            if do_thresh:
                # thresh = 70 # grey level at warehouse
                thresh = 50 # grey level on the desk
                temp_mono_img[temp_mono_img >= thresh] = 255
                temp_mono_img[temp_mono_img < thresh] = 0
                # ret, binary = cv.threshold(self.mono_left_img, 60, 255,
            # denoise
            if do_denoise:
                ksz = 5
                kernel = np.ones((ksz,ksz),np.uint8)
                # self.mono_left_img = cv.dilate(self.mono_left_img,kernel,iterations=1)
                temp_mono_img = cv.erode(temp_mono_img,kernel,iterations=2)
            # Convert the grayscale image to binary
                # cv.THRESH_OTSU)
            binary = temp_mono_img.copy()
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
            for c in contours:
                x, y, w, h = cv.boundingRect(c)

                # select contour by areay or height...
                if h > 100 and (w > 10 and w < 90):
                # if h > 150 and (w > 50 and w < 100):
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
            new_msg.header = self.mono_left_msg.header # Use the header of the mono image
            new_msg.height = mono_img.shape[0]
            new_msg.width = mono_img.shape[1]
            new_msg.encoding = "bgr8"  # encoding: color to show the template ROI
            new_msg.step = new_msg.width * 3  # BGR8: 3 bytes per pixel
            new_msg.data = frameLeftColor.tobytes() # Convert to bytes

            self.publisher.publish(new_msg)

        if self.cv_window_name is not None:
            cv.waitKey(1)
        else:
            self.get_logger().error("error TODO be specific!.")


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
