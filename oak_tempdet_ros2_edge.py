#!/usr/bin/env python3
# republish the image after the template has been detected
# tested on rapsberry pi in virtualenvironment envDepthAI
# with an oak camera connected to the pi, run the publisher:
# python mono_preview_pub.py
# in another terminal or tab in tmux, run the publisher with template detection (this file):
# the published image with template is published and can be seen in rviz with topic
# templatematch_image
# added: publish the centroid of the detected template ROI (both as Point and PointStamped)
# added margin at the border of the image to reduce template search ROI 

import cv2 as cv
# import depthai as dai
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int32
import numpy as np
# for point cloud
from geometry_msgs.msg import Point,PointStamped
import os
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# import sys, getopt # no more needed 

# subscriber then publisher
class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        my_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(
            Image,
            '/mono_left',
            self.image_callback,
            my_qos)
        # subscribe current detect object
        self.edge_value_subscription = self.create_subscription(
            Int32,
            '/intersect_type',
            self.detect_edge_value_callback,
            my_qos)
        self.publisher = self.create_publisher(
            Image,
            '/templatematch_image',  # Publish the image with template to a new topic
            my_qos)
        self.point_publisher = self.create_publisher(
            Point,
            '/templateCOG',  # template Center of Gravity
            1)
        self.pointstamped_pub = self.create_publisher(
            PointStamped,
            '/templateCOG_stampd',  # template Center of Gravity
            1)
        # self.timer_ = self.create_timer(1.0,self.image_callback)

        # 0 = middle structure, 1 = left edge, 2 = right edge
        self.detect_edge_value = 0
        self.seq = 0
        if 'DISPLAY' in os.environ:
            self.cv_window_name = "templateImage"
            cv.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)  # Allow resizing
        else:
            self.cv_window_name = None
        # fn1 = '10x18cm_at_d_90cm.png'
        fn1 = 'img/yangmei_printed.png'
        fn_left = 'img/Ruifang_Down_Template.jpg'
        fn_right = 'img/Ruifang_Top_Template.jpg'
        # fn_left = 'img/yangmei_printed_crop_left.jpg'
        # fn_right = 'img/yangmei_printed_crop_right.jpg'
        # fn_left = 'img/yangmei_edge_left.jpg'
        # fn_right = 'img/yangmei_edge_right.jpg'
        # fn1 = 'cropped_image.png'

        self.templt_img = cv.imread(cv.samples.findFile(fn1), cv.IMREAD_GRAYSCALE)
        self.templt_left_img = cv.imread(cv.samples.findFile(fn_left), cv.IMREAD_GRAYSCALE)
        self.templt_right_img = cv.imread(cv.samples.findFile(fn_right), cv.IMREAD_GRAYSCALE)
        feature_name = 'orb-flann'
        # feature_name = 'sift'
        if feature_name == 'sift':
            self.detector = cv.SIFT_create()
            norm = cv.NORM_L2
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            scoreType = cv.ORB_HARRIS_SCORE
            self.detector = cv.ORB_create(edgeThreshold=10, patchSize=31, 
                                     nlevels=8, fastThreshold=20, 
                                     scaleFactor=1.2, WTA_K=2,
                                     scoreType=scoreType,
                                     firstLevel=0, nfeatures=5000)
            self.detector_edge = cv.ORB_create(edgeThreshold=4, patchSize=31, 
                                     nlevels=4, fastThreshold=10, 
                                     scaleFactor=1.2, WTA_K=2,
                                     scoreType=scoreType,
                                     firstLevel=1, nfeatures=6000)
            # detector = cv.ORB_create(400)
            # check the ORB parameters
            # for attribute in dir(detector):
            #     if not attribute.startswith("get"):
            #         continue
            #     param = attribute.replace("get", "")
            #     get_param = getattr(backend, attribute)
            #     val = get_param()
            #     print(f'param= {val}')
            norm = cv.NORM_HAMMING
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        self.matcher = cv.FlannBasedMatcher(flann_params, {})  
        # self.matcher = cv.BFMatcher(norm)


        if self.templt_img is None:
            print('Failed to load fn1:', fn1)
            sys.exit(1)
        if self.detector is None:
            print('unknown feature:', feature_name)
            sys.exit(1)
        # self.tempcog = np.array([[1.0,2.0,3.0]],dtype=np.float32)

        # TODO: return template COG 
        self.kp1, self.desc1 = self.detector.detectAndCompute(self.templt_img, None)
        self.kp_left, self.desc_left = self.detector_edge.detectAndCompute(self.templt_left_img, None)
        self.kp_right, self.desc_right = self.detector_edge.detectAndCompute(self.templt_right_img, None)
    def detect_edge_value_callback(self, msg):
        try:
            self.detect_edge_value = msg.data
            self.get_logger().info(f'get edge detect value: {self.detect_edge_value}')
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")
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
                # may use reduced ROI
                margn = 0
                if margn == 0:
                    if self.detect_edge_value == 0:
                        kp2, desc2 = self.detector.detectAndCompute(cv_image, None)
                    else:
                        kp2, desc2 = self.detector_edge.detectAndCompute(cv_image, None)
                else:
                    cv_image_roi = cv_image[margn:height - margn,margn:width-margn]
                    if self.detect_edge_value == 0:
                        kp2, desc2 = self.detector.detectAndCompute(cv_image_roi, None)
                    else:
                        kp2, desc2 = self.detector_edge.detectAndCompute(cv_image_roi, None)
                    

                # uncomment to show nb of features in the template
                # print('self.templt_img - %d features' % (len(self.kp1)))

                def match_and_draw(win,margin=0):
                # def match_and_draw(win):
                    p1, p2, kp_pairs, H, status, explore_template_image = [], [], [], None, None, None
                    if self.detect_edge_value == 0:
                        explore_template_image = self.templt_left_img
                        raw_matches = self.matcher.knnMatch(self.desc_left, trainDescriptors = desc2, k = 2) #2
                        # p1, p2, kp_pairs = filter_matches(self.kp1, kp2, raw_matches)
                        p1, p2, kp_pairs = filter_matches(self.kp_left, kp2, raw_matches,ratio = 0.9)
                        if len(p1) >= 4:
                            H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
                            # uncomment to show nb of inliers and matched features
                            # print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
                        else:
                            H, status = None, None
                            print('%d matches found, not enough for homography estimation' % len(p1))
                    else:
                        print('now detect edge')
                        if self.detect_edge_value == 1:
                            explore_template_image = self.templt_left_img
                            raw_matches = self.matcher.knnMatch(self.desc_left, trainDescriptors = desc2, k = 2)
                            p1, p2, kp_pairs = filter_matches(self.kp_left, kp2, raw_matches,ratio = 0.9)
                        elif self.detect_edge_value == 2:
                            explore_template_image = self.templt_right_img
                            raw_matches = self.matcher.knnMatch(self.desc_right, trainDescriptors = desc2, k = 2)
                            p1, p2, kp_pairs = filter_matches(self.kp_right, kp2, raw_matches,ratio = 0.9)
                        if len(p1) >= 4:
                            H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
                        else:
                            H, status = None, None
                            print('%d matches found, not enough for homography estimation' % len(p1))
                    
                    _vis, _tempcog = explore_match_simple(win, explore_template_image, cv_image, kp_pairs, H, margin = margn)

                    # cv.imshow('win',cv_image_roi)
                    if win is not None:
                        cv.imshow(win, _vis)
                    return _vis, _tempcog


                tempmatchimg,tempcog = match_and_draw(self.cv_window_name,margin=margn)

                # new ROS Image message following Chobits' model
                new_msg = Image()
                new_msg.header = msg.header  
                new_msg.height = height
                new_msg.width = width
                new_msg.encoding = "bgr8"  # encoding: color to show the template ROI
                new_msg.step = width * 3  # BGR8: 3 bytes per pixel
                new_msg.data = tempmatchimg.tobytes() # Convert to bytes

                self.publisher.publish(new_msg)

                # publish the template COG as a geometry_msgs
                # add fake depth(for testing function) to point tempcog
                # publish only if a valid COG is found
                if tempcog is not None:
                    tempcog3d = [float(tempcog[0]/100.0), float(tempcog[1]/100.0),1.0]
                    # tempcog3d = np.array([[1.0,2.0,3.0]],dtype=np.float32) # for test
                    print(f'tempcog = {tempcog}, template_COG = {tempcog3d}')
                    # point stamped to show in rviz2
                    pointstamped_msg = PointStamped()
                    pointstamped_msg.header = Header()
                    # pointstamped_msg.header.stamp = self.get_clock().now().to_msg()
                    pointstamped_msg.header.frame_id = 'map'
                    # pointstamped_msg.header.seq = self.seq
                    pointstamped_msg.point.x = tempcog3d[0]
                    pointstamped_msg.point.y = tempcog3d[1]
                    pointstamped_msg.point.z = tempcog3d[2]
                    self.pointstamped_pub.publish(pointstamped_msg)
                    self.get_logger().info(f" template COG stamped: x,y,z = {pointstamped_msg.point.x,pointstamped_msg.point.y,pointstamped_msg.point.z}")
                    # simple point (seems can't be shown in rviz)
                    point_msg = Point()
                    # normalized pixel coordinate
                    point_msg.x = tempcog[0] / width
                    point_msg.y = tempcog[1] / height
                    point_msg.z = 0
                    self.point_publisher.publish(point_msg)
                    self.get_logger().info(f" template COG: x = {point_msg.x,point_msg.y,point_msg.z}")
                else:
                    self.get_logger().info(f"no valide template COG")
                if self.cv_window_name is not None:
                    cv.waitKey(1)
            else:
                self.get_logger().error("Could not create OpenCV image. Check encoding and data.")

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv.SIFT_create()
        #detector = cv.SIFT_create(edgeThreshold=0.01)
        norm = cv.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv.xfeatures2d.SURF_create(800)
        norm = cv.NORM_L2
    elif chunks[0] == 'orb':
        # detector = cv.ORB_create(400)
        scoreType = cv.ORB_HARRIS_SCORE
        detector = cv.ORB_create(edgeThreshold=10, patchSize=31, 
                                 nlevels=8, fastThreshold=20, 
                                 scaleFactor=1.2, WTA_K=2,
                                 scoreType=scoreType,
                                 firstLevel=0, nfeatures=5000)
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

# TODO: seems win is not used here, cleanup
def explore_match_simple(win, img1, img2, kp_pairs, H = None,margin=0):
    '''
    Simplified version of simple match, without mouse interaction 
    and only show a polyline if the template was found.
    win: name of the window for imshow (string)
    img1 : template
    img2: target image
    H: result of findHomography
    margin
    return the visible image with template drown and the centroid (COG) of the template
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    asp_ratio = h1/w1 # aspect ratio of the input template image
    min_asp_ratio,max_asp_ratio = 0.9 * asp_ratio, 1.1*asp_ratio # min and max acceptable  aspect ratio
    min_h1,max_h1 = 70,150 # maximum apparent height of the tmplate in the image
    min_corner_angle,max_corner_angle = 75,110 # min and maximum acceptable angles of the corners of the matched template polygone

    vis = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    
    # draw polyline around detected match in the target image
    # simply draw a rectangle
    tempcog = None
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)  ) + margin
        # corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)  )
        cv.polylines(vis, [corners], True, (0, 255, 0),3)
        # x, y, w, h = cv.boundingRect(corners)
        # print(f'bng rect ={x,y,w,h} ')
        # cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2) # keep here for debg for now
        # if (h > min_h1) and (h < max_h1):
        #     if (w >  min_h1 / asp_ratio) and  (w < max_h1 / asp_ratio): 
        #         # if (h/w > min_asp_ratio) and (h/w < max_asp_ratio):
        #             # tempcog = (int(x+w/2),int(y+h/2))
        #         tempcog = (int(x+w/2),int(y+h/2))
        #
        # cv.circle(vis,tempcog,10,(200,0,0),-1,)
        # print(f'H={H},template centroid={tempcog}')
        # print(f'corners={corners,[corners]}')
        # compute angles and side length of polygon
        vertice_angles = []
        side_lengths = []
        nb_corners = len(corners)
        for i in range(nb_corners):
            # side lengths
            p1 = corners[i]
            p2 = corners[(i + 1) % nb_corners]  
            sidelen = np.linalg.norm(p2 - p1)
            side_lengths.append(sidelen)
            # angles
            p_prev = corners[(i - 1) % nb_corners]  # Previous corner
            p_curr = p1
            p_next = p2
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            dot_product = np.dot(v1, v2)
            magnitude_v1 = np.linalg.norm(v1)
            magnitude_v2 = np.linalg.norm(v2)
            cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
            angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) #clip to handle floating point errors
            angle_deg = np.degrees(angle_rad)
            vertice_angles.append(angle_deg)

        print(f'angles = {vertice_angles}, side length = {side_lengths}')
        # don't use the right up bounding rectangle anymore
        # x, y, w, h = cv.boundingRect(corners)
        # cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # tempcog = (int(x+w/2),int(y+h/2))
        tl_corner_x = np.min(corners[:,0])
        tl_corner_y = np.min(corners[:,1])
        avg_side1 = (side_lengths[0]+side_lengths[2]) /2
        avg_side2 = (side_lengths[1]+side_lengths[3]) /2
        poly_h = max(avg_side1,avg_side2) # assumes h > w !
        poly_w = min(avg_side1,avg_side2)
        cog_x = int(tl_corner_x + poly_w/2)
        cog_y = int(tl_corner_y + poly_h/2)
        # centroid from 
        # https://stackoverflow.com/a/75699662/6358973
        if nb_corners >=3:
            polygon2 = np.roll(corners, -1, axis=0)
            signed_areas = 0.5 * np.cross(corners, polygon2)
            centroids = (corners + polygon2) / 3.0
            if abs(sum(signed_areas)) < 1e-6:
                signed_areas[0] += 0.1
            # tempcog = (cog_x,cog_y)
            if np.all(np.asarray(vertice_angles) < 105.0) and np.all(np.asarray(vertice_angles) > 75.0):
                # if np.all(np.asarray(side_lengths) < 80) and np.all(np.asarray(side_lengths) > 20):
                # if np.all(np.asarray(side_lengths) < 200) and np.all(np.asarray(side_lengths) > 40):
                centroid = np.average(centroids, axis=0, weights=signed_areas)
                tempcog = (int(centroid[0]),int(centroid[1])) 
                cv.circle(vis,tempcog,10,(200,0,0),-1)
        
    # cv.imshow(win, vis)
    return vis,tempcog
    

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
