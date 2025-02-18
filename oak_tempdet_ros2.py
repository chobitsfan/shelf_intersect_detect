#!/usr/bin/env python3
# tested on rapsberry pi in virtualenvironment envDepthAI
# with an oak camera connected to the pi, run the publisher:
# python mono_preview_pub.py
# in another terminal or tab in tmux, run the publisher with template detection (this file):
# python 


import cv2 as cv
import depthai as dai
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
import os

# import sys, getopt # no more needed 

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/mono_left',
            self.image_callback,
            10)
        if 'DISPLAY' in os.environ:
            self.cv_window_name = "templateImage"
            cv.namedWindow(self.cv_window_name, cv.WINDOW_NORMAL)  # Allow resizing
        else:
            self.cv_window_name = None
        fn1 = '10x18cm_at_d_90cm.png'
        self.templt_img = cv.imread(cv.samples.findFile(fn1), cv.IMREAD_GRAYSCALE)
        feature_name = 'sift'
        self.detector = cv.SIFT_create()
        norm = cv.NORM_L2
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.matcher = cv.FlannBasedMatcher(flann_params, {})  

        if self.templt_img is None:
            print('Failed to load fn1:', fn1)
            sys.exit(1)
        if self.detector is None:
            print('unknown feature:', feature_name)
            sys.exit(1)

        self.kp1, self.desc1 = self.detector.detectAndCompute(self.templt_img, None)

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
                kp2, desc2 = self.detector.detectAndCompute(cv_image, None)

                # uncomment to show nb of features in the template
                # print('self.templt_img - %d features' % (len(self.kp1)))

                def match_and_draw(win):
                    raw_matches = self.matcher.knnMatch(self.desc1, trainDescriptors = desc2, k = 2) #2
                    p1, p2, kp_pairs = filter_matches(self.kp1, kp2, raw_matches)
                    if len(p1) >= 4:
                        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
                        # uncomment to show nb of inliers and matched features
                        # print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
                    else:
                        H, status = None, None
                        print('%d matches found, not enough for homography estimation' % len(p1))

                    _vis = explore_match_simple(win, self.templt_img, cv_image, kp_pairs, H)

                match_and_draw(self.cv_window_name)
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
        detector = cv.ORB_create(400)
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
def explore_match_simple(win, img1, img2, kp_pairs, H = None):
    '''
    Simplified version of simple match, without mouse interaction 
    and only show a polyline if the template was found.
    win: name of the window for imshow (string)
    img1 : template
    img2: target image
    H: result of findHomography
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    #vis = np.zeros(h2,w2, np.uint8)
    #vis = img2
    vis = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    
    # draw polyline around detected match in the target image
    # TODO: simply draw a rectangle
    if H is not None:
        print(f'H={H}')
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)  )
        cv.polylines(vis, [corners], True, (0, 255, 0),3)
        x, y, w, h = cv.boundingRect(corners)
        cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if win is not None:
        cv.imshow(win, vis)
    


def explore_match(win, img1, frame_gray, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = frame_gray.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = frame_gray
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
        status = status.reshape((len(kp_pairs), 1))
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv.circle(vis, (x1, y1), 2, col, -1)
            cv.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv.line(vis, (x1, y1), (x2, y2), green)

    cv.imshow(win, vis)

    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]

            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i][0]]
                cv.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv.drawKeypoints(cur_vis[:,w1:], kp2s, None, flags=4, color=kp_color)

        cv.imshow(win, cur_vis)
    cv.setMouseCallback(win, onmouse)
    return vis


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)

    # Destroy the window when the node is stopped
    if 'DISPLAY' in os.environ:
        cv.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
