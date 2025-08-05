import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain
import glob
import yaml
import pypose as pp
import torch

def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)
    
class StereoStream:
    def __init__(self, image_dirs, calib, stride = 2, maxsize=8):
        self.queue      = Queue(maxsize=maxsize)
        self.image_dirs = image_dirs

        self.calib = load_yaml(calib)
        self.fx_p, self.fy_p, self.cx_p, self.cy_p = np.array(self.calib['cam0']['intrinsics'])
        self.fx_s, self.fy_s, self.cx_s, self.cy_s = np.array(self.calib['cam1']['intrinsics'])

        self.stride     = stride

        # Load the intrinsics for primary camera
        self.K_p      = np.eye(3)
        self.K_p[0,0] = self.fx_p
        self.K_p[0,2] = self.cx_p
        self.K_p[1,1] = self.fy_p
        self.K_p[1,2] = self.cy_p
        self.distortion_coeffs_p = np.array(self.calib['cam0']['distortion_coeffs'])

        # Load the intrinsics for secondary camera
        self.K_s      = np.eye(3)
        self.K_s[0,0] = self.fx_s
        self.K_s[0,2] = self.cx_s
        self.K_s[1,1] = self.fy_s
        self.K_s[1,2] = self.cy_s
        self.distortion_coeffs_s = np.array(self.calib['cam1']['distortion_coeffs'])

        # Load the extrinsic calibration:
        # Rigid transformation from primary camera frame to secondary camera frame
        self.T_p_to_s = np.array(self.calib['T_cam0_to_cam1']).reshape(4,4)
        T_p_to_s_torch = torch.tensor(self.T_p_to_s, dtype=torch.float32)
        T_p_to_s_pp    = pp.from_matrix(T_p_to_s_torch, ltype=pp.SE3_type, check=False)
        t = T_p_to_s_pp.translation()           # shape (3,)
        quat = T_p_to_s_pp.rotation()
        self.p_to_s = torch.cat([t, quat], dim=0).unsqueeze(0)  # shape (1,7)

        # Collect images
        self.img_exts  = ["*.png", "*.jpeg", "*.jpg"]
        img_pairs = []

        for ext in self.img_exts:
            for f in glob.glob(os.path.join(self.image_dirs[0], ext)):
                fname = os.path.basename(f)
                timestamp = int(os.path.splitext(fname)[0])  # parse timestamp from name
                img_pairs.append((fname, timestamp))

        # Sort by timestamp
        img_pairs.sort(key=lambda x: x[1])

        # Apply stride (skip frames)
        img_pairs = img_pairs[::self.stride]

        # Split into lists
        self.img_list, self.img_times = zip(*img_pairs) if img_pairs else ([], [])
        self.img_list  = list(self.img_list)
        self.img_times = list(self.img_times)

    def stream(self):
        for t, img_name in enumerate(self.img_list):
            # The primary image has to exits
            path_to_primary_image = os.path.join(self.image_dirs[0], img_name)
            image_p      = cv2.imread(str(path_to_primary_image))
            image_p      = cv2.undistort(image_p, self.K_p, self.distortion_coeffs_p)
            intrinsics_p = np.array([self.fx_p, self.fy_p, self.cx_p, self.cy_p])

            # Get the secondary image
            path_to_secondary_image = os.path.join(self.image_dirs[1], img_name)
            image_s      = cv2.imread(str(path_to_secondary_image))
            image_s      = cv2.undistort(image_s, self.K_s, self.distortion_coeffs_s)
            intrinsics_s = np.array([self.fx_s, self.fy_s, self.cx_s, self.cy_s])

            h, w, _ = image_s.shape
            image_s = image_s[:h-h%16, :w-w%16]

            self.queue.put((t, image_p, intrinsics_p, image_s, intrinsics_s, self.p_to_s.clone()))
        self.queue.put((-1, image_p, intrinsics_p, image_s, intrinsics_s, self.p_to_s.clone()))

def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

