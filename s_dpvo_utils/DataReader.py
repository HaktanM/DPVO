import numpy as np
import cv2
import os
import cv2
import numpy as np
import glob
import yaml
import pypose as pp
import torch

def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)
    

class StereoReader:
    def __init__(self, image_dirs, calib, stride = 2, maxsize=8):
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


        # Compute rectification parameters
        # Get the image size
        img_name                = self.img_list[0]
        path_to_primary_image   = os.path.join(self.image_dirs[0], img_name)
        image_sample            = cv2.imread(str(path_to_primary_image))
        image_size              = (image_sample.shape[1], image_sample.shape[0])

        # --- Use OpenCV stereoRectify ---
        flags = cv2.CALIB_ZERO_DISPARITY  # align principal points -> zero vertical disparity
        alpha = 0.0  
        self.R_p_to_rp, self.R_s_rs, self.P1_rect, self.P2_rect, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(
            cameraMatrix1=self.K_p, distCoeffs1=self.distortion_coeffs_p,
            cameraMatrix2=self.K_s, distCoeffs2=self.distortion_coeffs_s,
            imageSize=image_size, R=self.T_p_to_s[:3, :3], T=self.T_p_to_s[:3, 3],
            flags=flags, alpha=alpha
        )

        # --- Compute remap (pixel transforms) ---
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(self.K_p, self.distortion_coeffs_p, self.R_p_to_rp, self.P1_rect, image_size, cv2.CV_32FC1)
        self.map2_x, self.map2_y = cv2.initUndistortRectifyMap(self.K_s, self.distortion_coeffs_s, self.R_s_rs, self.P2_rect, image_size, cv2.CV_32FC1)

        # Get the new intrinsics and extrinsics
        self.K_rp = self.P1_rect[:3,:3]
        self.K_rs = self.P2_rect[:3,:3]

        self.R_p_to_s   = self.T_p_to_s[:3, :3]
        self.R_rp_to_rs = self.R_s_rs @ self.R_p_to_s @ self.R_p_to_rp.transpose()
        self.q_rp_to_rs = pp.from_matrix(torch.from_numpy(self.R_rp_to_rs), ltype=pp.SO3_type, check=False).data.to(torch.float32)
        self.t_rp_in_rs = torch.from_numpy(self.T_p_to_s[:3, 3]).to(torch.float32)
        self.rp_to_rs   = torch.cat([self.t_rp_in_rs, self.q_rp_to_rs], dim=0).unsqueeze(0)  # shape (1,7)

        intrinsics_rp = np.array([self.K_rp[0,0], self.K_rp[1,1], self.K_rp[0,2], self.K_rp[1,2]])
        intrinsics_rs = np.array([self.K_rs[0,0], self.K_rs[1,1], self.K_rs[0,2], self.K_rs[1,2]])

        self.intrinsics_rp = torch.from_numpy(intrinsics_rp).cuda()
        self.intrinsics_rs = torch.from_numpy(intrinsics_rs).cuda()
        self.extrinsics    = self.rp_to_rs.cuda()


    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):

        # Get the image name
        img_name  = self.img_list[idx]

        # Read the primary and secondary images
        path_to_primary_image = os.path.join(self.image_dirs[0], img_name)
        image_p      = cv2.imread(str(path_to_primary_image))
        path_to_secondary_image = os.path.join(self.image_dirs[1], img_name)
        image_s      = cv2.imread(str(path_to_secondary_image))

        # --- Apply to stereo pair ---
        img1_rect = cv2.remap(image_p, self.map1_x, self.map1_y, interpolation=cv2.INTER_LINEAR)
        img2_rect = cv2.remap(image_s, self.map2_x, self.map2_y, interpolation=cv2.INTER_LINEAR)

        return img1_rect, img2_rect


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--imagedir_p', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/')
    parser.add_argument('--imagedir_s', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/')
    parser.add_argument('--calib', type=str, default='/home/haktanito/icra2026/DPVO/calib/euroc.yaml')
    args = parser.parse_args()

    image_dirs = (args.imagedir_p, args.imagedir_s)

    stereo_reader = StereoReader(image_dirs, calib=args.calib, stride = 1)
    
    
    for item in stereo_reader:
        img1_rect, img2_rect = item
        cv2.imshow("img1_rect", img1_rect)
        cv2.imshow("img2_rect", img2_rect)
        key = cv2.waitKey()
        if key == ord("q"):
            break


