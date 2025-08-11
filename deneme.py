import cv2
from s_dpvo_utils.DataReader import StereoReader

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
