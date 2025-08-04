from itertools import chain
from pathlib import Path
import argparse
from dpvo.stream import StereoReader
from multiprocessing import Process, Queue
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--imagedir_primary', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/')
parser.add_argument('--imagedir_secondary', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/')

parser.add_argument('--calib', type=str, default='calib/euroc.txt')

args = parser.parse_args()

imgdirs = (args.imagedir_primary, args.imagedir_secondary)

data_reader = StereoReader(image_dirs=imgdirs, calib=args.calib)
print(data_reader.img_times)

reader = Process(target=data_reader.stream, args=())
reader.start()

while 1:
    (t, image_p, intrinsics_p, image_s, intrinsics_p) = data_reader.queue.get()
    cv2.imshow("Primary", image_p)
    cv2.imshow("Secondary", image_s)
    key = cv2.waitKey()
    if key == ord("q"):
        break