import sys
sys.path.append('/home/haktanito/icra2026/RAFT-Stereo/core')
sys.path.append('/home/haktanito/icra2026/RAFT-Stereo')
from raft_stereo import RAFTStereo
from utils.utils import InputPadder

from multiprocessing import Process, Queue

import argparse
import glob
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import cv2

from s_dpvo_utils.DataReader import StereoReader

DEVICE = 'cuda'


def load_image(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.imread(imfile, cv2.IMREAD_COLOR_BGR).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def img2torch(img):
    img = img.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    img_dirs = (args.imagedir_p, args.imagedir_s)
    stereo_reader = StereoReader(img_dirs, calib=args.calib, stride = 1)


    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for item in stereo_reader:
            image_p, image_s = item

            image1 = img2torch(image_p)
            image2 = img2torch(image_s)
            
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()


            image1 = img2torch(image_p)
            image2 = img2torch(image_s)
            warped_image1 = warp_image_with_stereo_flow(image=image2, flow=flow_up)
            warped_image1 = warped_image1.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            image1        = image1.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            image2        = image2.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            plt.imsave(output_directory / f"estimated_flow.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')

            cv2.imshow("warped_image1", warped_image1)
            cv2.imshow("image1", image1)
            cv2.imshow("image2", image2)
            key = cv2.waitKey()
            if key == ord("q"):
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', default='/home/haktanito/icra2026/RAFT-Stereo/models/raftstereo-eth3d.pth', help="restore checkpoint")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs',  help="path to all first (left) frames", default="/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/*.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--imagedir_p', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/')
    parser.add_argument('--imagedir_s', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/')
    parser.add_argument('--calib', type=str, default='calib/euroc.yaml')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
