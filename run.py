import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch

from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
import matplotlib.pyplot as plt

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream, StereoStream
from dpvo.utils import Timer
from dpvo import altcorr, fastba, lietorch

from s_dpvo_utils.DataReader import StereoReader

import glob
import numpy as np

from s_dpvo_utils.stereo_raft_bridge import StereoMatcher
SKIP = 0


class LiveTrajectory():
    def __init__(self, ref_path = "/home/haktanito/icra2026/datasets/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"):
        # Read the GROUNDTRUTH
        self.ref_path = ref_path
        self.ref_data = np.loadtxt(self.ref_path, delimiter=",", comments="#")
        self.traj_ref = PoseTrajectory3D(
            positions_xyz=self.ref_data[:,1:4],
            orientations_quat_wxyz=self.ref_data[:, 4:8],
            timestamps=np.array(self.ref_data[:, 0])
        )

        self.scale = 1.0

        # Create the figure once
        plt.ion()  # interactive mode
        self.fig = plt.figure(figsize=(8, 8))
        self.plot_mode = plot.PlotMode.xz
        self.ax = plot.prepare_axis(self.fig, self.plot_mode)

    def getLivePlot(self, traj_est):
        traj_ref_sync, traj_est_sync = sync.associate_trajectories(self.traj_ref, traj_est)

        # Compute APE
        result = main_ape.ape(
            traj_ref_sync, traj_est_sync,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True
        )

        transformation_matrix = result.np_arrays['alignment_transformation_sim3']
        rotation_part = transformation_matrix[0:3, 0:3]
        self.scale = np.linalg.norm(rotation_part[0, :])

        # Clear axis instead of recreating figure
        self.ax.clear()
        self.ax.set_title(f"APE RMSE: {result.stats['rmse']:.4f} m")

        plot.traj(self.ax, self.plot_mode, traj_ref_sync, '--', 'gray', "Reference Trajectory")
        plot.traj(self.ax, self.plot_mode, traj_est_sync, '-', 'blue', "Estimated Trajectory")

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)   # refresh for 0.1s

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, args_sraft, network, image_dirs, calib, stride=1, skip=0, viz=False, timeit=False):

    slam = None
    
    # We need the list of images 
    images_list = sorted(glob.glob(os.path.join(image_dirs[0], "*.png")))[::stride]
    timestamps  = [float(x.split('/')[-1][:-4]) for x in images_list]

    stereo_reader = StereoReader(image_dirs, calib=calib, stride = stride)
    stereo_mathcer = StereoMatcher(args=args_sraft, DEVICE="cuda")

    livePlot = LiveTrajectory()
    for t, (image_p, image_s) in enumerate(stereo_reader):
        disparity, disparity_low = stereo_mathcer.getDisparity(img1=image_p, img2=image_s)
        warped1, valid_mask      = stereo_mathcer.warp_image1_with_stereo_flow()  

        # Convert to NumPy array
        disp = disparity.cpu().numpy()
        disp = np.squeeze(disp)

        # Optional: normalize to 0–255 for display
        disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        disp_norm = np.uint8(disp_norm)

        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # # Blend images
        # alpha = 0.6  # weight for original
        # beta = 1 - alpha  # weight for disparity
        # overlay_p = cv2.addWeighted(image_p, alpha, disp_color, beta, 0)

        cv2.imshow("image_p", image_p)
        cv2.imshow("image_s", image_s)
        cv2.imshow("warped1", warped1)
        # cv2.imshow("overlay_p", overlay_p)
        cv2.imshow("disp_color", disp_color)

        # valid_mask = valid_mask[:, :, None]  # shape becomes (H, W, 1)
        # error = np.abs(warped1.astype(np.float32) - image_s.astype(np.float32)).astype(np.uint8) * valid_mask
        # cv2.imshow("error", error)
 
        key = cv2.waitKey(1)
        if key == ord("q") or key == ord("Q"):
            break
        
        
        image_p = torch.from_numpy(image_p).permute(2,0,1).cuda()
        image_s = torch.from_numpy(image_s).permute(2,0,1).cuda()
        images  = (image_p, image_s, disparity)

        if slam is None:
            _, H, W = image_p.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, images, stereo_reader.intrinsics_rp, stereo_reader.intrinsics_rs, stereo_reader.extrinsics, livePlot=livePlot)

        # Get the estimated poses
        slam.traj = {}
        for i in range(slam.n):
            slam.traj[slam.pg.tstamps_[i]] = slam.pg.poses_[i]

        poses = [slam.get_pose(t) for t in range(slam.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()


        traj_est = PoseTrajectory3D(
            positions_xyz=poses[:,:3],
            orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
            timestamps=np.array(timestamps[:slam.counter])
        )

        if (slam.counter % 50 == 0) and (slam.counter>1):
            livePlot.getLivePlot(traj_est)
            # Path("saved_trajectories").mkdir(exist_ok=True)
            file_interface.write_tum_trajectory_file(f"estimated_trajectory.txt", traj_est)

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir_p', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/')
    parser.add_argument('--imagedir_s', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/')
    parser.add_argument('--calib', type=str, default='/home/haktanito/icra2026/DPVO/calib/euroc.yaml')
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true", default=True)
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--save_colmap', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()



    parser_sraft = argparse.ArgumentParser()
    parser_sraft.add_argument('--restore_ckpt', default='/home/haktanito/icra2026/RAFT-Stereo/models/raftstereo-eth3d.pth', help="restore checkpoint")
    parser_sraft.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser_sraft.add_argument('-l', '--left_imgs',  help="path to all first (left) frames", default="/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/*.png")
    parser_sraft.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/*.png")
    parser_sraft.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser_sraft.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser_sraft.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser_sraft.add_argument('--imagedir_p', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/')
    parser_sraft.add_argument('--imagedir_s', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/')
    parser_sraft.add_argument('--calib', type=str, default='calib/euroc.yaml')

    # Architecture choices
    parser_sraft.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser_sraft.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser_sraft.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser_sraft.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser_sraft.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser_sraft.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser_sraft.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser_sraft.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser_sraft.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args_sraft = parser_sraft.parse_args()


    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)
    img_dirs = (args.imagedir_p, args.imagedir_s)
    (poses, tstamps), (points, colors) = run(cfg, args_sraft, args.network, img_dirs, args.calib, args.stride, args.skip, args.viz, args.timeit)
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    
    
    if args.save_ply:
        save_ply(args.name, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(args.name, trajectory, points, colors)

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {args.name}", filename=f"trajectory_plots/{args.name}.pdf")


        

