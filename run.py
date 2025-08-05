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

import glob
import numpy as np
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
def run(cfg, network, image_dirs, calib, stride=1, skip=0, viz=False, timeit=False):

    slam = None
    
    # We need the list of images 
    images_list = sorted(glob.glob(os.path.join(image_dirs[0], "*.png")))[::stride]
    timestamps  = [float(x.split('/')[-1][:-4]) for x in images_list]

    stereo_stream = StereoStream(image_dirs, calib=calib, stride = stride)
    stereo_reader = Process(target=stereo_stream.stream)
    stereo_reader.start()

    livePlot = LiveTrajectory()
    while 1:
        (t, image_p, intrinsics_p, image_s, intrinsics_s, extrinsics) = stereo_stream.queue.get()
        if t < 0: break

        # cv2.imshow("Frame", image)
        cv2.imshow("Frame2", image_p)
        cv2.waitKey(10)

        image_p = torch.from_numpy(image_p).permute(2,0,1).cuda()
        image_s = torch.from_numpy(image_s).permute(2,0,1).cuda()
        images  = (image_p, image_s)
        intrinsics_p = torch.from_numpy(intrinsics_p).cuda()
        intrinsics_s = torch.from_numpy(intrinsics_s).cuda()
        extrinsics   = extrinsics.cuda()

        if slam is None:
            _, H, W = image_p.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, images, intrinsics_p, intrinsics_s, extrinsics)

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
    stereo_reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics_p, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir_p', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam0/data/')
    parser.add_argument('--imagedir_s', type=str, default='/home/haktanito/icra2026/datasets/MH_01_easy/mav0/cam1/data/')
    parser.add_argument('--calib', type=str, default='calib/euroc.yaml')
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

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)
    img_dirs = (args.imagedir_p, args.imagedir_s)
    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, img_dirs, args.calib, args.stride, args.skip, args.viz, args.timeit)
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    
    
    if args.save_ply:
        save_ply(args.name, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {args.name}", filename=f"trajectory_plots/{args.name}.pdf")


        

