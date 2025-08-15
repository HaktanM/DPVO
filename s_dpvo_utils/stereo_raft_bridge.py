import sys
sys.path.append('/home/haktanito/icra2026/RAFT-Stereo/core')
sys.path.append('/home/haktanito/icra2026/RAFT-Stereo')
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
import torch
import numpy as np
import torch.nn.functional as F


class StereoMatcher:
    def __init__(self, args, RES=4, DEVICE="cuda"):
        
        self.args = args
        self.DEVICE = DEVICE
        self.RES = RES

        self.model = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
        self.model.load_state_dict(torch.load(self.args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

    def getDisparity(self, img1, img2):
        self.img1 = self.img2torch(img1)
        self.img2 = self.img2torch(img2)

        # Dowsample the image
        self.img1 = F.interpolate(self.img1, scale_factor=1/self.RES, mode='bilinear', align_corners=False)
        self.img2 = F.interpolate(self.img2, scale_factor=1/self.RES, mode='bilinear', align_corners=False)

        self.padder = InputPadder(self.img1.shape, divis_by=32)
        img1, img2 = self.padder.pad(self.img1, self.img2)

        flow_down, flow_up = self.model(img1, img2, iters=self.args.valid_iters, test_mode=True)
        self.disparity = self.padder.unpad(flow_up).squeeze()
        flow_down = flow_down.squeeze()

        return self.disparity, flow_down[0]
    
    def img2torch(self, img):
        img = img.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.DEVICE)
    

    def warp_image1_with_stereo_flow(self):
        """
        Warp an image using a predicted optical flow.
        
        Args:
            image (torch.Tensor): Tensor of shape [C, H, W] in the same device as flow.
            flow (torch.Tensor): Tensor of shape [2, H, W] containing (u, v) displacements.
        
        Returns:
            torch.Tensor: Warped image of shape [C, H, W].
        """
        # Extract dimensions
        _, _, H, W = self.img1.shape
        
        # Create mesh grid of pixel coordinates
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x = x.to(self.disparity.device)
        y = y.to(self.disparity.device)
        
        # Add flow to pixel coordinates
        x_new = x - self.disparity
        y_new = y
        
        # Normalize coordinates to [-1, 1]
        x_norm = 2 * (x_new / (W - 1)) - 1
        y_norm = 2 * (y_new / (H - 1)) - 1
        
        # Stack and reshape for grid_sample
        grid = torch.stack((x_norm, y_norm), dim=-1)  # [H, W, 2]
        
        # Warp the image
        image_batch = self.img1  # [1, C, H, W]
        warped = F.grid_sample(image_batch, grid.unsqueeze(0), align_corners=True)
        warped = warped.squeeze(0)
        warped = warped.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # -------------------------
        # Create validity mask
        # -------------------------
        mask_ones = torch.ones((1, 1, H, W), device=self.disparity.device)
        mask_warped = F.grid_sample(mask_ones, grid.unsqueeze(0), align_corners=True)

        # Convert mask to binary (valid=1, invalid=0)
        valid_mask = (mask_warped > 0.999).squeeze().cpu().numpy().astype(np.uint8)  # [H, W]

        
        return warped, valid_mask
    

    def warp_image2_with_stereo_flow(self):
        """
        Warp an image using a predicted optical flow.
        
        Args:
            image (torch.Tensor): Tensor of shape [C, H, W] in the same device as flow.
            flow (torch.Tensor): Tensor of shape [2, H, W] containing (u, v) displacements.
        
        Returns:
            torch.Tensor: Warped image of shape [C, H, W].
        """
        # Extract dimensions
        _, _, H, W = self.img2.shape
        
        # Create mesh grid of pixel coordinates
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x = x.to(self.disparity.device)
        y = y.to(self.disparity.device)
        
        # Add flow to pixel coordinates
        x_new = x + self.disparity
        y_new = y
        
        # Normalize coordinates to [-1, 1]
        x_norm = 2 * (x_new / (W - 1)) - 1
        y_norm = 2 * (y_new / (H - 1)) - 1
        
        # Stack and reshape for grid_sample
        grid = torch.stack((x_norm, y_norm), dim=-1)  # [H, W, 2]
        
        # Warp the image
        image_batch = self.img2  # [1, C, H, W]
        warped = F.grid_sample(image_batch, grid.unsqueeze(0), align_corners=True)
        warped = warped.squeeze(0)
        warped = warped.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        return warped
        