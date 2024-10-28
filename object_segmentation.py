import torch
import numpy as np
import cv2
import open3d as o3d
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision.ops import box_convert
from dataclasses import dataclass
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
import matplotlib.pyplot as plt
import random


class ObjectSegmenter:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_tiny.pt",         
        sam2_model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
        box_threshold=0.35,
        text_threshold=0.25,
    ):
        """
        Initialize the ObjectSegmenter class by loading the models.
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.transform = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # load SAM2 model
        # NOTE: if you are using hydra be careful about the config path
        self.sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        # load Grounding DINO model
        self.grounding_model = load_model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=device
        )

    def prepare_image(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Preprocess the NumPy array image to prepare it for Grounding DINO input.
        """
        image_pil = F.to_pil_image(image_np)
        return self.transform(image_pil).to(self.device)

    def get_masks(self, image: np.ndarray, text_prompt: str):
        """
        Accepts an RGB image and text prompt, returns the segmentation masks for the object of interest.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be an RGB image with shape (H, W, 3).")
        image_tensor = self.prepare_image(image)
        self.sam2_predictor.set_image(image)
        h, w, _ = image.shape
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if len(boxes) == 0:
            # no objects detected
            return None, None, None, None
        
        scale_tensor = torch.tensor([w, h, w, h], device=boxes.device)
        boxes = boxes * scale_tensor
        input_boxes = box_convert(
            boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
        ).cpu().numpy()

        # gets masks for all objects
        with torch.no_grad():
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        labels = [label for label in labels]
        return masks, input_boxes, labels, scores

    def upscale_points(
        self,
        masks: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: np.ndarray,
        rgb_image: np.ndarray,
        points: np.ndarray,
        rgb_points: np.ndarray,
        num_samples: int = 1000,
        upscale_factor: int = 2,  # num of additonal points per original point
        noise_std: float = 0.002  # smaller deviation = less scatter
    ):
        """
        For each mask, sample points within the masked area and generate additional points nearby to increase density.
        
        Args:
            masks (np.ndarray): Segmentation masks (N, H, W).
            depth_image (np.ndarray): Depth image (H, W).
            intrinsics (np.ndarray): Camera intrinsics matrix (3, 3).
            rgb_image (np.ndarray): RGB image (H, W, 3).
            points (np.ndarray): Original point cloud (N, 3).
            rgb_points (np.ndarray): Colors for original points (N, 3).
            num_samples (int): Number of points to sample per mask.
            upscale_factor (int): Number of additional points to generate per original point.
            noise_std (float): Standard deviation for Gaussian noise added to generate new points.
        
        Returns:
            combined_points (np.ndarray): Combined original and upscaled 3D points.
            combined_colors (np.ndarray): Combined colors for the points.
        """
        print(f"Depth image shape: {depth_image.shape}")
        print(f"Intrinsics shape: {intrinsics.shape}")
        print(f"Masks shape: {masks.shape}")

        upscaled_points_list = []
        upscaled_colors_list = []
        original_handle_points_list = []
        original_handle_colors_list = []

        # if there is an extra dim like h, w, _, remove 
        if depth_image.ndim == 3 and depth_image.shape[2] == 1:
            depth_image = depth_image.squeeze(2)
            print(f"Depth image shape after squeezing: {depth_image.shape}")
        
        if depth_image.ndim != 2:
            raise ValueError(f"Expected depth_image to be 2D, but got shape {depth_image.shape}")

        h, w = depth_image.shape

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        for i, mask in enumerate(masks):
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue

            # sample points within the mask
            if len(y_indices) > num_samples:
                sampled_indices = np.random.choice(len(y_indices), num_samples, replace=False)
            else:
                sampled_indices = np.arange(len(y_indices))

            x_sampled = x_indices[sampled_indices]
            y_sampled = y_indices[sampled_indices]

            # depth of the sampled points
            depth_sampled = depth_image[y_sampled, x_sampled]

            # filter out invalid depth values
            valid_depth = depth_sampled > 0
            x_sampled = x_sampled[valid_depth]
            y_sampled = y_sampled[valid_depth]
            depth_sampled = depth_sampled[valid_depth]

            print(f"Mask {i+1}: {len(x_sampled)} valid sampled points")

            if len(depth_sampled) == 0:
                continue

            # from pixel coordinates to camera coordinates
            x_camera = (x_sampled - cx) * depth_sampled / fx
            y_camera = (y_sampled - cy) * depth_sampled / fy
            z_camera = depth_sampled

            sampled_points = np.vstack((x_camera, y_camera, z_camera)).T

            # rgb values for these points
            sampled_colors = rgb_image[y_sampled, x_sampled]

            # add original handle points
            original_handle_points_list.append(sampled_points)
            original_handle_colors_list.append(sampled_colors)

            # generate new points by gaussian sampling
            for _ in range(upscale_factor):
                noise = np.random.normal(0, noise_std, sampled_points.shape)
                new_points = sampled_points + noise
                new_colors = sampled_colors  # just set to same color

                upscaled_points_list.append(new_points)
                upscaled_colors_list.append(new_colors)

        if upscaled_points_list:
            upscaled_points = np.concatenate(upscaled_points_list, axis=0)
            upscaled_colors = np.concatenate(upscaled_colors_list, axis=0)
            print(f"Total upscaled points: {upscaled_points.shape[0]}")
        else:
            upscaled_points = np.array([])
            upscaled_colors = np.array([])
            print("No upscaled points generated.")

        if original_handle_points_list:
            original_handle_points = np.concatenate(original_handle_points_list, axis=0)
            original_handle_colors = np.concatenate(original_handle_colors_list, axis=0)
            print(f"Total original handle points: {original_handle_points.shape[0]}")
        else:
            original_handle_points = np.array([])
            original_handle_colors = np.array([])
            print("No original handle points found.")

        # combine original and upscaled points and colors
        if upscaled_points.size > 0 and original_handle_points.size > 0:
            combined_points = np.concatenate((original_handle_points, upscaled_points), axis=0)
            combined_colors = np.concatenate((original_handle_colors, upscaled_colors), axis=0)
            print(f"Combined point cloud shape: {combined_points.shape}")
        else:
            combined_points = original_handle_points if original_handle_points.size > 0 else upscaled_points
            combined_colors = original_handle_colors if original_handle_colors.size > 0 else upscaled_colors
            print("Combined point cloud includes only original or only upscaled points.")

        return combined_points, combined_colors

    def visualize_masks_bw(self, masks: np.ndarray):
        """
        Visualize segmentation masks as black and white images.
        Args:
            masks (np.ndarray): Segmentation masks (N, H, W).
        """
        num_masks = masks.shape[0]
        cols = min(4, num_masks)
        rows = (num_masks + cols - 1) // cols 

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i in range(num_masks):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(masks[i], cmap='gray')
            plt.title(f"Mask {i+1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_combined_mask(self, masks: np.ndarray):
        """
        Combine all segmentation masks into a single binary mask and visualize it.
        Args:
            masks (np.ndarray): Segmentation masks (N, H, W).
        """
        # combine all the masks into one
        combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)  # (H,W)

        plt.figure(figsize=(8, 8))
        plt.imshow(combined_mask, cmap='gray')
        plt.title("Combined Segmentation Mask")
        plt.axis('off')
        plt.show()

    def visualize_masks_overlay(self, image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, labels: list):
        """
        Visualize segmentation masks overlaid on the original image with bounding boxes.
        Args:
            image (np.ndarray): Original RGB image.
            masks (np.ndarray): Segmentation masks (N, H, W).
            boxes (np.ndarray): Bounding boxes for each mask.
            labels (list): Labels for each mask.
        """
        vis_image = image.copy()

        colors = []
        for _ in range(masks.shape[0]):
            colors.append([random.randint(0, 255) for _ in range(3)])

        for idx, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
            color = colors[idx]
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[mask > 0.5] = color

            # blend the colored mask with the original image
            alpha = 0.5  # transparency
            vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)

            # bounding boxes
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2, cv2.LINE_AA)

        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(vis_image_rgb)
        plt.title("Segmentation Masks Overlay")
        plt.axis('off')
        plt.show()

    def visualize_combined_pointcloud(
        self,
        original_points: np.ndarray,
        original_colors: np.ndarray,
        upscaled_points: np.ndarray,
        upscaled_colors: np.ndarray
    ):
        """
        Visualize combined original and upscaled point clouds with distinct colors.
        
        Args:
            original_points (np.ndarray): Original handle 3D points.
            original_colors (np.ndarray): Colors for original handle points.
            upscaled_points (np.ndarray): Upscaled handle 3D points.
            upscaled_colors (np.ndarray): Colors for upscaled handle points.
        """
        original_pc = o3d.geometry.PointCloud()
        original_pc.points = o3d.utility.Vector3dVector(original_points)
        original_pc.colors = o3d.utility.Vector3dVector(original_colors / 255.0)

        upscaled_pc = o3d.geometry.PointCloud()
        upscaled_pc.points = o3d.utility.Vector3dVector(upscaled_points)
        # assign red color to new points
        upscaled_color_fixed = np.tile([1.0, 0.0, 0.0], (upscaled_points.shape[0], 1))
        upscaled_pc.colors = o3d.utility.Vector3dVector(upscaled_color_fixed)

        combined_pc = original_pc + upscaled_pc

        # add a coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        combined_pc += coord_frame

        o3d.visualization.draw_geometries([combined_pc],
                                          window_name="Combined Original (RGB) & Upscaled (Red) Handles",
                                          width=800,
                                          height=600,
                                          left=50,
                                          top=50,
                                          point_show_normal=False)

    def visualize_interactive(
        self,
        combined_points: np.ndarray,
        combined_colors: np.ndarray
    ):
        """
        Visualize the combined point cloud interactively using Open3D.
        Args:
            combined_points (np.ndarray): Combined 3D points.
            combined_colors (np.ndarray): Combined colors for the points.
        """
        combined_pc = o3d.geometry.PointCloud()
        combined_pc.points = o3d.utility.Vector3dVector(combined_points)
        combined_pc.colors = o3d.utility.Vector3dVector(combined_colors / 255.0)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([combined_pc, coord_frame],
                                          window_name="Combined Original & Upscaled Handles",
                                          width=800,
                                          height=600,
                                          left=50,
                                          top=50,
                                          point_show_normal=False)

    def save_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        filename: str
    ):
        """
        Save a point cloud to a PLY file.
        
        Args:
            points (np.ndarray): 3D points (N, 3).
            colors (np.ndarray): RGB colors (N, 3).
            filename (str): Output filename (e.g., 'combined_handle.ply').
        """
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors / 255.0) # noramlize
        o3d.io.write_point_cloud(filename, pc)
        print(f"Point cloud saved to {filename}")


# small testing script
if __name__ == "__main__":
    data = np.load("pointcloud_data.npz")
    rgb_image = data["rgb_image"]  
    depth_image = data["depth_image"] 
    points = data["points"]
    rgb_points = data["rgb_points"]
    intrinsics = data["intrinsics"]

    H, W = depth_image.shape[:2]
    N = points.shape[0]
    expected_N = H * W
    print(f"Depth Image Shape: {depth_image.shape}")
    print(f"Point Cloud Shape: {points.shape}")
    print(f"Expected Number of Points (H*W): {expected_N}")
    print(f"Actual Number of Points: {N}")
    if N != expected_N:
        print("Warning: Number of points does not match H*W. Check point ordering or data integrity.")

    def visualize_pointcloud(points, colors, window_name="Point Cloud"):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors / 255.0)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        vis.add_geometry(pc)
        vis.run()
        vis.destroy_window()

    print("Visualizing original point cloud")
    visualize_pointcloud(points, rgb_points, window_name="Original Point Cloud")

    segmenter = ObjectSegmenter()

    # NOTE: BE SURE TO ADD A PERIOD TO WHAT YOUR TRYING TO DETECT
    text_prompt = "handles."

    masks, boxes, labels, scores = segmenter.get_masks(rgb_image, text_prompt)

    if masks is None:
        print("No objects detected in the image.")
        exit()

    # some nice visulizaiton methods 
    segmenter.visualize_masks_bw(masks)

    segmenter.visualize_combined_mask(masks)

    segmenter.visualize_masks_overlay(rgb_image, masks, boxes, labels)

    print("Upscaling")
    combined_points, combined_colors = segmenter.upscale_points(
        masks, depth_image, intrinsics, rgb_image, points, rgb_points, 
        num_samples=1000, upscale_factor=2, noise_std=0.001 
    )

    # if needed
    segmenter.save_pointcloud(combined_points, combined_colors, "combined_handle.ply")

    full_pc = o3d.geometry.PointCloud()
    full_pc.points = o3d.utility.Vector3dVector(points)
    full_pc.colors = o3d.utility.Vector3dVector(rgb_points / 255.0)

    handle_pc = o3d.geometry.PointCloud()
    handle_pc.points = o3d.utility.Vector3dVector(combined_points)
    handle_pc.colors = o3d.utility.Vector3dVector(combined_colors / 255.0)
    handle_pc.paint_uniform_color([1.0, 0.0, 0.0])  # assign whatever color you want, in this case (Red)

    combined_scene = [full_pc, handle_pc]

    print("Visualizing full scene with modified handles...")
    o3d.visualization.draw_geometries(
        combined_scene,
        window_name="Full Scene with Modified Handles",
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False
    )
    segmenter.visualize_interactive(combined_points, combined_colors)
