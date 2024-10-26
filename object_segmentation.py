import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T
import numpy as np
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
import cv2

class ObjectSegmenter:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_tiny.pt", # can adjust, but small = fast
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

        sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        self.grounding_model = load_model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=device
        )

    def prepare_image(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Preprocesses the NumPy array image to prepare it for Grounding DINO input.
        """
        image_pil = F.to_pil_image(image_np)
        return self.transform(image_pil).to(self.device)

    def get_masks(self, image: np.ndarray, text_prompt: str):
        """
        Accepts an RGB image and text prompt, returns the segmentation masks for the object of interest.

        Args:
            image (np.ndarray): Input RGB image as a NumPy array of shape (H, W, 3).
            text_prompt (str): Text prompt for the object of interest.

        Returns:
            masks (np.ndarray): Binary masks for the object, shape (N, H, W), where N is the number of objects detected.
            boxes (np.ndarray): Bounding boxes for objects in XYXY format, shape (N, 4).
            labels (list): List of labels corresponding to each mask.
            scores (np.ndarray): Confidence scores for each mask.
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
            # No objects detected
            return None, None, None, None

        scale_tensor = torch.tensor([w, h, w, h], device=boxes.device)  # Ensure scale tensor is on the same device as boxes
        boxes = boxes * scale_tensor
        input_boxes = box_convert(
            boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
        ).cpu().numpy()

        # for faster inference
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True  # Corrected line

            # get masks for all objects
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

    def upscale_points(self, masks: np.ndarray, num_samples: int = 500):
        """
        For each mask, sample points densely within the masked area.

        Args:
            masks (np.ndarray): Binary masks, shape (N, H, W).
            num_samples (int): Number of points to sample within each mask.

        Returns:
            List of arrays of sampled points for each mask.
        """
        upscaled_points = []
        for mask in masks:
            coords = np.column_stack(np.where(mask == 1))
            if coords.size > 0:
                sampled_indices = np.random.choice(coords.shape[0], num_samples, replace=True)
                sampled_points = coords[sampled_indices]
                upscaled_points.append(sampled_points)
            else:
                upscaled_points.append(np.array([]))  # No points to sample
        return upscaled_points


# ex for how to use

# image_np = cv2.imread('/home/raymond/projects/Grounded-SAM-2/notebooks/images/handle.jpg')
# image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Convert to RGB

# segmenter = ObjectSegmenter()

# text_prompt = "handle."

# # Get masks for the object
# masks, boxes, labels, scores = segmenter.get_masks(image_np, text_prompt)

# if masks is not None:
#     upscaled_points = segmenter.upscale_points(masks, num_samples=1000)

#     for i, points in enumerate(upscaled_points):
#         print(f"Object {i+1}: {len(points)} points sampled.")
# else:
#     print("No objects detected in the image.")
