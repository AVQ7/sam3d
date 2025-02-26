"""
Segment point clouds using SAM.
"""

from typing import List, Tuple

import cv2
import numpy as np
import PIL.Image as Image
import torch
from matplotlib import pyplot as plt
from ransac import get_bounding_box_ransac
from set_axes_equal import set_axes_equal
from transformers import SamModel, SamProcessor


class SamPointCloudSegmenter():
    def __init__(self, device='cpu', render_2d_results=False):
        self.model: SamModel = SamModel.from_pretrained("facebook/sam-vit-base") # type: ignore
        self.processor: SamProcessor = SamProcessor.from_pretrained("facebook/sam-vit-base") # type: ignore
        self.render_2d_results = render_2d_results
        self.device = device

    def _segment_image(self, image: Image.Image, input_points=None, input_boxes=None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        inputs = self.processor(images=[image], input_points=input_points, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks( # type: ignore
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),      # type: ignore
            inputs["reshaped_input_sizes"].cpu() # type: ignore
        )
        scores = outputs.iou_scores

        # Render intermediate segmentation result
        if self.render_2d_results:
            plt.imshow(image) # type: ignore # It gets converted to a Numpy array.
            if input_points is not None:
                x = [point[0] for point in input_points]
                y = [point[1] for point in input_points]
                plt.scatter(x, y, color='red', label='Input points')
            elif input_boxes is not None:
                for input_box in input_boxes[0]:
                    x1, y1, x2, y2 = input_box
                    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color='blue', label='Input box')
            
            for mask in masks:
                plt.imshow(mask[0, 0].detach().cpu().numpy(), alpha=0.5)

            plt.legend()
            plt.show()

        masks = [mask[0, 0] for mask in masks]

        return (masks, scores)

    def transfer_segmentation(self,
                              segmentation: np.ndarray,
                              base_point_cloud: np.ndarray,
                              supplementary_rgb_image: Image.Image,
                              supplementary_point_cloud: np.ndarray):
        """
        Takes a segmentation mask for a given point cloud, and generates a prompt to segment another point cloud.
        """
        segmented_points: np.ndarray = base_point_cloud[segmentation]
        valid_mask = ~(segmented_points == -10000).any(axis=-1)
        segmented_points = segmented_points[valid_mask] # (n1, 3)

        # Finds nearby points in the supplementary image.
        n1 = segmented_points.shape[0]
        # Filter out invalid points in the supplementary image.
        w = supplementary_rgb_image.width
        h = supplementary_rgb_image.height
        coordinates = np.zeros((h, w, 2))
        coordinates[..., 0] = np.arange(w)[None, :].repeat(h, axis=0)
        coordinates[..., 1] = np.arange(h)[:, None].repeat(w, axis=1)
        valid_mask = ~(supplementary_point_cloud == -10000).any(axis=-1)

        supplementary_point_cloud = supplementary_point_cloud[valid_mask] # (n2, 4)
        coordinates = coordinates[valid_mask]

        # Filter out outliers.
        min_pt, max_pt, _inliers = get_bounding_box_ransac(
            segmented_points,
            min_inliers=len(segmented_points) * 0.95,
            n_hypothetical_inliers=16,
            max_iterations=10
        )

        # print("Considering", supplementary_point_cloud.shape[0], "points in the supplementary point cloud...")

        # radius = np.linalg.norm(max_pt - min_pt, axis=-1) / 2
        # # Filter to only consider points within the radius
        # valid_mask = np.linalg.norm(supplementary_point_cloud - (max_pt + min_pt) / 2, axis=-1) < (radius + max_distance)
        # supplementary_point_cloud = supplementary_point_cloud[valid_mask]
        # coordinates = coordinates[valid_mask]

        # print("Filtered to", supplementary_point_cloud.shape[0], "points in the supplementary point cloud...")

        # n2 = supplementary_point_cloud.shape[0]
        # # Calculates distance between all pairs of points.
        # # Could definitely be optimized. Matrix shape: (n1, n2)
        # dists = np.linalg.norm(segmented_points[:, None, :].repeat(n2, axis=1) - supplementary_point_cloud[None, :, :].repeat(n1, axis=0), axis=-1)
        # Get the right coordinates
        # coordinates = coordinates[(dists < max_distance).any(axis=0)]

        # Filter to coordinates that fall in the same bounding box.
        mask = ((min_pt < supplementary_point_cloud) & (supplementary_point_cloud < max_pt)).all(axis=-1)
        coordinates = coordinates[mask]

        if not np.any(mask):
            print("No points found in the bounding box.")
            return (None, None)

        # Construct a bounding box
        x1 = np.min(coordinates[:, 0])
        y1 = np.min(coordinates[:, 1])
        x2 = np.max(coordinates[:, 0])
        y2 = np.max(coordinates[:, 1])

        print("Segmenting based on bounding box", [x1, y1, x2, y2])

        transferred_segmentation = self._segment_image(supplementary_rgb_image, input_boxes=[[[x1, y1, x2, y2]]])

        # TODO: Filter out points that are not in the shadow of the original segmentation.

        return transferred_segmentation

    def segment(self, base_rgb_image: Image.Image, base_point_cloud: np.ndarray, prompt_bounding_box: List[int], supplementary_rgb_images: List[Image.Image], supplementary_point_clouds: List[np.ndarray]):
        """
        Given a base RGB + point cloud image, and a prompt for that image,
        segment the point cloud using the SAM model. Then, fill out the point
        cloud using the other images.
        """

        base_segmentation_masks, base_segmentation_scores = self._segment_image(base_rgb_image, input_boxes=[[prompt_bounding_box]])

        mask = base_segmentation_masks[0].detach().cpu().numpy().astype(np.uint8)
            # Slightly erode the mask to account for segmentation error.
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), mask, iterations=3) # type: ignore
        mask = mask.astype(bool)

        base_segmentation_cloud = base_point_cloud[mask]
        base_segmentation_cloud_color = np.array(base_rgb_image)[mask]
        base_mask = mask

        point_clouds = [base_segmentation_cloud]
        colors = [base_segmentation_cloud_color]
        segmentation_masks = [base_segmentation_masks[0]]

        # Transfer the segmentation to the other point clouds.
        for supplementary_rgb_image, supplementary_point_cloud in zip(supplementary_rgb_images, supplementary_point_clouds):
            (transferred_segmentation_masks, transferred_segmentation_scores) = \
                self.transfer_segmentation(base_mask, base_point_cloud, supplementary_rgb_image, supplementary_point_cloud)
            
            if transferred_segmentation_masks is None:
                continue

            mask = transferred_segmentation_masks[0].detach().cpu().numpy().astype(np.uint8)
            # Slightly erode the mask to account for segmentation error.
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), mask, iterations=3) # type: ignore
            mask = mask.astype(bool)

            point_cloud = supplementary_point_cloud[mask]
            color = np.array(supplementary_rgb_image)[mask]

            # Add the resulting points.
            point_clouds.append(point_cloud)
            colors.append(color)
            segmentation_masks.append(transferred_segmentation_masks[0])

        point_cloud = np.concatenate(point_clouds).reshape(-1, 3)
        color = np.concatenate(colors).reshape(-1, 3)
        valid = ~(point_cloud == -10000).any(axis=-1)
        point_cloud = point_cloud[valid]
        color = color[valid]

        return (np.ascontiguousarray(point_cloud), np.ascontiguousarray(color), segmentation_masks)