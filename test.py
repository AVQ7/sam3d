import os
import open3d as o3d
import numpy as np
import cv2
import PIL.Image as Image
from matplotlib import pyplot as plt
from transformers import  SamModel, SamProcessor
import torch
from segment_point_cloud_copy import SamPointCloudSegmenter  # Import the class from your script

# Define the directory path where your files are stored
directory_path = "/home/huron/Downloads"  # Replace with your actual directory path

# List files in the directory
files = os.listdir(directory_path)

# Define the paths for your base and supplementary pairs
base_rgb_image_file = "/home/huron/Downloads/RobotPOV.jpg"
base_point_cloud_file = "/home/huron/Downloads/robotpov.pcd"
supplementary_rgb_image_file = "/home/huron/Downloads/KinectPOV.jpg"
supplementary_point_cloud_file = "/home/huron/Downloads/kinectpov.pcd"

# Assuming one base image-point cloud pair and one supplementary image-point cloud pair
for file in files:
    if file.endswith(".jpg") or file.endswith(".png"):
        if not base_rgb_image_file:
            base_rgb_image_file = os.path.join(directory_path, file)
        else:
            supplementary_rgb_image_file = os.path.join(directory_path, file)
    elif file.endswith(".ply") or file.endswith(".pcd"):
        if not base_point_cloud_file:
            base_point_cloud_file = os.path.join(directory_path, file)
        else:
            supplementary_point_cloud_file = os.path.join(directory_path, file)

# Ensure both the base and supplementary files are found
if not base_rgb_image_file or not base_point_cloud_file or not supplementary_rgb_image_file or not supplementary_point_cloud_file:
    raise ValueError("Could not find the required base and supplementary image-point cloud pairs in the directory.")

# Load the base RGB image
base_rgb_image = cv2.imread(base_rgb_image_file)
base_rgb_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2RGB)
base_rgb_image = cv2.resize(base_rgb_image, (1024, 1024))  # Resize for consistency
base_rgb_image_pil = Image.fromarray(base_rgb_image)

# Load the base point cloud (either .ply or .pcd)
base_pcd = o3d.io.read_point_cloud(base_point_cloud_file)
base_point_cloud_data = np.asarray(base_pcd.points)

# Load the supplementary RGB image
supplementary_rgb_image = cv2.imread(supplementary_rgb_image_file)
supplementary_rgb_image = cv2.cvtColor(supplementary_rgb_image, cv2.COLOR_BGR2RGB)
supplementary_rgb_image = cv2.resize(supplementary_rgb_image, (1024, 1024))  # Resize for consistency
supplementary_rgb_image_pil = Image.fromarray(supplementary_rgb_image)

# Load the supplementary point cloud (either .ply or .pcd)
supplementary_pcd = o3d.io.read_point_cloud(supplementary_point_cloud_file)
supplementary_point_cloud_data = np.asarray(supplementary_pcd.points)

# Example of a bounding box (adjust this as per your requirement)
bounding_box = [300, 300, 700, 700]  # Example coordinates of the bounding box [xmin, ymin, xmin, ymax]

# Create the SamPointCloudSegmenter instance
segmenter = SamPointCloudSegmenter(device='cpu', render_2d_results=True)

# Call the segment method
segmented_points, segmented_colors, segmentation_masks = segmenter.segment(
    base_rgb_image_pil, base_point_cloud_data, bounding_box, 
    [supplementary_rgb_image_pil], [supplementary_point_cloud_data]
)
print("Segmented point", segmented_points)
print("Segmented colors", segmented_colors)
print("Segmented point", segmentation_masks)
# Visualize the segmented point cloud
segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points)
segmented_pcd.colors = o3d.utility.Vector3dVector(segmented_colors / 255.0)  # Normalize colors

# Visualize with Open3D
o3d.visualization.draw_geometries([segmented_pcd.points])

# Optionally, visualize the segmentation mask (2D)
for points in segmentation_masks:
    plt.imshow(points[0, 0].detach().cpu().numpy(), alpha=0.5)
    plt.title("Segmentation Mask")
    plt.show()
