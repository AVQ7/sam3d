import open 3d as o3d         from typing import List
#point_cloud_file = "your_point_cloud_file.ply"  # Replace with your point cloud file
pcd = o3d.io.read_point_cloud(point_cloud_file)
# Convert the point cloud to a numpy array
point_cloud_data = np.asarray(pcd.points)
# Create an RGB image from the point cloud (you could use the point colors if available)
# For simplicity, let's just generate a dummy RGB image (you should replace this with a real RGB image if available)
rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.circle(rgb_image, (120, 120), 50, (255, 0, 0), -1)  # Add a red circle in the image
# Create a bounding box (in the image coordinates) for segmentation
bounding_box = [50, 50, 200, 200]  # Example coordinates of the bounding box
# Create supplementary RGB images (these could be other images related to the scene)
supplementary_rgb_images = [Image.fromarray(rgb_image)]
# Create supplementary point clouds (these could be from different viewpoints)
supplementary_point_clouds = [point_cloud_data]  # Using the same point cloud for simplicity
# Instantiate SamPointCloudSegmenter
segmenter = SamPointCloudSegmenter(device='cpu', render_2d_results=True)
# Segment the point cloud based on the real point cloud data
segmented_points, segmented_colors, segmentation_masks = segmenter.segment(
    Image.fromarray(rgb_image), point_cloud_data, bounding_box, supplementary_rgb_images, supplementary_point_clouds
)
# Visualize the result
plt.imshow(rgb_image)
plt.title("Segmented RGB Image")
plt.show()
# Visualize the segmented point cloud using Open3D
segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points)
segmented_pcd
