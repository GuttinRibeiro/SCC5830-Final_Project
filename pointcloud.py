import open3d as o3d
import os

out_map = o3d.io.read_image(os.path.join('outputs100', '9.jpg'))
color = o3d.io.read_image(os.path.join('outputs100', '9color.jpg'))
raw_map = o3d.io.read_image(os.path.join('outputs100', '9raw.jpg'))

rgbd_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, raw_map, convert_rgb_to_intensity=False)

rgbd_full = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, out_map, convert_rgb_to_intensity=False)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_raw,
    o3d.camera.PinholeCameraIntrinsic(width=544, height=384, fx=525.0, fy=525.0, cx=271.5, cy=191.5))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_full,
    o3d.camera.PinholeCameraIntrinsic(width=544, height=384, fx=525.0, fy=525.0, cx=271.5, cy=191.5))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])