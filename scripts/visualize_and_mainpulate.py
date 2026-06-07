import json
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

def load_pose_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def rotate_y(joints, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotation_matrix = np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    return np.dot(joints, rotation_matrix.T)

def scale_skeleton(joints, scale_factor):
    return joints * scale_factor

def translate_skeleton(joints, offset_x, offset_y, offset_z):
    translation = np.array([offset_x, offset_y, offset_z])
    return joints + translation

def plot_3d_skeleton(joints_np, title="3D Pose"):
    connections = mp.solutions.pose.POSE_CONNECTIONS
    
    xs = joints_np[:, 0]
    ys = -joints_np[:, 1] 
    zs = joints_np[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, zs, ys, c='red', marker='o', s=50)

    for start_idx, end_idx in connections:
        ax.plot([xs[start_idx], xs[end_idx]], 
                [zs[start_idx], zs[end_idx]], 
                [ys[start_idx], ys[end_idx]], 'blue', linewidth=2)

    ax.set_xlabel('Left / Right (X)')
    ax.set_ylabel('Depth (Z)')
    ax.set_zlabel('Up / Down (Y)')
    ax.set_title(title)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1]) 
    
    plt.show()

data = load_pose_data('yoga_poses_3d.json')

if len(data) > 0:
    sample = data[0]
    print(f"Visualizing Class: {sample['class_name']} | File: {sample['image_file']}")
    
    joints_np = np.array(sample['joints_3d'])

    plot_3d_skeleton(joints_np, title="Original Pose")

    rotated_joints = rotate_y(joints_np, 90)
    plot_3d_skeleton(rotated_joints, title="Pose Rotated 90° (Y-Axis)")
    
    scaled_joints = scale_skeleton(joints_np, 0.5)
    plot_3d_skeleton(scaled_joints, title="Pose Scaled 50%")
else:
    print("The JSON file is empty. Check your extraction script.")
