import cv2
import mediapipe as mp
import matplotlib.pyplot as plt


def extract_and_plot_3d_pose(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=2, 
        enable_segmentation=False, 
        min_detection_confidence=0.5
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_world_landmarks:
        landmarks = results.pose_world_landmarks.landmark
        
        xs = [landmark.x for landmark in landmarks]
        ys = [-landmark.y for landmark in landmarks] 
        zs = [landmark.z for landmark in landmarks]

        # Set up the 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(xs, zs, ys, c='red', marker='o', s=50)
        
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection[0], connection[1]
            ax.plot([xs[start_idx], xs[end_idx]], 
                    [zs[start_idx], zs[end_idx]], 
                    [ys[start_idx], ys[end_idx]], 'blue', linewidth=2)

        # Format the plot
        ax.set_xlabel('Left / Right (X)')
        ax.set_ylabel('Depth (Z)')
        ax.set_zlabel('Up / Down (Y)')
        ax.set_title('3D Yoga Pose Skeleton')
        
        # Equalize axes for a proportional view
        ax.set_box_aspect([1, 1, 1]) 
        plt.show()

        print(f"Nose 3D Coordinates (meters): X: {xs[0]:.2f}, Y: {ys[0]:.2f}, Z: {zs[0]:.2f}")
        
    else:
        print("No pose detected in the image. Try another yoga pose.")

extract_and_plot_3d_pose('/home/mustapha/.cache/kagglehub/datasets/shrutisaxena/yoga-pose-image-classification-dataset/versions/1/dataset/padangusthasana/19-0.png')
