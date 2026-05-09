import os
import cv2
import json
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

def batch_process_dataset(dataset_dir, output_json="yoga_poses_3d.json"):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    dataset_data = []
    
    base_path = Path(dataset_dir)
    
    image_paths = list(base_path.glob('*/*.*')) # Matches folder/image.jpg
    
    print(f"Found {len(image_paths)} images. Beginning extraction...")
    
    for img_path in tqdm(image_paths):
        class_name = img_path.parent.name
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            joints_3d = [[lm.x, lm.y, lm.z] for lm in landmarks]
            
            dataset_data.append({
                "class_name": class_name,
                "image_file": img_path.name,
                "image_path": str(img_path),
                "joints_3d": joints_3d
            })

    with open(output_json, 'w') as f:
        json.dump(dataset_data, f, indent=4)
        
    print(f"\nSuccessfully extracted {len(dataset_data)} poses.")
    print(f"Data saved to {output_json}")

dataset_path = '/home/mustapha/.cache/kagglehub/datasets/shrutisaxena/yoga-pose-image-classification-dataset/versions/1/dataset/'
batch_process_dataset(dataset_path)
