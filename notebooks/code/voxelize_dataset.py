import os
import subprocess
import torch
import sys
from PIL import Image
# adding the path to ensure imports work
sys.path.append("/fhome/vis3d02/TripoSR")

# importing from your local dataloader
from dataloader import dataset_yoga

# bypass broken cudnn initialization
torch.backends.cudnn.enabled = False 
torch.backends.cuda.matmul.allow_tf32 = True

# 1. configuration paths
REPO_PATH = "/fhome/vis3d02/TripoSR"
OUTPUT_BASE_DIR = "/fhome/vis3d02/outputs_triposr"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 2. processing loop
print("\n--- starting triposr processing ---")
os.chdir(REPO_PATH)

for i, (img_path, label_idx) in enumerate(dataset_yoga.samples):
    class_name = dataset_yoga.classes[label_idx]
    file_name = os.path.basename(img_path).split('.')[0]
    
    # create a unique directory for THIS specific image
    current_output_dir = os.path.join(OUTPUT_BASE_DIR, class_name, file_name)
    os.makedirs(current_output_dir, exist_ok=True)
    
    # check if mesh already exists to skip and save time
    final_mesh_path = os.path.join(current_output_dir, f"{file_name}.obj")
    if os.path.exists(final_mesh_path):
        continue

    print(f"[{i+1}/{len(dataset_yoga.samples)}] processing {class_name}: {file_name}")
    
    # temporary rgb image path
    temp_rgb_path = os.path.join(current_output_dir, "input_rgb.png")
    
    try:
        # force rgb conversion
        with Image.open(img_path) as img:
            rgb_img = img.convert('RGB')
            rgb_img.save(temp_rgb_path)

        # run triposr
        # triposr will automatically create current_output_dir/0/
        subprocess.run([
            "python", "run.py", 
            temp_rgb_path, 
            "--output-dir", current_output_dir,
            "--no-remove-bg",
            "--device", "cuda" # using cuda since we disabled cudnn above
        ], check=True)
        
        # fix: find the mesh inside the '0' folder triposr creates
        triposr_generated_mesh = os.path.join(current_output_dir, "0", "mesh.obj")
        
        if os.path.exists(triposr_generated_mesh):
            os.rename(triposr_generated_mesh, final_mesh_path)
            print(f"   success: {final_mesh_path}")
            
            # optional: cleanup the empty '0' folder
            # os.remove(os.path.join(current_output_dir, "0", "input.png"))
            # os.rmdir(os.path.join(current_output_dir, "0"))
            
    except Exception as e:
        print(f"   failed {file_name}: {e}")
    finally:
        if os.path.exists(temp_rgb_path):
            os.remove(temp_rgb_path)

print(f"\nprocessing complete. results in: {OUTPUT_BASE_DIR}")