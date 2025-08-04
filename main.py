#all the main code is in this file
#This script serves as the main entry point for generating 3D models from text or images

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from text_to_3d import generate_3d_from_text
from image_to_3d import generate_3d_from_image
import os

os.makedirs("outputs", exist_ok=True)
print("=== 3D Model Generator ===")
mode = input("Choose input type (text/image): ").strip().lower()
if mode == "text":
    prompt = input("Enter your text prompt (e.g., 'a red apple'): ")
    generate_3d_from_text(prompt)
elif mode == "image":
    image_path = input("Enter path to input image (e.g., corgi.png): ").strip()
    if not os.path.exists(image_path):
        print("[ERROR] Image not found.")
    else:
        generate_3d_from_image(image_path)
else:
    print("[ERROR] Invalid input type.")
