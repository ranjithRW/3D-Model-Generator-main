# 🧠 Photo or Text to 3D Model Generator – Prototype Assignment

This project is a functional **Python prototype** that generates simple 3D models (`.obj` files) from either:
- A **text prompt** (e.g., "a red apple")
- Or a **photo of an object** (e.g., a `.jpg` or `.png` image of a chair, toy, car, etc.)

It fulfills the requirements for the intern selection assignment round by demonstrating:
- AI-based 3D generation using open-source tools
- Clean, modular code structure
- Image preprocessing and 3D output handling
- The ability to work independently and document the solution

---

## 🚀 Features

✅ Accepts **text or image** input  
✅ Generates `.obj` 3D mesh models using **OpenAI's Shap-E**  
✅ Saves outputs locally for download or 3D printing  
✅ Runs on CPU and GPU (CUDA support optional)  
✅ Clean CLI interface and extensible architecture

---

## 🧰 Tech Stack

- **Language**: Python 3.8+
- **Model**: [Shap-E by OpenAI](https://github.com/openai/shap-e)
- **Libraries**: PyTorch, Pillow, Trimesh, NumPy, Matplotlib, TQDM

---

## 📁 Project Structure

photo_text_to_3d/
├── main.py # CLI interface to choose text/image input
├── text_to_3d.py # Generates 3D model from text prompt
├── image_to_3d.py # Generates 3D model from image file
├── outputs/ # Stores generated .obj models
├── requirements.txt # All dependencies
└── README.md # You're reading this

---

## 📸 How It Works

### 🧾 1. Input
- `Text prompt`: e.g., "a red apple"
- `Image`: clear single-object photo (`.jpg` or `.png`)

### ⚙️ 2. Processing
- **Text input** uses Shap-E’s `text300M` model to generate latent 3D space
- **Image input** uses Shap-E’s `image300M` model with optional preprocessing
- Latents are decoded into a **triangle mesh**
- The `.obj` file is saved to the `outputs/` directory

### 📦 3. Output
- `.obj` 3D file viewable in any 3D viewer (e.g., Blender, Meshlab)
- File is named based on input prompt or image index

---

## ▶️ Example Usage

### 🔡 Text Input

```bash
$ python main.py
Choose input type (text/image): text
Enter your text prompt: a red apple
[SUCCESS] 3D model saved at outputs/text_a_red_apple_0.obj

My Thought Process

I chose Shap-E because it supports both text-to-3D and image-to-3D generation using diffusion.

I modularized the code to separate CLI, text logic, and image logic for clarity.

I ensured generated models are saved with meaningful names and can be viewed in Blender or other mesh viewers.

Focused on simplicity, extensibility, and clarity of logic to show initiative and understanding.

⚙️ Installation
Step 1: Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

📦 Dependencies
See requirements.txt for full list:

torch
trimesh
pillow
matplotlib
numpy
scipy
tqdm
shap-e (installed via Git)

📌 Notes
This prototype was tested on macOS M2, but runs on Windows/Linux with Python 3.8+.

CUDA is supported (if available), otherwise runs on CPU.

Output .obj files are stored in the /outputs folder.



# 3D-Model-Generator
# 3D-Model-Generator
