

# 👷 Construction Site Safety Detection (AI-Powered)

## AI-Based Detection of Helmets & Safety Vests Using Custom YOLOv8

| 🎖️ **Conference Publication** | 📘 **Research Paper** | 💾 **Repository** | 🧑‍💻 **Lead Developer** |
| :--- | :--- | :--- | :--- |
| **ICCUBEA-2025** | `research_paper_ICCUBEA2025.pdf` | [GitHub Link](https://www.google.com/search?q=https://github.com/%3Cyour-username%3E/Construction-Site-Safety-Detection) | [Chandrakant Thakare](https://www.google.com/search?q=https://github.com/Chandrakant-Thakare) |

-----

## ✨ Project Overview

Construction sites are high-risk zones where timely adherence to **Personal Protective Equipment (PPE)** is vital. Ignoring safety equipment like helmets and vests is a primary cause of severe accidents.

This project introduces a **robust AI-driven monitoring solution** utilizing **custom-trained YOLOv8 models** to automatically detect safety compliance in real-time footage.

### Key Objectives:

  * Identify personnel wearing a **Safety Helmet** ($\mathbf{0}$).
  * Identify personnel wearing a **Reflective Safety Vest** ($\mathbf{1}$).
  * Enable real-time detection on images, video streams, and CCTV footage.

-----

## 🚀 Key Features

  * **Dual YOLOv8 Models:** Independently trained models for helmet and vest detection.
  * **High Accuracy:** Custom-labeled and trained models for superior performance on construction datasets.
  * **Versatile Inference:** Works seamlessly with images, videos, and real-time CCTV frames.
  * **Complete Pipeline:** Full training, evaluation, and inference pipeline documented in a Jupyter Notebook.
  * **Ready-to-Use Data:** Includes the YOLO-formatted (train/valid/test) dataset structure.
  * **Publication Assets:** Official research paper and conference presentation slides included.

-----

## 📂 Project Structure

A clean and logical structure makes the project easy to navigate and reproduce.

```
Construction-Site-Safety-Detection/
│
├── 🧠 enhanced_construction.ipynb        # Core Notebook: Training, Evaluation, & Inference
├── ⚙️ requirements.txt                    # Project Dependencies
├── 💾 data.yaml                          # YOLO Dataset Configuration File
│
├── 📜 research_paper_ICCUBEA2025.pdf     # Published Research Paper (ICCUBEA-2025)
├── 📈 presentation_ICCUBEA2025.pptx      # Conference Presentation Slides
│
├── 📦 models/
│   ├── helmet_detection.pt                # Trained YOLOv8 weights for Helmet Detection
│   └── safety_vest_detection.pt           # Trained YOLOv8 weights for Safety Vest Detection
│
└── 📊 dataset/
    ├── train/images/ & labels/          # Training Data
    ├── valid/images/ & labels/          # Validation Data
    └── test/images/ & labels/           # Testing Data
```

-----

## 🛠️ Installation & Setup

### **1. Clone the Repository**

Open your terminal and execute:

```bash
git clone https://github.com/<your-username>/Construction-Site-Safety-Detection.git
cd Construction-Site-Safety-Detection
```

### **2. Install Dependencies**

All required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

-----

## 🧠 Model Details & Training

The system uses **two highly focused, independently trained YOLOv8** models for maximum performance and clarity.

### Model Architecture

The models are based on the **YOLOv8** architecture, which is known for its balance of speed and accuracy.

| Model File | Purpose | Classes Detected |
| :--- | :--- | :--- |
| `helmet_detection.pt` | Detects **Safety Helmets** | Helmet ($\mathbf{0}$) |
| `safety_vest_detection.pt` | Detects **Reflective Safety Vests** | Safety Vest ($\mathbf{1}$) |

### Training Parameters

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Epochs** | 80–100 | Sufficient for convergence on the custom dataset. |
| **Image Size** | $640 \times 640$ | Optimal size for YOLOv8 inference and training. |
| **Batch Size** | 8 | Standard for GPU training. |
| **Optimizer** | Adam / SGD | Standard deep learning optimizers. |
| **Loss** | YOLO Default Loss | Combination of bounding box and classification loss. |

-----

## 📊 Dataset Structure & Labeling

The custom dataset strictly follows the **YOLO format** for seamless integration with the Ultralytics framework.

### Class Mapping

| Class ID | Object |
| :---: | :--- |
| $\mathbf{0}$ | **Helmet** |
| $\mathbf{1}$ | **Safety Vest** |

### YOLO Format Example

Each image in `images/` has a corresponding `.txt` label file in `labels/` with normalized bounding box coordinates:

```text
# Example label file content:
<class> <x_center> <y_center> <width> <height>
0 0.500 0.300 0.150 0.100  # Helmet
1 0.450 0.700 0.200 0.400  # Safety Vest
```

-----

## ▶️ Running Detection (Inference)

Inference can be run easily using the `ultralytics` library, as shown in the `enhanced_construction.ipynb` notebook.

```python
from ultralytics import YOLO

# Load the desired trained model
model = YOLO("models/helmet_detection.pt")  # Switch to safety_vest_detection.pt for vests

# Run detection on an image or video
results = model("path/to/your/test_image.jpg")

# Display the results with bounding boxes
results.show()
```

-----

## 🏋️‍♂️ Training Your Own Model

To retrain or fine-tune the model, use the provided configuration in the notebook:

```python
from ultralytics import YOLO

# Start from a base model (e.g., YOLOv8n for fast inference)
model = YOLO("yolov8n.pt")

# Initiate training using the data.yaml configuration
model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    name="my_custom_run"
)
```

-----

## 📈 Evaluation Metrics

The notebook automatically calculates and visualizes standard object detection metrics to validate the model's performance:

  * **Precision, Recall, and F1-score**
  * **Mean Average Precision** ($\text{mAP}@50$)
  * **Confusion Matrix**
  * **Training & Validation Loss Curves** (for convergence analysis)

-----

## 📜 Citation (APA Style)

Please cite our published work if you use this project in your research or commercial application:

```
Thakare, C., Jakate, S., & Warme, K. (2025). Enhancing Construction Site Safety Using Detection Models.
In Proceedings of ICCUBEA-2025 (PCCOE Pune).
```

-----

## 🤝 Contributors

  * **Chandrakant Thakare** — Lead Researcher & Developer
  * **Shubhankar Jakate** — Co-Researcher
  * **Kaustubh Warme** — Co-Researcher

-----

## 📄 License

This project is released under the **MIT License**. You are free to use, modify, and distribute the code, provided you include the original copyright and license notice.

-----

