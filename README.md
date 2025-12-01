

```markdown
# 🏗️ Construction Site Safety Detection  
### AI-Based Detection of Construction Helmets & Safety Vests Using YOLO Models  

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()  
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange)]()  
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)]()  
[![Conference](https://img.shields.io/badge/Published-ICCUBEA--2025-green)]()

---

## 📘 **Project Overview**
Construction sites are high-risk environments where ignoring basic safety equipment can lead to severe accidents.  
This project introduces an **AI-driven safety monitoring system** capable of detecting:

- 🪖 **Safety Helmets**  
- 🦺 **Reflective Safety Vests**  

using **custom-trained YOLOv8 models**.

The solution works on **images, video frames, and CCTV footage**, and can be integrated into real-time monitoring systems to improve safety compliance.

This project was **presented and published at ICCUBEA-2025 (PCCOE Pune)** and includes the official research paper and presentation.

---

## 🎯 **Key Features**
- ✔ YOLOv8-based Helmet & Vest detection  
- ✔ High-accuracy custom-trained models  
- ✔ Works with images, videos, and CCTV frames  
- ✔ Full training pipeline implemented in Jupyter Notebook  
- ✔ YOLO-formatted dataset (train/valid/test) included  
- ✔ Pretrained model weights (`.pt` files) provided via Git LFS  
- ✔ Research paper and conference presentation included  

---

## 🏆 **Conference Publication**
**Conference:** ICCUBEA-2025 — International Conference on Computing, Communication, Control & Automation  
**Institution:** Pimpri Chinchwad College of Engineering (PCCOE), Pune  

Files included:
- `research_paper_ICCUBEA2025.pdf`
- `presentation_ICCUBEA2025.pptx`

---

## 📂 **Project Structure**
```

Construction-Site-Safety-Detection/
│── enhanced_construction.ipynb      # Core notebook: training + evaluation + inference
│── requirements.txt                 # Required Python libraries
│── data.yaml                        # YOLO dataset configuration
│── research_paper_ICCUBEA2025.pdf   # Published research paper
│── presentation_ICCUBEA2025.pptx    # Conference presentation (optional)

│── models/
│     ├── helmet_detection.pt        # YOLOv8 trained model for helmets
│     └── safety_vest_detection.pt   # YOLOv8 trained model for vests

│── dataset/
│     ├── train/
│     │     ├── images/
│     │     └── labels/
│     ├── valid/
│     │     ├── images/
│     │     └── labels/
│     └── test/
│           ├── images/
│           └── labels/

│── .gitignore
└── README.md

````

---

## 📦 **Installation**
### **1️⃣ Clone the repository**
```bash
git clone https://github.com/<your-username>/Construction-Site-Safety-Detection.git
cd Construction-Site-Safety-Detection
````

### **2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🧠 **Model Details**

This project uses **two independently trained YOLOv8 models**:

| Model                      | Purpose                         |
| -------------------------- | ------------------------------- |
| `helmet_detection.pt`      | Detects safety helmets          |
| `safety_vest_detection.pt` | Detects reflective safety vests |

Training Details:

* Epochs: 80–100
* Image size: 640×640
* Batch size: 8
* Optimizer: Adam / SGD
* Loss: YOLO default loss
* Dataset: Custom-labeled dataset (train/valid/test split)

---

## 🎓 **Dataset Structure**

Dataset follows the **YOLO format**:

### Each split contains:

* `images/` — image files
* `labels/` — YOLO bounding box text files

### Class Mapping:

```
0 = Helmet
1 = Safety Vest
```

### Example folder layout:

```
dataset/train/images/
dataset/train/labels/
dataset/valid/images/
dataset/valid/labels/
dataset/test/images/
dataset/test/labels/
```

You can replace this dataset with your own following the same structure.

---

## ▶️ **Running Detection (Inference)**

You can run inference **directly inside the notebook**:

```python
from ultralytics import YOLO

model = YOLO("models/helmet_detection.pt")   # or safety_vest_detection.pt
results = model("test_image.jpg")
results.show()
```

The notebook (`enhanced_construction.ipynb`) includes:

* Inference examples
* Visualization of detection results
* Training logs & metrics

---

## 🏋️‍♂️ **Training Your Own Model**

Inside the notebook, you can retrain with:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Choose base model version
model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=8
)
```

---

## 📈 **Evaluation Metrics**

The notebook includes:

* ✔ Precision
* ✔ Recall
* ✔ F1-score
* ✔ mAP@50
* ✔ Confusion Matrix
* ✔ Training & validation loss curves

These metrics validate the performance and generalization of the trained models.

---


## 📜 **Citation (APA Style)**

```
Thakare, C., Jakate, S., & Warme, K. (2025). Enhancing Construction Site Safety Using Detection Models.
In Proceedings of ICCUBEA-2025 (PCCOE Pune).
```

---

## 🤝 **Contributors**

* **Chandrakant Thakare** — Lead Researcher & Developer
* **Shubhankar Jakate** — Co-Researcher
* **Kaustubh Warme** — Co-Researcher

---

## 📄 **License**

This project is released under the **MIT License**.
You may use, modify, and distribute it with proper credit.

---

## ⭐ **Support the Project**

If you found this project helpful, please ⭐ **star the repository** — your support encourages further research and development!

```


