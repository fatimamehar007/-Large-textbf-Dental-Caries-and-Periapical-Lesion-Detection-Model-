# Dental Caries and Periapical Lesion Detection using YOLOv8

This project implements an automated detection system for **dental caries (cavities)** and **periapical (PA) lesions** from dental X-ray images using the **YOLOv8s** architecture. The system handles dataset imbalance, image quality issues, and label integrity to build a robust model for real-world dental diagnostics.

## 📁 Dataset Overview

- **Source**: Dental X-ray images with annotations in YOLO format.
- **Classes**:
  - `0`: Cavity
  - `1`: Periapical Lesion (PA)
- **Initial Sample Distribution**:
  - Only Cavity: 2,583
  - Only PA: 552
  - Both: 1,041
- **Imbalance Ratio**: ~3.27 (Cavity:PA)
- **Splits**:
  - Training: 4,488 images
  - Validation: 307 images
  - Test: 206 images

---

## 🧪 Image Analysis

- **Noise Type**: Salt-and-pepper (speckle) noise.
- **Structures Identified**: Teeth (molars), roots, enamel/dentin, potential cavities or lesions.
- **Image Size (avg)**: 512x512
- **Format**: All images converted to `.jpg`.

---

## ⚙️ Preprocessing Pipeline

### ✅ Label Validation & Correction

- Checked YOLO format: `[class_id x_center y_center width height]`
- Fixed:
  - 336 invalid labels (format, range, or class errors)
  - Increased valid samples from 4,665 to 5,001
- Ensured 100% label integrity in all splits

### 🧹 Image Processing

- **Noise Reduction**:
  - Median blur (ksize=3)
  - Bilateral filter (d=5, σColor=30, σSpace=30)
- **Normalization**: Intensity scaled to [0, 255]
- **Contrast Check**: Std. deviation ≥ 15
- **Resizing**: 640x640 pixels

### 🔁 Augmentation

To address class imbalance:
- **Target per class**: 7,000 instances
- **Augmentation for PA (minority class)**:
  - Rotations, flipping (LR & UD), brightness/contrast shifts, CLAHE, Gaussian noise, mixup, mosaic, and random erasing
- **Final Augmented Dataset**:
  - Cavity: 13,713 instances
  - PA: 9,375 instances
  - Images: 9,979

---

## 🧠 Model Architecture: YOLOv8s

- Framework: [Ultralytics YOLOv8](https://docs.ultralytics.com)
- Architecture: 129 layers → 111 after fusion
- Params: 11.1M | GFLOPs: 28.4
- Classes: 2 (Cavity, PA)
- Pretrained weights: `yolov8s.pt`

---

## 🏋️ Training Configuration

| Setting            | Value                          |
|--------------------|--------------------------------|
| Epochs             | 150 (early stopped at 103)     |
| Batch Size         | 4                              |
| Image Size         | 640x640                        |
| Optimizer          | AdamW                          |
| Learning Rate      | 0.002 → 0.005 (cosine decay)   |
| Data Augmentation  | Enabled                        |
| Mixed Precision    | ✅ Automatic Mixed Precision (AMP) |
| Hardware           | Tesla P100 16GB GPU            |
| Total Time         | ~6.5 hours                     |

---

## 📊 Performance

### ✅ Best Model (Epoch 103)

- **mAP@50**: 0.812
- **mAP@50-95**: 0.295
- **Precision**: 0.829
- **Recall**: 0.768

### 🔍 Class-wise mAP@50

| Class   | Precision | Recall | mAP@50 | mAP@50-95 |
|---------|-----------|--------|--------|-----------|
| Cavity  | 0.821     | 0.659  | 0.713  | 0.253     |
| PA      | 0.895     | 0.719  | 0.880  | 0.328     |

### ⚡ Inference Speed

| Stage          | Time (per image) |
|----------------|------------------|
| Preprocessing  | 0.2ms → 1.3ms     |
| Inference      | 2.0ms → 3.4ms     |
| Postprocessing | 2.2ms → 1.7ms     |

---

## 📦 Outputs

- `best.pt`: Best model (epoch 103, 22.5MB)
- `last.pt`: Final model checkpoint
- `data.yaml`: Updated configuration file with new dataset paths
- Training plots and label visualizations saved in `dentalproject/train/`

---

## 🖼️ Visualization

Bounding boxes are drawn on images:
- **Cavity**: Red boxes
- **Periapical Lesion (PA)**: Blue boxes
- Includes confidence scores and labels

---

## 🏁 Conclusion

This project successfully:
- Balanced a highly imbalanced dental dataset
- Reduced image noise and improved contrast
- Built a high-accuracy detector using YOLOv8s
- Achieved >81% mAP@50 for two critical dental pathologies

---

## 🔗 Dependencies

- Python 3.11+
- PyTorch 2.6.0
- Ultralytics YOLOv8 (`pip install ultralytics`)
- OpenCV
- NumPy
- scikit-image
- albumentations
- tqdm

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/dental-caries-pa-detection.git
cd dental-caries-pa-detection

# Install dependencies
pip install -r requirements.txt

# Train model
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=150 imgsz=640

# Validate model
yolo task=detect mode=val model=best.pt data=data.yaml

# Inference
yolo task=detect mode=predict model=best.pt source=your_image.jpg
