# Water Pollution Detection – Week 3 Submission  
AICTE Internship | Sustainability – Clean Water & Sanitation

This repository contains the **Week-3 deliverables** for the Water Pollution Detection project.  
The goal is to classify water bodies as **Clean** or **Polluted** using deep learning and transfer learning.

A complete training workflow, evaluation reports, model outputs, and a simple Gradio-based UI have been provided.

---

## Week-3 Objectives

- Apply **Data Augmentation** to increase dataset diversity  
- Implement **Transfer Learning** using MobileNetV2  
- Perform **fine-tuning** to boost accuracy  
- Generate evaluation visuals:  
  - Accuracy & Loss curves  
  - Confusion Matrix  
  - Classification Report  
  - ROC Curve  
- Save the final trained model (`.h5` format)  
- Create a **simple UI using Gradio** to test predictions  
- Upload all Week-3 results, visuals, and model files to GitHub  

---

## Repository Structure
    water-pollution-detection/
    │── README.md
    │── water_pollution_training.ipynb # Model training + fine-tuning
    │── water_pollution_clean.h5 # Final trained model
    │── app.py # Simple Gradio-based UI
    │── output_images/ # All generated outputs
    │ ├── model_accuracy_loss_plot.png
    │ ├── confusion_matrix.png
    │ ├── roc_curve.png
    │ ├── sample1.png
    │ ├── sample_2.png
    │ ├── ui_screenshot.png # Simple Gradio UI preview
    │── requirements.txt


---

## Model Overview

### **Model Architecture**
- **Base Model:** MobileNetV2 (ImageNet weights, no top layers)
- **Custom Head:**
  - GlobalAveragePooling2D  
  - Dense(128, ReLU)  
  - Dropout layers  
  - Dense(1, Sigmoid)  

### **Training Strategy**
- Stage 1: Freeze CNN layers → Train classifier  
- Stage 2: Unfreeze last 20–25 layers → Fine-tune with lower LR  

### **Final Test Accuracy:** ~92–94%

---

## Dataset Information

**Dataset:** Clean vs Dirty Water (Kaggle)  
**Classes:**  
- `clean`  
- `dirty`  

Dataset Source:  
https://www.kaggle.com/datasets/elvinagammed/clean-dirty-water-dataset

Data was split into `train/` and `test/` before training.

---

## Evaluation Outputs (All in `output_images/`)

The following outputs have been generated and uploaded:

✔ `accuracy_plot.png`  
✔ `loss_plot.png`  
✔ `confusion_matrix.png`  
✔ `roc_curve.png`  
✔ `sample_predictions.png`  
✔ `ui_screenshot.png` *(Simple Gradio UI preview)*  

These illustrate the required Week-3 results:  
- Training stability  
- Validation performance  
- Misclassification behavior  
- Threshold discrimination (ROC)  
- Model usability via a simple UI

## Previous Week Submissions  
- **Week 1 Repository:** https://github.com/Ramkumar137/week-1 
- **Week 2 Repository:** https://github.com/Ramkumar137/week-2 

