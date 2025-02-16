# 🏷️ Fashion Product Classifier  

A deep learning-based **fashion product classifier** built using **EfficientNet/Xception** with transfer learning on **ImageNet**. The model is trained on a **merged dataset** combining multiple fashion datasets to improve accuracy and generalization.

---

## 📂 Dataset  

This project uses a **merged dataset** from five different sources:  

- **Apparel Dataset**  
- **Clothing dataset** (full, high-resolution)  
- **DeepFashion In-shop Clothes Retrieval Dataset**  
- **DeepFashion2 Original Dataset**  
- **Fashion Product Images Dataset**  

The dataset is preprocessed for class balance, augmentation, and label consistency.  

---

## 🚀 Features  

✅ **Pretrained on ImageNet** for robust feature extraction  
✅ **Handles imbalanced datasets** using **class weighting & augmentation**  
✅ **Fine-tuned EfficientNet/Xception** for high accuracy  
✅ **Streamlit-based frontend available** for easy model interaction 

---
## 📊 Model Training  

- **Base Model:** EfficientNet/Xception  
- **Loss Function:** Weighted Categorical Cross-Entropy with **class weights**  
- **Optimizer:** Adam with learning rate scheduling  
- **Augmentations:** Flip, rotation, brightness adjustments  
- **Class Balancing:** Applied **class weights** to compensate for dataset imbalance  
- **Evaluation Metrics:** Accuracy,F1-score  


## 💡 References  

- DeepFashion, DeepFashion2 Datasets  
- ImageNet Pretrained Models  
- TensorFlow/Keras Documentation  

