# 🐶 Dog Breed Classifier – AI Powered Web App

🔗 **Live App:** https://dogbreedclassifier-31.streamlit.app/

A modern **AI-powered web application** that predicts the **dog breed** from an uploaded image using a **deep learning Vision Transformer (ViT)** model.  
Built with **Streamlit + PyTorch + Hugging Face Transformers**, this project demonstrates an end-to-end **machine learning deployment** workflow.
 
---

## ✨ Features

- 📤 Upload dog images (JPG / PNG / JPEG) 
- 🧠 AI-based dog breed classification
- 🏆 Displays **top predicted breed**
- 📊 Shows **Top-5 breed confidence scores** in a bar chart
- ⚡ Fast inference with cached model loading
- 🎨 Clean, responsive, and user-friendly UI
- 🌐 Fully deployed and publicly accessible

---

## 🖼️ Demo

👉 Try it live here:  
**https://dogbreedclassifier-31.streamlit.app/**

---

## 🛠️ Tech Stack

| Category | Technologies |
|-------|-------------|
| Frontend | Streamlit |
| Deep Learning | PyTorch |
| Model | Vision Transformer (ViT) |
| ML Framework | Hugging Face Transformers |
| Image Processing | Pillow |
| Data Handling | Pandas |
| Deployment | Streamlit Cloud |

---

## 🧠 Model Details

- **Model Name:** `google/vit-base-patch16-224`
- **Architecture:** Vision Transformer (ViT)
- **Input:** RGB image (224×224)
- **Output:** ImageNet class probabilities
- **Inference:** CPU-based (Streamlit Cloud)

> ⚠️ Note: The model is trained on ImageNet classes. Some predictions may be visually similar dog breeds.

---

## 📊 How It Works

1. User uploads a dog image  
2. Image is preprocessed using Hugging Face `AutoImageProcessor`  
3. Vision Transformer predicts class probabilities  
4. Top-5 predictions are extracted  
5. Results are displayed with confidence scores  

---
