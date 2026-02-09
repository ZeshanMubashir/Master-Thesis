# Autonomous Monitoring and Classification of Solar PVs Anomalies using Deep Learning Methods

## Overview

This repository contains the Master's thesis project focused on the **Autonomous Monitoring and Classification of Solar PVs Anomalies using Deep Learning Methods**. The research aim is to develop and evaluate intelligent computer vision systems for the automated inspection of photovoltaic installations using thermal imaging.

**Author:** Zeshan Mubshir  
**University:** Norwegian University of Science and Technology (NTNU)  
**Supervisor:** Saleh Abdel-Afou Alaliyat  
**Date:** July 2025

---

## Abstract

Solar energy is a critical pillar of sustainable energy, yet widespread adoption requires effective monitoring strategies. This study develops advanced deep learning models for detecting and classifying anomalies (e.g., hot spots, cell cracks, diode failures, soiling) using the **InfraredSolarModules dataset** (20,000+ thermal images).

The research evaluates both traditional **CNNs** (ResNet50, VGG16, EfficientNet-B0) and state-of-the-art **Vision Transformers** (ViT, DeiT). Results show that Vision Transformers achieve superior performance, reaching over **98% accuracy** in binary classification.

---

## Visual Showcase

### Thermal Anomaly Detection

Below is an example of the thermal signatures used to identify defects in solar modules:
<img src="Figures/Solar%20PV/Original%20Dataset/image25_inferno.png" alt="Thermal Anomaly" width="400">

### Model Performance

Training progress of the **DeiT-B16** architecture across 11 anomaly classes:
![Training History](Figures/Vision%20Transformer/DeiT-B16/11_Class/training_history_11class_deit.png)

### Confusion Matrix

Evaluation of the 12-class classification model (11 anomaly types + normal):
![Confusion Matrix](Figures/Vision%20Transformer/DeiT-B16/confusion_matrix_12class_deit.png)

---

## Key Features & Methodology

- **Deep Learning Architectures:** Comparative analysis between CNNs (VGG16, ResNet50) and Transformers (ViT, DeiT).
- **Comprehensive Classification:** Evaluated across Binary, 11-class, and 12-class scenarios.
- **Explainability:** Implementation of Grad-CAM and t-SNE for visualizing model decision-making.
- **Dataset:** Utilization of the large-scale InfraredSolarModules dataset.

## Results Summary

- **High Accuracy:** >98% for anomaly detection.
- **Transformer Superiority:** Vision Transformers significantly outperformed traditional CNNs in feature extraction and classification tasks.
- **Automated Inspection:** Proof-of-concept for scalable, cost-effective industrial monitoring solutions.

---
*This project was completed as part of the Master's program in Simulation and Visualization at NTNU.*
