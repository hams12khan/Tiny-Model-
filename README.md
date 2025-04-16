# Tiny-Model-
This project demonstrates **knowledge distillation**, a technique to compress a large deep learning model (teacher) into a smaller, faster model (student), using the MNIST dataset.
## 📌 Project Summary

We train a **ResNet50** model as the teacher and a **SimpleCNN** as the student. The student model learns both from the hard labels (true labels) and the soft labels (outputs from the teacher), using a combined distillation loss.

## 🧠 Techniques Used

- **Knowledge Distillation**
- **Transfer Learning**
- **Custom CNN Architecture**
- **KL Divergence + Cross Entropy Loss**
- **Accuracy Tracking**
