# 🐾 Wildlife Intrusion Detection & Alert System (Image-Based)

## 📌 Overview
This project detects wildlife intrusion from images using a deep learning–based object detection model and generates real-time alerts to reduce human–wildlife conflict. The system is designed to be fast, reliable, and suitable for real-world safety applications.

---

## 🎯 Problem Statement
Wild animals entering human-populated areas pose serious safety risks. Manual monitoring systems are slow and ineffective.

**Goal:**  
Automatically detect wild animals from images and alert users with sound and SMS notifications along with location details.

---

## 💡 Proposed Solution
An image is processed using a trained YOLOv8 object detection model. If an animal is detected with sufficient confidence:
- A sound alert is triggered
- An SMS notification is sent using Twilio
- Location details are included for quick response

---

## 🛠️ Tech Stack
- **Deep Learning Model:** YOLOv8  
- **Frontend / Application:** Streamlit  
- **Programming Language:** Python  
- **Dataset & Annotation:** Roboflow  
- **Alerts:** Twilio SMS API  
- **Image Processing:** OpenCV  

---

## 🔄 System Flow
1. User uploads an image through the Streamlit interface  
2. Image is passed to the YOLOv8 model  
3. Model detects animals and outputs confidence scores  
4. If confidence exceeds the threshold:  
   - Alert sound is played  
   - SMS alert is sent with location information  

---

## 📂 Dataset
- **Source:** Roboflow  
- **Total annotated images:** 4,762  
- **Training images:** 4,362  
- **Testing images:** 364  

All images are annotated with bounding boxes for wildlife classes.

---

## 📊 Evaluation Metrics
- Precision  
- Recall  
- F1 Score  
- Confidence Threshold  
- Confusion Matrix  

These metrics help balance false alerts versus missed detections, which is critical for safety systems.

---

## 🧩 Application Modules
- **Model Inference Module:** Loads the trained YOLOv8 model  
- **Detection Logic:** Filters predictions based on confidence  
- **Alert Module:** Triggers sound alerts and sends SMS via Twilio  
- **Location Module:** Attaches area/location details to alerts  

---

## 🚨 Alert System
- 🔔 Sound alert for immediate warning  
- 📩 SMS alert via Twilio  
- 📍 Location information included  

Alerts are generated only for high-confidence detections.

---

## 📌 Conclusion
This project focuses on building a reliable wildlife detection and alert system rather than just training a model. Emphasis is placed on evaluation, confidence tuning, and real-world deployment considerations.
