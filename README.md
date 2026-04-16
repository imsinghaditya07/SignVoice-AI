#  SignVoice AI

### *Bridging Voices Beyond Silence*

---
**GitHub Link - https://github.com/imsinghaditya07/SignVoice-AI

## 👥 Team BYTE BREAKER

*Anand Kumar Jha (Team Leader | (Backend Developer) 

*Aditya Singh (AI & System Architect | UI/UX Designer)

*Debashrita Mandal  (Integration & Deployment)

*Aaryan Lal Das (ML Model Train | Data & Training)

---
## 📌 Overview

**SignVoice AI** is a real-time, AI-powered **bi-directional communication system** designed to bridge the gap between deaf/mute individuals and people who rely on spoken language.

The platform enables seamless communication by converting:

* ✋ **Sign Language → Text**
* ⌨️ **Text → Sign Language (visual form)**

This system focuses on accessibility, simplicity, and real-world usability, making communication inclusive for everyone.

---

## 🎯 Problem Statement

* Millions of deaf individuals struggle to communicate daily
* Most people do not understand sign language
* Lack of affordable and accessible assistive tools
* Communication barriers in hospitals, schools, and public services

---

## 💡 Our Solution

SignVoice AI provides a **two-way AI communication bridge** that allows:

* Deaf users to express themselves using gestures
* Hearing users to communicate using text

👉 Without requiring either person to learn the other’s language.

---

## ⚙️ Key Features

* 🔄 Bi-directional communication
* ⚡ Real-time gesture recognition
* 🧠 AI-powered prediction
* 📷 Webcam-based sign detection
* 🖼️ Sign language visualization
* 📱 Responsive and simple UI
* 🌍 Accessible for rural users
* 🚨 Emergency gesture detection (HELP)

---

## 🧠 Tech Stack

### 👨‍💻 Frontend

* React.js / Next.js
* Tailwind CSS
* Framer Motion

### 🤖 Backend

* Python
* Flask / FastAPI

### 🧪 AI & Machine Learning

* OpenCV
* MediaPipe (Hand Tracking)
* Scikit-learn (Random Forest Model)

### 🔊 Processing

* Text Processing Engine
* Dataset Mapping (Sign Images / GIFs)

---

## 🧩 Functional Architecture

### 🔄 Bi-Directional System

```
SIGN LANGUAGE → TEXT

Webcam → MediaPipe → Landmark Extraction → Feature Processing → ML Model → Prediction → Text Output


TEXT → SIGN LANGUAGE

Text Input → Text Processing → Sign Mapping Engine → Dataset (Images/GIFs) → Visual Output
```

---

## ⚙️ System Components

* **Input Layer**

  * Webcam (gesture input)
  * Text input

* **Processing Layer**

  * MediaPipe hand tracking
  * Feature extraction

* **AI Layer**

  * Machine learning model (Random Forest)

* **Output Layer**

  * Text display
  * Sign language visuals

---

## 🚀 How It Works

### 🟢 Sign → Text

1. Webcam captures hand gesture
2. MediaPipe detects 21 landmarks
3. Features are extracted
4. ML model predicts gesture
5. Output displayed as text

---

### 🔵 Text → Sign

1. User enters text
2. Text is processed
3. System maps text to sign visuals
4. Output displayed as sign language

---

## 📊 Dataset

* Custom dataset created using MediaPipe landmarks
* Multiple gesture samples collected
* Labels include common words like: HELLO, YES, NO, HELP

---

## 📈 Performance

* Real-time processing
* Lightweight model
* Works on standard devices
* Accuracy: ~85–95% (depends on conditions)

---

## 🌍 Impact

* Enables communication for deaf individuals
* Useful in hospitals, schools, and public spaces
* Promotes inclusivity and accessibility

---

## 🔮 Future Scope

* Sentence-level recognition (LSTM / Deep Learning)
* Multi-language support
* Mobile application
* 3D avatar-based sign display
* Offline AI model

---

## 🏆 Innovation Highlights

* Real-time bi-directional communication
* Low-cost and scalable solution
* Human-centered AI design
* Practical real-world application

---

## 🎯 Vision

> “To create a world where communication is not limited by ability.”

---

## 📜 License

This project is developed for educational and research purposes.

---

## ❤️ Acknowledgements

* MediaPipe
* OpenCV
* Scikit-learn
* WHO (for data references)
* Open-source community

---

## ⭐ Final Note

**SignVoice AI is not just a project — it is a bridge that connects silence to expression.**

If you like this project, give it a ⭐ and support inclusive technology!
