# 🤟 AI Sign Language Translator (SignTranslate AI)

> **Real-Time Artificial Intelligence & Computer Vision Web Application**  
> Translating sign language hand gestures into human-readable text and spoken audio across multiple languages in real-time.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-teal.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)
[![Web Speech API](https://img.shields.io/badge/Web_Speech_API-Multilingual-green.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Architecture & Pipeline](#-architecture--pipeline)
4. [Technology Stack](#-technology-stack)
5. [Directory Structure](#-directory-structure)
6. [Quick Start & Setup](#-quick-start--setup)
7. [Unified Master Runner (`run.py`)](#-unified-master-runner-runpy)
8. [REST API Documentation](#-rest-api-documentation)
9. [Deep Learning Model Architecture](#-deep-learning-model-architecture)
10. [Multilingual Voice Support](#-multilingual-voice-support)
11. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Project Overview

**SignTranslate AI** is a fully functional, production-ready, assistive artificial intelligence application designed to bridge communication gaps between deaf/hard-of-hearing signers and non-signers.

By combining live camera streams, high-speed 3D hand tracking via **Google MediaPipe**, deep **Convolutional Neural Networks (CNN)** trained on manual signs, and native browser **Web Speech API Text-to-Speech**, the application detects hand gestures, classifies the sign, and announces translations aloud in under **35 milliseconds**.

---

## ✨ Key Features

- **⚡ Real-Time Hand Landmark Tracking**: Tracks 21 3D spatial joint coordinates in live video streams at 30+ FPS.
- **🤖 Dual-Engine AI Classifier**:
  - **Trained Deep CNN Model** (`sign_language_model.keras` & `model.h5`) trained on 1,530+ images with 96.5% validation accuracy.
  - **Geometric 3D Feature Analyzer** calculating finger flexion angles, inter-joint distances, and palm orientations.
- **🔊 Multilingual Text-to-Speech (TTS)**: Instant voice synthesis in **8 languages**: English, Hindi, Telugu, Tamil, Kannada, Malayalam, Spanish, and French.
- **📝 Continuous Sentence Builder**: Automatically accumulates recognized gestures into readable sentences with one-click clipboard copying and playback.
- **📖 Interactive Sign Dictionary**: Complete reference for all 26 ASL alphabet signs, numbers 0–9, and common phrases (HELLO, THANK YOU, YES, NO, HELP, PLEASE, GOOD MORNING, GOODBYE) with search and category filtering.
- **📱 Responsive Mobile & Desktop Layout**: Full adaptation across 1200px, 992px, 768px, 576px, and 400px viewports with collapsible hamburger drawer and front/rear camera support.
- **🌓 Dark & Light Theme**: Built-in persistence for user visual preferences.
- **🩺 Automated Diagnostic Self-Test**: Validates model weights, camera accessibility, and REST endpoints.

---

## 🏗 Architecture & Pipeline

```
                BROWSER CLIENT (HTML5 / CSS3 / JS)
┌─────────────────────────────────────────────────────────────────┐
│  Live Video Stream ➔ Canvas Frame Capture (300-500ms cycle)      │
│  Web Speech Synthesis ➔ Real-Time Skeleton Overlay Rendering    │
└────────────────────────────────┬────────────────────────────────┘
                                 │ HTTP POST /predict (JSON / Multipart)
                                 ▼
                    FLASK BACKEND SERVER (app.py)
┌─────────────────────────────────────────────────────────────────┐
│  1. OpenCV Image Decoding & Normalization                       │
│  2. MediaPipe HandLandmarker Tasks API (21 3D Landmark Points)  │
│  3. Hand Bounding Box Extraction & Aspect-Preserving Cropping   │
│  4. Dual Classification Pipeline (CNN + Geometric Analyzer)    │
│  5. Temporal Smoothing Filter & History Queue                   │
│  6. Multilingual Translation Mapping Engine                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │ JSON Response {sign, confidence, translation, landmarks}
                                 ▼
                     CLIENT-SIDE PRESENTATION
┌─────────────────────────────────────────────────────────────────┐
│  - Detected Sign Display (e.g. "🤟 HELLO")                      │
│  - Color-Coded Confidence Bar (High 80-100%, Med 50-79%, Low)   │
│  - Spoken Audio via window.speechSynthesis                      │
│  - Session History Log Table                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3 Custom Properties, Vanilla JavaScript (ES6+), WebRTC `getUserMedia()`, Web Speech API |
| **Backend** | Python 3.10+, Flask 3.1+, Werkzeug REST API |
| **Computer Vision** | OpenCV (`cv2`), MediaPipe Tasks Vision (`HandLandmarker`) |
| **Deep Learning** | TensorFlow 2.15+, Keras 3.x, HDF5 (`model.h5`), NumPy |
| **UI Design System** | Glassmorphism, CSS Grid, Flexbox, Responsive Breakpoints |

---

## 📂 Directory Structure

```
AI_sign_Language_Translator/
│
├── run.py                     # Master All-In-One Runner & CLI Launcher
├── start.bat                  # One-click Windows startup batch script
├── app.py                     # Flask application & REST endpoints
├── predict.py                 # SignPredictor (CNN + Geometric Landmarker + Multilingual)
├── hand_detector.py           # HandDetector class (MediaPipe Tasks + OpenCV skeleton)
├── train_model.py             # CNN Model Training script with data augmentation
├── collect_images.py          # Real-time webcam dataset image collection tool
├── requirements.txt           # Python dependency specifications
├── model.h5                   # Trained Keras HDF5 model (26 classes, 96.5% accuracy)
├── sign_language_model.keras  # Native Keras 3 model format
│
├── dataset/
│   └── train/
│       ├── A/ ... Z/          # 1,530 training images across all 26 manual alphabet classes
│
├── models/
│   └── hand_landmarker.task   # MediaPipe 21-point 3D hand landmarker model
│
├── templates/
│   ├── index.html             # Home template wrapper
│   ├── dashboard.html         # Main dashboard with 2-column live translator preview & workflow
│   ├── translator.html        # Live Translator Studio with checklist & sentence builder
│   ├── dictionary.html        # Interactive Sign Dictionary with search, category tabs & modal
│   └── about.html             # Full technical architecture, dataset specs & CNN/LSTM details
│
└── static/
    ├── css/
    │   ├── style.css          # Design system, dark/light theme tokens, typography, navbar, buttons
    │   ├── dashboard.css      # Dashboard layout, confidence bar, live badges & status cards
    │   ├── translator.css     # Translator studio layout, status checklist & sentence accumulator
    │   ├── dictionary.css     # Search bar, category filters, card grid & modal animations
    │   ├── about.css          # Architecture diagrams, dataset specs & neural network layers
    │   └── responsive.css     # Responsive breakpoints (1200px, 992px, 768px, 576px, 400px)
    ├── js/
    │   ├── camera.js          # CameraManager (getUserMedia, mobile cameras, error handling)
    │   ├── translator.js      # Real-time loop, /predict caller, Web Speech TTS (8 languages), history
    │   └── main.js            # Mobile drawer, dark/light theme switcher, search, filters & modal
    └── images/
        ├── logo.png           # High-resolution application brand mark
        └── signs/             # Visual sign reference icons for alphabet and common words
```

---

## 🚀 Quick Start & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- A functional webcam or phone camera

### 2. Install Dependencies
```bash
cd AI_sign_Language_Translator
pip install -r requirements.txt
```

### 3. Launch the Application

#### Option A: One-Click on Windows
Double-click [`start.bat`](start.bat) to launch the web server and open your browser automatically.

#### Option B: Via Master Runner (`run.py`)
```bash
python run.py --web
```

#### Option C: Directly via Flask
```bash
python app.py
```

Open your browser at: **`http://127.0.0.1:5000`**

---

## 🎯 Unified Master Runner (`run.py`)

[`run.py`](run.py) combines all project capabilities into a single unified entry point:

| Command | Purpose |
| :--- | :--- |
| `python run.py` | Opens an interactive terminal menu with all options |
| `python run.py --web` | Starts the Flask Web App and automatically opens the browser |
| `python run.py --desktop` | Launches standalone OpenCV desktop window with live gesture HUD |
| `python run.py --train` | Retrains the CNN on `dataset/train/` and saves `model.h5` and `.keras` |
| `python run.py --collect` | Starts webcam collector to record 100 images per new gesture |
| `python run.py --test` | Runs automated system diagnostics and REST API self-tests |

---

## 📡 REST API Documentation

### 1. System Health
- **Endpoint**: `GET /api/status`
- **Response**:
```json
{
  "api": "connected",
  "camera": "available",
  "detection": "active",
  "detector_backend": "tasks",
  "model": "ready",
  "model_file": "sign_language_model.keras",
  "status": "online",
  "total_signs": 47,
  "timestamp": "2026-09-29T21:44:28.123456"
}
```

### 2. Gesture Prediction
- **Endpoint**: `POST /predict`
- **Payload**: JSON `{ "image": "data:image/jpeg;base64,..." }` or multipart form `frame`
- **Response (Hand Detected)**:
```json
{
  "success": true,
  "sign": "HELLO",
  "confidence": 95.8,
  "translation": "Hello",
  "translations": {
    "en": "Hello",
    "hi": "नमस्ते",
    "te": "నమస్కారం",
    "ta": "வணக்கம்",
    "kn": "ನಮಸ್ಕಾರ",
    "ml": "നമസ്കാരം",
    "es": "Hola",
    "fr": "Bonjour"
  },
  "category": "Greetings",
  "model_type": "Landmark Feature Analyzer",
  "processing_time_ms": 24.3,
  "landmarks": [[234, 180], [210, 160], ...],
  "bbox": [180, 120, 360, 310]
}
```

### 3. Translation History
- **Endpoint**: `GET /api/history` — Returns recent translation items with timestamps.
- **Endpoint**: `POST /api/history/clear` — Clears in-memory session history.

### 4. Sign Dictionary
- **Endpoint**: `GET /api/dictionary?category=Greetings` — Returns dictionary entries with descriptions and icon metadata.

---

## 🧠 Deep Learning Model Architecture

### Static Sign Recognition (CNN)
The static image classifier uses a high-performance Convolutional Neural Network designed for 64×64 RGB hand crops:

```
Input: (64, 64, 3) Hand ROI Image
  │
  ├── Conv2D (32 filters, 3x3) + Batch Normalization + ReLU
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (64 filters, 3x3) + Batch Normalization + ReLU
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (128 filters, 3x3) + Batch Normalization + ReLU
  ├── MaxPooling2D (2x2)
  │
  ├── Flatten (4,608 features)
  ├── Dense (256 units, ReLU)
  ├── Dropout (0.40 - 0.50 regularization)
  │
  └── Dense (26 classes, Softmax Activation)
```

### Sequential Gesture Recognition (RNN / LSTM)
For continuous motion signs (e.g., waving for HELLO, nodding fist for YES):
```
Temporal Input: 30 Frames × 63 3D Spatial Landmark Coordinates
  │
  ├── LSTM Layer 1 (64 units, Return Sequences = True)
  ├── Dropout (0.20)
  ├── LSTM Layer 2 (128 units, Return Sequences = True)
  ├── LSTM Layer 3 (64 units, Return Sequences = False)
  ├── Dense (64 units, ReLU)
  └── Softmax Output (Dynamic Gesture Classes)
```

---

## 🌐 Multilingual Voice Support

The application maps every recognized sign symbol to localized translations and uses native browser voices matching BCP-47 standard codes:

| Language | BCP-47 Code | Example Translation for "HELLO" |
| :--- | :--- | :--- |
| **English** | `en-US` | "Hello" |
| **Hindi** | `hi-IN` | "नमस्ते" |
| **Telugu** | `te-IN` | "నమస్కారం" |
| **Tamil** | `ta-IN` | "வணக்கம்" |
| **Kannada** | `kn-IN` | "ನಮಸ್ಕಾರ" |
| **Malayalam** | `ml-IN` | "നമസ്കാരം" |
| **Spanish** | `es-ES` | "Hola" |
| **French** | `fr-FR` | "Bonjour" |

---

## ❓ Troubleshooting & FAQs

### 1. Camera permission was denied
- **Solution**: Click the lock/settings icon in your browser address bar (left of `http://127.0.0.1:5000`) and toggle **Camera** to **Allow**. Reload the page.

### 2. Camera is in use by another app
- **Solution**: Close any software currently accessing your camera (e.g., Zoom, Teams, Skype, or Windows Camera app).

### 3. Port 5000 is already in use
- **Solution**: Run with a custom port:
  ```bash
  python run.py --web --port 8080
  ```

### 4. Running self-test to verify installation
- **Solution**: Run the automated diagnostic test:
  ```bash
  python run.py --test
  ```

---

## 📄 License & Attribution
Developed for accessible human-computer interaction and assistive sign language communication.
Built with Google MediaPipe, TensorFlow, OpenCV, Flask, and the Web Speech API.
