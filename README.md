# IR_VIS_FUSION: Infrared & Visible Image Fusion

An end-to-end deep learning framework and web application for Infrared (IR) and Visible (VIS) image fusion, enhanced with real-time object detection capabilities (YOLOv11).

## 📌 Project Overview

This project provides a full-stack solution for fusing complementary information from thermal infrared images (which capture thermal signatures) and visible spectrum images (which capture high-resolution detail and color). The fused results highlight key targets under varying lighting conditions and nighttime environments.

### Key Features
- **Deep Learning Fusion**: Combines infrared and visible spectrum details into a single high-quality image.
- **Target Detection**: Built-in YOLOv11 target detection on fused image outputs.
- **Interactive UI**: React & Vite modern frontend interface for single and batch image processing.
- **RESTful API**: Fast and light PyTorch backend powered by FastAPI.

---

## 🏗️ Project Architecture

```
IR_VIS_FUSION/
├── backend/            # FastAPI & PyTorch Image Fusion Engine
│   ├── main.py         # API Server Endpoints
│   ├── Modules.py      # Neural Network Blocks & Modules
│   ├── net.py          # Deep Learning Model Definitions
│   ├── utils.py        # Image Processing & Helper Utilities
│   └── requirements.txt# Backend Dependencies
├── frontend/           # React + Vite Web UI
│   ├── src/            # Components & Application Logic
│   ├── package.json    # Frontend Dependencies
│   └── vite.config.ts  # Vite Configuration
└── README.md           # Project Documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- Node.js 16+ & npm

---

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend server
python main.py
```
The backend server will run by default on `http://localhost:8000`.

---

### 2. Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
Open your browser and navigate to `http://localhost:5173`.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.
