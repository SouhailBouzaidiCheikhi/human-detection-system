# Human Detection & Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A real-time facial recognition system with database management**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Screenshots](#-screenshots) • [Troubleshooting](#-troubleshooting)

</div>

## 🎯 Overview

Human Detection & Recognition System is a comprehensive facial recognition application built with **Streamlit** and **SQLite**. It enables real-time face detection, registration, and recognition through a web camera interface. The system maintains a database of registered individuals and provides tools for managing and analyzing detection data.

### Key Capabilities
- ✅ **Real-time face detection** using web camera
- ✅ **Automatic face recognition** of registered individuals
- ✅ **Database management** for registered persons
- ✅ **Detection statistics** and analytics
- ✅ **Easy registration** of new individuals

---

## ✨ Features

### 🎥 **Real-time Detection**
- Live camera feed with face detection
- Automatic recognition of registered persons
- Visual feedback with bounding boxes and labels
- Confidence scoring for recognitions

### 👤 **Person Management**
- Register new individuals through camera or image upload
- Edit/delete person information
- Search functionality in database
- Export data capabilities

### 📊 **Analytics & Statistics**
- Detection statistics dashboard
- Visual charts of top visitors
- Daily/weekly/monthly detection trends
- System performance metrics

### ⚙️ **System Configuration**
- Adjustable detection settings
- Model selection (HOG vs CNN)
- Database management tools
- System status monitoring

---

## 🏗️ System Architecture

<img width="1294" height="1188" alt="Screenshot 2025-12-23 063724" src="https://github.com/user-attachments/assets/4d5f6e54-c136-4f04-8971-2c399d0ddd97" />

**Technology Stack:**
- **Frontend**: Streamlit
- **Face Recognition**: face-recognition (dlib + OpenCV)
- **Database**: SQLite
- **Image Processing**: OpenCV, Pillow
- **Data Visualization**: Plotly

---

## 📥 Installation Guide

### Prerequisites
- Python 3.8 or higher
- Web camera (for real-time detection)
- CMake (for dlib installation)

### Step-by-Step Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/human-detection-system.git
cd human-detection-system
```
2. **Create Virtual Environment (Optional but recommended)**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```
3. **Install CMake (Required for dlib)**
- Windows: Download from cmake.org
- Mac: brew install cmake
- Linux: sudo apt-get install cmake

4. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```
**If you encounter issues with dlib, try:**
```bash
pip install cmake
pip install dlib==19.24.2
pip install face-recognition==1.3.0
pip install opencv-python==4.8.1.78
pip install streamlit==1.28.0
```
5. **Verify Installation**
```bash
python -c "import face_recognition; import streamlit; print('✓ All packages installed successfully!')"
```
## 🚀 Usage Guide
**Starting the Application**
```bash
streamlit run app.py
```
The application will open in your default web browser at http://localhost:8501

**First-Time Setup**
- Open the application in your browser
- Navigate to 📷 Real-time Detection page
- Allow camera access when prompted
- Start registering individuals

**Pages Overview**

# Face Recognition System

A comprehensive face recognition system with real-time detection, database management, and registration capabilities.

## 🏠 Dashboard

<img width="3822" height="1882" alt="Screenshot 2025-12-23 061629" src="https://github.com/user-attachments/assets/e646df25-46e9-437e-aa71-63cbc0ed6595" />

The dashboard provides an overview of system statistics including:
- Total registered persons
- Today's detections
- Recent detections list
- System status

## 📷 Real-time Detection

This page has two tabs:

### 🎥 Live Camera Tab

<img width="3822" height="1951" alt="Screenshot 2025-12-23 061658" src="https://github.com/user-attachments/assets/22fa4ec2-9241-4b2e-8fea-9f4ab4f7cdc5" />

- Real-time face detection from web camera
- Green boxes: Recognized individuals
- Red boxes: Unknown faces (click to register)
- Confidence scores displayed

### 📤 Upload & Register Tab

<img width="3833" height="1803" alt="Screenshot 2025-12-23 061708" src="https://github.com/user-attachments/assets/a6fd442e-8d8f-4f02-b108-cce76cc3a2a3" />

- Upload images for registration
- Visual feedback of detected faces
- Registration form with person details
- Tips for better face detection

## 👥 Manage Database

Three tabs for database management:

### 📋 View All Tab

<img width="3839" height="1901" alt="Screenshot 2025-12-23 061719" src="https://github.com/user-attachments/assets/b42eddaa-8bd1-4a87-8de8-77875d364a36" />

- List of all registered persons
- Details including last seen and detection count
- Expandable view for each person

### 🔍 Search Tab

<img width="3839" height="1851" alt="Screenshot 2025-12-23 061754" src="https://github.com/user-attachments/assets/23c6c852-144f-42e8-ab54-93d0bd35e8c5" />

- Search persons by name
- Real-time search results
- Detailed person information

### ✏️ Edit/Delete Tab

<img width="3838" height="1879" alt="Screenshot 2025-12-23 061838" src="https://github.com/user-attachments/assets/e13e5e31-7baf-426b-a6a1-1a22774a13ef" />

- Select person to edit
- Update name, role, and information
- Delete persons from database

## ⚙️ Settings

<img width="3838" height="2005" alt="Screenshot 2025-12-23 061849" src="https://github.com/user-attachments/assets/f67e3a71-0855-4618-8bf8-0965eb8962b3" />

- Face detection model selection (HOG/CNN)
- Database management tools
- System information and statistics
- Refresh database option

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```
##📁 Project Structure

<img width="1005" height="811" alt="Screenshot 2025-12-23 063849" src="https://github.com/user-attachments/assets/c8f1048f-5f6c-403c-96b7-a18d451bae4b" />

##🔧 Technical Details

- Face Detection: Uses dlib's HOG or CNN model
- Face Encoding: 128-dimensional face embedding
- Face Comparison: Euclidean distance between encodings
- Recognition: Matching against known face database

**Database Schema**

```bash
-- Persons Table
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    face_encoding TEXT NOT NULL,
    registration_date TEXT NOT NULL,
    last_seen TEXT,
    total_detections INTEGER DEFAULT 0,
    metadata TEXT
);

-- Detection Logs Table
CREATE TABLE detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    detection_time TEXT NOT NULL,
    confidence REAL
);
```
**Performance Considerations**
- HOG Model: Faster, suitable for real-time (CPU)
- CNN Model: More accurate, requires more resources
- Image Resolution: Optimal at 640x480 pixels
- Database Size: Efficient up to thousands of persons

##📄 Acknowledgments
- Built with Streamlit
- Face recognition powered by face_recognition
- Uses dlib for machine learning
- OpenCV for computer vision

##📧 Contact
- GitHub Issues: Create an issue
- Project Link: https://github.com/SouhailBouzaidiCheikhi/human-detection-system

##⭐ Support
If you find this project useful, please give it a star on GitHub!
