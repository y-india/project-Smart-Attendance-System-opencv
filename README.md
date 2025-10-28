# 🎯 Smart Attendance System — Powered by Face Recognition & OpenCV

A real-time **Smart Attendance System** that uses **Face, Eye, and Mask detection** to automatically log attendance with date and time.  
Built with advanced **OpenCV**, **Python**, and **Machine Learning techniques**, this system is ideal for **schools, offices, and organizations** aiming to automate daily attendance tracking.

---

## 🚀 Features

- 👁️ Detects **Face**, **Eyes**, and **Mask** in real time  
- 🧠 Uses **LBPH Face Recognizer** for reliable recognition  
- 🕒 Logs attendance automatically with **name, time, and confidence level**  
- 📅 Creates **daily attendance files** in the `attendance_logs/` folder  
- 🔁 Prevents multiple attendance entries for the same person per day  
- 🧍‍♂️ Easy to train with your own employee or student images  
- ⚡ Lightweight and works on most laptops or desktops

---

## 🧩 Tech Stack

| Component | Description |
|:-----------|:-------------|
| 🐍 **Python 3.10+** | Core programming language |
| 📸 **OpenCV (cv2 + contrib)** | Face detection and recognition |
| 📊 **Pandas** | Attendance logging and CSV management |
| 🔢 **NumPy** | Image matrix operations |
| 💾 **OS / datetime** | File handling and date-based logging |

---

## 🏗️ Project Structure


- **#1dataset_creator.py** → Creates and stores face datasets for each registered person  
- **#2train_recognizer.py** → Trains the LBPH face recognition model  
- **#3main.py** → Runs real-time attendance detection and logging  

### 📦 Folders

- **/trainer** → Stores trained model files  
  - `trainer.yml` – Trained recognizer  
  - `names.npy` – ID-name mapping  

- **/datasets** → Contains captured face images for each user  
  - `user_1/`, `user_2/` etc.  

- **/attendance_records** → Stores daily attendance CSV files  
  - `2025-10-28.csv`, `2025-10-29.csv`, etc.  

- **/snapshots** → Saves captured snapshots (if enabled)  

### 🧾 Other Files
- **README.md** → Documentation and setup guide  
- **.gitignore** → Files to exclude from Git commits  















---

## 🧭 How to Use (For Organizations Only)

This project is designed to be **professionally installed and configured** for your organization.

To use this system in your **school, office, or event**, you must:
1. Contact me for setup and installation.
2. I will configure:
   - Your organization’s dataset (employees/students)
   - Model training
   - Real-time attendance logging
   - Folder organization & automation for daily reports

---

## 💼 Hire Me to Set It Up

Want this system installed for your organization?

I provide:
- ✅ Complete project setup on your organization’s CCTV or webcam feed
- ✅ Dataset creation and model training
- ✅ Daily CSV attendance automation
- ✅ Minor customizations (UI, naming, etc.)

📧 **Email:** [y.india.main@gmail.com](mailto:y.india.main@gmail.com)



---

## ⚙️ Demo (Developers Only)

If you are a developer or student and just want to run a **demo version**:

```bash
python main.py
