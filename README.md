# 🚗 Pakistani Smart ANPR System (YOLOv11 + PaddleOCR)

A high-accuracy, real-time Automatic Number Plate Recognition (ANPR) system specifically optimized for Pakistani vehicle number plates (including Islamabad, Punjab, Sindh, and other regional formats). Built using **Ultralytics YOLOv11** for precise plate detection, **PaddleOCR** for robust text extraction, and an automated **OpenCV / SQLite** pipeline to capture and store clean, high-sharpness vehicle records.

---

## 🌟 Key Features

* **Custom-Trained YOLOv11:** Fine-tuned object detection model (`yolov11n.pt`) specifically trained to detect vehicle number plates in challenging real-world conditions (night lighting, motion blur, varying angles).
* **Blur & Sharpness Filtering:** Automatically filters out blurry or unstable frames using Laplacian variance, ensuring only the clearest frames are processed.
* **Intelligent Track-Based Best Crop Selection:** Tracks vehicles across multiple frames (`track_id`), dynamically retaining and updating the sharpest cropped plate image before performing OCR.
* **Advanced Text Cleaning & Regex:** Specially designed for Pakistani license plate formats. Automatically handles common OCR errors, strips out city/provincial markers (e.g., `ICT`, `ISLAMABAD`, `KARACHI`, `LAHORE`), and removes blacklisted words.
* **Frame Stabilization Buffer:** Implements a multi-frame stabilization window to prevent early, premature, or false extractions (such as partial reads or artifacts like `COCLA`) from being logged.
* **Local Database Storage:** Automatically logs newly detected and registered plates into a local SQLite database along with confidence scores and snapshot paths.
* **100% Offline / Local Execution:** Runs completely locally on your machine without relying on any external cloud APIs.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Object Detection & Tracking:** Ultralytics YOLOv11 (`weights/yolov11n.pt`)
* **OCR Engine:** PaddleOCR (Offline Mode)
* **Image Processing:** OpenCV, NumPy
* **Database:** SQLite (`database.py`)
* **Interface:** Streamlit (`app.py`)

---

## 📂 Project Structure

```text
license_plate_project/
│
├── weights/
│   └── yolov11n.pt               # Custom trained YOLOv11 model weights
├── outputs/
│   └── cropped_plates/           # Saved snapshots of verified plates
├── database.py                   # SQLite database handler
├── processor.py                  # Core ANPR pipeline & logic class
├── app.py                        # Application entry point (Streamlit)
└── README.md                     # Project Documentation


# 1. Clone the Repository (or navigate to your project directory)
cd path/to/your/license_plate_project

# 2. Create and Activate a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# 3. Upgrade Pip
pip install --upgrade pip

# 4. Install Required Dependencies
pip install ultralytics paddleocr opencv-python numpy streamlit

# 5. Run the Application
streamlit run app.py

🚀 How It Works

    Detection & Tracking: YOLOv11 detects the license plate bounding box and assigns a persistent track_id to the vehicle.

    Quality Control: The system calculates the frame sharpness using a Laplacian variance check. Blurry frames are instantly dropped.

    Best Frame Selection: The system continuously monitors the tracking session and keeps the sharpest plate crop.

    OCR & Text Cleaning: PaddleOCR extracts raw text from multiple enhanced variants of the crop. The regex engine cleans up OCR confusions, strips unwanted regional text, and validates the format against Pakistani plate standards.

    Stabilization & Logging: Once the detected text stabilizes across consecutive frames, the clean plate number is verified against the database. If it's a new entry, the highest-quality cropped image is saved to disk and logged.
