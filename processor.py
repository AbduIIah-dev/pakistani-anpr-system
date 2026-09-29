import os

# Environment variables to prevent OneDNN/MKLDNN crashes
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import re
import logging
from collections import Counter, defaultdict
import numpy as np
from ultralytics import YOLO
from database import DatabaseHandler
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.ERROR)

class ANPRProcessor:
    def __init__(self, model_path="fine_tune_best_plate.pt", output_dir="outputs/cropped_plates"):
        print("[INFO] Loading YOLO Model & ANPR Engine for Pakistani Plates...")
        self.model = YOLO(model_path)
        
        self.reader = PaddleOCR(
            use_angle_cls=False, 
            lang='en', 
            enable_mkldnn=False, 
            use_gpu=False
        )
        
        self.output_dir = output_dir
        self.db = DatabaseHandler()
        
        self.track_buffers = defaultdict(list)
        self.track_best_crops = {}         # Yahan variable properly initialize kar diya hai
        self.committed_tracks = set()      
        self.recently_detected = {} 
        self.COOLDOWN_FRAMES = 150  
        self.frame_counter = 0     
        
        self.blacklist_words = [
            "LICENSE", "STATE", "PLATE", "GOVT", "VEHICLE", 
            "REGISTRATION", "PAKISTAN", "SINDH", "PUNJAB", 
            "ISLAMABAD", "ICT", "KARACHI", "LAHORE", "PESHAWAR", "QUETTA", "COCLAS"
        ]
        
        self.MIN_PLATE_AREA = 1000          
        self.MIN_BLUR_VAR = 35.0            
        self.STABILIZATION_FRAMES = 10      
        self.MAX_BUFFER_SIZE = 15
        
        os.makedirs(self.output_dir, exist_ok=True)

    def reset_session(self):
        self.track_buffers.clear()
        self.track_best_crops.clear()     # Reset session mein bhi clear ho jayega
        self.committed_tracks.clear()
        self.recently_detected.clear()
        self.frame_counter = 0

    def get_padded_crop(self, frame, box, pad_percent=0.12):
        h_frame, w_frame, _ = frame.shape
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * pad_percent), int(bh * pad_percent)
        nx1, ny1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        nx2, ny2 = min(w_frame, x2 + pad_x), min(h_frame, y2 + pad_y)
        return frame[ny1:ny2, nx1:nx2]

    def calculate_sharpness(self, img_gray):
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()

    def enhance_image_variants(self, cropped_plate):
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(resized, 11, 17, 17)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        v1 = clahe.apply(denoised)
        
        _, v2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        v3 = cv2.filter2D(v1, -1, kernel)
        
        return [
            cv2.cvtColor(v3, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(v1, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR),
            cropped_plate
        ]

    def clean_text(self, raw_text):
        clean = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper().strip()
        for word in self.blacklist_words:
            clean = clean.replace(word, '')
            
        if not clean:
            return ""

        digit_to_alpha = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '5': 'S',
            '8': 'B'
        }

        chars = list(clean)
        for i in range(min(3, len(chars))):
            if chars[i].isdigit() and chars[i] in digit_to_alpha:
                chars[i] = digit_to_alpha[chars[i]]

        cleaned_text = "".join(chars)

        # District code (1 ya 2 digits) ko hata kar main format (Letters + 4 Digits) banana
        match = re.match(r'^([A-Z]{2,3})\d{1,2}(\d{4})$', cleaned_text)
        if match:
            cleaned_text = match.group(1) + match.group(2)

        return cleaned_text

    def is_valid_plate(self, text):
        if not text or len(text) < 5 or len(text) > 10:
            return False
        if len(set(text)) == 1:
            return False
        return bool(re.match(r'^[A-Z]{2,3}[A-Z0-9]{3,7}$', text))

    def _extract_paddle_text(self, ocr_out):
        raw_text = ""
        if not ocr_out:
            return raw_text
        for item in ocr_out:
            if isinstance(item, list):
                for res in item:
                    if isinstance(res, tuple) and len(res) > 0 and isinstance(res[0], str):
                        raw_text += res[0]
                    elif isinstance(res, list) and len(res) > 1 and isinstance(res[1], (list, tuple)):
                        raw_text += str(res[1][0])
            elif isinstance(item, tuple) and len(item) > 0:
                raw_text += str(item[0])
        return raw_text

    def robust_multi_ocr(self, cropped_plate):
        variants = self.enhance_image_variants(cropped_plate)
        candidates = []
        
        for var in variants:
            try:
                out = self.reader.ocr(var)
                txt = self.clean_text(self._extract_paddle_text(out))
                if self.is_valid_plate(txt):
                    candidates.append(txt)
            except Exception:
                continue

        if not candidates:
            return ""

        valid_candidates = [c for c in candidates if len(c) >= 6]
        if valid_candidates:
            return max(valid_candidates, key=len)

        counts = Counter(candidates)
        return counts.most_common(1)[0][0]

    def process_video(self, frame, conf_threshold=0.20):
        self.frame_counter += 1
        annotated_frame = frame.copy()
        detections = []
        results = self.model.track(source=frame, conf=conf_threshold, persist=True, verbose=False)

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                if confidence < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else 1

                crop_w, crop_h = (x2 - x1), (y2 - y1)
                if (crop_w * crop_h) < self.MIN_PLATE_AREA:
                    continue

                cropped_plate = self.get_padded_crop(frame, (x1, y1, x2, y2), pad_percent=0.12)
                if cropped_plate.size == 0 or crop_h < 5 or crop_w < 5:
                    continue

                # 1. Sharpness / Blur Calculation
                gray_crop = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                sharpness = self.calculate_sharpness(gray_crop)

                # Agar frame blur hai toh skip kar do
                if sharpness < self.MIN_BLUR_VAR:
                    continue

                # 2. Track ki sabse best aur saaf image ko maintain karna
                if track_id not in self.track_best_crops:
                    self.track_best_crops[track_id] = {"crop": cropped_plate, "sharpness": sharpness}
                else:
                    if sharpness > self.track_best_crops[track_id]["sharpness"]:
                        self.track_best_crops[track_id] = {"crop": cropped_plate, "sharpness": sharpness}

                # Ab sirf sabse behtareen crop par OCR chalega
                best_crop_to_process = self.track_best_crops[track_id]["crop"]
                plate_number = self.robust_multi_ocr(best_crop_to_process)

                if self.is_valid_plate(plate_number):
                    self.track_buffers[track_id].append(plate_number)
                    if len(self.track_buffers[track_id]) > self.MAX_BUFFER_SIZE:
                        self.track_buffers[track_id].pop(0)

                candidates = self.track_buffers[track_id]
                live_winner = max(set(candidates), key=candidates.count) if candidates else plate_number

                # Box aur Text ko screen par show karne ke liye
                display_text = live_winner if live_winner else "Detecting..."
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, display_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if track_id not in self.committed_tracks:
                    if len(self.track_buffers[track_id]) >= 10 and len(live_winner) >= 5:
                        
                        if live_winner in self.recently_detected:
                            if (self.frame_counter - self.recently_detected[live_winner]) < self.COOLDOWN_FRAMES:
                                continue 

                        self.recently_detected[live_winner] = self.frame_counter
                        is_in_db = self.db.is_plate_registered(live_winner)
                        status_str = "Already Registered" if is_in_db else "New Entry Saved"

                        final_best_crop = self.track_best_crops[track_id]["crop"]
                        image_name = f"{live_winner}.jpg"
                        save_path = os.path.join(self.output_dir, image_name)
                        
                        if status_str == "New Entry Saved":
                            cv2.imwrite(save_path, final_best_crop)
                            self.db.insert_plate(live_winner, save_path, confidence)

                        self.committed_tracks.add(track_id)
                        detections.append({
                            "plate_number": live_winner,
                            "confidence": confidence,
                            "status": status_str,
                            "crop": final_best_crop
                        })

        return annotated_frame, detections

    def process_image(self, frame, conf_threshold=0.10):
        self.reset_session()
        annotated_frame = frame.copy()
        detections = []
        results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cropped_plate = self.get_padded_crop(frame, (x1, y1, x2, y2), pad_percent=0.12)
                if cropped_plate.size == 0:
                    continue
    
                gray_crop = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                if self.calculate_sharpness(gray_crop) < self.MIN_BLUR_VAR:
                    continue

                plate_number = self.robust_multi_ocr(cropped_plate)
    
                if not self.is_valid_plate(plate_number):
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    continue
    
                is_registered = self.db.is_plate_registered(plate_number)
                box_color = (0, 165, 255) if is_registered else (0, 255, 0)
    
                image_name = f"{plate_number}_{int(np.random.randint(1000,9999))}.jpg"
                save_path = os.path.join(self.output_dir, image_name)
                cv2.imwrite(save_path, cropped_plate)
                
                if not is_registered:
                    self.db.insert_plate(plate_number, save_path, confidence)
                    status_str = "New Entry Saved"
                else:
                    status_str = "Already Registered"
    
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, plate_number, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    
                detections.append({
                    "plate_number": plate_number,
                    "confidence": confidence,
                    "status": status_str,
                    "crop": cropped_plate
                })
    
        return annotated_frame, detections

    def process_frame(self, frame, conf_threshold=0.20, is_video=True):
        if not is_video:
            return self.process_image(frame, conf_threshold=conf_threshold)
        else:
            return self.process_video(frame, conf_threshold=conf_threshold)
