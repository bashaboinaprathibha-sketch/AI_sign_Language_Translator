"""
AI Sign Language Translator - Dataset Image Collector
Captures hand gesture training samples from webcam and saves them in dataset/train/<SIGN>
"""

import os
import cv2
import time
from hand_detector import HandDetector

DATASET_PATH = "dataset/train"
NUM_IMAGES = 100

def collect_images():
    print("==================================================")
    print(" AI Sign Language - Dataset Image Collector")
    print("==================================================")
    sign = input("Enter the sign/symbol to collect (e.g., A, B, HELLO, 1): ").strip().upper()
    if not sign:
        print("Invalid sign name.")
        return

    save_path = os.path.join(DATASET_PATH, sign)
    os.makedirs(save_path, exist_ok=True)
    print(f"Target directory: {save_path}")
    print(f"Target count: {NUM_IMAGES} images")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    detector = HandDetector()
    count = len(os.listdir(save_path))
    started = False

    print("\nControls:")
    print("  SPACE : Start auto-capture")
    print("  Q     : Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = detector.detect(frame)
        display = detector.draw_landmarks(frame, res)

        # Status text
        status_color = (16, 185, 129) if started else (245, 158, 11)
        cv2.putText(display, f"Sign: {sign}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(display, f"Images: {count}/{NUM_IMAGES}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        prompt = "Recording... Keep hand steady" if started else "Press SPACE to start recording"
        cv2.putText(display, prompt, (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (239, 68, 68) if started else (200, 200, 200), 2)

        cv2.imshow("Sign Language Dataset Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            started = True
        elif key == ord('q'):
            break

        if started and count < NUM_IMAGES:
            # Crop hand if detected, or center
            if res.get("detected"):
                crop = detector.crop_hand(frame, res["primary_hand"]["bbox"], target_size=(64, 64))
            else:
                crop = cv2.resize(frame, (64, 64))

            filename = os.path.join(save_path, f"{sign}_{count + 1}.jpg")
            cv2.imwrite(filename, crop)
            count += 1
            time.sleep(0.08)

        if count >= NUM_IMAGES:
            print(f"\n✅ Completed collecting {NUM_IMAGES} images for sign '{sign}'!")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()
