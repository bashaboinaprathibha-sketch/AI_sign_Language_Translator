import os
import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    """
    Robust, modular hand detector supporting both MediaPipe Tasks API
    (hand_landmarker.task) and MediaPipe Solutions API as fallback.
    """
    def __init__(self, model_path="models/hand_landmarker.task", num_hands=2, min_confidence=0.5):
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_confidence = min_confidence
        self.detector_type = "none"
        self.task_detector = None
        self.solutions_detector = None

        self._init_detector()

    def _init_detector(self):
        # 1. Try MediaPipe Tasks API if task model exists
        if os.path.exists(self.model_path):
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=self.min_confidence,
                    min_hand_presence_confidence=self.min_confidence,
                    min_tracking_confidence=self.min_confidence
                )
                self.task_detector = vision.HandLandmarker.create_from_options(options)
                self.detector_type = "tasks"
                print(f"[HandDetector] Initialized MediaPipe Tasks HandLandmarker from '{self.model_path}'")
                return
            except Exception as e:
                print(f"[HandDetector] Warning: Could not initialize Tasks API: {e}. Falling back to Solutions Hands.")

        # 2. Try MediaPipe Solutions API
        try:
            mp_hands = mp.solutions.hands
            self.solutions_detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.num_hands,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence
            )
            self.detector_type = "solutions"
            print("[HandDetector] Initialized MediaPipe Solutions Hands.")
        except Exception as e:
            print(f"[HandDetector] Warning: Solutions API failed: {e}. Hand detector operating in fallback mode.")
            self.detector_type = "none"

    def detect(self, frame):
        """
        Detect hands in BGR image.
        Returns:
            dict containing:
                'detected': bool
                'num_hands': int
                'hands': list of hand info dicts
                'primary_hand': first detected hand or None
        """
        if frame is None or frame.size == 0:
            return {"detected": False, "num_hands": 0, "hands": [], "primary_hand": None}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_list = []

        if self.detector_type == "tasks" and self.task_detector:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.task_detector.detect(mp_image)

                if result and result.hand_landmarks:
                    for idx, hand in enumerate(result.hand_landmarks):
                        landmarks = []
                        pixel_lms = []
                        xs, ys = [], []

                        for lm in hand:
                            landmarks.append([float(lm.x), float(lm.y), float(lm.z)])
                            px = int(np.clip(lm.x * w, 0, w - 1))
                            py = int(np.clip(lm.y * h, 0, h - 1))
                            pixel_lms.append([px, py])
                            xs.append(px)
                            ys.append(py)

                        x_min = max(0, min(xs))
                        y_min = max(0, min(ys))
                        x_max = min(w, max(xs))
                        y_max = min(h, max(ys))

                        handedness = "Right"
                        if hasattr(result, "handedness") and result.handedness and idx < len(result.handedness):
                            handedness = result.handedness[idx][0].category_name

                        hands_list.append({
                            "landmarks": landmarks,
                            "pixel_landmarks": pixel_lms,
                            "bbox": [x_min, y_min, x_max, y_max],
                            "handedness": handedness
                        })
            except Exception as e:
                print(f"[HandDetector] Tasks detection error: {e}")

        elif self.detector_type == "solutions" and self.solutions_detector:
            try:
                result = self.solutions_detector.process(rgb)
                if result and result.multi_hand_landmarks:
                    for idx, hand in enumerate(result.multi_hand_landmarks):
                        landmarks = []
                        pixel_lms = []
                        xs, ys = [], []

                        for lm in hand.landmark:
                            landmarks.append([float(lm.x), float(lm.y), float(lm.z)])
                            px = int(np.clip(lm.x * w, 0, w - 1))
                            py = int(np.clip(lm.y * h, 0, h - 1))
                            pixel_lms.append([px, py])
                            xs.append(px)
                            ys.append(py)

                        x_min = max(0, min(xs))
                        y_min = max(0, min(ys))
                        x_max = min(w, max(xs))
                        y_max = min(h, max(ys))

                        handedness = "Right"
                        if result.multi_handedness and idx < len(result.multi_handedness):
                            handedness = result.multi_handedness[idx].classification[0].label

                        hands_list.append({
                            "landmarks": landmarks,
                            "pixel_landmarks": pixel_lms,
                            "bbox": [x_min, y_min, x_max, y_max],
                            "handedness": handedness
                        })
            except Exception as e:
                print(f"[HandDetector] Solutions detection error: {e}")

        detected = len(hands_list) > 0
        primary_hand = hands_list[0] if detected else None

        return {
            "detected": detected,
            "num_hands": len(hands_list),
            "hands": hands_list,
            "primary_hand": primary_hand
        }

    def crop_hand(self, frame, bbox=None, target_size=(64, 64), padding=30):
        """
        Crops the hand region of interest (ROI) with clean padding and aspect ratio handling,
        resizing to target_size for CNN inference.
        """
        if frame is None or frame.size == 0:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        if bbox is None:
            # Fallback to center crop
            cx, cy = w // 2, h // 2
            size = min(w, h) // 2
            x1, y1 = max(0, cx - size // 2), max(0, cy - size // 2)
            x2, y2 = min(w, cx + size // 2), min(h, cy + size // 2)
        else:
            x1, y1, x2, y2 = bbox
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            # Make it square if possible
            bw = x2 - x1
            bh = y2 - y1
            diff = abs(bw - bh)
            if bw > bh:
                y1 = max(0, y1 - diff // 2)
                y2 = min(h, y2 + (diff - diff // 2))
            elif bh > bw:
                x1 = max(0, x1 - diff // 2)
                x2 = min(w, x2 + (diff - diff // 2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        return resized

    def draw_landmarks(self, frame, detection_result, draw_bbox=True):
        """Draw hand landmarks, bones, and bounding box on frame."""
        annotated = frame.copy()
        if not detection_result or not detection_result.get("detected"):
            return annotated

        # Hand skeleton connections (21 landmarks)
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Palm base
        ]

        for hand in detection_result["hands"]:
            pixel_lms = hand["pixel_landmarks"]
            bbox = hand["bbox"]

            # Draw bones
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if p1_idx < len(pixel_lms) and p2_idx < len(pixel_lms):
                    pt1 = tuple(pixel_lms[p1_idx])
                    pt2 = tuple(pixel_lms[p2_idx])
                    cv2.line(annotated, pt1, pt2, (16, 185, 129), 2)  # Teal line

            # Draw landmark dots
            for idx, pt in enumerate(pixel_lms):
                color = (255, 255, 255) if idx in [4, 8, 12, 16, 20] else (59, 130, 246)
                radius = 5 if idx in [4, 8, 12, 16, 20] else 3
                cv2.circle(annotated, tuple(pt), radius, color, -1)
                cv2.circle(annotated, tuple(pt), radius + 1, (15, 23, 42), 1)

            # Draw bounding box
            if draw_bbox and bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (99, 102, 241), 2)
                label = f"{hand.get('handedness', 'Hand')}"
                cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (99, 102, 241), 2)

        return annotated

# ============================================================
# STANDALONE EXECUTION (PRESERVES ORIGINAL SCRIPT BEHAVIOR)
# ============================================================
if __name__ == "__main__":
    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    print("\n[HandDetector] Live webcam test started. Press 'q' to quit.")
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        res = detector.detect(frame)
        annotated = detector.draw_landmarks(frame, res)

        cv2.putText(annotated, f"Hands: {res['num_hands']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (16, 185, 129), 2)

        cv2.imshow("AI Sign Language Translator - Hand Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
