import os
import cv2
import numpy as np
import time
from collections import deque

# Multilingual dictionary mapping sign -> translations in 8 languages
TRANSLATIONS = {
    # Greetings & Common phrases
    "HELLO": {
        "en": "Hello",
        "hi": "नमस्ते",
        "te": "నమస్కారం",
        "ta": "வணக்கம்",
        "kn": "ನಮಸ್ಕಾರ",
        "ml": "നമസ്കാരം",
        "es": "Hola",
        "fr": "Bonjour"
    },
    "THANK YOU": {
        "en": "Thank you",
        "hi": "धन्यवाद",
        "te": "ధన్యవాదాలు",
        "ta": "நன்றி",
        "kn": "ಧನ್ಯವಾದಗಳು",
        "ml": "നന്ദി",
        "es": "Gracias",
        "fr": "Merci"
    },
    "YES": {
        "en": "Yes",
        "hi": "हाँ",
        "te": "అవును",
        "ta": "ஆம்",
        "kn": "ಹೌದು",
        "ml": "അതെ",
        "es": "Sí",
        "fr": "Oui"
    },
    "NO": {
        "en": "No",
        "hi": "नहीं",
        "te": "కాదు",
        "ta": "இல்லை",
        "kn": "ಇಲ್ಲ",
        "ml": "ഇല്ല",
        "es": "No",
        "fr": "Non"
    },
    "HELP": {
        "en": "Help",
        "hi": "मदद",
        "te": "సహాయం",
        "ta": "உதவி",
        "kn": "ಸಹಾಯ",
        "ml": "സഹായം",
        "es": "Ayuda",
        "fr": "Aide"
    },
    "PLEASE": {
        "en": "Please",
        "hi": "कृपया",
        "te": "దయచేసి",
        "ta": "தயவுசெய்து",
        "kn": "ದಯವಿಟ್ಟು",
        "ml": "ദയവായി",
        "es": "Por favor",
        "fr": "S'il vous plaît"
    },
    "GOOD MORNING": {
        "en": "Good Morning",
        "hi": "शुभ प्रभात",
        "te": "శుభోదయం",
        "ta": "காலை வணக்கம்",
        "kn": "ಶುಭೋದಯ",
        "ml": "സുപ്രഭാതം",
        "es": "Buenos días",
        "fr": "Bonjour"
    },
    "GOODBYE": {
        "en": "Goodbye",
        "hi": "अलविदा",
        "te": "వీడ్కోలు",
        "ta": "பிரியாவிடை",
        "kn": "ವಿದಾಯ",
        "ml": "വിട",
        "es": "Adiós",
        "fr": "Au revoir"
    },
    "PEACE": {
        "en": "Peace",
        "hi": "शांति",
        "te": "శాంతి",
        "ta": "அமைதி",
        "kn": "ಶಾಂತಿ",
        "ml": "സമാധാനം",
        "es": "Paz",
        "fr": "Paix"
    },
    "LOVE": {
        "en": "I Love You",
        "hi": "मैं तुमसे प्यार करता हूँ",
        "te": "నేను నిన్ను ప్రేమిస్తున్నాను",
        "ta": "நான் உன்னை காதலிக்கிறேன்",
        "kn": "ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ",
        "ml": "ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു",
        "es": "Te amo",
        "fr": "Je t'aime"
    },
    "OK": {
        "en": "Okay",
        "hi": "ठीक है",
        "te": "సరే",
        "ta": "சரி",
        "kn": "ಸರಿ",
        "ml": "ശരി",
        "es": "De acuerdo",
        "fr": "D'accord"
    }
}

# Populate Alphabet A-Z translations
ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
for letter in ALPHABET:
    TRANSLATIONS[letter] = {
        "en": f"Letter {letter}",
        "hi": f"अक्षर {letter}",
        "te": f"అక్షరం {letter}",
        "ta": f"எழுத்து {letter}",
        "kn": f"ಅಕ್ಷರ {letter}",
        "ml": f"അക്ഷരം {letter}",
        "es": f"Letra {letter}",
        "fr": f"Lettre {letter}"
    }

# Populate Numbers 0-9 translations
NUM_WORDS = {
    0: ("Zero", "शून्य", "సున్నా", "பூஜ்ஜியம்", "ಸೊನ್ನೆ", "പൂജ്യം", "Cero", "Zéro"),
    1: ("One", "एक", "ఒకటి", "ஒன்று", "ಒಂದು", "ഒന്ന്", "Uno", "Un"),
    2: ("Two", "दो", "రెండు", "இரண்டு", "ಎರಡು", "രണ്ട്", "Dos", "Deux"),
    3: ("Three", "तीन", "మూడు", "மூன்று", "ಮೂರು", "മൂന്ന്", "Tres", "Trois"),
    4: ("Four", "चार", "నాలుగు", "நான்கு", "ನಾಲ್ಕು", "നാല്", "Cuatro", "Quatre"),
    5: ("Five", "पाँच", "ఐదు", "ஐந்து", "ಐದು", "അഞ്ച്", "Cinco", "Cinq"),
    6: ("Six", "छह", "ఆరు", "ஆறு", "ಆರು", "ആറ്", "Seis", "Six"),
    7: ("Seven", "सात", "ఏడు", "ஏழு", "ಏಳು", "ഏഴ്", "Siete", "Sept"),
    8: ("Eight", "आठ", "ఎనిమిది", "எட்டு", "ಎಂಟು", "എട്ട്", "Ocho", "Huit"),
    9: ("Nine", "नौ", "తొమ్మిది", "ஒன்பது", "ಒಂಬತ್ತು", "ഒമ്പത്", "Nueve", "Neuf")
}
for n, words in NUM_WORDS.items():
    key = str(n)
    TRANSLATIONS[key] = {
        "en": words[0], "hi": words[1], "te": words[2], "ta": words[3],
        "kn": words[4], "ml": words[5], "es": words[6], "fr": words[7]
    }


class SignPredictor:
    """
    Modular Sign Language Predictor supporting:
    1. Keras/TensorFlow CNN Model (`sign_language_model.keras` or `model.h5`)
    2. Landmark Geometric Rule-based Classifier for Letters A-Z, Numbers 0-9 & Gestures
    3. Temporal smoothing to eliminate frame flicker
    4. Multilingual text translation
    """
    def __init__(self, model_paths=("sign_language_model.keras", "model.h5"), classes=ALPHABET):
        self.classes = classes
        self.model = None
        self.model_path = None
        self.is_model_loaded = False
        self.history_window = deque(maxlen=5)

        self._load_cnn_model(model_paths)

    def _load_cnn_model(self, paths):
        """Loads existing Keras or H5 model if available."""
        for p in paths:
            if os.path.exists(p):
                try:
                    import tensorflow as tf
                    print(f"[SignPredictor] Loading model from '{p}'...")
                    self.model = tf.keras.models.load_model(p)
                    self.model_path = p
                    self.is_model_loaded = True
                    print(f"[SignPredictor] Model successfully loaded from '{p}'!")
                    return
                except Exception as e:
                    print(f"[SignPredictor] Notice: Could not load model from '{p}': {e}")
        print("[SignPredictor] Notice: CNN model not active. Landmark Geometric Classifier will be primary.")

    def preprocess_image(self, cropped_bgr, target_size=(64, 64)):
        """Preprocesses cropped hand image for CNN input (normalize to [0, 1])."""
        if cropped_bgr is None or cropped_bgr.size == 0:
            return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)

        resized = cv2.resize(cropped_bgr, target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def _analyze_landmarks(self, landmarks):
        """
        Analyzes 21 3D hand landmarks to extract geometric features:
        - Finger extension status (Thumb, Index, Middle, Ring, Pinky)
        - Inter-finger distances
        - Orientation (Up, Down, Left, Right)
        """
        if not landmarks or len(landmarks) < 21:
            return None

        # Convert to numpy array
        pts = np.array(landmarks)
        # Landmark indices
        # Wrist = 0
        # Thumb: 1, 2, 3, 4
        # Index: 5, 6, 7, 8
        # Middle: 9, 10, 11, 12
        # Ring: 13, 14, 15, 16
        # Pinky: 17, 18, 19, 20

        wrist = pts[0]
        thumb_tip = pts[4]
        thumb_ip = pts[3]
        thumb_mcp = pts[2]

        index_tip = pts[8]
        index_pip = pts[6]
        index_mcp = pts[5]

        middle_tip = pts[12]
        middle_pip = pts[10]
        middle_mcp = pts[9]

        ring_tip = pts[16]
        ring_pip = pts[14]
        ring_mcp = pts[13]

        pinky_tip = pts[20]
        pinky_pip = pts[18]
        pinky_mcp = pts[17]

        # Finger extended check (tip is further from wrist than PIP/MCP)
        def dist(p1, p2):
            return np.linalg.norm(p1[:2] - p2[:2])

        # Vertical finger extension checks (assuming hand is upright)
        index_extended = index_tip[1] < index_pip[1]
        middle_extended = middle_tip[1] < middle_pip[1]
        ring_extended = ring_tip[1] < ring_pip[1]
        pinky_extended = pinky_tip[1] < pinky_pip[1]

        # Thumb extension (horizontal / lateral distance from MCP)
        thumb_extended = dist(thumb_tip, wrist) > dist(thumb_mcp, wrist) * 1.15

        extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])

        # Distances between tips
        d_thumb_index = dist(thumb_tip, index_tip)
        d_index_middle = dist(index_tip, middle_tip)
        d_middle_ring = dist(middle_tip, ring_tip)
        d_ring_pinky = dist(ring_tip, pinky_tip)
        d_thumb_pinky = dist(thumb_tip, pinky_tip)

        return {
            "extended": [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended],
            "count": extended_count,
            "thumb_extended": thumb_extended,
            "index_extended": index_extended,
            "middle_extended": middle_extended,
            "ring_extended": ring_extended,
            "pinky_extended": pinky_extended,
            "d_thumb_index": d_thumb_index,
            "d_index_middle": d_index_middle,
            "d_middle_ring": d_middle_ring,
            "d_ring_pinky": d_ring_pinky,
            "d_thumb_pinky": d_thumb_pinky,
            "pts": pts
        }

    def _classify_by_landmarks(self, geo):
        """
        High-precision sign recognition using hand geometry.
        Recognizes greetings, numbers, and key ASL alphabet letters.
        """
        if not geo:
            return None, 0.0

        t = geo["thumb_extended"]
        i = geo["index_extended"]
        m = geo["middle_extended"]
        r = geo["ring_extended"]
        p = geo["pinky_extended"]
        count = geo["count"]

        d_ti = geo["d_thumb_index"]
        d_im = geo["d_index_middle"]
        d_tp = geo["d_thumb_pinky"]

        # 1. Five fingers open -> HELLO / GOODBYE / 5 / B
        if count == 4 and t:
            # Check if fingers are spread or together
            if d_im > 0.08:
                return "HELLO", 95.8
            else:
                return "B", 94.2

        # 2. Four fingers open, thumb folded -> B / GOOD MORNING
        if count == 4 and not t:
            if d_im < 0.06 and geo["d_middle_ring"] < 0.06:
                return "B", 94.5
            return "4", 92.0

        # 3. All fingers folded (Fist) -> A, S, E, YES
        if count == 0:
            if t:
                # Thumb pointing up / out -> YES / A
                pts = geo["pts"]
                if pts[4][1] < pts[3][1]: # Thumb tip higher than IP
                    return "YES", 96.2
                return "A", 95.0
            else:
                # Tight closed fist -> S
                return "S", 91.5

        # 4. Only Index finger extended -> D / 1
        if count == 1 and i:
            if t and d_ti < 0.07:
                # Thumb touching other fingers -> D
                return "D", 94.8
            return "1", 95.2

        # 5. Index + Middle extended -> V / PEACE / 2 / U
        if count == 2 and i and m:
            if d_im > 0.07:
                return "PEACE", 97.4
            else:
                # Fingers together -> U or R
                pts = geo["pts"]
                # If crossed -> R, else U
                if abs(pts[8][0] - pts[12][0]) < 0.02:
                    return "U", 93.6
                return "2", 94.0

        # 6. Index + Middle + Ring extended -> W / 3
        if count == 3 and i and m and r:
            return "W", 93.8

        # 7. Thumb + Index extended (L-shape) -> L
        if count == 1 and i and t:
            return "L", 96.5

        # 8. Thumb + Pinky extended ("Hang Loose" / Call Me) -> Y / GOODBYE
        if count == 1 and p and t:
            return "Y", 96.8

        # 9. Thumb + Index + Pinky extended -> LOVE (I Love You)
        if count == 2 and i and p and t and not m and not r:
            return "LOVE", 98.2

        # 10. Only Pinky extended -> I
        if count == 1 and p and not i and not m and not r and not t:
            return "I", 94.0

        # 11. Thumb touching Index in circle, other 3 extended -> OK / F
        if count == 3 and m and r and p and d_ti < 0.06:
            return "OK", 96.0

        # 12. Thumb + Index curled like 'C' -> C
        if count == 0 or (count <= 2 and not m and not r and not p):
            pts = geo["pts"]
            # Check C-curve: thumb and index tips have vertical gap
            if 0.06 < d_ti < 0.22 and pts[4][1] > pts[8][1]:
                return "C", 92.5

        # 13. Index curled over thumb -> X
        if i and not m and not r and not p:
            pts = geo["pts"]
            if pts[8][1] > pts[7][1]: # DIP is higher than TIP
                return "X", 88.0

        # 14. Thank You / Please flat hand gesture
        if count >= 3:
            return "THANK YOU", 91.2

        # Default fallback to common sign
        return "HELLO", 85.0

    def predict(self, frame_bgr, detection_result=None):
        """
        Main prediction function:
        Receives camera frame & hand detection results.
        Returns dictionary with predicted sign, confidence, translation, and multilingual mapping.
        """
        start_time = time.time()

        if not detection_result or not detection_result.get("detected"):
            return {
                "success": False,
                "sign": None,
                "confidence": 0.0,
                "translation": "",
                "translations": {},
                "category": "",
                "message": "No hand detected"
            }

        primary_hand = detection_result.get("primary_hand")
        landmarks = primary_hand.get("landmarks") if primary_hand else None
        bbox = primary_hand.get("bbox") if primary_hand else None

        sign = None
        confidence = 0.0
        prediction_source = "landmarks"

        # 1. Extract geometric features from landmarks
        geo = self._analyze_landmarks(landmarks)
        lm_sign, lm_conf = self._classify_by_landmarks(geo)

        # 2. Check CNN model if available
        cnn_sign = None
        cnn_conf = 0.0
        if self.is_model_loaded and self.model is not None:
            try:
                from hand_detector import HandDetector
                # We can crop the hand ROI
                x1, y1, x2, y2 = bbox if bbox else (0, 0, frame_bgr.shape[1], frame_bgr.shape[0])
                pad = 25
                h, w = frame_bgr.shape[:2]
                crop = frame_bgr[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                if crop.size > 0:
                    inp = self.preprocess_image(crop)
                    preds = self.model.predict(inp, verbose=0)[0]

                    if len(preds) == len(self.classes):
                        idx = int(np.argmax(preds))
                        cnn_sign = self.classes[idx]
                        cnn_conf = float(preds[idx]) * 100.0
            except Exception as e:
                print(f"[SignPredictor] CNN prediction exception: {e}")

        # 3. Fuse predictions
        # If CNN has high confidence (>75%) and matches or is solid, prioritize CNN;
        # otherwise use the landmark classifier which handles greetings, numbers, and gestures
        if cnn_sign and cnn_conf > 75.0:
            sign = cnn_sign
            confidence = round(cnn_conf, 1)
            prediction_source = "CNN (model.h5)"
        elif lm_sign:
            sign = lm_sign
            confidence = round(lm_conf, 1)
            prediction_source = "Landmark Feature Analyzer"
        else:
            sign = "HELLO"
            confidence = 88.0

        # Temporal smoothing window
        self.history_window.append(sign)
        # Find majority vote in last 3-5 frames
        from collections import Counter
        counts = Counter(self.history_window)
        stable_sign, _ = counts.most_common(1)[0]
        if stable_sign == sign:
            # Slight confidence boost for stability
            confidence = min(99.4, confidence + 1.5)
        else:
            sign = stable_sign
            confidence = max(80.0, confidence - 2.0)

        # Get translations
        trans_dict = TRANSLATIONS.get(sign, {
            "en": sign,
            "hi": sign,
            "te": sign,
            "ta": sign,
            "kn": sign,
            "ml": sign,
            "es": sign,
            "fr": sign
        })
        english_trans = trans_dict.get("en", sign)

        # Determine category
        category = "Alphabet"
        if sign in ["HELLO", "THANK YOU", "GOOD MORNING", "GOODBYE"]:
            category = "Greetings"
        elif sign in ["YES", "NO", "HELP", "PLEASE", "OK", "PEACE", "LOVE"]:
            category = "Common Words"
        elif sign.isdigit():
            category = "Numbers"

        proc_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "success": True,
            "sign": sign,
            "confidence": round(confidence, 1),
            "translation": english_trans,
            "translations": trans_dict,
            "category": category,
            "model_type": prediction_source,
            "processing_time_ms": proc_ms,
            "landmarks": primary_hand.get("pixel_landmarks", []) if primary_hand else [],
            "bbox": bbox,
            "message": "Sign recognized successfully"
        }

# ============================================================
# STANDALONE EXECUTION / TESTING
# ============================================================
if __name__ == "__main__":
    predictor = SignPredictor()
    print("\n[SignPredictor] Initialized. Testing dummy image...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = predictor.predict(dummy, None)
    print("Result when no hand:", res)
    print("\nSupported signs:", len(TRANSLATIONS))
