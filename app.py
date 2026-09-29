import os
import sys
import base64
import time
from datetime import datetime

import numpy as np
import cv2

from flask import Flask, render_template, request, jsonify


# ============================================================
# IMPORT AI MODULES
# ============================================================

try:
    from hand_detector import HandDetector
except ImportError:
    HandDetector = None


try:
    from predict import SignPredictor, TRANSLATIONS
except ImportError:
    SignPredictor = None
    TRANSLATIONS = {}


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# AI INITIALIZATION
# ============================================================

print("\n[AI Sign Language Translator] Initializing AI models...")

hand_detector = None
sign_predictor = None


# Hand detector
try:

    if HandDetector:

        model_task_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "hand_landmarker.task"
        )

        if os.path.exists(model_task_path):

            hand_detector = HandDetector(
                model_path=model_task_path
            )

            print(
                "[AI Pipeline] Hand detector initialized."
            )

        else:

            print(
                "[AI Pipeline] hand_landmarker.task not found."
            )

except Exception as e:

    print(
        f"[AI Pipeline] Warning initializing "
        f"hand detector: {e}"
    )


# Sign predictor
try:

    if SignPredictor:

        sign_predictor = SignPredictor()

        print(
            "[AI Pipeline] Sign predictor initialized."
        )

except Exception as e:

    print(
        f"[AI Pipeline] Warning initializing "
        f"sign predictor: {e}"
    )


# ============================================================
# TRANSLATION HISTORY
# ============================================================

translation_history = []

MAX_HISTORY = 50


# ============================================================
# DICTIONARY
# ============================================================

DICTIONARY_ITEMS = [

    {
        "sign": "HELLO",
        "name": "Hello / Hi",
        "category": "Greetings",
        "description": "Open hand with fingers spread.",
        "icon": "wave"
    },

    {
        "sign": "THANK YOU",
        "name": "Thank You",
        "category": "Greetings",
        "description": "Hand moves forward from chin.",
        "icon": "hand-heart"
    },

    {
        "sign": "YES",
        "name": "Yes",
        "category": "Common Words",
        "description": "Closed fist with thumb upright.",
        "icon": "thumbs-up"
    },

    {
        "sign": "NO",
        "name": "No",
        "category": "Common Words",
        "description": "Index and middle fingers move toward thumb.",
        "icon": "x-circle"
    },

    {
        "sign": "HELP",
        "name": "Help",
        "category": "Common Words",
        "description": "Fist with thumb up on open palm.",
        "icon": "life-buoy"
    },

    {
        "sign": "PLEASE",
        "name": "Please",
        "category": "Common Words",
        "description": "Flat hand makes circular motion.",
        "icon": "smile"
    },

    {
        "sign": "OK",
        "name": "Okay",
        "category": "Common Words",
        "description": "Thumb and index finger form an O.",
        "icon": "check-circle"
    }

]


# Add A-Z
for code in range(ord("A"), ord("Z") + 1):

    letter = chr(code)

    DICTIONARY_ITEMS.append({
        "sign": letter,
        "name": f"Letter {letter}",
        "category": "Alphabet",
        "description":
            f"ASL manual alphabet hand shape for {letter}.",
        "icon": "type"
    })


# ============================================================
# IMAGE DECODER
# ============================================================

def decode_image_from_request(req):

    # Multipart file
    if "frame" in req.files:

        file = req.files["frame"]

        file_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        return cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


    if "image" in req.files:

        file = req.files["image"]

        file_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        return cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


    # JSON
    if req.is_json:

        data = req.get_json(
            silent=True
        ) or {}

        image_data = (
            data.get("image")
            or data.get("frame")
        )

        if image_data:

            if "," in image_data:

                image_data = image_data.split(
                    ",",
                    1
                )[1]

            try:

                decoded = base64.b64decode(
                    image_data
                )

                np_arr = np.frombuffer(
                    decoded,
                    np.uint8
                )

                return cv2.imdecode(
                    np_arr,
                    cv2.IMREAD_COLOR
                )

            except Exception as e:

                print(
                    f"Image decoding error: {e}"
                )


    # Form data
    raw_data = (
        req.form.get("image")
        or req.form.get("frame")
    )

    if raw_data:

        if "," in raw_data:

            raw_data = raw_data.split(
                ",",
                1
            )[1]

        try:

            decoded = base64.b64decode(
                raw_data
            )

            np_arr = np.frombuffer(
                decoded,
                np.uint8
            )

            return cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

        except Exception as e:

            print(
                f"Form image decoding error: {e}"
            )


    return None


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template(
        "dashboard.html",
        active_page="home"
    )


@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        active_page="dashboard"
    )


@app.route("/translator")
def translator():

    return render_template(
        "translator.html",
        active_page="translator"
    )


@app.route("/dictionary")
def dictionary():

    return render_template(
        "dictionary.html",
        active_page="dictionary",
        signs=DICTIONARY_ITEMS
    )


@app.route("/about")
def about():

    return render_template(
        "about.html",
        active_page="about"
    )


# ============================================================
# STATUS API
# ============================================================

@app.route("/api/status")
def api_status():

    return jsonify({

        "status": "online",

        "api": "connected",

        "model":
            "ready"
            if sign_predictor
            else "loading",

        "camera": "available",

        "voice": "available",

        "detection":
            "active"
            if hand_detector
            else "waiting",

        "total_signs":
            len(DICTIONARY_ITEMS),

        "timestamp":
            datetime.now().isoformat()

    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    global translation_history

    try:

        frame = decode_image_from_request(
            request
        )


        if frame is None:

            return jsonify({

                "success": False,

                "sign": None,

                "confidence": 0,

                "translation": "",

                "message":
                    "Invalid or missing camera frame"

            }), 400


        # Hand detection
        if not hand_detector:

            return jsonify({

                "success": False,

                "sign": None,

                "confidence": 0,

                "translation": "",

                "message":
                    "Hand detector unavailable"

            }), 503


        detection_result = (
            hand_detector.detect(frame)
        )


        if not detection_result:

            return jsonify({

                "success": False,

                "sign": None,

                "confidence": 0,

                "translation": "",

                "message":
                    "No hand detected"

            })


        if not detection_result.get(
            "detected"
        ):

            return jsonify({

                "success": False,

                "sign": None,

                "confidence": 0,

                "translation": "",

                "message":
                    "No hand detected"

            })


        # AI prediction
        if not sign_predictor:

            return jsonify({

                "success": False,

                "sign": None,

                "confidence": 0,

                "translation": "",

                "message":
                    "AI predictor unavailable"

            }), 503


        result = sign_predictor.predict(
            frame,
            detection_result
        )


        # Save history
        if (
            result.get("success")
            and result.get("sign")
        ):

            record = {

                "id":
                    len(translation_history) + 1,

                "time":
                    datetime.now().strftime(
                        "%I:%M %p"
                    ),

                "sign":
                    result.get("sign"),

                "translation":
                    result.get(
                        "translation",
                        result.get("sign")
                    ),

                "confidence":
                    f"{result.get('confidence', 0)}%",

                "category":
                    result.get(
                        "category",
                        "General"
                    )

            }

            translation_history.insert(
                0,
                record
            )


            if len(
                translation_history
            ) > MAX_HISTORY:

                translation_history = (
                    translation_history[
                        :MAX_HISTORY
                    ]
                )


        return jsonify(result)


    except Exception as e:

        print(
            f"[Error in /predict]: {e}"
        )

        return jsonify({

            "success": False,

            "sign": None,

            "confidence": 0,

            "translation": "",

            "message":
                str(e)

        }), 500


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
def get_history():

    return jsonify({

        "success": True,

        "count":
            len(translation_history),

        "history":
            translation_history

    })


@app.route(
    "/api/history/clear",
    methods=["POST"]
)
def clear_history():

    global translation_history

    translation_history = []

    return jsonify({

        "success": True,

        "message":
            "Translation history cleared successfully."

    })


# ============================================================
# DICTIONARY API
# ============================================================

@app.route(
    "/api/dictionary",
    methods=["GET"]
)
def get_dictionary():

    category = request.args.get(
        "category"
    )

    items = DICTIONARY_ITEMS


    if category and category.lower() != "all":

        items = [

            item

            for item in DICTIONARY_ITEMS

            if item["category"].lower()
            == category.lower()

        ]


    return jsonify({

        "success": True,

        "count":
            len(items),

        "signs":
            items

    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        f"\n🚀 AI Sign Language Translator running at:"
        f" http://127.0.0.1:{port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )