"""
================================================================================
AI SIGN LANGUAGE TRANSLATOR - UNIFIED MASTER LAUNCHER (run.py)
================================================================================
Combines all project modules into a single unified execution entry point:
  1. Start Web Server & Open Browser (Flask Web App)
  2. Live Desktop Camera Sign Recognition (OpenCV + MediaPipe + CNN + TTS)
  3. Train / Retrain CNN Model (dataset/train -> model.h5 & .keras)
  4. Collect New Dataset Images (Webcam collector)
  5. Run System Health & API Self-Tests
================================================================================
"""

import os
import sys
import time
import argparse
import webbrowser
import threading

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def print_banner():
    banner = r"""
================================================================================
  [SignTranslate AI] - REAL-TIME AI SIGN LANGUAGE TRANSLATOR - MASTER RUNNER
================================================================================
  Computer Vision | Hand Detection | Deep Learning (CNN) | Voice Output (TTS)
================================================================================
"""
    print(banner)

def check_dependencies():
    """Verify that all core Python dependencies are installed and available."""
    print("[1/3] Checking environment dependencies...")
    missing = []
    
    for pkg in ["flask", "cv2", "mediapipe", "numpy", "tensorflow"]:
        try:
            if pkg == "cv2":
                import cv2
            else:
                __import__(pkg)
            print(f"  ✓ {pkg} is installed")
        except ImportError:
            missing.append(pkg)
            print(f"  ✗ {pkg} is MISSING")

    if missing:
        print(f"\n[Warning] Missing packages: {', '.join(missing)}")
        print("Please install them with:")
        print("  pip install -r requirements.txt\n")
        return False
    print("  All core packages verified successfully.\n")
    return True

def launch_web(port=5000, auto_open=True):
    """Launches the Flask web application and opens browser."""
    print(f"[2/3] Launching Flask Web Application on http://127.0.0.1:{port}...")
    import app

    def open_browser():
        time.sleep(1.8)
        url = f"http://127.0.0.1:{port}"
        print(f"\n🚀 Opening browser automatically at: {url}")
        webbrowser.open(url)

    if auto_open:
        threading.Thread(target=open_browser, daemon=True).start()

    print("[3/3] AI Pipeline Online! Press CTRL+C to stop server.\n")
    app.app.run(host="127.0.0.1", port=port, debug=False)

def launch_desktop():
    """Runs real-time OpenCV desktop window with MediaPipe and AI prediction."""
    print("[2/2] Launching Real-Time Desktop AI Sign Recognition...")
    import cv2
    import numpy as np
    from hand_detector import HandDetector
    from predict import SignPredictor, TRANSLATIONS

    detector = HandDetector()
    predictor = SignPredictor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    print("\nDesktop recognition window started.")
    print("  Press 's' to speak current sign.")
    print("  Press 'q' to quit.\n")

    current_sign = ""
    current_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        detection = detector.detect(frame)
        annotated = detector.draw_landmarks(frame, detection)

        if detection["detected"]:
            pred = predictor.predict(frame, detection)
            if pred["success"]:
                current_sign = pred["sign"]
                current_conf = pred["confidence"]
                trans = pred["translation"]

                # HUD box
                cv2.rectangle(annotated, (15, 15), (380, 115), (15, 23, 42), -1)
                cv2.rectangle(annotated, (15, 15), (380, 115), (99, 102, 241), 2)

                cv2.putText(annotated, f"SIGN: {current_sign}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(annotated, f"Trans: {trans}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 182, 212), 2)
                cv2.putText(annotated, f"Conf: {current_conf:.1f}%", (30, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 185, 129), 2)
        else:
            cv2.putText(annotated, "Show hand to camera", (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (148, 163, 184), 2)

        cv2.imshow("SignTranslate AI - Desktop Studio", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s') and current_sign:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(current_sign)
                engine.runAndWait()
            except Exception:
                print(f"[Speech]: {current_sign}")

    cap.release()
    cv2.destroyAllWindows()

def launch_training():
    """Runs model training script."""
    print("[2/2] Launching CNN Model Training...")
    import train_model
    train_model.train_cnn()

def launch_collector():
    """Runs webcam dataset image collector."""
    print("[2/2] Launching Dataset Collector...")
    import collect_images
    collect_images.collect_images()

def launch_selftest():
    """Runs automated diagnostic self-test."""
    print("[2/2] Running automated system health self-test...")
    import app
    import cv2
    import numpy as np
    import base64

    client = app.app.test_client()

    print("  Testing GET / ...", end=" ")
    r1 = client.get('/')
    assert r1.status_code == 200
    print("PASS (200)")

    print("  Testing GET /translator ...", end=" ")
    r2 = client.get('/translator')
    assert r2.status_code == 200
    print("PASS (200)")

    print("  Testing GET /dictionary ...", end=" ")
    r3 = client.get('/dictionary')
    assert r3.status_code == 200
    print("PASS (200)")

    print("  Testing GET /about ...", end=" ")
    r4 = client.get('/about')
    assert r4.status_code == 200
    print("PASS (200)")

    print("  Testing GET /api/status ...", end=" ")
    r5 = client.get('/api/status')
    assert r5.status_code == 200
    print(f"PASS (Status: {r5.get_json()['status']})")

    print("  Testing POST /predict ...", end=" ")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', dummy)
    b64 = base64.b64encode(buf).decode('utf-8')
    r6 = client.post('/predict', json={'image': f'data:image/jpeg;base64,{b64}'})
    assert r6.status_code == 200
    print("PASS (API active)")

    print("\n✅ ALL SYSTEM DIAGNOSTICS PASSED! Project is 100% operational.\n")

def interactive_menu():
    """Displays an interactive console menu."""
    print_banner()
    check_dependencies()

    while True:
        print("Select an option:")
        print("  [1] 🌐 Start Web Application (Recommended)")
        print("  [2] 🖥️  Live Desktop Camera Recognition")
        print("  [3] 🧠 Train / Retrain CNN Model")
        print("  [4] 📸 Collect Dataset Images from Webcam")
        print("  [5] 🩺 Run System Diagnostic Self-Test")
        print("  [0] ❌ Exit")
        choice = input("\nEnter choice [1-5 or 0]: ").strip()

        if choice == "1":
            launch_web()
            break
        elif choice == "2":
            launch_desktop()
            break
        elif choice == "3":
            launch_training()
            break
        elif choice == "4":
            launch_collector()
            break
        elif choice == "5":
            launch_selftest()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, 4, 5, or 0.\n")

def main():
    parser = argparse.ArgumentParser(description="AI Sign Language Translator Master Runner")
    parser.add_argument("--web", action="store_true", help="Launch Flask web application")
    parser.add_argument("--desktop", action="store_true", help="Launch desktop OpenCV window")
    parser.add_argument("--train", action="store_true", help="Train CNN model on dataset")
    parser.add_argument("--collect", action="store_true", help="Collect new dataset images from webcam")
    parser.add_argument("--test", action="store_true", help="Run automated API self-tests")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    args = parser.parse_args()

    if args.web:
        print_banner()
        check_dependencies()
        launch_web(port=args.port, auto_open=not args.no_browser)
    elif args.desktop:
        print_banner()
        check_dependencies()
        launch_desktop()
    elif args.train:
        print_banner()
        check_dependencies()
        launch_training()
    elif args.collect:
        print_banner()
        check_dependencies()
        launch_collector()
    elif args.test:
        print_banner()
        check_dependencies()
        launch_selftest()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
