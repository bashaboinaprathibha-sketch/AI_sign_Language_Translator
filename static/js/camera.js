/**
 * AI SIGN LANGUAGE TRANSLATOR - CAMERA MANAGER
 * Handles MediaDevices getUserMedia stream, error states, and canvas frame captures.
 */

class CameraManager {
    constructor(videoElementId, options = {}) {
        this.video = document.getElementById(videoElementId);
        this.stream = null;
        this.isActive = false;
        this.width = options.width || 640;
        this.height = options.height || 480;
        this.facingMode = options.facingMode || "user"; // 'user' or 'environment'

        // Hidden canvas used for extracting frame snapshots
        this.captureCanvas = document.createElement("canvas");
        this.captureCanvas.width = this.width;
        this.captureCanvas.height = this.height;
        this.captureCtx = this.captureCanvas.getContext("2d", { willReadFrequently: true });

        // Event callbacks
        this.onStart = options.onStart || (() => {});
        this.onStop = options.onStop || (() => {});
        this.onError = options.onError || (() => {});
    }

    /**
     * Request webcam access and start live stream.
     */
    async startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const err = new Error("Your browser does not support camera access.");
            this.handleCameraError(err);
            return false;
        }

        const constraints = {
            audio: false,
            video: {
                width: { ideal: this.width },
                height: { ideal: this.height },
                facingMode: this.facingMode
            }
        };

        try {
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            if (this.video) {
                this.video.srcObject = this.stream;
                await this.video.play();
                this.video.style.display = "block";
            }
            this.isActive = true;
            this.onStart();
            console.log("[CameraManager] Camera stream started successfully.");
            return true;
        } catch (error) {
            this.handleCameraError(error);
            return false;
        }
    }

    /**
     * Stop webcam stream and release hardware tracks.
     */
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (e) {
                    console.warn("[CameraManager] Error stopping track:", e);
                }
            });
            this.stream = null;
        }

        if (this.video) {
            this.video.pause();
            this.video.srcObject = null;
            this.video.style.display = "none";
        }

        this.isActive = false;
        this.onStop();
        console.log("[CameraManager] Camera stopped.");
    }

    /**
     * Captures current video frame as base64 JPEG image string.
     * @returns {string|null} base64 data URL
     */
    captureFrameBase64(quality = 0.8) {
        if (!this.isActive || !this.video || this.video.readyState < 2) {
            return null;
        }

        const vw = this.video.videoWidth || this.width;
        const vh = this.video.videoHeight || this.height;

        if (this.captureCanvas.width !== vw || this.captureCanvas.height !== vh) {
            this.captureCanvas.width = vw;
            this.captureCanvas.height = vh;
        }

        // Draw video frame to canvas
        this.captureCtx.drawImage(this.video, 0, 0, vw, vh);
        return this.captureCanvas.toDataURL("image/jpeg", quality);
    }

    /**
     * Captures current video frame as Blob for multipart form upload.
     * @returns {Promise<Blob|null>}
     */
    async captureFrameBlob(quality = 0.8) {
        if (!this.isActive || !this.video || this.video.readyState < 2) {
            return null;
        }

        const vw = this.video.videoWidth || this.width;
        const vh = this.video.videoHeight || this.height;

        if (this.captureCanvas.width !== vw || this.captureCanvas.height !== vh) {
            this.captureCanvas.width = vw;
            this.captureCanvas.height = vh;
        }

        this.captureCtx.drawImage(this.video, 0, 0, vw, vh);

        return new Promise(resolve => {
            this.captureCanvas.toBlob(blob => resolve(blob), "image/jpeg", quality);
        });
    }

    /**
     * Graceful camera error messaging per specifications.
     */
    handleCameraError(error) {
        let userMessage = "Unable to access camera.";

        if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
            userMessage = "Camera permission is required. Please grant camera permission in your browser.";
        } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
            userMessage = "Camera is unavailable. Please connect a camera and try again.";
        } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
            userMessage = "Camera is already in use by another application. Please close other camera apps.";
        } else if (error.name === "OverconstrainedError") {
            userMessage = "Requested camera resolution is not supported by your camera hardware.";
        } else if (error.message && error.message.includes("does not support")) {
            userMessage = "Your browser does not support camera access.";
        }

        console.error("[CameraManager] Error:", error);
        this.onError(userMessage, error);
    }
}

// Global camera instance exporter
window.CameraManager = CameraManager;
