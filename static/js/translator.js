/**
 * AI SIGN LANGUAGE TRANSLATOR - REAL-TIME PREDICTOR & SPEECH ENGINE
 * Orchestrates real-time frame transmission to /predict, confidence UI,
 * Web Speech API Text-to-Speech in 8 languages, and history sync.
 */

class TranslatorEngine {
    constructor() {
        this.cameraManager = null;
        this.predictionInterval = null;
        this.intervalMs = 400; // 400ms prediction cycle (2.5 FPS)
        this.isProcessing = false;
        this.autoSpeak = false;
        this.lastSpokenSign = null;
        this.signStabilityCount = 0;
        this.currentLanguage = localStorage.getItem("signtranslate_lang") || "en-US";
        this.langKey = this._getLangKey(this.currentLanguage);
        this.drawLandmarksEnabled = true;

        // UI Element References
        this.elements = {
            // Camera
            video: document.getElementById("camera"),
            overlayCanvas: document.getElementById("cameraOverlayCanvas"),
            placeholder: document.getElementById("cameraPlaceholder"),
            startBtn: document.getElementById("startCamera"),
            stopBtn: document.getElementById("stopCamera"),
            captureBtn: document.getElementById("captureFrame"),
            cameraStatusDot: document.getElementById("cameraStatusDot"),
            cameraStatusText: document.getElementById("cameraStatusText"),

            // AI Prediction
            predictionSign: document.getElementById("predictionSign"),
            predictionSignIcon: document.getElementById("predictionSignIcon"),
            confidenceValue: document.getElementById("confidenceValue"),
            confidenceBar: document.getElementById("confidenceBar"),
            translationText: document.getElementById("translationText"),
            multilingualSub: document.getElementById("multilingualSub"),
            predictionStatus: document.getElementById("predictionStatus"),
            aiModelStatusText: document.getElementById("aiModelStatusText"),
            aiModelBadge: document.getElementById("aiModelBadge"),

            // Actions & Audio
            speakBtn: document.getElementById("speakButton"),
            stopVoiceBtn: document.getElementById("stopVoiceButton"),
            langSelector: document.getElementById("langSelector"),
            autoSpeakToggle: document.getElementById("autoSpeakToggle"),
            toggleLandmarksBtn: document.getElementById("toggleLandmarksBtn"),

            // Status Cards
            cardCameraStatus: document.getElementById("cardCameraStatus"),
            cardCameraDot: document.getElementById("cardCameraDot"),
            cardModelStatus: document.getElementById("cardModelStatus"),
            cardModelDot: document.getElementById("cardModelDot"),
            cardDetectionStatus: document.getElementById("cardDetectionStatus"),
            cardDetectionDot: document.getElementById("cardDetectionDot"),
            cardVoiceStatus: document.getElementById("cardVoiceStatus"),
            cardApiStatus: document.getElementById("cardApiStatus"),
            cardApiDot: document.getElementById("cardApiDot"),

            // History
            historyTableBody: document.getElementById("historyTableBody"),
            clearHistoryBtn: document.getElementById("clearHistoryBtn"),
            emptyHistoryMsg: document.getElementById("emptyHistoryMsg"),

            // Sentence Builder (Translator Studio)
            sentenceOutput: document.getElementById("sentenceOutput"),
            speakSentenceBtn: document.getElementById("speakSentenceBtn"),
            copySentenceBtn: document.getElementById("copySentenceBtn"),
            clearSentenceBtn: document.getElementById("clearSentenceBtn"),

            // Checklist items
            chkCamera: document.getElementById("chkCamera"),
            chkHand: document.getElementById("chkHand"),
            chkFeatures: document.getElementById("chkFeatures"),
            chkModel: document.getElementById("chkModel"),
            chkPrediction: document.getElementById("chkPrediction")
        };

        this.overlayCtx = this.elements.overlayCanvas ? this.elements.overlayCanvas.getContext("2d") : null;
        this.recentHistory = [];
        this.sentenceWords = [];

        this._init();
    }

    _init() {
        this._setupCameraManager();
        this._bindEvents();
        this._checkApiStatus();
        this._loadHistory();

        // Restore language selection in dropdown
        if (this.elements.langSelector) {
            this.elements.langSelector.value = this.currentLanguage;
        }
    }

    _getLangKey(bcp47) {
        const map = {
            "en-US": "en",
            "hi-IN": "hi",
            "te-IN": "te",
            "ta-IN": "ta",
            "kn-IN": "kn",
            "ml-IN": "ml",
            "es-ES": "es",
            "fr-FR": "fr"
        };
        return map[bcp47] || "en";
    }

    _setupCameraManager() {
        if (!this.elements.video) return;

        this.cameraManager = new CameraManager("camera", {
            width: 640,
            height: 480,
            onStart: () => {
                if (this.elements.placeholder) this.elements.placeholder.style.display = "none";
                if (this.elements.overlayCanvas) {
                    this.elements.overlayCanvas.style.display = "block";
                    this.elements.overlayCanvas.width = this.elements.video.videoWidth || 640;
                    this.elements.overlayCanvas.height = this.elements.video.videoHeight || 480;
                }
                this._updateCameraStatus(true, "Camera Active");
                this._setChecklist("chkCamera", true);
                this.startPredictionLoop();
            },
            onStop: () => {
                if (this.elements.placeholder) this.elements.placeholder.style.display = "flex";
                if (this.elements.overlayCanvas) {
                    this.elements.overlayCanvas.style.display = "none";
                    if (this.overlayCtx) this.overlayCtx.clearRect(0, 0, this.elements.overlayCanvas.width, this.elements.overlayCanvas.height);
                }
                this._updateCameraStatus(false, "Camera Off");
                this._setChecklist("chkCamera", false);
                this._setChecklist("chkHand", false);
                this.stopPredictionLoop();
                this._resetPredictionDisplay();
            },
            onError: (msg) => {
                this._updateCameraStatus(false, "Camera Error");
                alert(msg);
            }
        });
    }

    _bindEvents() {
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener("click", () => this.cameraManager.startCamera());
        }

        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener("click", () => this.cameraManager.stopCamera());
        }

        if (this.elements.captureBtn) {
            this.elements.captureBtn.addEventListener("click", () => this._manualCaptureFrame());
        }

        if (this.elements.speakBtn) {
            this.elements.speakBtn.addEventListener("click", () => this.speakCurrentTranslation());
        }

        if (this.elements.stopVoiceBtn) {
            this.elements.stopVoiceBtn.addEventListener("click", () => this.stopVoice());
        }

        if (this.elements.langSelector) {
            this.elements.langSelector.addEventListener("change", (e) => {
                this.currentLanguage = e.target.value;
                this.langKey = this._getLangKey(this.currentLanguage);
                localStorage.setItem("signtranslate_lang", this.currentLanguage);
                console.log("[TranslatorEngine] Language changed to:", this.currentLanguage);
            });
        }

        if (this.elements.autoSpeakToggle) {
            this.elements.autoSpeakToggle.addEventListener("change", (e) => {
                this.autoSpeak = e.target.checked;
            });
        }

        if (this.elements.toggleLandmarksBtn) {
            this.elements.toggleLandmarksBtn.addEventListener("click", () => {
                this.drawLandmarksEnabled = !this.drawLandmarksEnabled;
                this.elements.toggleLandmarksBtn.classList.toggle("active", this.drawLandmarksEnabled);
                if (!this.drawLandmarksEnabled && this.overlayCtx) {
                    this.overlayCtx.clearRect(0, 0, this.elements.overlayCanvas.width, this.elements.overlayCanvas.height);
                }
            });
        }

        if (this.elements.clearHistoryBtn) {
            this.elements.clearHistoryBtn.addEventListener("click", () => this.clearHistory());
        }

        // Sentence Builder controls
        if (this.elements.speakSentenceBtn) {
            this.elements.speakSentenceBtn.addEventListener("click", () => {
                const text = this.sentenceWords.join(" ");
                if (text) this.speakText(text);
            });
        }

        if (this.elements.copySentenceBtn) {
            this.elements.copySentenceBtn.addEventListener("click", () => {
                const text = this.sentenceWords.join(" ");
                if (text) {
                    navigator.clipboard.writeText(text);
                    const original = this.elements.copySentenceBtn.innerHTML;
                    this.elements.copySentenceBtn.innerHTML = "✓ Copied!";
                    setTimeout(() => { this.elements.copySentenceBtn.innerHTML = original; }, 1500);
                }
            });
        }

        if (this.elements.clearSentenceBtn) {
            this.elements.clearSentenceBtn.addEventListener("click", () => {
                this.sentenceWords = [];
                this._renderSentence();
            });
        }
    }

    _updateCameraStatus(online, text) {
        if (this.elements.cameraStatusText) this.elements.cameraStatusText.innerText = text;
        if (this.elements.cameraStatusDot) {
            this.elements.cameraStatusDot.className = "status-indicator-dot " + (online ? "online" : "offline");
        }
        if (this.elements.cardCameraStatus) this.elements.cardCameraStatus.innerText = online ? "Online" : "Offline";
        if (this.elements.cardCameraDot) {
            this.elements.cardCameraDot.className = "status-indicator-dot " + (online ? "online" : "offline");
        }
    }

    _setChecklist(elementKey, active) {
        if (this.elements[elementKey]) {
            this.elements[elementKey].classList.toggle("active", active);
        }
    }

    startPredictionLoop() {
        if (this.predictionInterval) clearInterval(this.predictionInterval);

        this.predictionInterval = setInterval(() => {
            this.processCurrentFrame();
        }, this.intervalMs);

        console.log(`[TranslatorEngine] Real-time prediction loop active (${this.intervalMs}ms interval).`);
    }

    stopPredictionLoop() {
        if (this.predictionInterval) {
            clearInterval(this.predictionInterval);
            this.predictionInterval = null;
        }
        this.isProcessing = false;
    }

    async processCurrentFrame() {
        if (this.isProcessing || !this.cameraManager || !this.cameraManager.isActive) {
            return;
        }

        const base64Image = this.cameraManager.captureFrameBase64(0.75);
        if (!base64Image) return;

        this.isProcessing = true;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: base64Image })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this._handlePredictionResult(data);

        } catch (error) {
            console.warn("[TranslatorEngine] Prediction request failed:", error);
            if (this.elements.cardApiStatus) this.elements.cardApiStatus.innerText = "Error";
            if (this.elements.cardApiDot) this.elements.cardApiDot.className = "status-indicator-dot error";
        } finally {
            this.isProcessing = false;
        }
    }

    async _manualCaptureFrame() {
        if (!this.cameraManager || !this.cameraManager.isActive) {
            alert("Please start the camera first.");
            return;
        }
        await this.processCurrentFrame();
    }

    _handlePredictionResult(data) {
        if (this.elements.cardApiStatus) this.elements.cardApiStatus.innerText = "Connected";
        if (this.elements.cardApiDot) this.elements.cardApiDot.className = "status-indicator-dot connected";

        if (data && data.success && data.sign) {
            // Hand detected & predicted
            this._setChecklist("chkHand", true);
            this._setChecklist("chkFeatures", true);
            this._setChecklist("chkModel", true);
            this._setChecklist("chkPrediction", true);

            if (this.elements.cardDetectionStatus) this.elements.cardDetectionStatus.innerText = "Active";
            if (this.elements.cardDetectionDot) this.elements.cardDetectionDot.className = "status-indicator-dot active";

            this._displayPrediction(data);
            this._drawOverlay(data.landmarks, data.bbox);

            // Auto-speak check: stable for 3 consecutive cycles
            if (data.sign === this.lastSpokenSign) {
                this.signStabilityCount++;
            } else {
                this.lastSpokenSign = data.sign;
                this.signStabilityCount = 1;
            }

            if (this.autoSpeak && this.signStabilityCount === 3 && data.confidence >= 80) {
                this.speakCurrentTranslation();
            }

            // Append to sentence accumulator if confident
            this._maybeAddToSentence(data.sign, data.confidence);

            // Record to history table
            this._addHistoryRecord(data.sign, data.translation, data.confidence);

        } else {
            // No hand detected
            this._setChecklist("chkHand", false);
            this._setChecklist("chkFeatures", false);
            this._setChecklist("chkPrediction", false);

            if (this.elements.cardDetectionStatus) this.elements.cardDetectionStatus.innerText = "Waiting";
            if (this.elements.cardDetectionDot) this.elements.cardDetectionDot.className = "status-indicator-dot waiting";

            if (this.elements.predictionStatus) {
                this.elements.predictionStatus.innerText = data.message || "Waiting for hand sign...";
            }

            // Clear overlay canvas
            if (this.overlayCtx && this.elements.overlayCanvas) {
                this.overlayCtx.clearRect(0, 0, this.elements.overlayCanvas.width, this.elements.overlayCanvas.height);
            }
        }
    }

    _displayPrediction(data) {
        const sign = data.sign;
        const confidence = parseFloat(data.confidence) || 0;

        // Detected sign
        if (this.elements.predictionSign) {
            this.elements.predictionSign.innerText = sign;
        }

        // Icon representation
        if (this.elements.predictionSignIcon) {
            const icons = {
                "HELLO": "👋", "THANK YOU": "🙏", "YES": "👍", "NO": "👎",
                "HELP": "🆘", "PLEASE": "🤲", "GOODBYE": "👋", "PEACE": "✌️",
                "LOVE": "🤟", "OK": "👌", "GOOD MORNING": "🌅"
            };
            this.elements.predictionSignIcon.innerText = icons[sign] || "🤟";
        }

        // Confidence display & color-coded progress
        if (this.elements.confidenceValue) {
            this.elements.confidenceValue.innerText = `${Math.round(confidence)}%`;
        }

        if (this.elements.confidenceBar) {
            this.elements.confidenceBar.style.width = `${Math.min(100, Math.max(0, confidence))}%`;
            this.elements.confidenceBar.className = "confidence-fill " +
                (confidence >= 80 ? "high" : confidence >= 50 ? "medium" : "low");
        }

        // Multilingual Translation
        const translations = data.translations || {};
        const localizedTrans = translations[this.langKey] || data.translation || sign;

        if (this.elements.translationText) {
            this.elements.translationText.innerText = localizedTrans;
        }

        if (this.elements.multilingualSub) {
            if (this.langKey !== "en") {
                this.elements.multilingualSub.innerText = `English: ${data.translation || sign}`;
            } else {
                this.elements.multilingualSub.innerText = `Sign Category: ${data.category || 'ASL Gesture'}`;
            }
        }

        if (this.elements.predictionStatus) {
            this.elements.predictionStatus.innerText = `Recognized via ${data.model_type || 'AI Model'}`;
        }
    }

    _resetPredictionDisplay() {
        if (this.elements.predictionSign) this.elements.predictionSign.innerText = "—";
        if (this.elements.predictionSignIcon) this.elements.predictionSignIcon.innerText = "🤟";
        if (this.elements.confidenceValue) this.elements.confidenceValue.innerText = "0%";
        if (this.elements.confidenceBar) {
            this.elements.confidenceBar.style.width = "0%";
            this.elements.confidenceBar.className = "confidence-fill";
        }
        if (this.elements.translationText) this.elements.translationText.innerText = "Waiting for translation...";
        if (this.elements.multilingualSub) this.elements.multilingualSub.innerText = "";
        if (this.elements.predictionStatus) this.elements.predictionStatus.innerText = "Waiting for camera input";
    }

    _drawOverlay(landmarks, bbox) {
        if (!this.drawLandmarksEnabled || !this.overlayCtx || !this.elements.overlayCanvas) return;

        const ctx = this.overlayCtx;
        const cvs = this.elements.overlayCanvas;
        ctx.clearRect(0, 0, cvs.width, cvs.height);

        if (!landmarks || landmarks.length === 0) return;

        // Skeleton connections
        const BONES = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12],
            [9, 13], [13, 14], [14, 15], [15, 16],
            [13, 17], [17, 18], [18, 19], [19, 20],
            [0, 17]
        ];

        // Draw bones
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#10b981";
        ctx.beginPath();
        for (const [p1, p2] of BONES) {
            if (landmarks[p1] && landmarks[p2]) {
                ctx.moveTo(landmarks[p1][0], landmarks[p1][1]);
                ctx.lineTo(landmarks[p2][0], landmarks[p2][1]);
            }
        }
        ctx.stroke();

        // Draw points
        for (let i = 0; i < landmarks.length; i++) {
            const [x, y] = landmarks[i];
            const isFingertip = [4, 8, 12, 16, 20].includes(i);
            ctx.beginPath();
            ctx.arc(x, y, isFingertip ? 6 : 4, 0, 2 * Math.PI);
            ctx.fillStyle = isFingertip ? "#ffffff" : "#6366f1";
            ctx.fill();
            ctx.strokeStyle = "#0f172a";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Draw bounding box
        if (bbox && bbox.length === 4) {
            const [x1, y1, x2, y2] = bbox;
            ctx.strokeStyle = "#6366f1";
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }
    }

    _maybeAddToSentence(sign, confidence) {
        if (!this.elements.sentenceOutput || confidence < 85) return;

        const lastWord = this.sentenceWords[this.sentenceWords.length - 1];
        if (lastWord !== sign) {
            this.sentenceWords.push(sign);
            if (this.sentenceWords.length > 25) this.sentenceWords.shift();
            this._renderSentence();
        }
    }

    _renderSentence() {
        if (!this.elements.sentenceOutput) return;
        if (this.sentenceWords.length === 0) {
            this.elements.sentenceOutput.innerHTML = '<span class="sentence-placeholder">Recognized signs will accumulate here as a continuous sentence...</span>';
        } else {
            this.elements.sentenceOutput.innerText = this.sentenceWords.join(" ");
        }
    }

    /**
     * Web Speech API - Text-to-Speech Engine
     */
    speakCurrentTranslation() {
        const text = this.elements.translationText ? this.elements.translationText.innerText : "";
        if (!text || text === "Waiting for translation..." || text.trim() === "") {
            alert("No translation available to speak.");
            return;
        }
        this.speakText(text);
    }

    speakText(text) {
        if (!('speechSynthesis' in window)) {
            alert("Web Speech API is not supported in your browser.");
            return;
        }

        // Cancel previous speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = this.currentLanguage;
        utterance.rate = 0.95;
        utterance.pitch = 1.0;

        // Try to match specific language voice if available
        const voices = window.speechSynthesis.getVoices();
        const matchingVoice = voices.find(v => v.lang === this.currentLanguage || v.lang.startsWith(this.langKey));
        if (matchingVoice) {
            utterance.voice = matchingVoice;
        }

        window.speechSynthesis.speak(utterance);
        console.log(`[TTS] Speaking "${text}" in ${this.currentLanguage}`);
    }

    stopVoice() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            console.log("[TTS] Voice stopped.");
        }
    }

    _addHistoryRecord(sign, translation, confidence) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Prevent duplicate spam
        if (this.recentHistory.length > 0 && this.recentHistory[0].sign === sign) {
            return;
        }

        const record = {
            time: timeStr,
            sign: sign,
            translation: translation || sign,
            confidence: `${Math.round(confidence)}%`
        };

        this.recentHistory.unshift(record);
        if (this.recentHistory.length > 30) this.recentHistory.pop();

        this._renderHistory();
    }

    async _loadHistory() {
        try {
            const res = await fetch("/api/history");
            const data = await res.json();
            if (data && data.success && Array.isArray(data.history)) {
                this.recentHistory = data.history;
                this._renderHistory();
            }
        } catch (e) {
            console.warn("[TranslatorEngine] History API load error:", e);
        }
    }

    _renderHistory() {
        if (!this.elements.historyTableBody) return;

        if (this.recentHistory.length === 0) {
            this.elements.historyTableBody.innerHTML = `
                <tr>
                    <td colspan="4">
                        <div class="empty-history-msg">
                            <span class="empty-history-icon">📋</span>
                            <p>No translations recorded yet. Start camera to translate signs.</p>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        this.elements.historyTableBody.innerHTML = this.recentHistory.map(item => `
            <tr>
                <td><span class="sign-tag">${item.sign}</span></td>
                <td><strong>${item.translation}</strong></td>
                <td><span class="conf-badge">${item.confidence}</span></td>
                <td><small>${item.time}</small></td>
            </tr>
        `).join("");
    }

    async clearHistory() {
        try {
            await fetch("/api/history/clear", { method: "POST" });
            this.recentHistory = [];
            this._renderHistory();
            console.log("[TranslatorEngine] History cleared.");
        } catch (e) {
            console.error("Failed to clear history:", e);
        }
    }

    async _checkApiStatus() {
        try {
            const res = await fetch("/api/status");
            const status = await res.json();
            if (this.elements.cardModelStatus) {
                this.elements.cardModelStatus.innerText = status.model === "ready" ? "Ready" : "Loading";
            }
            if (this.elements.cardModelDot) {
                this.elements.cardModelDot.className = "status-indicator-dot " + (status.model === "ready" ? "ready" : "loading");
            }
            if (this.elements.aiModelStatusText) {
                this.elements.aiModelStatusText.innerText = status.model === "ready" ? "AI Model Active" : "AI Loading";
            }
            if (this.elements.cardVoiceStatus) {
                this.elements.cardVoiceStatus.innerText = ('speechSynthesis' in window) ? "Available" : "Not Supported";
            }
            if (this.elements.cardApiStatus) {
                this.elements.cardApiStatus.innerText = "Connected";
            }
            if (this.elements.cardApiDot) {
                this.elements.cardApiDot.className = "status-indicator-dot connected";
            }
        } catch (e) {
            console.warn("[TranslatorEngine] API status check failed:", e);
            if (this.elements.cardApiStatus) this.elements.cardApiStatus.innerText = "Disconnected";
            if (this.elements.cardApiDot) this.elements.cardApiDot.className = "status-indicator-dot disconnected";
        }
    }
}

// Global initialization
document.addEventListener("DOMContentLoaded", () => {
    window.translatorApp = new TranslatorEngine();
});
