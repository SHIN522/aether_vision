import {
  FilesetResolver,
  HandLandmarker,
  ObjectDetector
} from "./vision_bundle.js";

// ==========================================================================
// STATE VARIABLES
// ==========================================================================
let activeSource = "webcam"; // "webcam" or "file"
let handLandmarker = null;
let objectDetector = null;
let webcamStream = null;
let isModelsLoaded = false;
let isPlaying = false;
let selectedCameraId = null;
let isRecording = false;
let activeDetections = [];

// Canvas Elements
const outputCanvas = document.getElementById("output-canvas");
const ctx = outputCanvas.getContext("2d");
const backgroundCanvas = document.getElementById("background-canvas");
const bgCtx = backgroundCanvas.getContext("2d");

// Helper elements
const webcamVideo = document.getElementById("webcam-video");
const uploadVideo = document.getElementById("upload-video");
let activeVideo = webcamVideo; // Pointer to current active video source

// Recording variables
let mediaRecorder = null;
let recordedChunks = [];
let recordStreamType = "processed"; // "processed" or "raw"

// Invisibility Cloak State
let isBgCaptured = false;
let bgImageData = null;

// Settings
let selectedEffect = "cloak";
let effectOpacity = 1.0;
let drawOutline = true;
let enableObjectDetection = true;
let detectionThreshold = 0.3;

// Pinch Gesture State
let isPinchLatched = false;
let modeShiftMessage = "";
let modeShiftMessageTime = 0;
const effectsList = ["cloak", "crt_scanlines", "line_halftone", "dither", "nokia", "crt_synth", "ascii_depth", "thermal", "wireframe"];
let pinchCycleCount = 0;

// Matrix Rain Effect State
let matrixChars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$+-*/=%#@&";
let matrixDrops = [];
const matrixFontSize = 14;

// FPS Counter State
let lastFpsUpdate = 0;
let fpsFrames = 0;
let currentFps = 0;

// Offscreen buffer for fast pixel manipulation (Thermal/Sobel)
const offscreenCanvas = document.createElement("canvas");
const offscreenCtx = offscreenCanvas.getContext("2d");
offscreenCanvas.width = 320;
offscreenCanvas.height = 180;

// ==========================================================================
// UI ELEMENTS REFERENCE
// ==========================================================================
const handStatus = document.getElementById("hand-status");
const objectStatus = document.getElementById("object-status");
const fpsCounter = document.getElementById("fps-counter");
const viewportLoader = document.getElementById("viewport-loader");
const btnSourceWebcam = document.getElementById("btn-source-webcam");
const btnSourceFile = document.getElementById("btn-source-file");
const fileUploadZone = document.getElementById("file-upload-zone");
const videoFileInput = document.getElementById("video-file-input");
const btnCaptureBg = document.getElementById("btn-capture-bg");
const btnResetBg = document.getElementById("btn-reset-bg");
const bgStatusText = document.getElementById("bg-status-text");
const effectSelect = document.getElementById("effect-select");
const effectOpacityInput = document.getElementById("effect-opacity");
const opacityValLabel = document.getElementById("opacity-val");
const toggleOutlineCheckbox = document.getElementById("toggle-outline");
const toggleObjectDetectionCheckbox = document.getElementById("toggle-object-detection");
const detectorThresholdInput = document.getElementById("detector-threshold");
const thresholdValLabel = document.getElementById("threshold-val");
const btnRecord = document.getElementById("btn-record");
const recordingBadge = document.getElementById("recording-badge");
const recordingText = document.getElementById("recording-text");
const videoPlaybackControls = document.getElementById("video-playback-controls");
const playPauseBtn = document.getElementById("play-pause-btn");
const videoProgress = document.getElementById("video-progress");
const videoTimeLabel = document.getElementById("video-time");
const videoDurationLabel = document.getElementById("video-duration");
const muteBtn = document.getElementById("mute-btn");
const cameraSelect = document.getElementById("camera-select");
const cameraSelectContainer = document.getElementById("camera-select-container");
const btnStartApp = document.getElementById("btn-start-app");
const loaderSpinner = document.getElementById("loader-spinner");
const loaderText = document.getElementById("loader-text");
const debugLog = document.getElementById("debug-log");

function logDebug(message) {
  if (debugLog) {
    debugLog.style.display = "block";
    debugLog.innerHTML += `[${new Date().toLocaleTimeString()}] ${message}<br>`;
  }
  console.log(`[DEBUG] ${message}`);
}

// ==========================================================================
// ASYNC CORE MODELS LOADER
// ==========================================================================
async function initAIModels() {
  logDebug("Initializing AI Vision Engine...");
  try {
    logDebug("Loading WebAssembly vision runtime from local directory...");
    const vision = await FilesetResolver.forVisionTasks(
      "./wasm"
    );
    logDebug("WASM runtime loaded successfully.");

    logDebug("Loading local Hand Landmarker model (models/hand_landmarker.task)...");
    // Initialize Hand Landmarker
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "./models/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    });
    logDebug("Hand Landmarker model loaded successfully.");
    handStatus.innerHTML = '<span class="indicator green"></span> HAND TRACKING: ACTIVE';

    logDebug("Loading local Object Detector model (models/efficientdet_lite0.tflite)...");
    // Initialize Object Detector
    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "./models/efficientdet_lite0.tflite",
        delegate: "GPU"
      },
      scoreThreshold: detectionThreshold,
      runningMode: "VIDEO"
    });
    logDebug("Object Detector model loaded successfully.");
    objectStatus.innerHTML = '<span class="indicator green"></span> OBJECT DETECTOR: ACTIVE';

    isModelsLoaded = true;
    logDebug("All AI models loaded successfully.");
    viewportLoader.style.opacity = "0";
    setTimeout(() => viewportLoader.style.display = "none", 500);

    // Start video loop
    startAppLoop();
  } catch (error) {
    logDebug(`CRITICAL ERROR during AI model load: ${error.name} - ${error.message}`);
    console.error("AI Models initialization failed:", error);
    viewportLoader.innerHTML = `<p style="color:var(--accent-red)">Model Load Error: Check internet connection.</p>`;
  }
}

// ==========================================================================
// WEBCAM & MEDIA HANDLING
// ==========================================================================
async function startWebcam() {
  if (webcamStream) return;
  
  logDebug("Initializing webcam stream capture...");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    logDebug("FAIL: Webcam APIs (navigator.mediaDevices.getUserMedia) are undefined in this context.");
    alert("Webcam APIs are NOT available. Please make sure you are accessing the app via http://localhost:8000 and NOT by double-clicking the index.html file (file:///), which blocks camera access.");
    return;
  }
  
  try {
    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        deviceId: selectedCameraId ? { exact: selectedCameraId } : undefined
      },
      audio: false
    };
    logDebug("Requesting getUserMedia stream with video-only constraints...");
    webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
    logDebug("SUCCESS: Webcam stream captured successfully.");
    webcamVideo.srcObject = webcamStream;
    activeVideo = webcamVideo;
    
    // Wait for video meta to load
    webcamVideo.onloadedmetadata = () => {
      logDebug(`Webcam resolution loaded: ${webcamVideo.videoWidth}x${webcamVideo.videoHeight}`);
      webcamVideo.play();
      isPlaying = true;
      adjustCanvasDimensions();
    };
    
    // Refresh the camera list (this populates labels once permission is active)
    await updateCameraList();
  } catch (err) {
    logDebug(`FAIL: webcam capture failed with error: ${err.name} - ${err.message}`);
    console.error("Webcam access error:", err);
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      alert("Webcam permission denied. Please click the camera icon in your browser's address bar, choose 'Always Allow', and refresh the page.");
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      alert("Your camera is already in use by another application (e.g. Zoom, Discord, OBS, Teams, or the Windows Camera App). Please close those applications and refresh the page.");
    } else {
      alert(`Webcam failed to start: ${err.message}. Check that the camera is plugged in and not blocked in system privacy settings.`);
    }
  }
}

async function updateCameraList() {
  logDebug("Enumerating media input devices...");
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    logDebug("WARNING: navigator.mediaDevices.enumerateDevices is NOT supported in this context (insecure context or old browser).");
    cameraSelect.innerHTML = "";
    const option = document.createElement("option");
    option.text = "Camera access unsupported";
    option.value = "";
    cameraSelect.appendChild(option);
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === "videoinput");
    
    // Clear the select options
    cameraSelect.innerHTML = "";
    
    if (videoDevices.length === 0) {
      logDebug("WARNING: No camera input devices found.");
      const option = document.createElement("option");
      option.text = "No cameras detected";
      option.value = "";
      cameraSelect.appendChild(option);
      return;
    }
    
    logDebug(`SUCCESS: Found ${videoDevices.length} camera(s) in system.`);
    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      if (selectedCameraId === device.deviceId || (!selectedCameraId && index === 0)) {
        option.selected = true;
        if (!selectedCameraId) selectedCameraId = device.deviceId;
      }
      cameraSelect.appendChild(option);
      logDebug(` -> Camera ${index+1}: ${option.text} (${device.deviceId.substring(0,8)}...)`);
    });
  } catch (err) {
    logDebug(`FAIL: enumerating input devices failed: ${err.name} - ${err.message}`);
    console.error("Error enumerating devices:", err);
  }
}

// Add change listener to cameraSelect
cameraSelect.addEventListener("change", async (e) => {
  const newDeviceId = e.target.value;
  if (newDeviceId === selectedCameraId) return;
  
  selectedCameraId = newDeviceId;
  
  if (activeSource === "webcam") {
    // Restart camera stream with new device
    stopWebcam();
    await startWebcam();
  }
});

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
    webcamVideo.srcObject = null;
  }
}

function adjustCanvasDimensions() {
  // Sync canvas size to match the active video's natural dimensions
  const width = activeVideo.videoWidth || 640;
  const height = activeVideo.videoHeight || 480;
  
  outputCanvas.width = width;
  outputCanvas.height = height;
  backgroundCanvas.width = width;
  backgroundCanvas.height = height;
  
  // Reset Matrix drops based on width
  const columns = Math.ceil(width / matrixFontSize);
  matrixDrops = [];
  for (let i = 0; i < columns; i++) {
    matrixDrops[i] = Math.random() * -100;
  }
}

// ==========================================================================
// VIDEO FILE UPLOAD HANDLING
// ==========================================================================
fileUploadZone.addEventListener("click", () => videoFileInput.click());

fileUploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  fileUploadZone.style.borderColor = "var(--accent-cyan)";
});

fileUploadZone.addEventListener("dragleave", () => {
  fileUploadZone.style.borderColor = "rgba(0, 242, 254, 0.3)";
});

fileUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  fileUploadZone.style.borderColor = "rgba(0, 242, 254, 0.3)";
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    loadUploadedVideo(e.dataTransfer.files[0]);
  }
});

videoFileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    loadUploadedVideo(e.target.files[0]);
  }
});

function loadUploadedVideo(file) {
  const url = URL.createObjectURL(file);
  
  // Stop webcam if it's active
  stopWebcam();
  
  uploadVideo.src = url;
  activeVideo = uploadVideo;
  
  uploadVideo.onloadedmetadata = () => {
    adjustCanvasDimensions();
    uploadVideo.play();
    isPlaying = true;
    
    // Show playback controls
    videoPlaybackControls.style.display = "flex";
    playPauseBtn.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;
    
    // Set duration slider
    videoProgress.max = Math.floor(uploadVideo.duration);
    videoDurationLabel.textContent = formatTime(uploadVideo.duration);
  };
}

// Custom Video Controls
playPauseBtn.addEventListener("click", () => {
  if (uploadVideo.paused) {
    uploadVideo.play();
    isPlaying = true;
    playPauseBtn.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;
  } else {
    uploadVideo.pause();
    isPlaying = false;
    playPauseBtn.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
  }
});

uploadVideo.addEventListener("timeupdate", () => {
  videoProgress.value = Math.floor(uploadVideo.currentTime);
  videoTimeLabel.textContent = formatTime(uploadVideo.currentTime);
});

videoProgress.addEventListener("input", (e) => {
  uploadVideo.currentTime = e.target.value;
});

muteBtn.addEventListener("click", () => {
  uploadVideo.muted = !uploadVideo.muted;
  if (uploadVideo.muted) {
    muteBtn.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M3.63 3.63L2.37 4.89 7.48 10H3v4h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.03c1.38-.31 2.63-.95 3.69-1.81l2.42 2.42 1.27-1.27L3.63 3.63zM12 4L9.91 6.09 12 8.18V4zM16.5 12c0-.94-.3-1.81-.8-2.53l1.45-1.45c.85 1.14 1.35 2.54 1.35 3.98 0 1.94-.92 3.67-2.36 4.79l-1.43-1.43c.5-.72.79-1.58.79-2.51zM14 3.23v2.06c2.89.86 5 3.54 5 6.71 0 1.22-.32 2.37-.88 3.37l1.48 1.48c1.01-1.45 1.4-3.23 1.4-5.06s-2.99-7.86-7-8.77z"/></svg>`;
  } else {
    muteBtn.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77zM3 9v6h4l5 5V4L7 9H3z"/></svg>`;
  }
});

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

// ==========================================================================
// SOURCE INPUT SWITCHER
// ==========================================================================
btnSourceWebcam.addEventListener("click", () => {
  if (activeSource === "webcam") return;
  activeSource = "webcam";
  btnSourceWebcam.classList.add("active");
  btnSourceFile.classList.remove("active");
  fileUploadZone.style.display = "none";
  videoPlaybackControls.style.display = "none";
  cameraSelectContainer.style.display = "block";
  
  uploadVideo.pause();
  startWebcam();
});

btnSourceFile.addEventListener("click", () => {
  if (activeSource === "file") return;
  activeSource = "file";
  btnSourceFile.classList.add("active");
  btnSourceWebcam.classList.remove("active");
  fileUploadZone.style.display = "block";
  cameraSelectContainer.style.display = "none";
  
  stopWebcam();
  isPlaying = false;
  
  // Clear canvas
  ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
});

// ==========================================================================
// INVISIBILITY CLOAK BG CAPTURING
// ==========================================================================
btnCaptureBg.addEventListener("click", () => {
  if (!activeVideo.videoWidth) return;
  
  // Draw the current video frame onto backgroundCanvas
  bgCtx.drawImage(activeVideo, 0, 0, backgroundCanvas.width, backgroundCanvas.height);
  
  isBgCaptured = true;
  btnResetBg.disabled = false;
  bgStatusText.textContent = "STATUS: BACKGROUND LOCKED";
  bgStatusText.style.color = "var(--accent-cyan)";
  bgStatusText.style.borderColor = "var(--accent-cyan)";
});

btnResetBg.addEventListener("click", () => {
  isBgCaptured = false;
  bgCtx.clearRect(0, 0, backgroundCanvas.width, backgroundCanvas.height);
  btnResetBg.disabled = true;
  bgStatusText.textContent = "STATUS: BACKGROUND NOT CAPTURED";
  bgStatusText.style.color = "var(--text-secondary)";
  bgStatusText.style.borderColor = "rgba(255, 255, 255, 0.08)";
});

// ==========================================================================
// EFFECT & SETTING SWITCHERS
// ==========================================================================
effectSelect.addEventListener("change", (e) => {
  selectedEffect = e.target.value;
});

effectOpacityInput.addEventListener("input", (e) => {
  effectOpacity = e.target.value / 100;
  opacityValLabel.textContent = `${e.target.value}%`;
});

toggleOutlineCheckbox.addEventListener("change", (e) => {
  drawOutline = e.target.checked;
});

toggleObjectDetectionCheckbox.addEventListener("change", (e) => {
  enableObjectDetection = e.target.checked;
});

detectorThresholdInput.addEventListener("input", (e) => {
  detectionThreshold = e.target.value / 100;
  thresholdValLabel.textContent = `${e.target.value}%`;
  
  // Dynamic update to the active detector threshold
  if (objectDetector && isModelsLoaded) {
    objectDetector.setOptions({ scoreThreshold: detectionThreshold });
  }
});

// ==========================================================================
// VIDEO RECORDING MODULE
// ==========================================================================
btnRecord.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

function startRecording() {
  recordedChunks = [];
  const radioRecordingType = document.querySelector('input[name="recording-type"]:checked').value;
  recordStreamType = radioRecordingType;

  let stream = null;
  if (recordStreamType === "processed") {
    // Capture canvas frame
    stream = outputCanvas.captureStream(30);
  } else {
    // Capture raw webcam stream or raw video stream
    if (activeSource === "webcam" && webcamStream) {
      stream = webcamStream;
    } else if (activeSource === "file" && uploadVideo.src) {
      // In firefox/chrome we can capture from video element
      stream = uploadVideo.captureStream ? uploadVideo.captureStream(30) : uploadVideo.mozCaptureStream(30);
    }
  }

  if (!stream) {
    alert("Unable to capture video stream. Ensure camera/video source is active.");
    return;
  }

  try {
    let options = { mimeType: "video/webm;codecs=vp9,opus" };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options = { mimeType: "video/webm;codecs=vp8,opus" };
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options = { mimeType: "video/webm" };
      }
    }

    mediaRecorder = new MediaRecorder(stream, options);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = saveRecording;
    
    // Start recording with 1s timeslices
    mediaRecorder.start(1000);
    isRecording = true;
    
    // Update UI
    btnRecord.textContent = "STOP RECORDING";
    btnRecord.classList.add("recording");
    recordingBadge.style.display = "flex";
    recordingText.textContent = `REC: ${recordStreamType.toUpperCase()}`;
  } catch (err) {
    console.error("Recording start failed:", err);
    alert("Failed to start recording on this browser.");
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    
    // Update UI
    btnRecord.textContent = "START RECORDING";
    btnRecord.classList.remove("recording");
    recordingBadge.style.display = "none";
  }
}

function saveRecording() {
  if (recordedChunks.length === 0) return;
  
  const blob = new Blob(recordedChunks, { type: "video/webm" });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = url;
  a.download = `aether-vision-${recordStreamType}-${Date.now()}.webm`;
  document.body.appendChild(a);
  a.click();
  
  setTimeout(() => {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, 100);
}

// ==========================================================================
// PIXEL EFFECT SHADERS (THERNAL, WIREFRAME EDGE DETECTION, GLITCH)
// ==========================================================================

// Thermal Vision Shader
function applyThermalFilter(srcCtx, destCtx, x, y, width, height) {
  const imgData = srcCtx.getImageData(x, y, width, height);
  const data = imgData.data;
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i+1];
    const b = data[i+2];
    
    // Luminance
    const v = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    
    // Map luminance to thermal ramp (blue -> magenta -> orange -> yellow)
    if (v < 64) {
      data[i] = 0;              // R
      data[i+1] = 0;            // G
      data[i+2] = v * 4;        // B
    } else if (v < 128) {
      data[i] = (v - 64) * 4;   // R
      data[i+1] = 0;            // G
      data[i+2] = 255;          // B
    } else if (v < 192) {
      data[i] = 255;            // R
      data[i+1] = (v - 128) * 4;// G
      data[i+2] = 255 - (v - 128) * 4; // B
    } else {
      data[i] = 255;            // R
      data[i+1] = 255;          // G
      data[i+2] = (v - 192) * 4;// B
    }
  }
  
  destCtx.putImageData(imgData, x, y);
}

// Sobel Edge Detection Shader (Cyberpunk Wireframe)
function applySobelFilter(srcCtx, destCtx, x, y, width, height) {
  const imgData = srcCtx.getImageData(x, y, width, height);
  const data = imgData.data;
  const w = width;
  const h = height;
  
  // Gray scale conversion array
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < data.length; i += 4) {
    gray[i/4] = 0.2126 * data[i] + 0.7152 * data[i+1] + 0.0722 * data[i+2];
  }
  
  const edgeData = destCtx.createImageData(w, h);
  const edge = edgeData.data;
  
  // Sobel convolution
  for (let cy = 1; cy < h - 1; cy++) {
    for (let cx = 1; cx < w - 1; cx++) {
      const idx = cy * w + cx;
      
      const valX = 
        -gray[idx - w - 1] + gray[idx - w + 1] +
        -2 * gray[idx - 1] + 2 * gray[idx + 1] +
        -gray[idx + w - 1] + gray[idx + w + 1];
        
      const valY = 
        -gray[idx - w - 1] - 2 * gray[idx - w] - gray[idx - w + 1] +
        gray[idx + w - 1] + 2 * gray[idx + w] + gray[idx + w + 1];
        
      const gMagnitude = Math.sqrt(valX * valX + valY * valY);
      
      const pixelIdx = idx * 4;
      if (gMagnitude > 50) {
        // Neon cyan edge
        edge[pixelIdx] = 0;
        edge[pixelIdx+1] = 242;
        edge[pixelIdx+2] = 254;
        edge[pixelIdx+3] = 255;
      } else {
        // Dark grid background
        edge[pixelIdx] = 5;
        edge[pixelIdx+1] = 8;
        edge[pixelIdx+2] = 17;
        edge[pixelIdx+3] = 220;
      }
    }
  }
  
  destCtx.putImageData(edgeData, x, y);
}

// 4x4 Bayer Dither Matrix for retro Macintosh Ordered Dithering
const bayerMatrix = [
  [   0, 128,  32, 160 ],
  [ 192,  64, 224,  96 ],
  [  48, 176,  16, 144 ],
  [ 240, 112, 208,  80 ]
];

function applyBayerDither(srcCtx, destCtx, x, y, width, height) {
  const imgData = srcCtx.getImageData(x, y, width, height);
  const data = imgData.data;
  
  for (let cy = 0; cy < height; cy++) {
    for (let cx = 0; cx < width; cx++) {
      const idx = (cy * width + cx) * 4;
      const r = data[idx];
      const g = data[idx+1];
      const b = data[idx+2];
      
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const matrixX = cx % 4;
      const matrixY = cy % 4;
      const threshold = bayerMatrix[matrixY][matrixX];
      
      // Map to 1-bit glowing cyan/dark blue theme
      if (luma > threshold) {
        data[idx] = 0;         // Red
        data[idx+1] = 242;     // Green
        data[idx+2] = 254;     // Blue (neon cyan)
      } else {
        data[idx] = 5;
        data[idx+1] = 8;
        data[idx+2] = 17;      // dark HUD background
      }
    }
  }
  destCtx.putImageData(imgData, x, y);
}

// Analog CRT Video Synthesizer Glitch Filter
function applyCRTSynthFilter(srcCtx, destCtx, x, y, width, height) {
  const imgData = srcCtx.getImageData(x, y, width, height);
  const data = imgData.data;
  const outData = new Uint8ClampedArray(data); // clone source pixels
  
  const time = performance.now() / 120;
  const syncBarY = Math.floor((performance.now() / 12) % height);
  
  for (let cy = 0; cy < height; cy++) {
    // Sinusoidal wavy distortion of rows
    const wave = Math.sin(cy / 6 + time) * 6;
    
    // Shearing horizontal shift near the VHS tracking sync bar
    const distToSync = Math.abs(cy - syncBarY);
    const shear = distToSync < 12 ? (12 - distToSync) * 2.5 : 0;
    
    const shift = Math.floor(wave + shear);
    
    for (let cx = 0; cx < width; cx++) {
      const idx = (cy * width + cx) * 4;
      
      let srcX = cx + shift;
      if (srcX < 0) srcX = 0;
      if (srcX >= width) srcX = width - 1;
      
      // Chromatic aberration shifts (Red offset left, Blue offset right)
      const redOffset = 3;
      const blueOffset = -3;
      
      let redX = srcX + redOffset;
      let blueX = srcX + blueOffset;
      if (redX < 0) redX = 0; if (redX >= width) redX = width - 1;
      if (blueX < 0) blueX = 0; if (blueX >= width) blueX = width - 1;
      
      const srcIdx = (cy * width + srcX) * 4;
      const redIdx = (cy * width + redX) * 4;
      const blueIdx = (cy * width + blueX) * 4;
      
      // Glow/blowout color intensities
      data[idx] = Math.min(255, outData[redIdx] * 1.35);          // Red
      data[idx+1] = Math.min(255, outData[srcIdx+1] * 1.1);       // Green
      data[idx+2] = Math.min(255, outData[blueIdx+2] * 1.45);      // Blue
      
      // Draw VHS horizontal tracking lines
      if (distToSync < 3) {
        data[idx] *= 0.15;
        data[idx+1] *= 0.15;
        data[idx+2] *= 0.15;
      }
    }
  }
  destCtx.putImageData(imgData, x, y);
}

// Charcoal Sketch Outline Filter (Inverse Sobel on Warm Parchment Paper)
function applyCharcoalFilter(srcCtx, destCtx, x, y, width, height) {
  const imgData = srcCtx.getImageData(x, y, width, height);
  const data = imgData.data;
  
  // Create output buffer
  const edgeData = destCtx.createImageData(width, height);
  const edge = edgeData.data;
  
  // Grayscale helper
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < data.length; i += 4) {
    gray[i/4] = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
  }
  
  // Sobel Edge operator
  for (let cy = 1; cy < height - 1; cy++) {
    for (let cx = 1; cx < width - 1; cx++) {
      const idx = cy * width + cx;
      
      const val00 = gray[idx - width - 1];
      const val01 = gray[idx - width];
      const val02 = gray[idx - width + 1];
      
      const val10 = gray[idx - 1];
      const val12 = gray[idx + 1];
      
      const val20 = gray[idx + width - 1];
      const val21 = gray[idx + width];
      const val22 = gray[idx + width + 1];
      
      const gx = (val02 + 2 * val12 + val22) - (val00 + 2 * val10 + val20);
      const gy = (val20 + 2 * val21 + val22) - (val00 + 2 * val01 + val02);
      
      const gMagnitude = Math.sqrt(gx * gx + gy * gy);
      
      // Inverse value: high magnitude edges become dark/black, flat areas are paper color
      const edgeIntensity = Math.max(0, 255 - gMagnitude * 2.2);
      
      const pixelIdx = idx * 4;
      if (edgeIntensity < 130) {
        // Charcoal gray stroke
        edge[pixelIdx] = 32;       // Red
        edge[pixelIdx+1] = 32;     // Green
        edge[pixelIdx+2] = 40;     // Blue
        edge[pixelIdx+3] = 255;
      } else {
        // Warm textured sketch paper background
        edge[pixelIdx] = 244;      // Red
        edge[pixelIdx+1] = 241;    // Green
        edge[pixelIdx+2] = 224;    // Blue
        edge[pixelIdx+3] = 255;
      }
    }
  }
  
  destCtx.putImageData(edgeData, x, y);
}

// Compute the geometric center of a polygon
function getPolygonCentroid(vertices) {
  if (!vertices || vertices.length === 0) return { x: 0, y: 0 };
  let cx = 0;
  let cy = 0;
  vertices.forEach(pt => {
    cx += pt.x;
    cy += pt.y;
  });
  return {
    x: cx / vertices.length,
    y: cy / vertices.length
  };
}

// Liquid Metallic Chrome Reflection Filter
function applyChromeFilter(srcCtx, destCtx, x, y, width, height) {
  const imgData = srcCtx.getImageData(x, y, width, height);
  const data = imgData.data;
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i+1];
    const b = data[i+2];
    
    const luma = 0.299 * r + 0.587 * g + 0.114 * b;
    
    // High frequency sine wave reflection bands
    const v = Math.abs(Math.sin(luma * 0.048)) * 255;
    
    // Metallic chrome reflection mapping (silver base with iridescent magenta/cyan bands)
    data[i] = Math.min(255, v * 0.72 + 65);       // Red (Silver with warm reflection)
    data[i+1] = Math.min(255, v * 0.88 + 25);     // Green (Slightly lower green)
    data[i+2] = Math.min(255, v * 1.0 + 15);      // Blue (Bright blue specular band)
  }
  
  destCtx.putImageData(imgData, x, y);
}

// Full-Screen ASCII Depth & Object Telemetry Map
function drawASCIIDepthMap() {
  let hx = outputCanvas.width / 2;
  let hy = outputCanvas.height / 2;
  let handW = 150; // default proxy width
  
  if (handPolygon && handPolygon.length > 0) {
    const centroid = getPolygonCentroid(handPolygon);
    hx = centroid.x;
    hy = centroid.y;
    
    // Compute hand bounding box width to estimate depth scale
    let minX = Infinity, maxX = -Infinity;
    handPolygon.forEach(pt => {
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
    });
    handW = maxX - minX;
  }
  
  // Hand scale depth proxy: close hand = larger grid scale
  const depthScale = Math.min(1.6, Math.max(0.6, handW / 185));
  
  // Configure grid columns/rows dynamically based on depth
  const cols = Math.floor(75 * depthScale);
  const rows = Math.floor(55 * depthScale);
  
  offscreenCtx.drawImage(activeVideo, 0, 0, cols, rows);
  const imgData = offscreenCtx.getImageData(0, 0, cols, rows);
  const pixels = imgData.data;
  
  // Clear main canvas with dark HUD backdrop
  ctx.fillStyle = "rgba(5, 8, 17, 0.96)";
  ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
  
  const cellW = outputCanvas.width / cols;
  const cellH = outputCanvas.height / rows;
  
  // Set font
  ctx.font = "bold 9px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  
  const chars = "@%#*+=-:. ";
  
  for (let r = 0; r < rows; r += 2) {
    for (let c = 0; c < cols; c++) {
      const idx = (r * cols + c) * 4;
      const red = pixels[idx];
      const green = pixels[idx+1];
      const blue = pixels[idx+2];
      const luma = 0.299 * red + 0.587 * green + 0.114 * blue;
      
      const cx = c * cellW + cellW/2;
      const cy = r * cellH + cellH/2;
      
      // Compute distance to hand center
      const dx = cx - hx;
      const dy = cy - hy;
      const dist = Math.sqrt(dx*dx + dy*dy);
      
      // Calculate 3D perspective dome warp (depth perception)
      const maxWarpDist = 200 * depthScale;
      const warp = Math.max(0, 1 - dist / maxWarpDist);
      
      // Boost luminance near hand center to simulate depth sensor scan beam
      const perceivedBrightness = Math.min(255, luma * (0.8 + warp * 0.95));
      if (perceivedBrightness < 22) continue;
      
      const charIdx = Math.floor((perceivedBrightness / 255) * (chars.length - 1));
      let char = chars[charIdx];
      
      // Check if coordinate falls inside any active object detection box
      let isInsideObject = false;
      let objectLabel = "";
      if (enableObjectDetection && activeDetections.length > 0) {
        for (let i = 0; i < activeDetections.length; i++) {
          const det = activeDetections[i];
          const box = det.boundingBox;
          if (box) {
            const bx = box.originX;
            const by = box.originY;
            const bw = box.width;
            const bh = box.height;
            if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh) {
              isInsideObject = true;
              objectLabel = det.categories[0].categoryName.toUpperCase();
              break;
            }
          }
        }
      }
      
      // Character mapping: inside objects, we occasionally inject label letters!
      if (isInsideObject && objectLabel && Math.random() < 0.25) {
        const letterIdx = Math.floor((cx + cy) % objectLabel.length);
        char = objectLabel[letterIdx];
      }
      
      // Color coding:
      if (isInsideObject) {
        // High contrast amber/red for detected objects
        ctx.fillStyle = `rgba(255, 78, 62, ${perceivedBrightness / 255})`;
      } else if (warp > 0) {
        // Cyan-to-blue glow gradient near hand to reflect depth
        const gInt = Math.floor(180 * (1 - warp) + 255 * warp);
        ctx.fillStyle = `rgba(0, ${gInt}, 254, ${perceivedBrightness / 255})`;
      } else {
        // Ambient dim phosphor green for background environment
        ctx.fillStyle = `rgba(0, 242, 140, ${perceivedBrightness / 255 * 0.7})`;
      }
      
      ctx.fillText(char, cx, cy);
    }
  }
  
  // Draw terminal HUD labels for object detection
  if (enableObjectDetection && activeDetections.length > 0) {
    activeDetections.forEach(det => {
      const box = det.boundingBox;
      if (box) {
        const bx = box.originX;
        const by = box.originY;
        const bw = box.width;
        const bh = box.height;
        
        ctx.strokeStyle = "rgba(255, 78, 62, 0.5)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]); // dashed retro HUD box
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);
      }
    });
  }
}

// Unified Shader Draw Pipeline
function drawShader(effect) {
  if (effect === "cloak") {
    if (isBgCaptured) {
      ctx.drawImage(backgroundCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
    } else {
      ctx.fillStyle = "rgba(5, 8, 17, 0.95)";
      ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
      ctx.fillStyle = "var(--accent-cyan)";
      ctx.font = "14px 'Space Grotesk'";
      ctx.textAlign = "center";
      ctx.fillText("CAPTURE BACKGROUND TO ACTIVATE CLOAK", outputCanvas.width / 2, outputCanvas.height / 2);
    }
  }
  else if (effect === "crt_scanlines") {
    ctx.save();
    ctx.filter = "grayscale(100%) brightness(1.1) sepia(20%)";
    ctx.drawImage(activeVideo, 0, 0, outputCanvas.width, outputCanvas.height);
    ctx.filter = "none";
    
    ctx.globalCompositeOperation = "color";
    ctx.fillStyle = "rgba(0, 242, 254, 0.35)";
    ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    ctx.globalCompositeOperation = "source-over";
    
    // Draw scanlines
    ctx.fillStyle = "rgba(5, 8, 17, 0.28)";
    for (let sy = 0; sy < outputCanvas.height; sy += 3) {
      ctx.fillRect(0, sy, outputCanvas.width, 1.5);
    }
    
    // Phosphor sweep line
    const sweepY = (performance.now() / 3.5) % (outputCanvas.height + 120) - 60;
    const sweepGrad = ctx.createLinearGradient(0, sweepY - 60, 0, sweepY);
    sweepGrad.addColorStop(0, "rgba(0, 242, 254, 0)");
    sweepGrad.addColorStop(1, "rgba(0, 242, 254, 0.18)");
    ctx.fillStyle = sweepGrad;
    ctx.fillRect(0, sweepY - 60, outputCanvas.width, 60);
    ctx.restore();
  }
  else if (effect === "line_halftone") {
    // Clear light cream pop-art background
    ctx.fillStyle = "#fffcf0";
    ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    const sampleW = 80;
    const sampleH = 60;
    offscreenCtx.drawImage(activeVideo, 0, 0, sampleW, sampleH);
    const imgData = offscreenCtx.getImageData(0, 0, sampleW, sampleH);
    const pixels = imgData.data;
    
    ctx.strokeStyle = "#1b2a1a";
    const cellW = outputCanvas.width / sampleW;
    const cellH = outputCanvas.height / sampleH;
    
    // Draw diagonal grid segments
    for (let sy = 0; sy < sampleH; sy += 2) {
      for (let sx = 0; sx < sampleW; sx += 2) {
        const idx = (sy * sampleW + sx) * 4;
        const r = pixels[idx];
        const g = pixels[idx+1];
        const b = pixels[idx+2];
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;
        
        // Thickness inversely proportional to brightness (pop halftone effect)
        const thickness = (1 - (luma / 255)) * 4.5;
        if (thickness < 0.5) continue;
        
        const cx = sx * cellW;
        const cy = sy * cellH;
        
        ctx.lineWidth = thickness;
        ctx.beginPath();
        ctx.moveTo(cx - cellW, cy - cellH);
        ctx.lineTo(cx + cellW, cy + cellH);
        ctx.stroke();
      }
    }
  }
  else if (effect === "dither") {
    offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    applyBayerDither(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  }
  else if (effect === "nokia") {
    // Green LCD screen
    ctx.fillStyle = "#c2d0a7";
    ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    const sampleW = 84;
    const sampleH = 48;
    offscreenCtx.drawImage(activeVideo, 0, 0, sampleW, sampleH);
    const imgData = offscreenCtx.getImageData(0, 0, sampleW, sampleH);
    const pixels = imgData.data;
    
    const cellW = outputCanvas.width / sampleW;
    const cellH = outputCanvas.height / sampleH;
    
    ctx.fillStyle = "#1b2a1a";
    for (let sy = 0; sy < sampleH; sy++) {
      for (let sx = 0; sx < sampleW; sx++) {
        const idx = (sy * sampleW + sx) * 4;
        const r = pixels[idx];
        const g = pixels[idx+1];
        const b = pixels[idx+2];
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;
        
        if (luma < 115) {
          ctx.fillRect(sx * cellW, sy * cellH, cellW - 0.5, cellH - 0.5);
        }
      }
    }
  }
  else if (effect === "crt_synth") {
    offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    applyCRTSynthFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  }
  else if (effect === "ascii_depth") {
    drawASCIIDepthMap();
  }
  else if (effect === "thermal") {
    offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    applyThermalFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  }
  else if (effect === "wireframe") {
    offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    applySobelFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  }
  else if (effect === "charcoal") {
    offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    applyCharcoalFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  }
  else if (effect === "chrome") {
    offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    applyChromeFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  }
}
function getDistance(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = p1.z - p2.z;
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

// Check if hand is closed (Fist)
function isFist(landmarks) {
  const wrist = landmarks[0];
  const indexFolded = getDistance(landmarks[8], wrist) < getDistance(landmarks[5], wrist);
  const middleFolded = getDistance(landmarks[12], wrist) < getDistance(landmarks[9], wrist);
  const ringFolded = getDistance(landmarks[16], wrist) < getDistance(landmarks[13], wrist);
  const pinkyFolded = getDistance(landmarks[20], wrist) < getDistance(landmarks[17], wrist);
  return indexFolded && middleFolded && ringFolded && pinkyFolded;
}

// Check if index finger is extended
function isIndexExtended(landmarks) {
  const wrist = landmarks[0];
  return getDistance(landmarks[8], wrist) > getDistance(landmarks[6], wrist);
}

// Check if index and thumb are pinching (touching)
function isPinching(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  return getDistance(thumbTip, indexTip) < 0.035;
}

// Cycle active shader effect (direction: 1 = next, -1 = prev)
function shiftEffect(direction) {
  let currentIndex = effectsList.indexOf(selectedEffect);
  let newIndex = (currentIndex + direction + effectsList.length) % effectsList.length;
  selectedEffect = effectsList[newIndex];
  
  // Increment cycle count for rotating 3D box faces!
  if (direction > 0) {
    pinchCycleCount++;
  } else {
    pinchCycleCount = (pinchCycleCount - 1 + effectsList.length) % effectsList.length;
  }
  
  // Sync HTML drop-down
  effectSelect.value = selectedEffect;
  
  // Setup HUD alert
  const effectNames = {
    cloak: "INVISIBILITY CLOAK",
    crt_scanlines: "CRT MONITOR SCANLINES",
    line_halftone: "LINE HALFTONE SCREEN",
    dither: "MACINTOSH ORDERED DITHER",
    nokia: "NOKIA 3310 LCD SCREEN",
    crt_synth: "VHS ANALOG SYNTH GLITCH",
    ascii_depth: "3D ASCII DEPTH TELEMETRY",
    thermal: "THERMAL HEATMAP",
    wireframe: "CYBERPUNK WIREFRAME"
  };
  modeShiftMessage = `MODE SHIFT // ${effectNames[selectedEffect] || selectedEffect.toUpperCase()}`;
  modeShiftMessageTime = performance.now();
}

// ==========================================================================
// CORE APP LOOP (HAND TRACKING & OBJECT DETECTION ENGINE)
// ==========================================================================
let lastTime = 0;

function startAppLoop() {
  function renderFrame(time) {
    if (!isPlaying) {
      requestAnimationFrame(renderFrame);
      return;
    }

    // FPS Calculations
    fpsFrames++;
    if (time > lastFpsUpdate + 1000) {
      currentFps = Math.round((fpsFrames * 1000) / (time - lastFpsUpdate));
      fpsCounter.textContent = `FPS: ${currentFps}`;
      fpsFrames = 0;
      lastFpsUpdate = time;
    }

    // 1. Draw live feed frame on output canvas
    ctx.drawImage(activeVideo, 0, 0, outputCanvas.width, outputCanvas.height);

    const timestamp = performance.now();
    let handPolygon = null;

    // 2. Perform Hand Tracking
    if (handLandmarker && isModelsLoaded) {
      const handResults = handLandmarker.detectForVideo(activeVideo, timestamp);
      
      if (handResults.landmarks && handResults.landmarks.length > 0) {
        let anyHandIsFist = false;
        let allHandsActive = true;
        
        for (let i = 0; i < handResults.landmarks.length; i++) {
          const l = handResults.landmarks[i];
          if (isFist(l)) {
            anyHandIsFist = true;
          }
          if (!isIndexExtended(l)) {
            allHandsActive = false;
          }
        }
        
        let anyHandIsPinching = false;
        for (let i = 0; i < handResults.landmarks.length; i++) {
          if (isPinching(handResults.landmarks[i])) {
            anyHandIsPinching = true;
          }
        }

        // Handle Pinch Latch to change effects
        if (anyHandIsPinching) {
          if (!isPinchLatched) {
            shiftEffect(1); // Cycle to next effect
            isPinchLatched = true;
          }
        } else {
          isPinchLatched = false;
        }

        // Only draw polygon if no hand is a fist and all hands have extended indexes
        if (!anyHandIsFist && allHandsActive) {
          const points = [];
          const l0 = handResults.landmarks[0];
          
          if (handResults.landmarks.length >= 2) {
            // Two hands case: Get Index Tip (8) and Thumb Tip (4) from both hands
            const l1 = handResults.landmarks[1];
            
            points.push({ x: l0[4].x * outputCanvas.width, y: l0[4].y * outputCanvas.height }); // Hand 1 Thumb
            points.push({ x: l0[8].x * outputCanvas.width, y: l0[8].y * outputCanvas.height }); // Hand 1 Index
            points.push({ x: l1[8].x * outputCanvas.width, y: l1[8].y * outputCanvas.height }); // Hand 2 Index
            points.push({ x: l1[4].x * outputCanvas.width, y: l1[4].y * outputCanvas.height }); // Hand 2 Thumb
          } else {
            // One hand case: Gather Wrist (0), Thumb Tip (4), Index Tip (8), and Pinky Tip (20)
            points.push({ x: l0[0].x * outputCanvas.width, y: l0[0].y * outputCanvas.height });   // Wrist
            points.push({ x: l0[4].x * outputCanvas.width, y: l0[4].y * outputCanvas.height });   // Thumb
            points.push({ x: l0[8].x * outputCanvas.width, y: l0[8].y * outputCanvas.height });   // Index
            points.push({ x: l0[20].x * outputCanvas.width, y: l0[20].y * outputCanvas.height }); // Pinky
          }
          
          handPolygon = points;
        }
      } else {
        // Reset pinch latch if no hands detected
        isPinchLatched = false;
      }
    }

    // 3. Render Hand-tracked visual effect (Full-screen ASCII Depth or 3D Layered Box HUD)
    if (handPolygon) {
      if (selectedEffect === "ascii_depth") {
        // Renders ASCII Depth map across the entire viewport
        drawASCIIDepthMap();
      } else {
        // Draw the 3D layered rectangle around the 2D polygon
        
        // Calculate 2D bounding box from hand polygon
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        handPolygon.forEach(pt => {
          if (pt.x < minX) minX = pt.x;
          if (pt.y < minY) minY = pt.y;
          if (pt.x > maxX) maxX = pt.x;
          if (pt.y > maxY) maxY = pt.y;
        });
        
        const pad = 18;
        minX = Math.max(0, minX - pad);
        minY = Math.max(0, minY - pad);
        maxX = Math.min(outputCanvas.width, maxX + pad);
        maxY = Math.min(outputCanvas.height, maxY + pad);
        
        const W = maxX - minX;
        const H = maxY - minY;
        const CX = minX + W/2;
        const CY = minY + H/2;
        
        // Calculate 3D perspective projection for back face
        // Back face center is offset by a dynamic angle swinging over time
        const time = performance.now() / 1500;
        const angle = Math.sin(time) * 0.25 + 0.55; // swing angle in radians
        const depth = 45;
        const dx = depth * Math.cos(angle);
        const dy = depth * Math.sin(angle);
        const scale = 0.82; // scaled down back face for 3D depth perception
        
        // Define Vertices
        const fTL = { x: minX, y: minY };
        const fTR = { x: maxX, y: minY };
        const fBR = { x: maxX, y: maxY };
        const fBL = { x: minX, y: maxY };
        
        const bTL = { x: CX + (minX - CX)*scale + dx, y: CY + (minY - CY)*scale + dy };
        const bTR = { x: CX + (maxX - CX)*scale + dx, y: CY + (minY - CY)*scale + dy };
        const bBR = { x: CX + (maxX - CX)*scale + dx, y: CY + (maxY - CY)*scale + dy };
        const bBL = { x: CX + (minX - CX)*scale + dx, y: CY + (maxY - CY)*scale + dy };
        
        // 5 Faces: Front, Top, Right, Bottom, Left
        const faces = [
          { name: "front", poly: [fTL, fTR, fBR, fBL] },
          { name: "top", poly: [fTL, fTR, bTR, bTL] },
          { name: "right", poly: [fTR, fBR, bBR, bTR] },
          { name: "bottom", poly: [fBL, fBR, bBR, bBL] },
          { name: "left", poly: [fTL, fBL, bBL, bTL] }
        ];
        
        // Filter out cloak and ascii_depth from side faces
        const activeFiltersList = effectsList.filter(e => e !== "cloak" && e !== "ascii_depth");
        
        // Render each face of the 3D box
        faces.forEach((face, fIdx) => {
          ctx.save();
          ctx.globalAlpha = effectOpacity;
          
          // Clip rendering to this face's polygon path
          ctx.beginPath();
          ctx.moveTo(face.poly[0].x, face.poly[0].y);
          for (let i = 1; i < face.poly.length; i++) {
            ctx.lineTo(face.poly[i].x, face.poly[i].y);
          }
          ctx.closePath();
          ctx.clip();
          
          // Front face gets the user's selected effect.
          // Other side faces get rotating effects shifted by pinchCycleCount!
          let faceEffect = selectedEffect;
          if (face.name !== "front") {
            const effIdx = (fIdx - 1 + pinchCycleCount) % activeFiltersList.length;
            faceEffect = activeFiltersList[effIdx];
          }
          
          // Draw shader inside the face
          drawShader(faceEffect);
          ctx.restore();
          
          // Draw face borders/outline
          if (drawOutline) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(face.poly[0].x, face.poly[0].y);
            for (let i = 1; i < face.poly.length; i++) {
              ctx.lineTo(face.poly[i].x, face.poly[i].y);
            }
            ctx.closePath();
            
            // Front face gets bright cyan border, side faces get dim teal borders
            ctx.strokeStyle = (face.name === "front") ? "var(--accent-cyan)" : "rgba(0, 242, 254, 0.4)";
            ctx.lineWidth = (face.name === "front") ? 2 : 1;
            if (face.name === "front") {
              ctx.shadowColor = "var(--accent-cyan)";
              ctx.shadowBlur = 8;
            }
            ctx.stroke();
            ctx.restore();
          }
        });
        
        // Draw 3D wireframe connecting edges (connecting front and back faces)
        if (drawOutline) {
          ctx.save();
          ctx.strokeStyle = "rgba(0, 242, 254, 0.55)";
          ctx.lineWidth = 1.5;
          ctx.setLineDash([2, 3]); // dashed depth connector lines
          
          const corners = [
            [fTL, bTL],
            [fTR, bTR],
            [fBR, bBR],
            [fBL, bBL]
          ];
          
          corners.forEach(edge => {
            ctx.beginPath();
            ctx.moveTo(edge[0].x, edge[0].y);
            ctx.lineTo(edge[1].x, edge[1].y);
            ctx.stroke();
          });
          ctx.setLineDash([]);
          
          // Draw glowing corner points on the front face
          ctx.fillStyle = "#ffffff";
          ctx.shadowColor = "var(--accent-cyan)";
          ctx.shadowBlur = 10;
          [fTL, fTR, fBR, fBL].forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
            ctx.fill();
          });
          
          // Render HUD coordinates next to front corners
          ctx.fillStyle = "var(--accent-cyan)";
          ctx.font = "8px monospace";
          ctx.shadowBlur = 0;
          ctx.fillText(`TL [${Math.round(fTL.x)},${Math.round(fTL.y)}]`, fTL.x - 45, fTL.y - 8);
          ctx.fillText(`BR [${Math.round(fBR.x)},${Math.round(fBR.y)}]`, fBR.x + 10, fBR.y + 12);
          ctx.restore();
        }
      }
    }

    // 4. Perform AI Object Detection & HUD overlays
    if (enableObjectDetection && objectDetector && isModelsLoaded) {
      const detectResults = objectDetector.detectForVideo(activeVideo, timestamp);
      activeDetections = detectResults.detections || [];
      
      if (activeDetections.length > 0) {
        detectResults.detections.forEach(det => {
          const category = det.categories[0];
          if (!category) return;

          const label = category.categoryName;
          const score = Math.round(category.score * 100);
          const box = det.boundingBox;

          if (box) {
            // Draw neon bounding box with HUD corners
            ctx.save();
            ctx.strokeStyle = "var(--accent-cyan)";
            ctx.lineWidth = 2;
            
            // Corner indicators for cyberpunk HUD feel
            const x = box.originX;
            const y = box.originY;
            const w = box.width;
            const h = box.height;
            const len = Math.min(20, w * 0.2); // length of corner bars

            // Draw bounding rectangle
            ctx.strokeRect(x, y, w, h);

            // Bounding box HUD corner brackets
            ctx.beginPath();
            ctx.lineWidth = 4;
            // Top Left
            ctx.moveTo(x + len, y); ctx.lineTo(x, y); ctx.lineTo(x, y + len);
            // Top Right
            ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
            // Bottom Left
            ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h);
            // Bottom Right
            ctx.moveTo(x + w, y + h - len); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - len, y + h);
            ctx.stroke();

            // Label Tag Box (Top-Left of bounding box)
            const textString = `${label.toUpperCase()} ${score}%`;
            ctx.font = "bold 11px 'Space Grotesk'";
            const textWidth = ctx.measureText(textString).width;
            
            ctx.fillStyle = "rgba(5, 8, 17, 0.85)";
            ctx.fillRect(x, y - 22, textWidth + 16, 22);
            ctx.strokeStyle = "var(--accent-cyan)";
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y - 22, textWidth + 16, 22);

            // Write Category and Confidence Score
            ctx.fillStyle = "var(--accent-cyan)";
            ctx.fillText(textString, x + 8, y - 7);
            
            ctx.restore();
          }
        });
      }
    } else {
      activeDetections = [];
    }

    // 5. Draw HUD Message for Mode Shift
    if (modeShiftMessage && timestamp - modeShiftMessageTime < 1200) {
      ctx.save();
      ctx.font = "bold 18px 'Space Grotesk'";
      ctx.fillStyle = "rgba(5, 8, 17, 0.8)";
      const textWidth = ctx.measureText(modeShiftMessage).width;
      
      const boxX = outputCanvas.width / 2 - textWidth / 2 - 20;
      const boxY = 40;
      const boxW = textWidth + 40;
      const boxH = 36;
      
      // Draw transparent HUD box
      ctx.fillRect(boxX, boxY, boxW, boxH);
      ctx.strokeStyle = "var(--accent-cyan)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      
      // Draw neon corner indicator accents
      ctx.beginPath();
      ctx.lineWidth = 3;
      const cLen = 6;
      ctx.moveTo(boxX + cLen, boxY); ctx.lineTo(boxX, boxY); ctx.lineTo(boxX, boxY + cLen);
      ctx.moveTo(boxX + boxW - cLen, boxY); ctx.lineTo(boxX + boxW, boxY); ctx.lineTo(boxX + boxW, boxY + cLen);
      ctx.stroke();

      // Write text
      ctx.fillStyle = "var(--accent-cyan)";
      ctx.textAlign = "center";
      ctx.shadowColor = "var(--accent-cyan)";
      ctx.shadowBlur = 8;
      ctx.fillText(modeShiftMessage, outputCanvas.width / 2, boxY + 24);
      ctx.restore();
    }

    requestAnimationFrame(renderFrame);
  }
  
  requestAnimationFrame(renderFrame);
}

// ==========================================================================
// INITIALIZATION ON DOCUMENT LOAD
// ==========================================================================
window.addEventListener("DOMContentLoaded", () => {
  // Populate the camera select list on load safely
  try {
    updateCameraList();
  } catch (e) {
    console.error("Failed to update camera list:", e);
  }
  
  // Wait for user gesture to activate the vision engine
  if (btnStartApp) {
    btnStartApp.addEventListener("click", async () => {
      btnStartApp.style.display = "none";
      loaderSpinner.style.display = "block";
      loaderText.textContent = "Initializing AI Neural Networks...";
      
      // Start camera feed and AI model initialization in parallel
      await startWebcam();
      await initAIModels();
    });
  }
});
