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
const effectsList = ["cloak", "holo_glitch", "ascii_codex", "rgb_shift", "neon_pulse", "night_vision", "thermal", "wireframe"];

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

// Pure-JS RGB Shift Chromatic Aberration Filter
function applyRGBShiftFilter(srcCtx, destCtx, x, y, width, height, shiftAmount) {
  const imgData = srcCtx.getImageData(x, y, width, height);
  const data = imgData.data;
  const outData = new Uint8ClampedArray(data); // clone of source pixels
  
  for (let cy = 0; cy < height; cy++) {
    for (let cx = 0; cx < width; cx++) {
      const idx = (cy * width + cx) * 4;
      
      // Shift Red channel to the left
      const rx = Math.max(0, cx - shiftAmount);
      const rIdx = (cy * width + rx) * 4;
      data[idx] = outData[rIdx]; // Red
      
      // Shift Blue channel to the right
      const bx = Math.min(width - 1, cx + shiftAmount);
      const bIdx = (cy * width + bx) * 4;
      data[idx+2] = outData[bIdx+2]; // Blue
      
      // Green (idx+1) and Alpha (idx+3) remain unchanged
    }
  }
  
  destCtx.putImageData(imgData, x, y);
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

// Helper for distance between two 3D landmarks
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
  
  // Sync HTML drop-down
  effectSelect.value = selectedEffect;
  
  // Setup HUD alert
  const effectNames = {
    cloak: "INVISIBILITY CLOAK",
    duotone: "NEON DUOTONE",
    scan_grid: "CYBERPUNK SCAN GRID",
    thermal: "THERMAL HEATMAP",
    wireframe: "CYBERPUNK WIREFRAME"
  };
  modeShiftMessage = `MODE SHIFT // ${effectNames[selectedEffect]}`;
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

    // 3. Render Hand-tracked effects inside the Polygon
    if (handPolygon) {
      // Setup alpha opacity for the shader overlays
      ctx.save();
      ctx.globalAlpha = effectOpacity;

      // Canvas clipping to hand polygon path
      ctx.beginPath();
      ctx.moveTo(handPolygon[0].x, handPolygon[0].y);
      for (let i = 1; i < handPolygon.length; i++) {
        ctx.lineTo(handPolygon[i].x, handPolygon[i].y);
      }
      ctx.closePath();
      ctx.clip();

      // Apply selected visual shader
      if (selectedEffect === "cloak") {
        // Draw pre-captured background image
        if (isBgCaptured) {
          ctx.drawImage(backgroundCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
        } else {
          // Draw visual notification text if background is not locked
          ctx.fillStyle = "rgba(0, 242, 254, 0.05)";
          ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
          ctx.fillStyle = "var(--accent-cyan)";
          ctx.font = "16px 'Space Grotesk'";
          ctx.textAlign = "center";
          ctx.fillText("CAPTURE BACKGROUND TO ACTIVATE CLOAK", outputCanvas.width / 2, outputCanvas.height / 2);
        }
      } 
      else if (selectedEffect === "holo_glitch") {
        // Draw desaturated, cyan tinted hologram base
        ctx.save();
        ctx.filter = "grayscale(100%) brightness(0.9)";
        ctx.drawImage(activeVideo, 0, 0, outputCanvas.width, outputCanvas.height);
        ctx.filter = "none";
        
        ctx.globalCompositeOperation = "color";
        ctx.fillStyle = "rgba(0, 242, 254, 0.45)";
        ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
        ctx.globalCompositeOperation = "source-over";
        
        // Random horizontal slice glitches
        const now = performance.now();
        if (Math.random() < 0.35) {
          const numSlices = Math.floor(Math.random() * 4) + 2;
          for (let s = 0; s < numSlices; s++) {
            const sy = Math.random() * outputCanvas.height;
            const sh = Math.random() * 35 + 8;
            const disp = (Math.random() - 0.5) * 35; // displacement
            ctx.drawImage(
              outputCanvas, 
              0, sy, outputCanvas.width, sh, 
              disp, sy, outputCanvas.width, sh
            );
          }
        }
        
        // Scroll scanline
        ctx.strokeStyle = "rgba(0, 242, 254, 0.2)";
        ctx.lineWidth = 1.5;
        const scanY = (now / 3) % (outputCanvas.height + 100) - 50;
        ctx.beginPath();
        ctx.moveTo(0, scanY);
        ctx.lineTo(outputCanvas.width, scanY);
        ctx.stroke();
        ctx.restore();
      }
      else if (selectedEffect === "ascii_codex") {
        // Fill base with dark green-tinted background
        ctx.fillStyle = "rgba(5, 12, 8, 0.95)";
        ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
        
        // Draw video downsampled onto offscreen canvas for fast pixel scanning
        const sampleW = 60;
        const sampleH = 45;
        offscreenCtx.drawImage(activeVideo, 0, 0, sampleW, sampleH);
        const imgData = offscreenCtx.getImageData(0, 0, sampleW, sampleH);
        const pixels = imgData.data;
        
        // Draw matrix green/cyan ASCII symbols based on luma
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        
        const cellW = outputCanvas.width / sampleW;
        const cellH = outputCanvas.height / sampleH;
        const chars = "@%#*+=-:. ";
        
        for (let sy = 0; sy < sampleH; sy += 2) { // step by 2 for spacing and speed
          for (let sx = 0; sx < sampleW; sx++) {
            const idx = (sy * sampleW + sx) * 4;
            const r = pixels[idx];
            const g = pixels[idx+1];
            const b = pixels[idx+2];
            
            const luma = 0.299 * r + 0.587 * g + 0.114 * b;
            if (luma < 25) continue; // skip dark pixels
            
            const charIdx = Math.floor((luma / 255) * (chars.length - 1));
            const char = chars[charIdx];
            
            const cx = sx * cellW + cellW / 2;
            const cy = sy * cellH + cellH / 2;
            
            // Neon matrix green color with brightness mapping opacity
            ctx.fillStyle = `rgba(0, 255, 140, ${luma / 255})`;
            ctx.fillText(char, cx, cy);
          }
        }
      }
      else if (selectedEffect === "rgb_shift") {
        // Render shift using offscreen buffer and pure-JS filter
        offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        const shiftVal = Math.floor(Math.sin(performance.now() / 120) * 8) + 4; // Sinusoidal shift 4-12px
        applyRGBShiftFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height, shiftVal);
        ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
      }
      else if (selectedEffect === "neon_pulse") {
        // High-contrast monochromatic stylized base
        ctx.save();
        ctx.filter = "grayscale(100%) contrast(160%) brightness(75%)";
        ctx.drawImage(activeVideo, 0, 0, outputCanvas.width, outputCanvas.height);
        ctx.filter = "none";
        
        // Centroid wave rings
        const centroid = getPolygonCentroid(handPolygon);
        const now = performance.now() / 1000;
        
        ctx.lineWidth = 3;
        ctx.shadowBlur = 10;
        
        // Render 3 pulsing rings with phase shifts
        for (let r = 0; r < 3; r++) {
          const phase = (now * 160 + r * 80) % 240;
          const alpha = 1 - (phase / 240);
          
          // Cyan Ring
          ctx.strokeStyle = `rgba(0, 242, 254, ${alpha * 0.75})`;
          ctx.shadowColor = "rgba(0, 242, 254, 0.8)";
          ctx.beginPath();
          ctx.arc(centroid.x, centroid.y, phase, 0, Math.PI * 2);
          ctx.stroke();
          
          // Magenta Ring (delayed phase offset)
          ctx.strokeStyle = `rgba(255, 0, 128, ${alpha * 0.5})`;
          ctx.shadowColor = "rgba(255, 0, 128, 0.8)";
          ctx.beginPath();
          ctx.arc(centroid.x, centroid.y, Math.max(0, phase - 25), 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }
      else if (selectedEffect === "night_vision") {
        // Separated phosphor-green tint filter
        ctx.save();
        ctx.filter = "brightness(1.15) contrast(1.25) sepia(100%) hue-rotate(85deg) saturate(320%)";
        ctx.drawImage(activeVideo, 0, 0, outputCanvas.width, outputCanvas.height);
        ctx.filter = "none";
        
        // Overlay digital noise
        const staticData = offscreenCtx.createImageData(60, 45);
        const pix = staticData.data;
        for (let i = 0; i < pix.length; i += 4) {
          const rand = Math.random() * 255;
          pix[i] = rand;
          pix[i+1] = rand;
          pix[i+2] = rand;
          pix[i+3] = 30; // low opacity static
        }
        offscreenCtx.putImageData(staticData, 0, 0);
        ctx.globalCompositeOperation = "overlay";
        ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
        ctx.globalCompositeOperation = "source-over";
        
        // Draw HUD targeting sights at centroid
        const centroid = getPolygonCentroid(handPolygon);
        ctx.strokeStyle = "rgba(50, 255, 50, 0.85)";
        ctx.lineWidth = 1.5;
        
        // Reticle circles
        ctx.beginPath();
        ctx.arc(centroid.x, centroid.y, 25, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.arc(centroid.x, centroid.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(50, 255, 50, 0.85)";
        ctx.fill();
        
        // Crosshair ticks
        ctx.beginPath();
        ctx.moveTo(centroid.x - 38, centroid.y); ctx.lineTo(centroid.x - 12, centroid.y);
        ctx.moveTo(centroid.x + 12, centroid.y); ctx.lineTo(centroid.x + 38, centroid.y);
        ctx.moveTo(centroid.x, centroid.y - 38); ctx.lineTo(centroid.x, centroid.y - 12);
        ctx.moveTo(centroid.x, centroid.y + 12); ctx.lineTo(centroid.x, centroid.y + 38);
        ctx.stroke();
        
        // Cyber coordinates text overlays
        ctx.fillStyle = "rgba(50, 255, 50, 0.85)";
        ctx.font = "bold 9px monospace";
        ctx.fillText("SYS_LOCK // WEBCAM_TRACK", centroid.x + 42, centroid.y - 15);
        ctx.fillText(`X: ${Math.round(centroid.x)} | Y: ${Math.round(centroid.y)}`, centroid.x + 42, centroid.y - 3);
        ctx.fillText("RADAR_SWEEP: ACTIVE", centroid.x + 42, centroid.y + 9);
        ctx.restore();
      }
      else if (selectedEffect === "thermal") {
        // Render thermal effect via offscreen buffer for speed
        offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        applyThermalFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
      } 
      else if (selectedEffect === "wireframe") {
        // Render Sobel Wireframe filter
        offscreenCtx.drawImage(activeVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        applySobelFilter(offscreenCtx, offscreenCtx, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        ctx.drawImage(offscreenCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
      }
      ctx.restore();

      // Draw Glowing outline border
      if (drawOutline) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(handPolygon[0].x, handPolygon[0].y);
        for (let i = 1; i < handPolygon.length; i++) {
          ctx.lineTo(handPolygon[i].x, handPolygon[i].y);
        }
        ctx.closePath();

        ctx.strokeStyle = "var(--accent-cyan)";
        ctx.lineWidth = 3;
        ctx.shadowColor = "var(--accent-cyan)";
        ctx.shadowBlur = 12;
        ctx.stroke();

        // Draw glowing points at vertex corners
        ctx.fillStyle = "#ffffff";
        handPolygon.forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
      }
    }

    // 4. Perform AI Object Detection & HUD overlays
    if (enableObjectDetection && objectDetector && isModelsLoaded) {
      const detectResults = objectDetector.detectForVideo(activeVideo, timestamp);
      
      if (detectResults.detections && detectResults.detections.length > 0) {
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
