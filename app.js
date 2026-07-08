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
let enableObjectDetection = false;
let detectionThreshold = 0.3;
let handPolygon = null;
let lastObjectDetectionTime = 0;
let enableGlitchTrack = false;
let handHistory = [];
let wasHandTrackedLastFrame = false;
let lerpedBox = null;

// Glitch Accumulator Canvas for permanent non-lagging trails
const glitchAccumulatorCanvas = document.createElement("canvas");
const glitchAccumulatorCtx = glitchAccumulatorCanvas.getContext("2d");

// Motion Blob Tracker State
let enableMotionBlob = false;
let motionBlobMode = "squares";
let prevFrameData = null;
let trackedBlobs = [];
let blobIdCounter = 1;

const motionCanvas = document.createElement("canvas");
const motionCtx = motionCanvas.getContext("2d");
motionCanvas.width = 160;
motionCanvas.height = 90;

// WebGL GPU Shader Engine State
let glCanvas = null;
let gl = null;
let glProgram = null;
let glTexture = null;

// Pinch Gesture State
let isPinchLatched = false;
let modeShiftMessage = "";
let modeShiftMessageTime = 0;
const effectsList = ["cloak", "crt_scanlines", "dither", "nokia", "thermal", "wireframe"];
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

// High-performance ML tracking buffer (640x360)
const detectionCanvas = document.createElement("canvas");
const detectionCtx = detectionCanvas.getContext("2d");
detectionCanvas.width = 640;
detectionCanvas.height = 360;

// Pre-rendered offscreen caches for high performance (Render-To-Texture)
const filterEffects = ["crt_scanlines", "dither", "nokia", "thermal", "wireframe"];
const filterCache = {};
filterEffects.forEach(eff => {
  const canvas = document.createElement("canvas");
  canvas.width = (eff === "nokia") ? 84 : 640;
  canvas.height = (eff === "nokia") ? 48 : 360;
  const ctx = canvas.getContext("2d");
  filterCache[eff] = { canvas, ctx, isRendered: false };
});

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
// WEBGL GPU FILTER ENGINE
// ==========================================================================
function initWebGL() {
  logDebug("Initializing WebGL GPU Filter Engine...");
  glCanvas = document.createElement("canvas");
  glCanvas.width = outputCanvas.width;
  glCanvas.height = outputCanvas.height;
  
  gl = glCanvas.getContext("webgl", { alpha: false, depth: false, antialias: false, preserveDrawingBuffer: true });
  if (!gl) {
    logDebug("WARNING: WebGL context not available. Falling back to CPU pixel filters.");
    return;
  }
  logDebug("SUCCESS: WebGL context acquired successfully.");

  const vertices = new Float32Array([
    -1.0, -1.0,   0.0, 0.0, // bottom-left
     1.0, -1.0,   1.0, 0.0, // bottom-right
    -1.0,  1.0,   0.0, 1.0, // top-left
    -1.0,  1.0,   0.0, 1.0, // top-left
     1.0, -1.0,   1.0, 0.0, // bottom-right
     1.0,  1.0,   1.0, 1.0  // top-right
  ]);

  const vertexShaderSource = `
    attribute vec2 a_position;
    attribute vec2 a_texCoord;
    varying vec2 v_texCoord;
    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
      v_texCoord = a_texCoord;
    }
  `;

  const fragmentShaderSource = `
    precision mediump float;
    varying vec2 v_texCoord;
    uniform sampler2D u_image;
    uniform int u_effect;
    uniform float u_time;
    uniform vec2 u_resolution;

    float getLuma(vec3 color) {
      return dot(color, vec3(0.299, 0.587, 0.114));
    }

    void main() {
      vec2 uv = vec2(v_texCoord.x, 1.0 - v_texCoord.y); // flip vertically
      
      if (u_effect == 0) { // cloak / pass-through
        gl_FragColor = texture2D(u_image, uv);
      }
      else if (u_effect == 1) { // CRT Scanlines
        vec4 color = texture2D(u_image, uv);
        float luma = getLuma(color.rgb);
        vec3 tinted = vec3(luma * 1.1, luma * 0.95, luma * 0.75);
        float scanline = sin(uv.y * u_resolution.y * 2.0) * 0.12;
        tinted -= scanline;
        float sweepY = mod(u_time * 0.05, 1.2) - 0.1;
        float dist = abs(uv.y - sweepY);
        if (dist < 0.05) {
          tinted += (1.0 - dist / 0.05) * vec3(0.12, 0.12, 0.0);
        }
        gl_FragColor = vec4(tinted, 1.0);
      }
      else if (u_effect == 2) { // Bayer Dither
        vec4 color = texture2D(u_image, uv);
        float luma = getLuma(color.rgb);
        
        int x = int(mod(gl_FragCoord.x, 4.0));
        int y = int(mod(gl_FragCoord.y, 4.0));
        float threshold = 0.0;
        
        if (y == 0) {
          if (x == 0) threshold = 0.0;
          else if (x == 1) threshold = 0.50;
          else if (x == 2) threshold = 0.125;
          else threshold = 0.625;
        }
        else if (y == 1) {
          if (x == 0) threshold = 0.75;
          else if (x == 1) threshold = 0.25;
          else if (x == 2) threshold = 0.875;
          else threshold = 0.375;
        }
        else if (y == 2) {
          if (x == 0) threshold = 0.1875;
          else if (x == 1) threshold = 0.6875;
          else if (x == 2) threshold = 0.0625;
          else threshold = 0.5625;
        }
        else {
          if (x == 0) threshold = 0.9375;
          else if (x == 1) threshold = 0.4375;
          else if (x == 2) threshold = 0.8125;
          else threshold = 0.3125;
        }
        
        if (luma > threshold) {
          gl_FragColor = vec4(0.0, 0.95, 1.0, 1.0); // neon cyan
        } else {
          gl_FragColor = vec4(0.02, 0.03, 0.07, 1.0); // dark blue background
        }
      }
      else if (u_effect == 3) { // Nokia LCD
        vec2 nokiaRes = vec2(84.0, 48.0);
        vec2 blockCoord = floor(uv * nokiaRes);
        vec2 insideBlock = fract(uv * nokiaRes);
        
        if (insideBlock.x < 0.08 || insideBlock.y < 0.08) {
          gl_FragColor = vec4(0.1, 0.16, 0.1, 1.0); // grid border
        } else {
          vec2 sampleUV = (blockCoord + 0.5) / nokiaRes;
          vec4 color = texture2D(u_image, sampleUV);
          float luma = getLuma(color.rgb);
          if (luma < 0.45) {
            gl_FragColor = vec4(0.1, 0.16, 0.1, 1.0); // dark green
          } else {
            gl_FragColor = vec4(0.76, 0.81, 0.65, 1.0); // light green background
          }
        }
      }
      else if (u_effect == 4) { // Thermal Heatmap
        vec4 color = texture2D(u_image, uv);
        float v = getLuma(color.rgb);
        vec3 thermal;
        if (v < 0.25) {
          thermal = vec3(0.0, 0.0, v * 4.0);
        } else if (v < 0.50) {
          thermal = vec3((v - 0.25) * 4.0, 0.0, 1.0);
        } else if (v < 0.75) {
          thermal = vec3(1.0, (v - 0.50) * 4.0, 1.0 - (v - 0.50) * 4.0);
        } else {
          thermal = vec3(1.0, 1.0, (v - 0.75) * 4.0);
        }
        gl_FragColor = vec4(thermal, 1.0);
      }
      else if (u_effect == 5) { // Sobel Wireframe
        float stepX = 1.0 / u_resolution.x;
        float stepY = 1.0 / u_resolution.y;
        
        float t00 = getLuma(texture2D(u_image, uv + vec2(-stepX, -stepY)).rgb);
        float t10 = getLuma(texture2D(u_image, uv + vec2(0.0, -stepY)).rgb);
        float t20 = getLuma(texture2D(u_image, uv + vec2(stepX, -stepY)).rgb);
        float t01 = getLuma(texture2D(u_image, uv + vec2(-stepX, 0.0)).rgb);
        float t21 = getLuma(texture2D(u_image, uv + vec2(stepX, 0.0)).rgb);
        float t02 = getLuma(texture2D(u_image, uv + vec2(-stepX, stepY)).rgb);
        float t12 = getLuma(texture2D(u_image, uv + vec2(0.0, stepY)).rgb);
        float t22 = getLuma(texture2D(u_image, uv + vec2(stepX, stepY)).rgb);
        
        float gx = (t20 + 2.0 * t21 + t22) - (t00 + 2.0 * t01 + t02);
        float gy = (t02 + 2.0 * t12 + t22) - (t00 + 2.0 * t10 + t20);
        float g = sqrt(gx * gx + gy * gy);
        
        if (g > 0.18) {
          gl_FragColor = vec4(0.0, 0.95, 1.0, 1.0); // neon cyan edge
        } else {
          gl_FragColor = vec4(0.02, 0.03, 0.07, 0.85); // dark blue background
        }
      }
    }
  `;

  function compileShader(source, type) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("Shader compile error:", gl.getShaderInfoLog(s));
      return null;
    }
    return s;
  }

  const vs = compileShader(vertexShaderSource, gl.VERTEX_SHADER);
  const fs = compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER);
  if (!vs || !fs) return;

  glProgram = gl.createProgram();
  gl.attachShader(glProgram, vs);
  gl.attachShader(glProgram, fs);
  gl.linkProgram(glProgram);
  if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(glProgram));
    return;
  }

  gl.useProgram(glProgram);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

  const a_position = gl.getAttribLocation(glProgram, "a_position");
  gl.enableVertexAttribArray(a_position);
  gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 16, 0);

  const a_texCoord = gl.getAttribLocation(glProgram, "a_texCoord");
  gl.enableVertexAttribArray(a_texCoord);
  gl.vertexAttribPointer(a_texCoord, 2, gl.FLOAT, false, 16, 8);

  glTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, glTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
}

function renderWebGLShader(effectName) {
  if (!gl || !glProgram) return;

  if (glCanvas.width !== outputCanvas.width || glCanvas.height !== outputCanvas.height) {
    glCanvas.width = outputCanvas.width;
    glCanvas.height = outputCanvas.height;
    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  }

  gl.useProgram(glProgram);

  gl.bindTexture(gl.TEXTURE_2D, glTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, activeVideo);

  let effectInt = 0;
  if (effectName === "crt_scanlines") effectInt = 1;
  else if (effectName === "dither") effectInt = 2;
  else if (effectName === "nokia") effectInt = 3;
  else if (effectName === "thermal") effectInt = 4;
  else if (effectName === "wireframe") effectInt = 5;

  const u_effect = gl.getUniformLocation(glProgram, "u_effect");
  gl.uniform1i(u_effect, effectInt);

  const u_time = gl.getUniformLocation(glProgram, "u_time");
  gl.uniform1f(u_time, performance.now() / 10.0);

  const u_resolution = gl.getUniformLocation(glProgram, "u_resolution");
  gl.uniform2f(u_resolution, glCanvas.width, glCanvas.height);

  gl.drawArrays(gl.TRIANGLES, 0, 6);
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

    // Initialize WebGL engine
    try {
      initWebGL();
    } catch (e) {
      console.error("WebGL initialization failed:", e);
    }

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
  fileUploadZone.style.borderColor = "rgba(236, 243, 158, 0.3)";
});

fileUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  fileUploadZone.style.borderColor = "rgba(236, 243, 158, 0.3)";
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

const toggleGlitchTrackCheckbox = document.getElementById("toggle-glitch-track");
if (toggleGlitchTrackCheckbox) {
  toggleGlitchTrackCheckbox.addEventListener("change", (e) => {
    enableGlitchTrack = e.target.checked;
    if (!enableGlitchTrack) {
      handHistory = [];
      if (glitchAccumulatorCanvas.width > 0 && glitchAccumulatorCanvas.height > 0) {
        glitchAccumulatorCtx.fillStyle = "#000000";
        glitchAccumulatorCtx.fillRect(0, 0, glitchAccumulatorCanvas.width, glitchAccumulatorCanvas.height);
      }
    }
  });
}

const toggleMotionBlobCheckbox = document.getElementById("toggle-motion-blob");
if (toggleMotionBlobCheckbox) {
  toggleMotionBlobCheckbox.addEventListener("change", (e) => {
    enableMotionBlob = e.target.checked;
    if (!enableMotionBlob) {
      trackedBlobs = [];
      prevFrameData = null;
    }
  });
}

const blobModeSelect = document.getElementById("blob-mode-select");
if (blobModeSelect) {
  blobModeSelect.addEventListener("change", (e) => {
    motionBlobMode = e.target.value;
  });
}

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
  
  const time = performance.now() / 110;
  const syncBarY = Math.floor((performance.now() / 10) % height);
  
  // Random horizontal tearing bursts
  const noiseBurst = Math.random() < 0.16;
  const noiseBurstY = Math.floor(Math.random() * (height - 20));
  const noiseBurstH = Math.floor(Math.random() * 15) + 4;
  
  for (let cy = 0; cy < height; cy++) {
    // Sinusoidal wavy distortion of rows
    const wave = Math.sin(cy / 5.5 + time) * 6.5;
    
    // Shearing horizontal shift near the VHS tracking sync bar
    const distToSync = Math.abs(cy - syncBarY);
    const shear = distToSync < 15 ? (15 - distToSync) * 2.8 : 0;
    
    // Horizontal noise burst offset
    const burstOffset = (noiseBurst && cy >= noiseBurstY && cy < noiseBurstY + noiseBurstH) ? (Math.random() - 0.5) * 35 : 0;
    
    // Bottom tracking noise jitter
    const bottomJitter = (cy > height - 12) ? (Math.random() - 0.5) * 16 : 0;
    
    const shift = Math.floor(wave + shear + burstOffset + bottomJitter);
    
    for (let cx = 0; cx < width; cx++) {
      const idx = (cy * width + cx) * 4;
      
      // Bottom VHS tape tracking noise (snow static)
      if (cy > height - 10 && Math.random() < 0.55) {
        const snow = Math.random() * 255;
        data[idx] = snow;
        data[idx+1] = snow;
        data[idx+2] = snow;
        continue;
      }
      
      let srcX = cx + shift;
      if (srcX < 0) srcX = 0;
      if (srcX >= width) srcX = width - 1;
      
      // Chromatic aberration shifts (Red offset left, Blue offset right)
      const redOffset = 4;
      const blueOffset = -4;
      
      let redX = srcX + redOffset;
      let blueX = srcX + blueOffset;
      if (redX < 0) redX = 0; if (redX >= width) redX = width - 1;
      if (blueX < 0) blueX = 0; if (blueX >= width) blueX = width - 1;
      
      const srcIdx = (cy * width + srcX) * 4;
      const redIdx = (cy * width + redX) * 4;
      const blueIdx = (cy * width + blueX) * 4;
      
      // Glow/blowout color intensities
      data[idx] = Math.min(255, outData[redIdx] * 1.4);          // Red
      data[idx+1] = Math.min(255, outData[srcIdx+1] * 1.15);      // Green
      data[idx+2] = Math.min(255, outData[blueIdx+2] * 1.55);     // Blue
      
      // Draw VHS horizontal tracking lines
      if (distToSync < 3) {
        data[idx] *= 0.12;
        data[idx+1] *= 0.12;
        data[idx+2] *= 0.12;
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
  ctx.fillStyle = "rgba(19, 42, 19, 0.96)";
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
        const scaleX = outputCanvas.width / (activeVideo.videoWidth || 640);
        const scaleY = outputCanvas.height / (activeVideo.videoHeight || 360);
        for (let i = 0; i < activeDetections.length; i++) {
          const det = activeDetections[i];
          const box = det.boundingBox;
          if (box) {
            const bx = box.originX * scaleX;
            const by = box.originY * scaleY;
            const bw = box.width * scaleX;
            const bh = box.height * scaleY;
            if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh) {
              const cat = det.categories && det.categories[0];
              if (cat && cat.categoryName) {
                isInsideObject = true;
                objectLabel = cat.categoryName.toUpperCase();
                break;
              }
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
        ctx.fillStyle = `rgba(236, 243, 158, ${perceivedBrightness / 255})`;
      } else {
        // Ambient dim phosphor green for background environment
        ctx.fillStyle = `rgba(144, 169, 85, ${(perceivedBrightness / 255) * 0.75})`;
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

let currentRenderCtx = null;

// Unified Shader Draw Pipeline with WebGL GPU Acceleration
function drawShader(effect) {
  const drawCtx = currentRenderCtx || ctx;
  if (effect === "cloak") {
    if (isBgCaptured) {
      drawCtx.drawImage(backgroundCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
    } else {
      drawCtx.fillStyle = "rgba(19, 42, 19, 0.48)";
      drawCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
      drawCtx.fillStyle = "var(--accent-cyan)";
      drawCtx.font = "14px 'Space Grotesk'";
      drawCtx.textAlign = "center";
      drawCtx.fillText("CAPTURE BACKGROUND TO ACTIVATE CLOAK", outputCanvas.width / 2, outputCanvas.height / 2);
    }
    return;
  }

  // GPU Acceleration via WebGL (High-definition, 60+ FPS, Zero CPU pixel loops)
  if (gl && glProgram) {
    try {
      renderWebGLShader(effect);
      drawCtx.drawImage(glCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
      return;
    } catch (e) {
      console.warn("WebGL draw failed, falling back to CPU:", e);
    }
  }

  // CPU Fallback
  const cache = filterCache[effect];
  if (!cache) return;

  if (!cache.isRendered) {
    const cCtx = cache.ctx;
    const cW = cache.canvas.width;
    const cH = cache.canvas.height;

    if (effect === "crt_scanlines") {
      cCtx.save();
      cCtx.filter = "grayscale(100%) brightness(1.1) sepia(20%)";
      cCtx.drawImage(activeVideo, 0, 0, cW, cH);
      cCtx.filter = "none";
      
      cCtx.globalCompositeOperation = "color";
      cCtx.fillStyle = "rgba(236, 243, 158, 0.35)";
      cCtx.fillRect(0, 0, cW, cH);
      cCtx.globalCompositeOperation = "source-over";
      
      // Draw scanlines
      cCtx.fillStyle = "rgba(19, 42, 19, 0.28)";
      for (let sy = 0; sy < cH; sy += 3) {
        cCtx.fillRect(0, sy, cW, 1.5);
      }
      
      // Phosphor sweep line
      const sweepY = (performance.now() / 3.5) % (cH + 40) - 20;
      const sweepGrad = cCtx.createLinearGradient(0, sweepY - 20, 0, sweepY);
      sweepGrad.addColorStop(0, "rgba(236, 243, 158, 0)");
      sweepGrad.addColorStop(1, "rgba(236, 243, 158, 0.18)");
      cCtx.fillStyle = sweepGrad;
      cCtx.fillRect(0, sweepY - 20, cW, 20);
      cCtx.restore();
    }
    else if (effect === "dither") {
      cCtx.drawImage(activeVideo, 0, 0, cW, cH);
      applyBayerDither(cCtx, cCtx, 0, 0, cW, cH);
    }
    else if (effect === "nokia") {
      cCtx.drawImage(activeVideo, 0, 0, cW, cH);
      const imgData = cCtx.getImageData(0, 0, cW, cH);
      const pixels = imgData.data;
      cCtx.fillStyle = "#c2d0a7";
      cCtx.fillRect(0, 0, cW, cH);
      cCtx.fillStyle = "#1b2a1a";
      for (let sy = 0; sy < cH; sy++) {
        for (let sx = 0; sx < cW; sx++) {
          const idx = (sy * cW + sx) * 4;
          const r = pixels[idx];
          const g = pixels[idx+1];
          const b = pixels[idx+2];
          const luma = 0.299 * r + 0.587 * g + 0.114 * b;
          if (luma < 115) {
            cCtx.fillRect(sx, sy, 1, 1);
          }
        }
      }
    }
    else if (effect === "thermal") {
      cCtx.drawImage(activeVideo, 0, 0, cW, cH);
      applyThermalFilter(cCtx, cCtx, 0, 0, cW, cH);
    }
    else if (effect === "wireframe") {
      cCtx.drawImage(activeVideo, 0, 0, cW, cH);
      applySobelFilter(cCtx, cCtx, 0, 0, cW, cH);
    }
    cache.isRendered = true;
  }

  drawCtx.drawImage(cache.canvas, 0, 0, outputCanvas.width, outputCanvas.height);
}

function draw3DBox(ctxToDraw, boxItem, opacity) {
  if (!boxItem) return;
  
  // 5 Faces: Front, Top, Right, Bottom, Left
  const faces = [
    { name: "front", poly: [boxItem.fTL, boxItem.fTR, boxItem.fBR, boxItem.fBL] },
    { name: "top", poly: [boxItem.fTL, boxItem.fTR, boxItem.bTR, boxItem.bTL] },
    { name: "right", poly: [boxItem.fTR, boxItem.fBR, boxItem.bBR, boxItem.bTR] },
    { name: "bottom", poly: [boxItem.fBL, boxItem.fBR, boxItem.bBR, boxItem.bBL] },
    { name: "left", poly: [boxItem.fTL, boxItem.fBL, boxItem.bBL, boxItem.bTL] }
  ];
  
  // Filter out cloak from side faces
  const activeFiltersList = effectsList.filter(e => e !== "cloak");
  
  // Set target rendering context
  currentRenderCtx = ctxToDraw;
  
  // Render each face of the 3D box
  faces.forEach((face, fIdx) => {
    ctxToDraw.save();
    ctxToDraw.globalAlpha = opacity;
    
    // Clip rendering to this face's polygon path
    ctxToDraw.beginPath();
    ctxToDraw.moveTo(face.poly[0].x, face.poly[0].y);
    for (let i = 1; i < face.poly.length; i++) {
      ctxToDraw.lineTo(face.poly[i].x, face.poly[i].y);
    }
    ctxToDraw.closePath();
    ctxToDraw.clip();
    
    // Front face gets selected effect, others rotating shifted by pinchCycleCount
    let faceEffect = selectedEffect;
    if (face.name !== "front") {
      const effIdx = (fIdx - 1 + boxItem.pinchCycleCount) % activeFiltersList.length;
      faceEffect = activeFiltersList[effIdx];
    }
    
    drawShader(faceEffect);
    ctxToDraw.restore();
    
    // Draw borders/outline
    if (drawOutline) {
      ctxToDraw.save();
      ctxToDraw.globalAlpha = opacity;
      ctxToDraw.beginPath();
      ctxToDraw.moveTo(face.poly[0].x, face.poly[0].y);
      for (let i = 1; i < face.poly.length; i++) {
        ctxToDraw.lineTo(face.poly[i].x, face.poly[i].y);
      }
      ctxToDraw.closePath();
      
      ctxToDraw.strokeStyle = (face.name === "front") ? "var(--accent-cyan)" : "rgba(236, 243, 158, 0.4)";
      ctxToDraw.lineWidth = (face.name === "front") ? 2 : 1;
      if (face.name === "front") {
        ctxToDraw.shadowColor = "var(--accent-cyan)";
        ctxToDraw.shadowBlur = 8;
      }
      ctxToDraw.stroke();
      ctxToDraw.restore();
    }
  });
  
  // Draw depth connectors
  if (drawOutline) {
    ctxToDraw.save();
    ctxToDraw.globalAlpha = opacity;
    ctxToDraw.strokeStyle = "rgba(236, 243, 158, 0.55)";
    ctxToDraw.lineWidth = 1.5;
    ctxToDraw.setLineDash([2, 3]);
    
    const corners = [
      [boxItem.fTL, boxItem.bTL],
      [boxItem.fTR, boxItem.bTR],
      [boxItem.fBR, boxItem.bBR],
      [boxItem.fBL, boxItem.bBL]
    ];
    
    corners.forEach(edge => {
      ctxToDraw.beginPath();
      ctxToDraw.moveTo(edge[0].x, edge[0].y);
      ctxToDraw.lineTo(edge[1].x, edge[1].y);
      ctxToDraw.stroke();
    });
    ctxToDraw.setLineDash([]);
    
    // Draw glowing vertices
    ctxToDraw.fillStyle = "#ffffff";
    ctxToDraw.shadowColor = "var(--accent-cyan)";
    ctxToDraw.shadowBlur = 10;
    [boxItem.fTL, boxItem.fTR, boxItem.fBR, boxItem.fBL].forEach(pt => {
      ctxToDraw.beginPath();
      ctxToDraw.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctxToDraw.fill();
    });
    
    // Render HUD labels
    ctxToDraw.fillStyle = "var(--accent-cyan)";
    ctxToDraw.font = "8px monospace";
    ctxToDraw.shadowBlur = 0;
    ctxToDraw.fillText(`TL [${Math.round(boxItem.fTL.x)},${Math.round(boxItem.fTL.y)}]`, boxItem.fTL.x - 45, boxItem.fTL.y - 8);
    ctxToDraw.fillText(`BR [${Math.round(boxItem.fBR.x)},${Math.round(boxItem.fBR.y)}]`, boxItem.fBR.x + 10, boxItem.fBR.y + 12);
    ctxToDraw.restore();
  }
  
  // Reset target rendering context
  currentRenderCtx = null;
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
// MOTION BLOB TRACKER ENGINE
// ==========================================================================
function processMotionBlobTracker() {
  if (!enableMotionBlob) return;

  // 1. Draw video downscaled to motion calculation canvas (160x90) for fast CPU processing
  motionCtx.drawImage(activeVideo, 0, 0, 160, 90);
  const currImgData = motionCtx.getImageData(0, 0, 160, 90);
  const currData = currImgData.data;

  if (!prevFrameData) {
    prevFrameData = new Uint8ClampedArray(currData);
    return;
  }

  // 2. Compute 16x9 movement density grid to filter noise and locate moving blobs
  const gridCols = 16;
  const gridRows = 9;
  const blockW = 10;
  const blockH = 10;
  const motionGrid = [];

  for (let r = 0; r < gridRows; r++) {
    motionGrid[r] = [];
    for (let c = 0; c < gridCols; c++) {
      let activePixels = 0;
      
      // Check difference inside this 10x10 block
      for (let by = 0; by < blockH; by++) {
        const py = r * blockH + by;
        for (let bx = 0; bx < blockW; bx++) {
          const px = c * blockW + bx;
          const idx = (py * 160 + px) * 4;
          
          const lumaCurr = 0.299 * currData[idx] + 0.587 * currData[idx+1] + 0.114 * currData[idx+2];
          const lumaPrev = 0.299 * prevFrameData[idx] + 0.587 * prevFrameData[idx+1] + 0.114 * prevFrameData[idx+2];
          
          if (Math.abs(lumaCurr - lumaPrev) > 22) {
            activePixels++;
          }
        }
      }
      
      // If > 18% of pixels in this block are moving, mark block as active
      motionGrid[r][c] = (activePixels > 18);
    }
  }

  // Save current frame data for next frame differencing comparison
  prevFrameData.set(currData);

  // 3. Group adjacent active grid blocks into blobs (Connected Components Clustering via BFS)
  const visited = [];
  for (let r = 0; r < gridRows; r++) {
    visited[r] = [];
    for (let c = 0; c < gridCols; c++) {
      visited[r][c] = false;
    }
  }

  const currentFrameBlobs = [];

  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      if (motionGrid[r][c] && !visited[r][c]) {
        // Start BFS to gather all connected blocks
        let minR = r, maxR = r;
        let minC = c, maxC = c;
        const queue = [{r, c}];
        visited[r][c] = true;

        while (queue.length > 0) {
          const curr = queue.shift();
          
          // 4-way check
          const neighbors = [
            {r: curr.r - 1, c: curr.c},
            {r: curr.r + 1, c: curr.c},
            {r: curr.r, c: curr.c - 1},
            {r: curr.r, c: curr.c + 1}
          ];

          for (let i = 0; i < neighbors.length; i++) {
            const n = neighbors[i];
            if (n.r >= 0 && n.r < gridRows && n.c >= 0 && n.c < gridCols) {
              if (motionGrid[n.r][n.c] && !visited[n.r][n.c]) {
                visited[n.r][n.c] = true;
                queue.push(n);
                
                if (n.r < minR) minR = n.r;
                if (n.r > maxR) maxR = n.r;
                if (n.c < minC) minC = n.c;
                if (n.c > maxC) maxC = n.c;
              }
            }
          }
        }

        // Calculate normalized bounding coordinates (0.0 to 1.0)
        const normMinX = minC / gridCols;
        const normMaxX = (maxC + 1) / gridCols;
        const normMinY = minR / gridRows;
        const normMaxY = (maxR + 1) / gridRows;

        // Scale normalized dimensions to outputCanvas pixel coordinates
        const x = normMinX * outputCanvas.width;
        const y = normMinY * outputCanvas.height;
        const w = (normMaxX - normMinX) * outputCanvas.width;
        const h = (normMaxY - normMinY) * outputCanvas.height;
        const cx = x + w / 2;
        const cy = y + h / 2;

        // Filter out tiny clusters (camera artifacts/noise) to ensure clean blob isolation
        if (w > outputCanvas.width * 0.05 && h > outputCanvas.height * 0.05) {
          currentFrameBlobs.push({ x, y, w, h, cx, cy });
        }
      }
    }
  }

  // 4. PERSISTENT MULTI-OBJECT TRACKING (Match current blobs with previous frame to maintain constant IDs)
  const nextTrackedBlobs = [];
  const maxMatchingDist = outputCanvas.width * 0.22; // max distance to map same object

  currentFrameBlobs.forEach(newBlob => {
    let bestMatch = null;
    let minDist = Infinity;

    trackedBlobs.forEach(oldBlob => {
      const dx = newBlob.cx - oldBlob.cx;
      const dy = newBlob.cy - oldBlob.cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist && dist < maxMatchingDist) {
        minDist = dist;
        bestMatch = oldBlob;
      }
    });

    if (bestMatch) {
      // Map existing ID and history path
      const history = [...bestMatch.history];
      history.push({ cx: newBlob.cx, cy: newBlob.cy });
      if (history.length > 25) history.shift();

      // Apply linear interpolation (lerp) on rendering coordinates (rx, ry, rw, rh) for smooth 60fps transitions
      const lf = 0.18; // lerp factor
      const rx = bestMatch.rx + (newBlob.x - bestMatch.rx) * lf;
      const ry = bestMatch.ry + (newBlob.y - bestMatch.ry) * lf;
      const rw = bestMatch.rw + (newBlob.w - bestMatch.rw) * lf;
      const rh = bestMatch.rh + (newBlob.h - bestMatch.rh) * lf;
      const rcx = rx + rw / 2;
      const rcy = ry + rh / 2;

      // Compute velocity vector (vx, vy)
      const vx = newBlob.cx - bestMatch.cx;
      const vy = newBlob.cy - bestMatch.cy;

      nextTrackedBlobs.push({
        id: bestMatch.id,
        x: newBlob.x,
        y: newBlob.y,
        w: newBlob.w,
        h: newBlob.h,
        cx: newBlob.cx,
        cy: newBlob.cy,
        rx, ry, rw, rh, rcx, rcy,
        vx, vy,
        history: history
      });

      // Remove from pool to prevent double matching
      trackedBlobs = trackedBlobs.filter(b => b.id !== bestMatch.id);
    } else {
      // Generate new Tracking ID
      nextTrackedBlobs.push({
        id: blobIdCounter++,
        x: newBlob.x,
        y: newBlob.y,
        w: newBlob.w,
        h: newBlob.h,
        cx: newBlob.cx,
        cy: newBlob.cy,
        rx: newBlob.x,
        ry: newBlob.y,
        rw: newBlob.w,
        rh: newBlob.h,
        rcx: newBlob.cx,
        rcy: newBlob.cy,
        vx: 0,
        vy: 0,
        history: [{ cx: newBlob.cx, cy: newBlob.cy }]
      });
    }
  });

  trackedBlobs = nextTrackedBlobs;

  // Helper function to draw glowing target brackets on a rectangle
  function drawTelemetryBrackets(bx, by, bw, bh, colorString) {
    ctx.save();
    ctx.strokeStyle = colorString;
    ctx.lineWidth = 2.0;
    ctx.strokeRect(bx, by, bw, bh);

    const len = Math.min(15, bw * 0.2);
    ctx.beginPath();
    ctx.lineWidth = 4.0;
    // Top-Left
    ctx.moveTo(bx + len, by); ctx.lineTo(bx, by); ctx.lineTo(bx, by + len);
    // Top-Right
    ctx.moveTo(bx + bw - len, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + len);
    // Bottom-Left
    ctx.moveTo(bx, by + bh - len); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + len, by + bh);
    // Bottom-Right
    ctx.moveTo(bx + bw, by + bh - len); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw - len, by + bh);
    ctx.stroke();
    ctx.restore();
  }

  // Helper function to draw a cool target acquisition crosshair
  function drawTargetCrosshair(cx, cy, colorString) {
    ctx.save();
    ctx.strokeStyle = colorString;
    ctx.lineWidth = 1.2;
    ctx.shadowColor = colorString;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    // Inner targeting circle
    ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    // Outer tick lines
    ctx.moveTo(cx - 14, cy); ctx.lineTo(cx + 14, cy);
    ctx.moveTo(cx, cy - 14); ctx.lineTo(cx, cy + 14);
    ctx.stroke();
    ctx.restore();
  }

  // 5. RENDER CHOSEN BLOB TRACKING MODE (Squares, Isolate, Track All)
  if (motionBlobMode === "isolate") {
    // Mode 1: Isolate active blobs on solid black background WITH glowing bounding boxes
    ctx.save();
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);

    trackedBlobs.forEach(blob => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(blob.rx, blob.ry, blob.rw, blob.rh);
      ctx.clip();
      
      // Draw full resolution (non-downscaled 4K) video frame inside clip
      ctx.drawImage(activeVideo, 0, 0, outputCanvas.width, outputCanvas.height);
      ctx.restore();

      // Render glowing telemetry box outline around isolated block
      drawTelemetryBrackets(blob.rx, blob.ry, blob.rw, blob.rh, "rgba(0, 242, 254, 0.85)");
    });
    ctx.restore();
  }
  else if (motionBlobMode === "squares") {
    // Mode 2: Draw telemetry overlay squares with tracking ID, sizes, velocity vector lines
    trackedBlobs.forEach(blob => {
      ctx.save();
      
      // Draw neon box and corner brackets
      drawTelemetryBrackets(blob.rx, blob.ry, blob.rw, blob.rh, "var(--accent-cyan)");
      
      // Draw target crosshair at centroid
      drawTargetCrosshair(blob.rcx, blob.rcy, "var(--accent-cyan)");

      // Bounding box ID tag
      const labelText = `OBJECT #${blob.id}`;
      ctx.font = "bold 11px 'Space Grotesk'";
      const textWidth = ctx.measureText(labelText).width;
      
      ctx.fillStyle = "rgba(19, 42, 19, 0.85)";
      ctx.fillRect(blob.rx, blob.ry - 20, textWidth + 16, 20);
      ctx.strokeStyle = "var(--accent-cyan)";
      ctx.lineWidth = 1;
      ctx.strokeRect(blob.rx, blob.ry - 20, textWidth + 16, 20);
      
      ctx.fillStyle = "var(--accent-cyan)";
      ctx.fillText(labelText, blob.rx + 8, blob.y - 6);

      // Cyberpunk-style overlay telemetry text under the box
      ctx.font = "8px monospace";
      ctx.fillStyle = "var(--accent-cyan)";
      ctx.shadowBlur = 0;
      
      // Size and Velocity metrics
      const sizeText = `SIZE: ${Math.round(blob.rw)}x${Math.round(blob.rh)} px`;
      const velocityVal = Math.sqrt(blob.vx * blob.vx + blob.vy * blob.vy) * 30; // convert to px/sec approx
      const velText = `VELOCITY: ${Math.round(velocityVal)} px/s`;
      
      ctx.fillText(sizeText, blob.rx + 4, blob.ry + blob.rh + 12);
      ctx.fillText(velText, blob.rx + 4, blob.ry + blob.rh + 22);

      // Draw direction vector arrow extending from centroid
      if (Math.abs(blob.vx) > 0.1 || Math.abs(blob.vy) > 0.1) {
        ctx.beginPath();
        ctx.moveTo(blob.rcx, blob.rcy);
        ctx.lineTo(blob.rcx + blob.vx * 3.5, blob.rcy + blob.vy * 3.5);
        ctx.strokeStyle = "rgba(255, 78, 62, 0.85)"; // orange vector line
        ctx.lineWidth = 2.0;
        ctx.stroke();
      }

      ctx.restore();
    });
  }
  else if (motionBlobMode === "track_all") {
    // Mode 3: Track all items with thin neon borders and fading vector path trails (light cycle effect)
    trackedBlobs.forEach(blob => {
      ctx.save();
      
      // Thin neon border
      ctx.strokeStyle = "rgba(236, 243, 158, 0.6)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(blob.rx, blob.ry, blob.rw, blob.rh);

      // Draw target crosshair at centroid
      drawTargetCrosshair(blob.rcx, blob.rcy, "var(--accent-cyan)");

      // Draw fading Tron-style motion trails
      if (blob.history.length > 1) {
        for (let i = 1; i < blob.history.length; i++) {
          const p1 = blob.history[i - 1];
          const p2 = blob.history[i];
          const pct = i / blob.history.length; // fade older segments
          
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(p1.cx, p1.cy);
          ctx.lineTo(p2.cx, p2.cy);
          ctx.strokeStyle = `rgba(0, 242, 254, ${pct * 0.85})`;
          ctx.lineWidth = 1.0 + pct * 2.5; // thicker/brighter closer to target
          ctx.stroke();
          ctx.restore();
        }
      }

      // Coordinate telemetry text labels
      ctx.fillStyle = "var(--accent-cyan)";
      ctx.font = "8px monospace";
      ctx.shadowBlur = 0;
      ctx.fillText(`LOC [${Math.round(blob.rcx)},${Math.round(blob.rcy)}]`, blob.rcx + 12, blob.rcy - 12);

      ctx.restore();
    });
  }
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

    // Reset shader pre-render cache flags for the new frame
    filterEffects.forEach(eff => {
      filterCache[eff].isRendered = false;
    });

    // 1. Draw live feed frame or clear to black when Glitch Track is active
    if (enableGlitchTrack) {
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    } else {
      ctx.drawImage(activeVideo, 0, 0, outputCanvas.width, outputCanvas.height);
    }

    const timestamp = performance.now();
    handPolygon = null;

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

    // 3. Render Hand-tracked visual effect (3D Layered Box HUD with optional Blob Glitch Track trails)
    if (handPolygon) {
      // Clear history of past tracking sessions if hand is newly re-detected to ensure trails overwrite instantly
      if (!wasHandTrackedLastFrame) {
        handHistory = [];
      }
      wasHandTrackedLastFrame = true;

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
      const time = performance.now() / 1500;
      const angle = Math.sin(time) * 0.25 + 0.55; // swing angle in radians
      const depth = 45;
      const dx = depth * Math.cos(angle);
      const dy = depth * Math.sin(angle);
      const scale = 0.82; // scaled down back face for 3D depth perception
      
      // Define Target Vertices
      const targetFTL = { x: minX, y: minY };
      const targetFTR = { x: maxX, y: minY };
      const targetFBR = { x: maxX, y: maxY };
      const targetFBL = { x: minX, y: maxY };
      
      const targetBTL = { x: CX + (minX - CX)*scale + dx, y: CY + (minY - CY)*scale + dy };
      const targetBTR = { x: CX + (maxX - CX)*scale + dx, y: CY + (minY - CY)*scale + dy };
      const targetBBR = { x: CX + (maxX - CX)*scale + dx, y: CY + (maxY - CY)*scale + dy };
      const targetBBL = { x: CX + (minX - CX)*scale + dx, y: CY + (maxY - CY)*scale + dy };

      // Initialize or interpolate coordinates for buttery-smooth 60 FPS motion
      if (!lerpedBox) {
        lerpedBox = {
          fTL: { ...targetFTL }, fTR: { ...targetFTR }, fBR: { ...targetFBR }, fBL: { ...targetFBL },
          bTL: { ...targetBTL }, bTR: { ...targetBTR }, bBR: { ...targetBBR }, bBL: { ...targetBBL }
        };
      } else {
        const lf = 0.25; // lerp speed coefficient
        lerpedBox.fTL.x += (targetFTL.x - lerpedBox.fTL.x) * lf;
        lerpedBox.fTL.y += (targetFTL.y - lerpedBox.fTL.y) * lf;
        lerpedBox.fTR.x += (targetFTR.x - lerpedBox.fTR.x) * lf;
        lerpedBox.fTR.y += (targetFTR.y - lerpedBox.fTR.y) * lf;
        lerpedBox.fBR.x += (targetFBR.x - lerpedBox.fBR.x) * lf;
        lerpedBox.fBR.y += (targetFBR.y - lerpedBox.fBR.y) * lf;
        lerpedBox.fBL.x += (targetFBL.x - lerpedBox.fBL.x) * lf;
        lerpedBox.fBL.y += (targetFBL.y - lerpedBox.fBL.y) * lf;
        
        lerpedBox.bTL.x += (targetBTL.x - lerpedBox.bTL.x) * lf;
        lerpedBox.bTL.y += (targetBTL.y - lerpedBox.bTL.y) * lf;
        lerpedBox.bTR.x += (targetBTR.x - lerpedBox.bTR.x) * lf;
        lerpedBox.bTR.y += (targetBTR.y - lerpedBox.bTR.y) * lf;
        lerpedBox.bBR.x += (targetBBR.x - lerpedBox.bBR.x) * lf;
        lerpedBox.bBR.y += (targetBBR.y - lerpedBox.bBR.y) * lf;
        lerpedBox.bBL.x += (targetBBL.x - lerpedBox.bBL.x) * lf;
        lerpedBox.bBL.y += (targetBBL.y - lerpedBox.bBL.y) * lf;
      }
    } else {
      wasHandTrackedLastFrame = false;
      lerpedBox = null;
    }

    // Render the active HUD box or cascading glitch trails
    if (enableGlitchTrack) {
      // Keep glitch accumulator canvas in sync with output canvas size
      if (glitchAccumulatorCanvas.width !== outputCanvas.width || glitchAccumulatorCanvas.height !== outputCanvas.height) {
        glitchAccumulatorCanvas.width = outputCanvas.width;
        glitchAccumulatorCanvas.height = outputCanvas.height;
        glitchAccumulatorCtx.fillStyle = "#000000";
        glitchAccumulatorCtx.fillRect(0, 0, glitchAccumulatorCanvas.width, glitchAccumulatorCanvas.height);
      }

      // Draw current box onto accumulator context (persistent canvas)
      if (handPolygon && lerpedBox) {
        const boxItem = {
          fTL: { ...lerpedBox.fTL }, fTR: { ...lerpedBox.fTR }, fBR: { ...lerpedBox.fBR }, fBL: { ...lerpedBox.fBL },
          bTL: { ...lerpedBox.bTL }, bTR: { ...lerpedBox.bTR }, bBR: { ...lerpedBox.bBR }, bBL: { ...lerpedBox.bBL },
          pinchCycleCount
        };
        draw3DBox(glitchAccumulatorCtx, boxItem, effectOpacity * 0.75);
      }

      // Render the accumulated permanent trails to the main screen
      ctx.drawImage(glitchAccumulatorCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
    } else {
      // Draw single active 3D box on the main canvas
      if (handPolygon && lerpedBox) {
        const boxItem = {
          fTL: { ...lerpedBox.fTL }, fTR: { ...lerpedBox.fTR }, fBR: { ...lerpedBox.fBR }, fBL: { ...lerpedBox.fBL },
          bTL: { ...lerpedBox.bTL }, bTR: { ...lerpedBox.bTR }, bBR: { ...lerpedBox.bBR }, bBL: { ...lerpedBox.bBL },
          pinchCycleCount
        };
        draw3DBox(ctx, boxItem, effectOpacity);
      }
    }

    // 4. Perform AI Object Detection & HUD overlays (Throttled for 60 FPS performance)
    if (enableObjectDetection && objectDetector && isModelsLoaded) {
      if (timestamp - lastObjectDetectionTime > 120) {
        lastObjectDetectionTime = timestamp;
        try {
          const detectResults = objectDetector.detectForVideo(activeVideo, timestamp);
          activeDetections = detectResults.detections || [];
        } catch (e) {
          console.warn("Object detection skipped for frame: ", e);
        }
      }
      
      if (activeDetections.length > 0) {
        activeDetections.forEach(det => {
          const category = det.categories && det.categories[0];
          if (!category) return;

          const label = category.categoryName;
          const score = Math.round(category.score * 100);
          const box = det.boundingBox;

          if (box) {
            // Draw neon bounding box with HUD corners
            ctx.save();
            ctx.strokeStyle = "var(--accent-cyan)";
            ctx.lineWidth = 2;
            
            // Scale bounding box from activeVideo size to full canvas size
            const scaleX = outputCanvas.width / (activeVideo.videoWidth || 640);
            const scaleY = outputCanvas.height / (activeVideo.videoHeight || 360);
            const x = box.originX * scaleX;
            const y = box.originY * scaleY;
            const w = box.width * scaleX;
            const h = box.height * scaleY;
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
            
            ctx.fillStyle = "rgba(19, 42, 19, 0.85)";
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

    // 4.5. Run Motion Blob Tracker (GPU-accelerated overlay mapping)
    processMotionBlobTracker();

    // 5. Draw HUD Message for Mode Shift
    if (modeShiftMessage && timestamp - modeShiftMessageTime < 1200) {
      ctx.save();
      ctx.font = "bold 18px 'Space Grotesk'";
      ctx.fillStyle = "rgba(19, 42, 19, 0.8)";
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
