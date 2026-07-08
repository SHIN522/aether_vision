# Aether Vision 🌌

Aether Vision is a real-time, browser-based **AI Creative Vision & Holographic HUD Application**. Inspired by interactive digital art installations (such as TouchDesigner) and client-side computer vision frameworks, it turns webcam feeds or uploaded video files into an interactive, gesture-controlled digital canvas.

---

## 🚀 Key Features

* **Gesture-Controlled Invisibility Cloak**: Form a closed boundary polygon using your fingers (both hands or single-hand gesture) to overlay a pre-captured empty background snapshot and make yourself "invisible".
* **Pinch-to-Rotate 3D Shaders**: Tap your index finger and thumb together in a pinch gesture to cycle through different visual effects and physically shift/rotate the shaders on the faces of the 3D HUD.
* **9 Creative Visual Shaders**:
  * 👤 *Invisibility Cloak*: Real-time background replacement.
  * 📺 *CRT Scanlines*: Monochrome/cyan CRT phosphorus monitor with horizontal scanlines and vertical sweeps.
  * 🖊️ *Line Halftone*: Diagonal publication-style halftone screening modulating thickness by pixel brightness.
  * 🎮 *Pixel Dither*: Retro GameBoy-style ordered 1-bit ordered dither using a 4x4 Bayer matrix.
  * 📱 *Nokia Screen*: Greenish-gray LCD monitor emulation with charcoal-green blocky pixels.
  * 📼 *CRT Glitch Synth*: VHS analog tape tracking simulation with horizontal waving, sync bars, and color channel splits.
  * 🧮 *ASCII Depth Map*: A full-screen 3D-perceived grid of characters warping around your hand, tracking detected surrounding objects.
  * 🌡️ *Thermal Heatmap*: Infrared thermal vision simulation.
  * 🩻 *Cyberpunk Wireframe*: High-contrast Sobel edge-detection filter mapping outlines in neon cyan.
* **Neural Object Tracking HUD**: Powered by Google's **EfficientDet Lite0** neural network to track everyday objects (e.g., humans) inside neon HUD bounding boxes with percentage confidence scores.
* **Dual-Mode Video Recorder**: Capture and download high-quality recordings of either the raw camera feed or the processed canvas overlay directly to your laptop as `.webm` files.
* **Dynamic Camera Selector**: Choose between multiple connected webcams in real-time.

---

## 🎨 Interactive Controls & Gestures

| Action / Gesture | Trigger | Result |
|---|---|---|
| **Form Polygon** | Extend Index & Thumb (1-hand / 2-hands) | Activates the tracking shape overlay |
| **Close Polygon** | Fold hand into a **Fist** | Immediately closes/disables the active effect |
| **Shift Effect** | Pinch Index & Thumb (Touch and release) | Automatically cycles to the next visual shader |

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SHIN522/aether_vision.git
cd aether_vision
```

### 2. Run Locally
Because the application loads local AI model files and MediaPipe WebAssembly modules via ES Modules, it requires a local web server (running straight from `index.html` file path will cause CORS issues).

You can start a simple server using Python:
```bash
# Python 3
python -m http.server 8000
```
Open your browser and navigate to:  
👉 **`http://localhost:8000`**

---

## 🧬 Technical Stack

* **Structure**: HTML5 Semantic Architecture.
* **Styling**: Vanilla CSS3 (Custom Glassmorphic sidebar panels, Glowing borders, and keyframe micro-animations).
* **Core Logic**: Modern ES6 JavaScript Modules.
* **Computer Vision**:
  * **MediaPipe Hand Landmarker** (Multihand tracking, depth estimation, and gesture detection).
  * **MediaPipe Object Detector** (EfficientDet Lite0).
* **Rendering & Performance**: HTML5 Canvas API. Heavy pixel manipulations (Thermal/Sobel filters) are downscaled to an offscreen `320x180` canvas buffer and rendered back upscaled, maintaining **30+ FPS** entirely client-side.
* **Recording**: Browser MediaRecorder API with canvas stream capturing (`canvas.captureStream(30)`).
