# VISIOAID-Voice-Assistant-for-blind-person
Real-time YOLO-based Object Detection with Distance Estimation & Voice Output

This project is a real-time vision assistant designed for visually impaired users.
It uses YOLOv8 for object detection, OpenCV for video capture, and pyttsx3 for offline text-to-speech.
When an object appears, the system announces its name and approximate distance. When the object disappears, it tells the user again.

✨ Features

🎥 Real-time camera input using OpenCV

🤖 Object detection powered by YOLOv8

📏 Distance estimation based on bounding-box height

🔊 Offline voice alerts using pyttsx3

🧠 Smart behaviour:

Announces the same object only a limited number of times

Prevents repeated announcements using a cooldown timer

Announces when the object disappears from view

🧵 Uses multi-threaded TTS to avoid blocking the detection loop

🛡️ Handles runtime errors safely without crashing the program

📦 Requirements

Install Python ≥ 3.8 and the following libraries:

pip install opencv-python ultralytics pyttsx3 numpy


If you face issues with TTS, reinstall pyttsx3:

pip install --upgrade pyttsx3

📁 Project Structure
project/
│
├── object.py      # Main application
├── yolov8x.pt     # YOLO model (downloaded automatically if missing)
└── README.md      # Documentation

▶️ How to Run

Connect a webcam

Place the YOLO model (yolov8x.pt) in the project folder

Run the script:

python object.py


A window will open showing real-time detection

Press Q to exit the application

⚙️ Configuration

You can modify the detection behaviour in the configuration section:

MODEL_PATH = "yolov8x.pt"
CAM_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MAX_SPEAK_COUNT = 2
DISTANCE_SCALAR = 5000.0
MIN_BOX_HEIGHT = 5
DISAPPEAR_HOLD_TIME = 0.5
SPEECH_COOLDOWN = 3.0

Meaning:

CAM_ID – camera index (0 = default webcam)

MAX_SPEAK_COUNT – how many times to announce each object

DISTANCE_SCALAR – tweak to adjust distance estimation

DISAPPEAR_HOLD_TIME – time after which a disappearance is announced

SPEECH_COOLDOWN – minimum delay between announcements

🧠 How It Works
1. Object Detection

The script uses:

results = model(frame, device=0)


YOLO returns bounding boxes, labels, and confidence scores.

2. Distance Calculation
distance = DISTANCE_SCALAR / box_height


Larger the box (closer object), smaller the distance.

3. Speech Using Multi-Threading

TTS runs on a separate worker thread, so the detection loop never freezes.

🙌 Troubleshooting
1. YOLO not found

Run:

pip install ultralytics

2. TTS not speaking

Install:

pip install pyttsx3

3. GPU not used

Make sure PyTorch + CUDA are installed:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

📘 Future Improvements

Add voice command recognition (Vosk / Whisper)

Support for object tracking IDs

Multi-language announcements

Obstacle direction ("left", "center", "right")

📜 License

This project is free to use and modify. Add your own license as required.

If you want, I can also generate:
✔ A project logo
✔ A GitHub description
✔ A setup.bat installer
✔ A full project folder structure
