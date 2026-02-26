# 🤟 ASL to Text Converter

A real-time **American Sign Language (ASL) fingerspelling** recognition web app. Show a hand sign to your webcam and the predicted letter appears instantly on screen. Hold a sign for 1.2 seconds to commit it to the text output.

Built with **MediaPipe Hands** + **Keras (.h5 model)** + **Flask** + vanilla HTML/CSS/JS.

---

## Demo

![ASL to Text UI](https://via.placeholder.com/900x400?text=ASL+to+Text+Converter+UI)

---

## How It Works

```
Webcam → Flask POST /api/recognize → MediaPipe (21 landmarks) → Keras MLP → Letter → UI
```

1. The browser captures a frame every ~80ms and POSTs it to the Flask backend
2. **MediaPipe Hands** detects the hand and extracts 21 landmark points (x, y) = 42 features
3. A pre-trained **Keras model** (`asl_model.h5`) classifies the 42 features into one of 29 classes
4. The predicted letter is returned and displayed instantly in the UI
5. **Hold-to-type**: holding the same sign for 1.2 seconds commits the letter to the text output

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand detection | MediaPipe Hands (Google) |
| Classifier | Keras MLP (`asl_model.h5`) |
| Backend | Flask (Python) |
| Frontend | HTML + CSS + JavaScript (no frameworks) |
| Transport | HTTP POST (`/api/recognize`) |

---

## Project Structure

```
asl-to-text/
├── app.py                  # Flask server + ModelAdapter
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   └── asl_model.h5        # Pre-trained Keras model (29 classes)
└── templates/
    └── index.html          # Full frontend UI
```

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/asl-to-text.git
cd asl-to-text
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## Usage

| Action | How |
|---|---|
| Start camera | Click **▶ Start** |
| Detect a letter | Show an ASL hand sign to the camera |
| Commit a letter | Hold the sign steady for **1.2 seconds** |
| Add a space | Hold the SPACE sign or click **␣ Space** |
| Delete last letter | Hold the DELETE sign or click **⌫** |
| Copy text | Click **📋 Copy** |
| Stop camera | Click **⏹ Stop** |

---

## Model Details

- **Input:** 42 features — raw `x, y` coordinates of 21 MediaPipe hand landmarks
- **Output:** 29 classes — A–Z + SPACE + DELETE + NOTHING
- **Architecture:** Keras MLP (Dense layers)
- **Framework:** TensorFlow / Keras (`.h5` format)

> The model was trained on the [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — 87,000 images across 29 classes.

---

## Tips for Best Recognition

- 💡 Good lighting on your hand makes a big difference
- ✋ Keep your hand fully in frame
- 🎯 Plain or simple backgrounds work best
- 📏 Hand roughly 30–60cm from the camera
- 🐢 Slow, deliberate signs are recognised more reliably than fast ones

---

## Requirements

```
flask>=3.0.0
mediapipe>=0.10.9
opencv-python>=4.9.0
numpy>=1.26.0
Pillow>=10.0.0
tensorflow>=2.15.0
```

Python 3.9–3.11 recommended.

---

## Known Limitations

- Static signs only (A–Z fingerspelling) — dynamic/motion signs (J, Z) are not supported
- Works best with one hand in frame at a time
- Accuracy may vary depending on lighting and camera quality

---

## License

MIT
