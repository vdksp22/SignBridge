# SignBridge

**Real-Time Hand Gesture Recognition for HELP and NOT_HELP Classification**

SignBridge is a computer-vision prototype that uses hand landmarks and machine learning to recognize two hand gestures: `HELP` and `NOT_HELP`. It processes webcam frames locally and displays predictions in real time, with an audible and visual alert when the HELP gesture is detected consistently.

## Features

- **Real-time hand tracking** using MediaPipe Hand Landmarker.
- **Landmark-based classification** using a Random Forest model.
- **Two gesture classes:** `HELP` and `NOT_HELP`.
- **Visual feedback** with hand landmarks, connections, and live predictions.
- **Local HELP alert** with a Windows notification sound and on-screen warning.
- **Lightweight inference** without requiring a dedicated GPU.

## Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy and pandas
- scikit-learn
- joblib

## System Architecture

```text
Webcam Input
    ↓
MediaPipe Hand Landmarker
    ↓
21 Hand Landmarks (x, y, z)
    ↓
63 Numerical Features
    ↓
Trained Random Forest Classifier
    ↓
HELP / NOT_HELP Prediction
    ↓
Visual Feedback + Local HELP Alert
```

## Machine Learning Methodology

1. Capture hand landmarks from webcam frames.
2. Store the 21 landmarks, each with x, y, and z coordinates, as 63 numerical features.
3. Assign each sample a `HELP` or `NOT_HELP` label.
4. Combine the samples and split them into training and test sets.
5. Train a Random Forest classifier using scikit-learn.
6. Serialize the trained classifier with joblib.
7. Load the saved classifier locally for real-time webcam inference.

### Initial Evaluation

The initial dataset contained 398 samples:

| Class | Samples |
|---|---:|
| HELP | 200 |
| NOT_HELP | 198 |
| **Total** | **398** |

An initial random 75/25 train-test split produced **100% accuracy on 100 test samples**.

This is a preliminary result, not a measure of proven real-world performance. Samples from the same recording session may be correlated. Further evaluation on independent recording sessions, different users, hand positions, and lighting conditions is needed.

## Project Structure

```text
SignBridge/
├── data/
│   └── README.md
├── docs/
├── models/
│   └── README.md
├── notebooks/
├── src/
│   ├── collect_data.py
│   ├── collect_landmarks.py
│   └── predict_sign.py
├── tests/
├── training/
├── .gitignore
├── README.md
└── requirements.txt
```

The dataset and model binaries are kept locally and excluded from Git. The MediaPipe task model and trained classifier must be available in the `models/` directory to run the corresponding scripts.

## Prerequisites

- Windows with Python 3.12 recommended for the current environment.
- A working webcam.
- Git and VS Code.
- The required model files, obtained separately.

## Installation

Clone the repository:

```bash
git clone https://github.com/vdksp22/SignBridge.git
cd SignBridge
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place these files in `models/` before running the corresponding scripts:

- `hand_landmarker.task` — MediaPipe hand-landmark detector.
- `signbridge_model.joblib` — trained SignBridge classifier.

The dataset and trained classifier are not included in the repository. Training instructions and dataset preparation should be followed separately.

## Usage

From the project root, activate the virtual environment and run the desired script.

**Visualize hand landmarks:**

```powershell
python .\src\collect_landmarks.py
```

**Collect training samples:**

```powershell
python .\src\collect_data.py
```

**Run live classification and the local HELP alert:**

```powershell
python .\src\predict_sign.py
```

Press `Q` while the webcam window is focused to exit.

## Limitations and Responsible Use

- The prototype recognizes only two trained classes.
- Predictions may be affected by lighting, camera angle, distance, and hand positioning.
- The initial evaluation uses a small dataset and may overestimate generalization.
- The HELP alert is local to the computer; it does not contact emergency services or notify another person.
- SignBridge is an experimental prototype, not a validated emergency-response or accessibility device.

## Future Improvements

- Evaluate using independent recording sessions and additional participants.
- Improve prediction stability and confidence handling.
- Expand the gesture vocabulary and support more diverse signing styles.
- Add automated tests and reproducible training workflows.
- Explore optional, consent-based notification mechanisms.

## Author

**Vedika Patil**

GitHub: [@vdksp22](https://github.com/vdksp22)

---

*Built as an applied computer-vision and machine-learning project.*
