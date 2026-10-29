
"""SignBridge: Live HELP / NOT_HELP sign classification."""

import os

import cv2
import joblib
import mediapipe as mp
import numpy as np

import time
import winsound
from collections import deque



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKER_PATH = os.path.join(
    PROJECT_ROOT, "models", "hand_landmarker.task"
)
CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT, "models", "signbridge_model.joblib"
)


def main():
    HELP_CONFIRM_FRAMES = 8
    ALERT_COOLDOWN_SECONDS = 5

    if not os.path.exists(LANDMARKER_PATH):
        raise FileNotFoundError(
            f"MediaPipe model not found: {LANDMARKER_PATH}"
        )

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Sign classifier not found: {CLASSIFIER_PATH}"
        )

    classifier = joblib.load(CLASSIFIER_PATH)
    help_history = deque(maxlen=HELP_CONFIRM_FRAMES)
    last_alert_time = 0.0
    alert_active = False


    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=LANDMARKER_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    previous_timestamp = -1

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as detector:
            print("SignBridge prediction active. Press Q to quit.")

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp = int(
                    cv2.getTickCount() * 1000 / cv2.getTickFrequency()
                )
                timestamp = max(timestamp, previous_timestamp + 1)
                previous_timestamp = timestamp

                result = detector.detect_for_video(image, timestamp)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                    # Preserve the exact feature order used during training.
                    features = np.array(
                        [
                            coordinate
                            for landmark in landmarks
                            for coordinate in (
                                landmark.x,
                                landmark.y,
                                landmark.z,
                            )
                        ],
                        dtype=np.float32,
                    ).reshape(1, -1)

                    prediction = classifier.predict(features)[0]
                    help_history.append(prediction == "HELP")
                    confirmed_help = (
                        len(help_history) == HELP_CONFIRM_FRAMES
                        and all(help_history)
                        )
                    current_time = time.monotonic()
                    
                    if confirmed_help:
                        alert_active = True
                        
                        if current_time - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                            last_alert_time = current_time
                        else:
                            alert_active = False


                    # Draw landmarks and hand connections.
                    for landmark in landmarks:
                        cv2.circle(
                            frame,
                            (
                                int(landmark.x * width),
                                int(landmark.y * height),
                            ),
                            4,
                            (0, 255, 0),
                            -1,
                        )

                    for connection in connections:
                        start = landmarks[connection.start]
                        end = landmarks[connection.end]

                        cv2.line(
                            frame,
                            (
                                int(start.x * width),
                                int(start.y * height),
                            ),
                            (
                                int(end.x * width),
                                int(end.y * height),
                            ),
                            (255, 180, 0),
                            2,
                        )

                    if alert_active:
                            message = "HELP DETECTED! ALERT ACTIVE"
                            color = (0, 0, 255)
                    else:
                            message = f"Prediction: {prediction}"
                            color = (
                                (0, 255, 0)
                                if prediction == "HELP"
                                else (0, 200, 255)
                            )

                else:
                    message = "No hand detected"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    message,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                cv2.imshow("SignBridge - Live Prediction", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
