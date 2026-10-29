
"""SignBridge: Live MediaPipe hand-landmark detection."""

import cv2
import mediapipe as mp


def main():
    model_path = "models/hand_landmarker.task"

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model_path
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
            print("Hand detection active. Press Q to quit.")

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

                    for landmark in landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    for connection in connections:
                        start = connection.start
                        end = connection.end
                        p1 = landmarks[start]
                        p2 = landmarks[end]
                        cv2.line(
                            frame,
                            (int(p1.x * width), int(p1.y * height)),
                            (int(p2.x * width), int(p2.y * height)),
                            (255, 180, 0),
                            2,
                        )

                    message, color = "Hand detected", (0, 255, 0)
                else:
                    message, color = "No hand detected", (0, 0, 255)

                cv2.putText(
                    frame, message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
                cv2.imshow("SignBridge - Hand Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
