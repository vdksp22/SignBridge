
"""SignBridge: Collect hand-landmark training data into CSV files."""

import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\nSignBridge Data Collector")
    print("1 = HELP")
    print("2 = NOT_HELP")
    print("Q = Quit\n")

    choice = input("Choose the label to collect: ").strip()

    labels = {"1": "HELP", "2": "NOT_HELP"}
    if choice not in labels:
        print("Invalid choice. Run the program again and enter 1 or 2.")
        return

    label = labels[choice]
    output_file = DATA_DIR / f"{label.lower()}_landmarks.csv"

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    header = [
        item
        for i in range(21)
        for item in (f"x{i}", f"y{i}", f"z{i}")
    ] + ["label"]

    file_exists = output_file.exists() and output_file.stat().st_size > 0
    previous_timestamp = -1
    saved_count = 0
    last_save_time = 0.0
    collecting = False

    try:
        with output_file.open(
            "a", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow(header)

            with mp.tasks.vision.HandLandmarker.create_from_options(
                options
            ) as detector:
                print(f"\nSelected label: {label}")
                print("Press S to start/stop collecting.")
                print("Press Q to quit.")
                print("HELP: palm toward camera, fingers extended.")
                print("NOT_HELP: use varied other hand poses and movements.")

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
                        cv2.getTickCount()
                        * 1000
                        / cv2.getTickFrequency()
                    )
                    timestamp = max(timestamp, previous_timestamp + 1)
                    previous_timestamp = timestamp

                    result = detector.detect_for_video(image, timestamp)

                    if result.hand_landmarks:
                        landmarks = result.hand_landmarks[0]

                        for point in landmarks:
                            cv2.circle(
                                frame,
                                (int(point.x * width), int(point.y * height)),
                                4,
                                (0, 255, 0),
                                -1,
                            )

                        for connection in (
                            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                        ):
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

                        now = time.monotonic()
                        if collecting and now - last_save_time >= 0.15:
                            row = []
                            for point in landmarks:
                                row.extend([point.x, point.y, point.z])
                            row.append(label)
                            writer.writerow(row)
                            csv_file.flush()
                            saved_count += 1
                            last_save_time = now

                    status = "COLLECTING" if collecting else "PAUSED"
                    color = (0, 0, 255) if collecting else (0, 200, 255)

                    cv2.putText(
                        frame,
                        f"Label: {label} | {status}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Saved this session: {saved_count}",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "S: start/pause | Q: quit",
                        (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("SignBridge - Data Collection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        collecting = not collecting
                        last_save_time = 0.0
                    elif key == ord("q"):
                        break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nSaved {saved_count} samples for {label}.")
    print(f"CSV location: {output_file}")


if __name__ == "__main__":
    main()
