
# SignBridge

**Real-Time HELP Gesture Detection Using Computer Vision**

SignBridge is an assistive-technology project that aims to recognize two predefined hand-gesture classes — `HELP` and `NOT_HELP` — from a camera feed.

The project explores hand-landmark extraction, machine-learning classification, robust evaluation, and real-time inference in a lightweight application.

## Objectives

- Detect and track hand landmarks from video.
- Train and evaluate a two-class gesture classifier.
- Measure generalization to previously unseen participants.
- Handle uncertain predictions instead of forcing every input into a class.
- Build a reproducible, testable and accessible application.

## Technology

- Python
- OpenCV
- MediaPipe
- scikit-learn
- Google Colab for cloud-based experiments
- Git and GitHub

Additional dependencies will be selected as the architecture develops.

## Repository Structure

- `src/` — application and inference code
- `training/` — model training and evaluation
- `notebooks/` — Google Colab notebooks
- `tests/` — automated tests
- `docs/` — architecture and technical documentation
- `models/` — instructions for obtaining model artifacts
- `data/` — dataset instructions

## Project Status

**Status: Initial development**

Model performance, latency and generalization results will be published after evaluation. No accuracy claims are made before testing.

## Responsible Use

SignBridge is an experimental prototype, not a certified emergency-response or medical device. Predictions may be incorrect. It should not be relied on as the sole means of requesting emergency assistance.

## Development

Development is performed locally in VS Code, while model training and computationally intensive experiments are intended to run in Google Colab.

Setup instructions will be added as the implementation becomes available.
