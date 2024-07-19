
# YO-FLO: YOLO-Like Object Detection with Florence Models

Welcome to YO-FLO, a proof-of-concept implementation of YOLO-like object detection using the Florence-2-base-ft model. Inspired by the powerful YOLO (You Only Look Once) object detection framework, YO-FLO leverages the capabilities of the Florence foundational vision model to achieve real-time inference while maintaining a lightweight footprint.

## Table of Contents

- Introduction
- Features
- Installation
- Usage
- Error Handling
- Contributing
- License

## Introduction

YO-FLO explores whether the new Florence foundational vision model can be implemented in a YOLO-like format for object detection. Florence-2 is designed by Microsoft as a unified vision-language model capable of handling diverse tasks such as object detection, captioning, and segmentation. To achieve this, it uses a sequence-to-sequence framework where images and task-specific prompts are processed to generate the desired text outputs. The model's architecture combines a DaViT vision encoder with a transformer-based multi-modal encoder-decoder, making it versatile and efficient.

Florence-2 has been trained on the extensive FLD-5B dataset, containing 126 million images and over 5 billion annotations, ensuring high-quality performance across multiple tasks. Despite its relatively small size, Florence-2 demonstrates strong zero-shot and fine-tuning capabilities, making it an excellent choice for real-time applications.

## Features

- Real-Time Object Detection: Achieve YOLO-like performance using the Florence-2-base-ft model.
- Class-Specific Detection: Specify the class of objects you want to detect (e.g., 'cat', 'dog').
- Expression Comprehension: Detect objects based on specific phrases, answering yes or no questions for cool and exotic results.
- Beep and Screenshot on Detection: Toggle options to beep and take screenshots when the target class or phrase is detected.
- Tkinter GUI: A user-friendly graphical interface for easy interaction.
- Cross-Platform Compatibility: Works on Windows, macOS, and Linux.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Installing Dependencies

```bash
pip install torch transformers pillow opencv-python colorama simpleaudio huggingface-hub
```

## Usage

### Running YO-FLO

To start YO-FLO, run the following command:

```bash
python yo-flo.py
```

### Menu Options

1. Select Model Path: Choose a local directory containing the Florence model.
2. Download Model from HuggingFace: Download and initialize the Florence-2-base-ft model from HuggingFace.
3. Set Class Name: Specify the class name you want to detect (leave blank to show all detections).
4. Set Phrase: Enter the phrase for comprehension detection (e.g., 'Is the person smiling?', 'Is the cat laying down?').
5. Toggle Beep on Detection: Enable or disable the beep sound on detection.
6. Toggle Screenshot on Detection: Enable or disable taking screenshots on detection.
7. Start Webcam Detection: Begin real-time object detection using your webcam.
8. Stop Webcam Detection: Stop the webcam detection and return to the menu.
9. Toggle Debug Mode: Enable or disable debug mode for detailed logging.
10. Exit: Exit the application.

### Example Workflow

1. Select Model Path or Download Model from HuggingFace.
2. Set Class Name to specify what you want to detect (e.g., 'cat', 'dog').
3. Set Phrase for specific phrase-based detections.
4. Toggle Beep on Detection if you want an audible alert.
5. Toggle Screenshot on Detection if you want to save screenshots of detections.
6. Start Webcam Detection to begin detecting objects in real-time.

## Error Handling

YO-FLO includes robust error handling to ensure smooth operation:

- Model Initialization Errors: Handles cases where the model path is incorrect or the model fails to load.
- Webcam Access Errors: Notifies if the webcam cannot be accessed.
- Image Processing Errors: Catches errors during frame processing and provides detailed messages.
- File Not Found Errors: Alerts if required files (e.g., beep sound file) are missing.

### Example Error Messages

- Error loading model: [detailed error message]
- Error running object detection: [detailed error message]
- Error plotting bounding boxes: [detailed error message]
- Error toggling beep: [detailed error message]
- Error saving screenshot: [detailed error message]
- OpenCV error: [detailed error message]

## Contributing

We welcome contributions to improve YO-FLO. Please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Create a new Pull Request.

## License

YO-FLO is licensed under the MIT License.

---

Thank you for using YO-FLO! We are excited to see what amazing applications you will build with this tool. Happy detecting!
