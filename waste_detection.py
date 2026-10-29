import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

# Load AI models
print("Loading MobileNetV2...")
classifier = MobileNetV2(weights="imagenet")

print("Loading YOLO...")
detector = YOLO("yolov8n.pt")

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam not found.")
    raise SystemExit(1)

mode = "mobilenet"

print("\nWaste Detection System Started")
print("Press M = MobileNetV2 classification")
print("Press Y = YOLO object detection")
print("Press Q = Quit")

try:
    while camera.isOpened():
        ok, frame = camera.read()

        if not ok:
            print("Cannot read webcam.")
            break

        if mode == "yolo":
            # Detect objects in the webcam frame
            results = detector(frame, verbose=False)
            frame = results[0].plot()

            cv2.putText(
                frame,
                "YOLO Object Detection",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        else:
            # Prepare image for MobileNetV2
            image = cv2.resize(frame, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = preprocess_input(
                np.expand_dims(
                    image.astype(np.float32), axis=0
                )
            )

            # Predict object category
            predictions = classifier.predict(
                image, verbose=0
            )

            _, name, score = decode_predictions(
                predictions, top=1
            )[0][0]

            cv2.putText(
                frame,
                f"Object: {name}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {score:.1%}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.imshow(
            "Waste Detection | M=MobileNet Y=YOLO Q=Quit",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("m"):
            mode = "mobilenet"
        elif key == ord("y"):
            mode = "yolo"

finally:
    camera.release()
    cv2.destroyAllWindows()