import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
model.trainable = False

PROTOTXT_PATH = "your path"
MODEL_PATH = 'your path'

if not os.path.exists(PROTOTXT_PATH):
    raise FileNotFoundError(f"Prototxt file not found at {PROTOTXT_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Caffe model not found at {MODEL_PATH}")

def load_model():
    model = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    return model

def detect_humans(frame, net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence * 100:.2f}%"
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    net = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_humans(frame, net)
        cv2.imshow("Human Body Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
