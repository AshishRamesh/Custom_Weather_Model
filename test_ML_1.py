import cv2
import numpy as np
import tensorflow as tf
import os
import threading
import queue

# Load your pre-trained TensorFlow Keras model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Load YOLO model
yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Function to preprocess the frames before feeding them into the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Adjust size according to your model's input size
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

# Function to draw predictions on the frame
def draw_predictions(frame, predictions, yolo_detections):
    for i, (label, score) in enumerate(predictions):
        text = f'{label}: {score:.2f}'
        cv2.putText(frame, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Draw YOLO detections
    for detection in yolo_detections:
        x, y, w, h = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Function to perform YOLO detection
def yolo_detect(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    yolo_detections = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            yolo_detections.append({'bbox': (x, y, w, h), 'label': label, 'confidence': confidence})
    
    return yolo_detections

# Check if the input is a video file or an image file
def is_video_file(filepath):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # Add other video formats if necessary
    _, ext = os.path.splitext(filepath)
    return ext.lower() in video_extensions

# Process image file
def process_image_file(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image.")
        return
    
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    predictions = [(f'Label_{i}', float(pred)) for i, pred in enumerate(predictions[0])]
    
    yolo_detections = yolo_detect(frame)
    frame_with_predictions = draw_predictions(frame, predictions, yolo_detections)
    cv2.imshow('Image with Predictions', frame_with_predictions)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

# Process video file
def process_video_file(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    def process_frame(queue):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            preprocessed_frame = preprocess_frame(frame)
            predictions = model.predict(preprocessed_frame)
            predictions = [(f'Label_{i}', float(pred)) for i, pred in enumerate(predictions[0])]
            
            yolo_detections = yolo_detect(frame)
            frame_with_predictions = draw_predictions(frame, predictions, yolo_detections)
            queue.put(frame_with_predictions)
    
    frame_queue = queue.Queue(maxsize=10)
    processing_thread = threading.Thread(target=process_frame, args=(frame_queue,))
    processing_thread.start()

    while cap.isOpened():
        if not frame_queue.empty():
            frame_with_predictions = frame_queue.get()
            cv2.imshow('Video with Predictions', frame_with_predictions)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    processing_thread.join()
    cap.release()
    cv2.destroyAllWindows()

# Main function to handle input path
def main():
    input_path = input("Enter the path to the video or image file: ").strip()
    if is_video_file(input_path):
        process_video_file(input_path)
    else:
        process_image_file(input_path)

# Example usage
if __name__ == '__main__':
    main()
