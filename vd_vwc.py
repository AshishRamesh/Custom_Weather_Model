import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5m', 'yolov5l', 'yolov5x' for larger models

# Define the function to process each frame and detect vehicles
def detect_vehicles_from_video(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame)

        # Render the results on the frame
        results.render()  # Updates results.imgs with boxes and labels

        # Convert the frame to the correct format for cv2.VideoWriter
        processed_frame = results.ims[0]
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Write the frame into the output video file
        out.write(processed_frame)

        # Display the frame (optional)
        cv2.imshow('Vehicle Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Specify the input and output video paths
input_video_path = r'test_vids\test_1.mp4'
output_video_path = 'output_video.mp4'

# Call the function to process the video
detect_vehicles_from_video(input_video_path, output_video_path)