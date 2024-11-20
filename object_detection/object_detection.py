!pip install -U -q ultralytics
!pip install -U -q gradio
!pip install opencv-python-headless
!pip install numpy
!pip install ultralytics
!pip install gradio


from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr

# Specify the path to the input video
input_path = "/content/sport-001.mp4"

def process_video(video_path):
    # Load the YOLO model for object detection and tracking
    model = YOLO("yolov8n.pt")

    # Open the video file
    video = cv2.VideoCapture(video_path)
    tracks = defaultdict(lambda: [])

    # Get video properties: FPS, width, and height
    video_fps = int(video.get(cv2.CAP_PROP_FPS))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and output file path for the processed video
    output_file = "tracked_video.mp4"
    video_writer = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, video_height)
    )

    # Loop through the frames of the video
    while video.isOpened():
        success, frame = video.read()

        if success:
            # Perform object tracking using YOLO
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()  # Extract bounding box coordinates
            object_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )
            # Annotate the frame with tracking data
            annotated_frame = results[0].plot()

            # Draw tracking lines for each detected object
            if object_ids:
                for box, obj_id in zip(boxes, object_ids):
                    x, y, w, h = box
                    track = tracks[obj_id]
                    track.append((float(x), float(y)))  # Append the center point of the box
                    if len(track) > 30:  # Retain only the last 30 frames for smoother lines
                        track.pop(0)

                    # Draw the tracking lines on the annotated frame
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=2,
                    )

            # Write the annotated frame to the output video
            video_writer.write(annotated_frame)
        else:
            break

    # Release resources: video reader and writer
    video.release()
    video_writer.release()

    return output_file

# Process the video and save the output
output_path = process_video(input_path)
print(f"Processed video saved to: {output_path}")

# Function for Gradio interface prediction
def predict(input_video):
    return process_video(input_video)

# Create a Gradio interface for the video processing pipeline
gr.Interface(
    fn=predict,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title="Object Detection"
).launch()
