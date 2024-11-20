****************************   Object Detection  *********************************************************

Overview
      This project demonstrates how to perform object detection and tracking on a video using the YOLOv8 model, OpenCV, and Gradio for a user-friendly interface. The processed video will have bounding boxes around detected objects and tracking lines for each object, showing their movement across frames.

Features
    YOLOv8 Model Integration: Utilizes the YOLOv8 model for object detection and tracking.
    Video Annotation: Adds bounding boxes and tracking lines to detected objects.
    Gradio Interface: Provides a web-based interface for video upload and processing.
    Output Video: Saves the processed video with annotations.
    Requirements
    Python Libraries
    Install the necessary Python packages using the following commands:

bash
  pip install -U ultralytics gradio opencv-python-headless numpy


Code Explanation
    Key Components
       1. YOLOv8 Model:
          YOLO("yolov8n.pt"): Loads the YOLOv8 model. Replace "yolov8n.pt" with other weights for different versions if needed.
       
       2.Video Processing:
        Reads the input video frame by frame.
        Performs object detection using YOLO.
        Tracks objects across frames by maintaining their center coordinates in a dictionary.
        
       3.Tracking Lines:
        Tracks the movement of each object and draws lines on the video to visualize object paths.

      4.Video Output:
        The processed video is saved as tracked_video.mp4 in the working directory.

      5.Gradio Interface:
        Allows users to upload a video file through a web-based interface and processes it.
        Displays the processed video as output.

Functions
1.process_video(video_path):
  Inputs: Path to the input video.
  Outputs: Path to the processed video with object tracking.

2.predict(input_video):
  Acts as a wrapper for process_video() for integration with the Gradio interface.
 
Usage
  Running the Script
    Local Execution:
        Save the script as object_tracking.py and execute it.
        Ensure the input video is available at the specified path (input_path).
        The processed video will be saved as tracked_video.mp4.
    Gradio Interface:
        Launch the Gradio app by running the script.
        Upload a video file through the interface.
        Download the processed video after it is displayed.
        Launch Gradio Interface
    Run the script, and the interface will launch locally. You will see:

        A file uploader for the input video.
        A player for the output video.
        
Sample Input
    Provide a video file such as sport-001.mp4.

Sample Output
    The output video (tracked_video.mp4) will include:

Dependencies
    Python 3.7+
    Required libraries:
    ultralytics
    opencv-python-headless
    numpy
    gradio
