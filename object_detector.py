import cv2  # OpenCV for camera capture and image display
import numpy as np  # NumPy for array handling and image conversion
import mediapipe as mp  # MediaPipe core library
from mediapipe.tasks import python  # MediaPipe Python task utilities
from mediapipe.tasks.python import vision  # MediaPipe vision tasks

# Visual styling constants for the detection overlay
MARGIN = 10  # Space between the bounding box and the text label
ROW_SIZE = 10  # Text line height for the overlay
FONT_SIZE = 1  # Font scale for the label text
FONT_THICKNESS = 1  # Line thickness for the text
TEXT_COLOR = (255, 0, 0)  # Red color for boxes and labels

# Storage for the latest detection result and the frame it came from
latest_result = None  # Holds the most recent detection output
latest_frame_bgr = None  # Holds the most recent frame in BGR format for drawing


def visualize(image, detection_result) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image."""
    image = image.copy()  # Copy the frame so we do not modify the original array

    for detection in detection_result.detections:  # Loop over each detected object
        bbox = detection.bounding_box  # Get the bounding box information
        start_point = (bbox.origin_x, bbox.origin_y)  # Top-left corner of the box
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)  # Bottom-right corner
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)  # Draw the rectangle

        category = detection.categories[0]  # Get the highest-confidence category
        category_name = category.category_name  # Human-readable label like "person"
        probability = round(category.score, 2)  # Round confidence to two decimals
        result_text = f"{category_name} ({probability})"  # Build the label text
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)  # Position for the text
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )  # Draw the label on the frame

    return image  # Return the annotated frame

def handle_result(result, output_image, timestamp):
    """Store the latest detection result and the image that produced it."""
    global latest_result, latest_frame_bgr  # Use the global variables declared above

    latest_result = result  # Save the detection result from the callback
    rgb_image = np.array(output_image.numpy_view())  # Convert the MediaPipe image to a NumPy array
    latest_frame_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV display

    print("Timestamp:", timestamp)  # Print the frame timestamp for debugging
    for detection in result.detections:  # Loop through detections and print them
        category = detection.categories[0]
        print(f"Label: {category.category_name}, Score: {round(category.score, 2)}")

# Create the base options for the MediaPipe detector using the model file
base_options = python.BaseOptions(
    model_asset_path="C:/Users/black/Coding/testing/handsfree/efficientdet_lite0.tflite"
)

# Configure the detector options for live-stream processing
options = vision.ObjectDetectorOptions(
    base_options=base_options,  # Use the model configuration created above
    running_mode=vision.RunningMode.LIVE_STREAM,  # Enable async live-stream mode
    max_results=5,  # Limit the number of detections shown
    result_callback=handle_result,  # Tell MediaPipe to call this function when results arrive
    score_threshold=0.5,  # Only show detections with confidence above 50%
)

# Create the detector object from the configured options
with vision.ObjectDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():  # Check whether the camera opened successfully
        raise RuntimeError("Cannot open camera")

    while True:  # Keep reading frames until the user quits
        ret, frame = cap.read()  # Read one frame from the camera
        if not ret:  # Stop if no frame could be read
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame horizontally for a more natural view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB for MediaPipe

        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)  # Create a timestamp in milliseconds
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # Wrap the frame as a MediaPipe image

        detector.detect_async(mp_image, timestamp)  # Send the frame to the detector asynchronously

        if latest_result is not None and latest_frame_bgr is not None:  # If a callback result is ready, show it
            annotated_frame = visualize(latest_frame_bgr, latest_result)  # Draw boxes on the latest callback frame
            cv2.imshow("Object Detection", annotated_frame)  # Display the annotated frame
        else:  # If no result has arrived yet, just show the raw camera frame
            cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) == ord("q"):  # Wait for a key press and quit if the user presses q
            break

    cap.release()  # Release the webcam when the loop ends
    cv2.destroyAllWindows()  # Close all OpenCV windows