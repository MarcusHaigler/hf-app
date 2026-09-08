import threading
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
frame_pending = False  # Whether a request is already in flight for async detection
pending_frame_bgr = None  # The frame that was sent to the detector for the pending request
pending_timestamp = None  # Timestamp associated with the pending request
state_lock = threading.Lock()  # Protect shared callback state


def visualize(image, detection_result) -> np.ndarray:
    """Draw hand landmarks and handedness labels on a BGR image."""
    image = image.copy()  # Copy the frame so we do not modify the original array

    if not hasattr(detection_result, "hand_landmarks"):
        return image

    for hand_landmarks in detection_result.hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, TEXT_COLOR, -1)

            if idx in (4, 8, 12, 16, 20):
                cv2.putText(
                    image,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE,
                    TEXT_COLOR,
                    FONT_THICKNESS,
                )

    for hand_index, handedness_list in enumerate(detection_result.handedness):
        if handedness_list:
            category = handedness_list[0]
            label = f"{category.category_name} ({round(category.score, 2)})"
            cv2.putText(
                image,
                label,
                (10, 30 + hand_index * 20),
                cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE,
                TEXT_COLOR,
                FONT_THICKNESS,
            )

    return image  # Return the annotated frame

def handle_result(result, output_image, timestamp):
    """Store the latest detection result and the image that produced it."""
    global latest_result, latest_frame_bgr, frame_pending, pending_frame_bgr, pending_timestamp

    with state_lock:
        if timestamp != pending_timestamp:
            # Ignore any stale callback that does not match the current pending request.
            return

        latest_result = result
        latest_frame_bgr = pending_frame_bgr
        frame_pending = False
        pending_frame_bgr = None
        pending_timestamp = None

# Create the base options for the MediaPipe detector using the model file
base_options = python.BaseOptions(
    model_asset_path='C:/Users/black/Coding/testing/handsfree/hand_landmarker.task'
)

# Configure the detector options for live-stream processing
options = vision.HandLandmarkerOptions(
    base_options=base_options,  # Use the model configuration created above
    running_mode=vision.RunningMode.LIVE_STREAM,  # Enable async live-stream mode
    result_callback=handle_result,  # Tell MediaPipe to call this function when results arrive
)

# Create the detector object from the configured options
with vision.HandLandmarker.create_from_options(options) as detector:
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

        frame_timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)  # Create a timestamp in milliseconds
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # Wrap the frame as a MediaPipe image

        with state_lock:
            if not frame_pending:
                pending_frame_bgr = frame.copy()
                pending_timestamp = frame_timestamp
                frame_pending = True
                detector.detect_async(mp_image, frame_timestamp)

            current_result = latest_result
            current_frame_bgr = latest_frame_bgr

        if current_result is not None and current_frame_bgr is not None:  # If a callback result is ready, show it
            annotated_frame = visualize(current_frame_bgr, current_result)  # Draw boxes on the latest callback frame
            cv2.imshow("Hand Detection", annotated_frame)  # Display the annotated frame
        else:  # If no result has arrived yet, just show the raw camera frame
            cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) == ord("q"):  # Wait for a key press and quit if the user presses q
            break

    cap.release()  # Release the webcam when the loop ends
    cv2.destroyAllWindows()  # Close all OpenCV windows