import threading
import cv2  # OpenCV for camera capture and image display
import numpy as np  # NumPy for array handling and image conversion
import mediapipe as mp  # MediaPipe core library
from mediapipe.tasks import python  # MediaPipe Python task utilities
from mediapipe.tasks.python import vision  # MediaPipe vision tasks

import queue

from dummy_cmds import *  # Import the dummy test function from the dummy_cmds module

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

# Track each hand separately so left and right hands can be evaluated independently.
left_hand_state = {"gesture": "Unknown", "active": False}
right_hand_state = {"gesture": "Unknown", "active": False}
left_border_state = {"active": False, "center": None, "radius": 0, "crossed": False}
right_border_state = {"active": False, "center": None, "radius": 0, "crossed": False}

border_active = False
border_center = None
border_radius = 0
border_crossed = False

gesture_thread = None
gesture_thread_lock = threading.Lock()
gesture_queue = queue.Queue()
current_gesture = None

def check_gesture_queue():
    '''
    Continuously monitor the gesture queue and print new gestures when they change.
    This function blocks on `gesture_queue.get()` so it runs as a background thread.
    '''

    global gesture_queue, current_gesture

    while True:
        try:
            # Block until a gesture is available
            last_gesture = gesture_queue.get()

            # Allow a None sentinel to stop the thread if ever used
            if last_gesture is None:
                break

            # Print when the gesture has changed (or first seen)
            if current_gesture is None or current_gesture != last_gesture:
                print(f'detected gesture: {last_gesture}')
                current_gesture = last_gesture
            # otherwise ignore duplicate consecutive gestures

        except Exception as exc:
            # Keep the thread alive on unexpected errors
            print('gesture monitor error:', exc)
            continue
    
def start_monitoring_gesture_queue():
    '''
    Start a background daemon thread to monitor the gesture queue. If the monitor
    is already running, do nothing.
    '''

    global gesture_thread

    with gesture_thread_lock:
        if gesture_thread is not None and getattr(gesture_thread, 'is_alive', lambda: False)():
            return
        gesture_thread = threading.Thread(target=check_gesture_queue, daemon=True)
        gesture_thread.start()

    print('gesture monitor started')

def clear_border(border_state=None):
    if border_state is None:
        global border_active, border_center, border_radius, border_crossed
        border_active = False
        border_center = None
        border_radius = 0
        border_crossed = False
        return

    border_state["active"] = False
    border_state["center"] = None
    border_state["radius"] = 0
    border_state["crossed"] = False

def draw_border(image, top_landmark, bottom_landmark, border_state=None):
    '''
    draw a circle around the epicenter of the hand gesture
    the circle will be proportional to the size of the hand
    '''

    global border_center, border_radius, border_crossed

    if border_state is None:
        border_state = {"active": False, "center": None, "radius": 0, "crossed": False}

    if image is None or top_landmark is None or bottom_landmark is None:
        return image

    if hasattr(top_landmark, 'x') and hasattr(top_landmark, 'y'):
        top_x, top_y = top_landmark.x, top_landmark.y
    else:
        top_x, top_y = top_landmark

    if hasattr(bottom_landmark, 'x') and hasattr(bottom_landmark, 'y'):
        bottom_x, bottom_y = bottom_landmark.x, bottom_landmark.y
    else:
        bottom_x, bottom_y = bottom_landmark

    # Convert normalized MediaPipe landmark coordinates to pixel coordinates.
    dx_px = (bottom_x - top_x) * image.shape[1]
    dy_px = (bottom_y - top_y) * image.shape[0]
    distance_px = np.hypot(dx_px, dy_px)

    # Midpoint between the wrist and index MCP points serves as the circle center.
    center_x = ((top_x + bottom_x) / 2.0) * image.shape[1]
    center_y = ((top_y + bottom_y) / 2.0) * image.shape[0]

    # Radius scales with the hand size and is kept at a sensible minimum.
    radius = max(25, distance_px * 1.5)
    border_state["center"] = (int(center_x), int(center_y))
    border_state["radius"] = int(radius)
    border_state["active"] = True
    border_state["crossed"] = False

    border_center = border_state["center"]
    border_radius = border_state["radius"]
    border_crossed = border_state["crossed"]

    border_color = (0, 0, 255) if border_state["crossed"] else (0, 255, 0)
    cv2.circle(image, border_state["center"], border_state["radius"], border_color, 2)

    return image

def monitor_border(hand_landmarks, image_shape=None, border_state=None):
    '''Compare hand landmark positions against the active border circle.
    Return True if any detected hand landmark is outside the border.
    '''
    global border_center, border_radius, border_crossed, border_active

    if border_state is None:
        border_state = {"active": border_active, "center": border_center, "radius": border_radius, "crossed": border_crossed}

    if hand_landmarks is None or border_state["center"] is None or border_state["radius"] <= 0:
        return False

    center_x, center_y = border_state["center"]
    width = image_shape[1] if image_shape is not None else 640
    height = image_shape[0] if image_shape is not None else 480

    for landmark in hand_landmarks:
        if landmark is None:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)
        distance = np.hypot(x - center_x, y - center_y)

        if distance >= border_state["radius"]:
            border_state["crossed"] = True
            border_state["active"] = False
            border_crossed = True
            border_active = False
            return True

    border_state["crossed"] = False
    border_crossed = False
    return False

def monitor_hand(hand_landmarks, image_shape=None, border_state=None):
    '''Wrapper used by the visualizer to decide whether the hand crossed the border.'''
    if monitor_border(hand_landmarks, image_shape, border_state):
        if border_state is not None:
            border_state["crossed"] = True
            border_state["active"] = False
        print("Hand crossed the border")
        return True

    if border_state is not None:
        border_state["crossed"] = False
    return False

def visualize(image, detection_result) -> np.ndarray:
    """Draw gesture, handedness, and hand landmarks on a BGR image."""

    global border_active, gesture_queue

    image = image.copy()  # Make a copy so the original frame is not modified.

    # Check whether the detection result contains hand landmark data before drawing anything.
    if not hasattr(detection_result, "hand_landmarks"):
        return image  # Return the unchanged image if no landmarks are available.

    # Define the list of landmark pairs that should be connected to form the hand skeleton.
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb connections.
        (5, 6), (6, 7), (7, 8),  # Index finger connections.
        (9, 10), (10, 11), (11, 12),  # Middle finger connections.
        (13, 14), (14, 15), (15, 16),  # Ring finger connections.
        (17, 18), (18, 19), (19, 20),  # Pinky finger connections.
        (0, 5), (5, 9), (9, 13), (13, 17),  # Connect fingers to the palm.
        (0, 17),  # Connect the thumb base to the pinky base.
    ]

    # Loop through each detected hand in the result.
    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
        if not hand_landmarks:  # Skip this hand if no landmarks were returned.
            continue  # Move to the next detected hand.

        # Infer handedness for this specific hand.
        handedness_label = "Unknown"
        if hand_index < len(detection_result.handedness):
            handedness_candidates = detection_result.handedness[hand_index]
            if handedness_candidates:
                handedness = handedness_candidates[0]
                handedness_label = f"{handedness.category_name} ({handedness.score:.2f})"

        # Select the matching state bucket for this hand.
        if handedness_label.lower().startswith("left"):
            hand_state = left_hand_state
            border_state = left_border_state
        elif handedness_label.lower().startswith("right"):
            hand_state = right_hand_state
            border_state = right_border_state
        else:
            hand_state = {"gesture": "Unknown", "active": False}
            border_state = None

        # Draw the hand skeleton by connecting each pair of landmarks with a line.
        for start_idx, end_idx in connections:
            start = hand_landmarks[start_idx]  # Get the first landmark in the pair.
            end = hand_landmarks[end_idx]  # Get the second landmark in the pair.
            x1 = int(start.x * image.shape[1])  # Convert the landmark's x coordinate to pixel space.
            y1 = int(start.y * image.shape[0])  # Convert the landmark's y coordinate to pixel space.
            x2 = int(end.x * image.shape[1])  # Convert the second landmark's x coordinate to pixel space.
            y2 = int(end.y * image.shape[0])  # Convert the second landmark's y coordinate to pixel space.
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green line between the two points.

        # Draw each landmark as a filled circle on the image.
        for landmark in hand_landmarks:
            x = int(landmark.x * image.shape[1])  # Convert the landmark x coordinate to pixel space.
            y = int(landmark.y * image.shape[0])  # Convert the landmark y coordinate to pixel space.
            cv2.circle(image, (x, y), 5, TEXT_COLOR, -1)  # Draw a red filled circle at the landmark.

        if border_state is not None and border_state["active"]:
            crossed = monitor_hand(hand_landmarks, image.shape, border_state)
            if crossed:
                cv2.circle(image, border_state["center"], border_state["radius"], (0, 0, 255), 2)
            elif border_state["center"] is not None:
                cv2.circle(image, border_state["center"], border_state["radius"], (0, 255, 0), 2)
        
        # Set a default gesture label in case no gesture is detected.
        gesture_label = "Unknown"
        if hand_index < len(detection_result.gestures):  # Check whether gesture data exists for this hand.
            gesture_candidates = detection_result.gestures[hand_index]  # Get the gesture candidates for this hand.

            if gesture_candidates:  # Use the top gesture if any candidates are available.
                gesture_label = gesture_candidates[0].category_name  # Store the highest-confidence gesture name.

            hand_state["gesture"] = gesture_label
            hand_state["active"] = True

            gesture_queue.put(gesture_label)

            '''
            if gesture_label == "Thumb_Up":
                dummy_thumbs_up_func()  # Call the dummy test function when a "Thumbs Up" gesture is detected.

            elif gesture_label == "Thumb_Down":
                dummy_thumbs_down_func()  # Call the dummy thumbs down function when a "Thumbs Down" gesture is detected.

            elif gesture_label == 'Open_Palm':
                # Create the border once when open palm is first detected, then keep it active.
                if border_state is not None and not border_state["active"]:
                    image = draw_border(image, hand_landmarks[5], hand_landmarks[0], border_state)
            '''
        # Calculate the vertical position for the text labels based on the hand index.
        text_y = 30 + hand_index * 40
        cv2.putText(  # Draw the gesture label near the top-left corner of the image.
            image,
            f"Gesture: {gesture_label}",
            (10, text_y), 
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )
        cv2.putText(  # Draw the handedness label beneath the gesture label.
            image,
            f"Handedness: {handedness_label}",
            (10, text_y + 20),
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    # Return the annotated image after all overlays have been drawn.
    return image

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
    model_asset_path='C:/Users/black/Coding/testing/handsfree/mediapipe_hand-tflite-float/gesture_recognizer.task'
)

# Configure the detector options for live-stream processing
options = vision.GestureRecognizerOptions(
    base_options=base_options,  # Use the model configuration created above
    running_mode=vision.RunningMode.LIVE_STREAM,  # Enable async live-stream mode
    result_callback=handle_result,  # Tell MediaPipe to call this function when results arrive
)

start_monitoring_gesture_queue()

# Create the detector object from the configured options
with vision.GestureRecognizer.create_from_options(options) as detector:
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

        # If no request is currently pending, send the frame to the detector for async processing
        with state_lock:
            if not frame_pending:
                pending_frame_bgr = frame.copy()
                pending_timestamp = frame_timestamp
                frame_pending = True
                detector.recognize_async(mp_image, frame_timestamp)

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