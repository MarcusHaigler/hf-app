# Import the camera, numerical, MediaPipe, and threading tools used by the
# asynchronous detection pipeline and the OpenCV display loop.
import threading
import cv2  # OpenCV for camera capture and image display
import numpy as np  # NumPy for array handling and image conversion
import mediapipe as mp  # MediaPipe core library
from mediapipe.tasks import python  # MediaPipe Python task utilities
from mediapipe.tasks.python import vision  # MediaPipe vision tasks

import queue

from dummy_cmds import *  # Import optional command functions used by gestures.

# These constants control the appearance of gesture labels and landmark dots.
MARGIN = 10  # Space between the bounding box and the text label
ROW_SIZE = 10  # Text line height for the overlay
FONT_SIZE = 1  # Font scale for the label text
FONT_THICKNESS = 1  # Line thickness for the text
TEXT_COLOR = (255, 0, 0)  # Red color for boxes and labels

# MediaPipe returns results asynchronously. These variables let the camera loop
# safely pair the newest result with the frame that produced it.
latest_result = None  # Holds the most recent detection output
latest_frame_bgr = None  # Holds the most recent frame in BGR format for drawing
frame_pending = False  # Whether a request is already in flight for async detection
pending_frame_bgr = None  # The frame that was sent to the detector for the pending request
pending_timestamp = None  # Timestamp associated with the pending request
state_lock = threading.Lock()  # Protect shared callback state

# Each hand owns its gesture and border state so two hands do not overwrite one
# another while the visualizer processes a detection result.
left_hand_state = {"gesture": "Unknown", "active": False}
right_hand_state = {"gesture": "Unknown", "active": False}
left_border_state = {"active": False, "center": None, "radius": 0, "crossed": False, "crossing_point": None, "crossing_direction": None}
right_border_state = {"active": False, "center": None, "radius": 0, "crossed": False, "crossing_point": None, "crossing_direction": None}

# The gesture-monitoring thread sets this flag; the visualizer consumes it on a
# later frame, where it has the landmarks and image needed to draw the border.
activate_border_flag = False
border_active = False
border_center = None
border_radius = 0
border_crossed = False

# Gesture labels travel from visualize() to this queue, then are consumed by a
# daemon thread that recognizes ordered gesture sequences.
gesture_thread = None
gesture_thread_lock = threading.Lock()
gesture_queue = queue.Queue()
last_gesture = None

def activate_border():
    '''
    set the border state to active so the main thread will automatically begin the border
    '''
    # This function is the action stored in neutral_state_mapping. It only
    # requests activation; visualize() performs the actual geometry calculation.
    global activate_border_flag, state_lock

    with state_lock:
        try:

            if not activate_border_flag:
                activate_border_flag = True

        except Exception as e:
            print(f'error: {e}')
            return

# The newest gesture is stored at index 0. Older gestures shift toward index 2,
# allowing the mapping below to match a three-gesture command sequence.
a = None
b = None 
c = None
active_gesture_chain = [a, b, c]

# Map a recognized gesture sequence to the action it should trigger.
neutral_state_mapping = {
    ("Open_Palm", "Closed_Fist", "Open_Palm"): activate_border,
}

def update_gesture_chain(new_val):
    '''
    Shift the existing gestures and place the newest gesture first.
    '''
    # The queue thread calls this after a new label arrives. Once the complete
    # chain matches a key, the associated callable is invoked immediately.
    global a, b, c, active_gesture_chain

    c = b
    b = a
    a = new_val
    active_gesture_chain[:] = [a, b, c]

    print(f'active gesture chain updated: {active_gesture_chain}')

    action = neutral_state_mapping.get(tuple(active_gesture_chain))
    if action is not None:
        
        action()
        print('action triggered')

def check_gesture_queue():
    '''
    Continuously monitor the gesture queue and print new gestures when they change.
    This function blocks on `gesture_queue.get()` so it runs as a background thread.
    '''

    # This is the consumer side of the queue. It filters repeated labels so a
    # held gesture advances the command sequence only once.
    global gesture_queue, last_gesture, active_gesture_chain

    while True:
        try:
            # Block until a gesture is available
            current_gesture = gesture_queue.get()
            if current_gesture == 'None':
                continue
            else:

                # Allow a None sentinel to stop the thread if ever used
                if current_gesture is None:
                    break

                # Print valid gestures
                if last_gesture is None or last_gesture != current_gesture:
                    print(f'detected gesture: {current_gesture}')
                    update_gesture_chain(current_gesture)
                        
                    last_gesture = current_gesture

        except Exception as exc:
            # Keep the thread alive on unexpected errors
            print('gesture monitor error:', exc)
            continue
    
def start_monitoring_gesture_queue():
    '''
    Start a background daemon thread to monitor the gesture queue. If the monitor
    is already running, do nothing.
    '''

    # Start one long-lived daemon consumer. The lock prevents duplicate monitor
    # threads when startup is called more than once.
    global gesture_thread

    with gesture_thread_lock:
        if gesture_thread is not None and getattr(gesture_thread, 'is_alive', lambda: False)():
            return
        gesture_thread = threading.Thread(target=check_gesture_queue, daemon=True)
        gesture_thread.start()

    print('gesture monitor started')

def clear_border(border_state=None):
    # Reset either the shared fallback border or the state belonging to one hand.
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
    border_state["crossing_point"] = None
    border_state["crossing_direction"] = None

def draw_border(image, top_landmark, bottom_landmark, border_state=None):
    '''
    draw a circle around the epicenter of the hand gesture
    the circle will be proportional to the size of the hand
    '''

    # Convert two normalized landmarks into a pixel-space circle, store that
    # circle in the hand state, and render it on the current frame.
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
    border_state["crossing_point"] = None
    border_state["crossing_direction"] = None

    border_center = border_state["center"]
    border_radius = border_state["radius"]
    border_crossed = border_state["crossed"]

    border_color = (0, 0, 255) if border_state["crossed"] else (0, 255, 0)
    cv2.circle(image, border_state["center"], border_state["radius"], border_color, 2)

    return image

def monitor_border(hand_landmarks, image_shape=None, border_state=None):
    '''
    Compare hand landmark positions against the active border circle.
    Return (True, direction) if a landmark is outside the border, otherwise
    return (False, None). Direction is relative to the circle epicenter.
    '''
    # Test every landmark against the stored circle. A single point outside the
    # radius ends the active border and reports a crossing to the caller.
    global border_center, border_radius, border_crossed, border_active

    if border_state is None:
        border_state = {"active": border_active, "center": border_center, "radius": border_radius, "crossed": border_crossed, "crossing_point": None, "crossing_direction": None}

    if hand_landmarks is None or border_state["center"] is None or border_state["radius"] <= 0:
        return False, None

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
            # Project the outside landmark back onto the circle. This gives the
            # exact pixel where the center-to-landmark path meets the border.
            if distance > 0:
                scale = border_state["radius"] / distance
                crossing_point = (
                    int(center_x + (x - center_x) * scale),
                    int(center_y + (y - center_y) * scale),
                )
            else:
                crossing_point = (center_x, center_y)

            # Image coordinates grow downward, so invert the vertical delta
            # before converting the vector angle into a compass direction.
            delta_x = crossing_point[0] - center_x
            delta_y = center_y - crossing_point[1]
            angle = (np.degrees(np.arctan2(delta_y, delta_x)) + 360) % 360
            direction_names = (
                "right", "up-right", "up", "up-left",
                "left", "down-left", "down", "down-right",
            )
            direction = direction_names[int((angle + 22.5) // 45) % 8]

            border_state["crossing_point"] = crossing_point
            border_state["crossing_direction"] = direction
            border_state["crossed"] = True
            border_state["active"] = False
            border_crossed = True
            border_active = False
            print(f"border crossed at {crossing_point} ({direction})")
            return True, direction

    border_state["crossed"] = False
    border_state["crossing_point"] = None
    border_state["crossing_direction"] = None
    border_crossed = False
    return False, None

def monitor_hand(hand_landmarks, image_shape=None, border_state=None):
    '''
    Wrapper used by the visualizer to decide whether the hand crossed the border
    '''

    # Keep border detection behind one small wrapper so visualization receives a
    # simple boolean and can decide how to draw the crossed/active state.
    global activate_border_flag

    crossed, direction = monitor_border(hand_landmarks, image_shape, border_state)
    if crossed:
        if border_state is not None:
            border_state["crossed"] = True
            border_state["active"] = False
        print(f"Hand crossed the border ({direction})")
        activate_border_flag = False
        return True, direction

    if border_state is not None:
        border_state["crossed"] = False
    return False, None

def visualize(image, detection_result) -> np.ndarray:
    """Draw gesture, handedness, and hand landmarks on a BGR image."""

    # This is the rendering and producer stage: it draws the completed result,
    # evaluates border movement, and publishes gesture labels to the queue.
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

        if activate_border_flag:
            # Create the border once when open palm is first detected, then keep it active.
            if border_state is not None and not border_state["active"]:
                image = draw_border(image, hand_landmarks[5], hand_landmarks[0], border_state)

        if border_state is not None and border_state["active"]:
            crossed, crossing_direction = monitor_hand(hand_landmarks, image.shape, border_state) # check if the hand has crossed
            if crossed:
                cv2.circle(image, border_state["center"], border_state["radius"], (0, 0, 255), 2) 
                if border_state["crossing_point"] is not None:
                    cv2.circle(image, border_state["crossing_point"], 7, (255, 0, 255), -1)
                    cv2.putText(image, crossing_direction, border_state["crossing_point"], cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (255, 0, 255), FONT_THICKNESS)
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
    # MediaPipe invokes this callback on its own thread. Only accept the result
    # belonging to the currently pending frame, then make it visible to the loop.
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

# Configure the MediaPipe recognizer and tell it to deliver live results through
# handle_result instead of blocking the camera loop.
base_options = python.BaseOptions(
    model_asset_path='C:/Users/black/Coding/testing/handsfree/mediapipe_hand-tflite-float/gesture_recognizer.task'
)

# Configure the detector options for live-stream processing
options = vision.GestureRecognizerOptions(
    base_options=base_options,  # Use the model configuration created above
    running_mode=vision.RunningMode.LIVE_STREAM,  # Enable async live-stream mode
    result_callback=handle_result,  # Tell MediaPipe to call this function when results arrive
)

# Start consuming gesture labels before frames begin producing them.
start_monitoring_gesture_queue()

# The context manager closes MediaPipe resources when the camera loop exits.
with vision.GestureRecognizer.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():  # Check whether the camera opened successfully
        raise RuntimeError("Cannot open camera")

    # Main producer/display loop: capture a frame, submit at most one pending
    # request, and display the newest completed detection without blocking.
    while True:
        ret, frame = cap.read()  # Read one frame from the camera
        if not ret:  # Stop if no frame could be read
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame horizontally for a more natural view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB for MediaPipe

        frame_timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)  # Create a timestamp in milliseconds
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # Wrap the frame as a MediaPipe image

        # Submit only one frame at a time. The callback clears frame_pending when
        # MediaPipe finishes, preventing requests from growing faster than results.
        with state_lock:
            if not frame_pending:
                pending_frame_bgr = frame.copy()
                pending_timestamp = frame_timestamp
                frame_pending = True
                detector.recognize_async(mp_image, frame_timestamp)

            current_result = latest_result
            current_frame_bgr = latest_frame_bgr

        # Display the annotated callback frame when available; otherwise keep the
        # camera feed responsive while the first asynchronous result is pending.
        if current_result is not None and current_frame_bgr is not None:
            annotated_frame = visualize(current_frame_bgr, current_result)  # Draw boxes on the latest callback frame
            cv2.imshow("Hand Detection", annotated_frame)  # Display the annotated frame
        else:  # If no result has arrived yet, just show the raw camera frame
            cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) == ord("q"):  # Wait for a key press and quit if the user presses q
            break

    cap.release()  # Release the webcam when the loop ends
    cv2.destroyAllWindows()  # Close all OpenCV windows