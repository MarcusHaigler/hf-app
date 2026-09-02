import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import queue
from dummy_cmds import *

class HandGestureDetector:
    """Encapsulates the gesture recognizer, gesture queue, and border logic."""

    # These constants control the appearance of gesture labels and landmark dots.
    MARGIN = 10
    ROW_SIZE = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)

    def __init__(self, model_path='C:/Users/black/Coding/testing/handsfree/mediapipe_hand-tflite-float/gesture_recognizer.task'):
        # MediaPipe returns results asynchronously. These attributes let the camera loop
        # safely pair the newest result with the frame that produced it.
        self.latest_result = None
        self.latest_frame_bgr = None
        self.frame_pending = False
        self.pending_frame_bgr = None
        self.pending_timestamp = None
        self.state_lock = threading.Lock()

        # Each hand owns its gesture and border state so two hands do not overwrite one
        # another while the visualizer processes a detection result.
        self.left_hand_state = {"gesture": "Unknown", "active": False}
        self.right_hand_state = {"gesture": "Unknown", "active": False}
        self.left_border_state = {
            "active": False,
            "center": None,
            "radius": 0,
            "outer_center": None,
            "outer_radius": 0,
            "crossed": False,
            "crossing_point": None,
            "crossing_direction": None,
        }
        self.right_border_state = {
            "active": False,
            "center": None,
            "radius": 0,
            "outer_center": None,
            "outer_radius": 0,
            "crossed": False,
            "crossing_point": None,
            "crossing_direction": None,
        }

        # The gesture-monitoring thread sets this flag; the visualizer consumes it on a
        # later frame, where it has the landmarks and image needed to draw the border.
        self.activate_border_flag = False
        self.border_active = False
        self.border_center = None
        self.border_radius = 0
        self.border_crossed = False
        self.outer_border_crossed = False
        self.outer_border_center = None
        self.outer_border_radius = 0

        # Gesture labels travel from visualize() to this queue, then are consumed by a
        # daemon thread that recognizes ordered gesture sequences.
        self.gesture_thread = None
        self.gesture_thread_lock = threading.Lock()
        self.gesture_queue = queue.Queue()
        self.last_gesture = None

        # The newest gesture is stored at index 0. Older gestures shift toward index 2,
        # allowing the mapping below to match a three-gesture command sequence.
        self.a = None
        self.b = None
        self.c = None
        self.active_gesture_chain = [self.a, self.b, self.c]

        # Map a recognized gesture sequence to the action it should trigger.
        self.neutral_state_mapping = {
            (("Open_Palm", None), ("Closed_Fist", None), ("Open_Palm", None)): self.activate_zborder,
        }

        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.handle_result,
        )

        self.start_monitoring_gesture_queue()

    def activate_border(self):
        """Request that the border be drawn on the next frame where landmarks are available."""
        with self.state_lock:
            if not self.activate_border_flag:
                self.activate_border_flag = True

    def update_gesture_chain(self, new_val):
        """Shift the existing gestures and execute the matching action if a sequence is complete."""
        self.c = self.b
        self.b = self.a
        self.a = new_val
        self.active_gesture_chain[:] = [self.a, self.b, self.c]

        print(f'active gesture chain updated: {self.active_gesture_chain}')

        action = self.neutral_state_mapping.get(tuple(self.active_gesture_chain))
        if action is not None:
            action()
            print('action triggered')

    def check_gesture_queue(self):
        """Continuously monitor the gesture queue and update the gesture sequence when new labels arrive."""
        while True:
            try:
                current_gesture, current_crossing_direction = self.gesture_queue.get()
                if current_gesture == 'None':
                    continue
                elif current_crossing_direction:
                    # pass the direction to the currently active control mode
                    pass

                if current_gesture is None:
                    break

                if self.last_gesture is None or self.last_gesture != current_gesture:
                    print(f'detected gesture: {current_gesture}')
                    self.update_gesture_chain(current_gesture, current_crossing_direction)
                    self.last_gesture = current_gesture

            except Exception as exc:
                print('gesture monitor error:', exc)
                continue

    def start_monitoring_gesture_queue(self):
        """Start a background daemon thread to monitor the gesture queue."""
        with self.gesture_thread_lock:
            if self.gesture_thread is not None and getattr(self.gesture_thread, 'is_alive', lambda: False)():
                return
            self.gesture_thread = threading.Thread(target=self.check_gesture_queue, daemon=True)
            self.gesture_thread.start()

        print('gesture monitor started')

    def clear_border(self, border_state=None):
        """Reset either the shared fallback border or the state belonging to one hand."""
        if border_state is None:
            self.border_active = False
            self.border_center = None
            self.border_radius = 0
            self.border_crossed = False
            self.outer_border_crossed = False
            self.outer_border_center = None
            self.outer_border_radius = 0
            return

        border_state["active"] = False
        border_state["center"] = None
        border_state["radius"] = 0
        border_state["outer_center"] = None
        border_state["outer_radius"] = 0
        border_state["crossed"] = False
        border_state["crossing_point"] = None
        border_state["crossing_direction"] = None

    def draw_border(self, image, top_landmark, bottom_landmark, border_state=None):
        """Draw an inner border and a larger trigger ring around the hand."""
        if border_state is None:
            border_state = {
                "active": False,
                "center": None,
                "radius": 0,
                "outer_center": None,
                "outer_radius": 0,
                "crossed": False,
                "crossing_point": None,
                "crossing_direction": None,
            }

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

        dx_px = (bottom_x - top_x) * image.shape[1]
        dy_px = (bottom_y - top_y) * image.shape[0]
        distance_px = np.hypot(dx_px, dy_px)

        center_x = ((top_x + bottom_x) / 2.0) * image.shape[1]
        center_y = ((top_y + bottom_y) / 2.0) * image.shape[0]

        inner_radius = max(25, distance_px * 1.75)
        outer_radius = int(max(25, distance_px * 3))
        center = (int(center_x), int(center_y))

        border_state["center"] = center
        border_state["radius"] = int(inner_radius)
        border_state["outer_center"] = center
        border_state["outer_radius"] = outer_radius
        border_state["active"] = True
        border_state["crossed"] = False
        border_state["crossing_point"] = None
        border_state["crossing_direction"] = None

        self.border_center = border_state["center"]
        self.border_radius = border_state["radius"]
        self.outer_border_center = border_state["outer_center"]
        self.outer_border_radius = border_state["outer_radius"]
        self.border_crossed = border_state["crossed"]
        self.outer_border_crossed = False

        border_color = (0, 0, 255) if border_state["crossed"] else (0, 255, 0)
        cv2.circle(image, border_state["center"], border_state["radius"], border_color, 2)
        cv2.circle(image, border_state["outer_center"], border_state["outer_radius"], (255, 255, 0), 1)

        return image

    def monitor_border(self, hand_landmarks, image_shape=None, border_state=None):
        """Check whether hand landmarks extend outside the inner or outer border."""
        if border_state is None:
            border_state = {
                "active": self.border_active,
                "center": self.border_center,
                "radius": self.border_radius,
                "outer_center": self.outer_border_center,
                "outer_radius": self.outer_border_radius,
                "crossed": self.border_crossed,
                "crossing_point": None,
                "crossing_direction": None,
            }

        check_center = border_state.get("center")
        check_radius = border_state.get("radius")
        check_outer_radius = border_state.get("outer_radius")

        if hand_landmarks is None or check_center is None or check_radius <= 0:
            return False, None

        center_x, center_y = check_center
        width = image_shape[1] if image_shape is not None else 640
        height = image_shape[0] if image_shape is not None else 480

        found_outside_inner = False
        direction = None
        crossing_point = None

        for landmark in hand_landmarks:
            if landmark is None:
                continue

            x = int(landmark.x * width)
            y = int(landmark.y * height)
            distance = np.hypot(x - center_x, y - center_y)

            if distance >= check_radius:
                found_outside_inner = True

                if distance > 0:
                    scale = check_radius / distance
                    crossing_point = (
                        int(center_x + (x - center_x) * scale),
                        int(center_y + (y - center_y) * scale),
                    )
                else:
                    crossing_point = (center_x, center_y)

                delta_x = crossing_point[0] - center_x
                delta_y = center_y - crossing_point[1]
                angle = (np.degrees(np.arctan2(delta_y, delta_x)) + 360) % 360
                direction_names = (
                    "right", "up-right", "up", "up-left",
                    "left", "down-left", "down", "down-right",
                )
                direction = direction_names[int((angle + 22.5) // 45) % 8]

                if check_outer_radius is not None and distance >= check_outer_radius:
                    border_state["crossing_point"] = crossing_point
                    border_state["crossing_direction"] = direction
                    border_state["crossed"] = True
                    border_state["active"] = False
                    self.border_crossed = True
                    self.border_active = False
                    self.outer_border_crossed = True
                    return True, direction

        if found_outside_inner:
            border_state["crossing_point"] = crossing_point
            border_state["crossing_direction"] = direction
            border_state["crossed"] = False
            border_state["active"] = True
            self.border_crossed = False
            self.border_active = True
            self.outer_border_crossed = False
            return False, direction

        border_state["crossed"] = False
        border_state["crossing_point"] = None
        border_state["crossing_direction"] = None
        self.border_crossed = False
        self.outer_border_crossed = False
        return False, None

    def monitor_hand(self, hand_landmarks, image_shape=None, border_state=None):
        """Wrapper used by the visualizer to decide whether the hand crossed the border."""
        crossed, direction = self.monitor_border(hand_landmarks, image_shape, border_state)
        if crossed:
            if border_state is not None:
                border_state["crossed"] = True
                border_state["active"] = False
            print(f"Hand crossed the border ({direction})")
            self.activate_border_flag = False
            return True, direction
        elif direction:
            print(f"Hand crossed the border ({direction})")
            return False, direction

        if border_state is not None:
            border_state["crossed"] = False
        return False, None

    def visualize(self, image, detection_result) -> np.ndarray:
        """Draw gesture, handedness, and hand landmarks on a BGR image."""
        image = image.copy()

        if not hasattr(detection_result, "hand_landmarks"):
            return image

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17),
            (0, 17),
        ]

        for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
            if not hand_landmarks:
                continue

            handedness_label = "Unknown"
            if hand_index < len(detection_result.handedness):
                handedness_candidates = detection_result.handedness[hand_index]
                if handedness_candidates:
                    handedness = handedness_candidates[0]
                    handedness_label = f"{handedness.category_name} ({handedness.score:.2f})"

            if handedness_label.lower().startswith("left"):
                hand_state = self.left_hand_state
                border_state = self.left_border_state
            elif handedness_label.lower().startswith("right"):
                hand_state = self.right_hand_state
                border_state = self.right_border_state
            else:
                hand_state = {"gesture": "Unknown", "active": False}
                border_state = None

            for start_idx, end_idx in connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1 = int(start.x * image.shape[1])
                y1 = int(start.y * image.shape[0])
                x2 = int(end.x * image.shape[1])
                y2 = int(end.y * image.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for landmark in hand_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, self.TEXT_COLOR, -1)

            if self.activate_border_flag:
                if border_state is not None and not border_state["active"]:
                    image = self.draw_border(image, hand_landmarks[5], hand_landmarks[0], border_state)

            possible_crossing_direction = None # store the crossing direction for later
            if border_state is not None and border_state["active"]:
                crossed, crossing_direction = self.monitor_hand(hand_landmarks, image.shape, border_state)
                if crossed:
                    possible_crossing_direction = crossing_direction
                    cv2.circle(image, border_state["center"], border_state["radius"], (0, 0, 255), 2)
                    cv2.circle(image, border_state["outer_center"], border_state["outer_radius"], (0, 0, 255), 1)
                    if border_state["crossing_point"] is not None:
                        cv2.circle(image, border_state["crossing_point"], 7, (255, 0, 255), -1)
                        cv2.putText(image, crossing_direction, border_state["crossing_point"], cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, (255, 0, 255), self.FONT_THICKNESS)
                elif border_state["center"] is not None:
                    cv2.circle(image, border_state["center"], border_state["radius"], (0, 255, 0), 2)
                    cv2.circle(image, border_state["outer_center"], border_state["outer_radius"], (255, 255, 0), 1)

            gesture_label = "Unknown"
            if hand_index < len(detection_result.gestures):
                gesture_candidates = detection_result.gestures[hand_index]

                if gesture_candidates:
                    gesture_label = gesture_candidates[0].category_name

                hand_state["gesture"] = gesture_label
                hand_state["active"] = True
                self.gesture_queue.put(gesture_label, possible_crossing_direction) # pass the gesture and crossing direction for later processing

            text_y = 30 + hand_index * 40
            cv2.putText(
                image,
                f"Gesture: {gesture_label}",
                (10, text_y),
                cv2.FONT_HERSHEY_PLAIN,
                self.FONT_SIZE,
                self.TEXT_COLOR,
                self.FONT_THICKNESS,
            )
            cv2.putText(
                image,
                f"Handedness: {handedness_label}",
                (10, text_y + 20),
                cv2.FONT_HERSHEY_PLAIN,
                self.FONT_SIZE,
                self.TEXT_COLOR,
                self.FONT_THICKNESS,
            )

        return image

    def handle_result(self, result, output_image, timestamp):
        """Store the latest detection result and the image that produced it."""
        with self.state_lock:
            if timestamp != self.pending_timestamp:
                return

            self.latest_result = result
            self.latest_frame_bgr = self.pending_frame_bgr
            self.frame_pending = False
            self.pending_frame_bgr = None
            self.pending_timestamp = None

    def run_camera_loop(self):
        """Open the camera, process frames, and display the annotated output."""
        with vision.GestureRecognizer.create_from_options(self.options) as detector:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise RuntimeError("Cannot open camera")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                with self.state_lock:
                    if not self.frame_pending:
                        self.pending_frame_bgr = frame.copy()
                        self.pending_timestamp = frame_timestamp
                        self.frame_pending = True
                        detector.recognize_async(mp_image, frame_timestamp)

                    current_result = self.latest_result
                    current_frame_bgr = self.latest_frame_bgr

                if current_result is not None and current_frame_bgr is not None:
                    annotated_frame = self.visualize(current_frame_bgr, current_result)
                    cv2.imshow("Hand Detection", annotated_frame)
                else:
                    cv2.imshow("Hand Detection", frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.run_camera_loop()