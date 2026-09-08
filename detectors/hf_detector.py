'''
Hands-Free Dectector class
detects hand gestures to interact with PC
'''

import threading
import cv2  # OpenCV for camera capture and image display
import numpy as np  # NumPy for array handling and image conversion
import mediapipe as mp  # MediaPipe core library
from mediapipe.tasks import python  # MediaPipe Python task utilities
from mediapipe.tasks.python import vision  # MediaPipe vision tasks
import queue


class HF_Detector:
    '''
    detect hand gestures and send requests to a backend to perform actions
    '''

    def __init__(self, model_path: str):
        '''
        initialize the hand gesture detector
        '''

        # Visual styling constants for the detection overlay
        self.MARGIN = 10  # Space between the bounding box and the text label
        self.ROW_SIZE = 10  # Text line height for the overlay
        self.FONT_SIZE = 1  # Font scale for the label text
        self.FONT_THICKNESS = 1  # Line thickness for the text
        self.TEXT_COLOR = (255, 0, 0)  # Red color for boxes and labels

        self.model_path = model_path
        self.latest_result = None  # Holds the most recent detection output
        self.latest_frame_bgr = None  # Holds the most recent frame in BGR format for drawing
        self.frame_pending = False  # Whether a request is already in flight for async detection
        self.pending_frame_bgr = None  # The frame that was sent to the detector for the pending request
        self.pending_timestamp = None  # Timestamp associated with the pending request
        self.state_lock = threading.Lock()  # Protect shared callback state

        # state constants
        self.gesture_chain_started = False  # Whether a gesture chain has started
        self.current_state = None  # Current state of the detector
        self.gesture_queue = queue.Queue(maxsize=5)  # Queue to store detected gestures for processing

        self.neutral_state_mappings = None # dict to hold the expected gesture sequences for valid gesture chains in the neutral state
        self.cursor_state_mappings = None # dict to hold the expected gesture sequences for valid gesture chains in the cursor state
        self.media_state_mappings = None # dict to hold the expected gesture sequences for valid gesture chains in the media state


        # Initialize MediaPipe Hands model
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def neutral_state_handler(self, gesture_label):
        '''
        logic for handling gestures in the neutral state

        gestures are registered in batches, specfically the last three registered gestures
        '''

        # confirm state is neutral
        if self.current_state != 'neutral':
            return
        
        # if a valid gesture was detected, check if chain is started
        if gesture_label:

            # if the chain is already started and the gesture is unique, add it to the gesture queue
            if self.gesture_chain_started and gesture_label != self.gesture_queue.queue[-1]:
                self.gesture_queue.put(gesture_label)

                # compare the last three gestures to the expected sequence for a valid gesture chain
                if self.gesture_queue.size() == 3:
                    # process the last three gestures
                    gesture_chain = list(self.gesture_queue.queue)
                    if gesture_chain in self.neutral_state_mappings:

                        # call backend function
                        pass

                        # remove last three gestures from the queue
                        for _ in range(3):
                            self.gesture_queue.get() 
            else:
                if gesture_label == 'open palm':
                    self.gesture_chain_started = True
                    self.gesture_queue.put(gesture_label)

    def handle_result(self, result, output_image, timestamp):
        """
        update the latest result and frame when the detection callback is invoked
        """

        with self.state_lock:
            if timestamp != pending_timestamp:
                # Ignore any stale callback that does not match the current pending request.
                return

            self.latest_result = result
            self.latest_frame_bgr = pending_frame_bgr
            self.frame_pending = False
            pending_frame_bgr = None
            pending_timestamp = None 

    def visualize(self, image, detection_result) -> np.ndarray:
        """
        Draw gesture, handedness, and hand landmarks on a BGR image
        """

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
                cv2.circle(image, (x, y), 5, self.TEXT_COLOR, -1)  # Draw a red filled circle at the landmark.

            # Set a default gesture label in case no gesture is detected.
            gesture_label = "Unknown"
            if hand_index < len(detection_result.gestures):  # Check whether gesture data exists for this hand.
                gesture_candidates = detection_result.gestures[hand_index]  # Get the gesture candidates for this hand.

                if gesture_candidates:  # Use the top gesture if any candidates are available.
                    gesture_label = gesture_candidates[0].category_name  # Store the highest-confidence gesture name.

                if self.current_state == 'neutral':
                    self.netural_state_handler(gesture_label)  # Handle the gesture in the neutral state.
                elif self.current_state == 'gesture_chain':
                    self.cursor_state_handler(gesture_label)  # Handle the gesture in the gesture chain state.
                elif self.current_state == 'gesture_chain_started':
                    self.media_state_handler(gesture_label)  # Handle the gesture in the gesture chain started state.

            # Set a default handedness label in case no handedness is detected.
            handedness_label = "Unknown"
            if hand_index < len(detection_result.handedness):  # Check whether handedness data exists for this hand.
                handedness_candidates = detection_result.handedness[hand_index]  # Get handedness candidates for this hand.
                if handedness_candidates:  # Use the top handedness result if available.
                    handedness = handedness_candidates[0]  # Get the highest-confidence handedness object.
                    handedness_label = f"{handedness.category_name} ({handedness.score:.2f})"  # Format the handedness text with confidence.

            # Calculate the vertical position for the text labels based on the hand index.
            text_y = 30 + hand_index * 40
            cv2.putText(  # Draw the gesture label near the top-left corner of the image.
                image,
                f"Gesture: {gesture_label}",
                (10, text_y), 
                cv2.FONT_HERSHEY_PLAIN,
                self.FONT_SIZE,
                self.TEXT_COLOR,
                self.FONT_THICKNESS,
            )
            cv2.putText(  # Draw the handedness label beneath the gesture label.
                image,
                f"Handedness: {handedness_label}",
                (10, text_y + 20),
                cv2.FONT_HERSHEY_PLAIN,
                self.FONT_SIZE,
                self.TEXT_COLOR,
                self.FONT_THICKNESS,
            )

        # Return the annotated image after all overlays have been drawn.
        return image
