'''
run simple functions on a separate thread
'''

import threading
import cv2  # OpenCV for camera capture and image display


_thumbs_up_thread_started = False
_thumbs_up_thread_lock = threading.Lock()

_thumbs_down_thread_started = False
_thumbs_down_thread_lock = threading.Lock()

def dummy_thumbs_up_func():
    global _thumbs_up_thread_started

    with _thumbs_up_thread_lock:
        if _thumbs_up_thread_started:
            return
        _thumbs_up_thread_started = True

    try:
        def thread_function():
            print("up")

        thread = threading.Thread(target=thread_function)
        thread.start()
        thread.join()
    finally:
        with _thumbs_up_thread_lock:
            _thumbs_up_thread_started = False

def dummy_thumbs_down_func():
    global _thumbs_down_thread_started

    with _thumbs_down_thread_lock:
        if _thumbs_down_thread_started:
            return
        _thumbs_down_thread_started = True

    try:
        def thread_function():
            print("down")

        thread = threading.Thread(target=thread_function)
        thread.start()
        thread.join()
    finally:
        with _thumbs_down_thread_lock:
            _thumbs_down_thread_started = False

