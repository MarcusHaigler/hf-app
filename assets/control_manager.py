import multiprocessing as mp
from .neutral import NeutralMode
from.cursor import CursorMode
from ctypes import c_int

NEUTRAL_MODE = 0
CURSOR_MODE = 1
MODE_IDS = {
    'neutral': NEUTRAL_MODE,
    'cursor': CURSOR_MODE,
}
MODE_NAMES = {mode_id: name for name, mode_id in MODE_IDS.items()}

class Control_Manager:
    '''
    entry point for interacting with the control classes
    monitor the incoming queue on a separate process, and execute functions depending on the active control mode
    '''

    def __init__(self, target_flag=None, lock=None):

        # used by the lower level control classes to safely update the target_flag in a thread-safe manner
        self.control_target_lock = lock
        self.control_target_flag = target_flag
        self.incoming_queue = mp.Queue() # Create a message queue used to send commands to the worker process.
        

        # create control class objects here
        self.neutral_mode = NeutralMode(self.control_target_flag, self.control_target_lock, self.incoming_queue)
        self.cursor_mode = CursorMode(self.control_target_flag, self.control_target_lock, self.incoming_queue)

        self.active_control_mappings = {
            'neutral' :self.neutral_mode.neutral_consumer,
            'cursor' : self.cursor_mode.cursor_consumer
        }

        self.active_control_mode = mp.Value(c_int, NEUTRAL_MODE)

    def active_mode_name(self):
        """Return the active mode name from the shared mode value."""
        return MODE_NAMES[self.active_control_mode.value]

    def start(self):
         # Start a separate process that owns the control mode state and executes commands.
        self.control_monitor_process = mp.Process(
            target=self.monitor_incoming_queue,
            daemon=True,
        )

        self.control_monitor_process.start()

    def monitor_incoming_queue(self):
        """
        use the active control mode and execute messages in the child process
        """

        print('control monitor started...')
        while True:
            
            # Wait until a new message arrives in the parent-to-child queue.
            message = self.incoming_queue.get()
            # The first element of each message is the command name.
            command = message[0]

            # A stop command tells the worker to shut itself down cleanly.
            if command == "stop":
                return

            # Protect the worker loop so one bad message cannot crash the entire process.
            try:
                # to set the active control mode
                if command == "set_mode":
                    new_mode = message[1]
                    if new_mode not in self.active_control_mappings:
                        raise ValueError(f"Invalid control mode: {new_mode}. Valid modes are: {list(self.active_control_mappings.keys())}")
                    else:
                        if new_mode == 'cursor':
                            self.cursor_mode.reset_sequence()
                        self.active_control_mode.value = MODE_IDS[new_mode]
                        print(f'control mode set to {new_mode}')

                # Handle an incoming gesture only when a mode is already active.
                elif command == "gesture":
                    gesture, crossing_direction, border_flag = message[1] # unpack values 
                    mode = MODE_NAMES[self.active_control_mode.value]
                    self.active_control_mappings[mode](gesture, crossing_direction, border_flag) # Pass the gesture to the currently active control mode for processing.
                    
            except Exception as exc:
                # Capture any exception and send a string form back out of the worker.
                print(f"error: {repr(exc)}")

    def submit(self, gesture, crossing_direction=None, border_flag=False):
        """
        Queue one gesture event without blocking the caller
        """
        self.incoming_queue.put(("gesture", (gesture, crossing_direction, border_flag)))

    def close(self):
        """Stop the worker and release its queue resources."""
        # Ask the child process to exit gracefully if it is still alive.
        if self.control_monitor_process.is_alive():
            self.incoming_queue.put(("stop",))
            self.control_monitor_process.join(timeout=2)
        # If the worker did not exit in time, force it to terminate.
        if self.control_monitor_process.is_alive():
            self.control_monitor_process.terminate()
        # Close both queues to free OS resources after shutdown.
        self.incoming_queue.close()
