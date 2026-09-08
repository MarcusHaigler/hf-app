import multiprocessing as mp
import queue
from assets.cursor import CursorMode
from assets.neutral import NeutralMode
from assets.media import MediaMode


class Control_Manager:
    '''
    entry point for interacting with the control classes
    monitor the incoming queue on a separate process, and execute functions depending on the active control mode
    '''

    def __init__(self):

        # create control class objects here
        self.neutral_mode = NeutralMode()

        self.active_control_mappings = {
            'neutral' :self.neutral_mode.neutral_consumer,
        }

        self.incoming_queue = mp.Queue() # Create a message queue used to send commands to the worker process.
        self.active_control_mode = None # the currently selected control mode

        # Start a separate process that owns the control mode state and executes commands.
        self.control_monitor_process = mp.Process(
            target=self.monitor_incoming_queue,
            args=(self.incoming_queue),
            daemon=True,
        )

        self.control_monitor_process.start()

    def monitor_incoming_queue(self):
        """
        use the active control mode and execute messages in the child process
        """

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
                    self.active_control_mode = message[1]

                # Handle an incoming gesture only when a mode is already active.
                elif command == "gesture" and self.active_control_mode is not None:
                    gesture, crossing_direction = message[1] # unpack values 
                    self.active_control_mappings[self.active_control_mode](gesture, crossing_direction) # Pass the gesture to the currently active control mode for processing.
                    
            except Exception as exc:
                # Capture any exception and send a string form back out of the worker.
                print(("error", repr(exc)))

    def set_mode(self, control_mode):
        """
        send a command to the worker process to set the active control mode
        """

        # Queue a command telling the worker to replace the active control mode.

        if control_mode not in self.active_control_mappings:
            raise ValueError(f"Invalid control mode: {control_mode}. Valid modes are: {list(self.active_control_mappings.keys())}")

        self.incoming_queue.put(("set_mode", control_mode))

    def submit(self, gesture, crossing_direction=None):
        """
        Queue one gesture event without blocking the caller
        """
        self.incoming_queue.put(("gesture", (gesture, crossing_direction)))

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

class ControlHelper:
    '''
    reusable functions that can be used by multiple control classes
    '''

    def activate_neutral_control_mode(self):
        '''
        reset the currently active controls variables, and
        set the current control mode to neutral
        '''
        # This helper currently acts as a no-op placeholder for neutral mode activation.
        return True

    def activate_border(self, target_lock=None, target_flag=None):
        '''
        change the target_flag to True using a lock to prevent race conditions.
        Returns the updated flag state so callers can persist the result.
        '''
        # If no flag was supplied, there is nothing to update.
        if target_flag is None:
            return None
        try:
            # If a lock is given, protect the flag update so only one thread changes it at a time.
            if target_lock is not None:
                with target_lock:
                    # Set the flag to true only if it is currently false.
                    if not target_flag:
                        target_flag = True
                    # Return the resulting state so the caller can save it.
                    return target_flag
            # If there is no lock, update the flag using normal Python assignment.
            if not target_flag:
                target_flag = True
            return target_flag
        except Exception as exc:
            # Print an error instead of failing silently when the border activation fails.
            print('error activating border:', exc)
            return target_flag

    def check_mapping(self, value, mapping):
        '''
        check if a value exists in a mapping, and execute its corresponding action if it does
        '''
        # Convert the incoming value into the tuple key used by the mapping.
        action = mapping.get(tuple(value))
        # If a matching callback exists, call it and return success.
        if action is not None:
            action()
            print('action triggered')
            return True
        # If no mapping entry exists, report that nothing happened.
        print('no action found')
        return False