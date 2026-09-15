

class ControlHelper:
    '''
    reusable functions that can be used by multiple control classes
    '''

    def __init__(self, target_flag=None, lock=None, target_queue=None):
        # used by the lower level control classes to safely update the target_flag in a thread-safe manner
        self.control_target_lock = lock
        self.control_target_flag = target_flag
        self.target_queue = target_queue

    def set_mode(self, val):
        """
        send a command to the worker process to set the active control mode
        - really only used by the neutral controls
        """

        # Queue a command telling the worker to replace the active control mode.
    
        self.target_queue.put(("set_mode", val))

    def _get_target_flag(self):
        """Return the current flag value whether it is a plain bool or a multiprocessing Value."""
        if self.control_target_flag is None:
            return None
        if hasattr(self.control_target_flag, 'value'):
            return self.control_target_flag.value
        return self.control_target_flag

    def _set_target_flag(self, value):
        """Update the underlying flag object without replacing the original reference."""
        if self.control_target_flag is None:
            return None
        if hasattr(self.control_target_flag, 'value'):
            self.control_target_flag.value = bool(value)
            return self.control_target_flag.value
        self.control_target_flag = bool(value)
        return self.control_target_flag

    def activate_neutral_control_mode(self):
        '''
        reset the currently active controls variables, and
        set the current control mode to neutral
        '''
        self.target_queue.put(("set_mode", 'neutral'))
        
    def activate_border(self):
        '''
        change the target_flag to True using a lock to prevent race conditions.
        Returns the updated flag state so callers can persist the result.
        '''
        current_flag = self._get_target_flag()

        # If no flag was supplied, there is nothing to update.
        if current_flag is None:
            print('no target flag supplied, cannot activate border')
            return None
        try:
            # If a lock is given, protect the flag update so only one thread changes it at a time.
            if self.control_target_lock is not None:
                with self.control_target_lock:
                    # Set the flag to true only if it is currently false.
                    if not current_flag:
                        return self._set_target_flag(True)
                    return self._get_target_flag()
            # If there is no lock, update the flag using normal Python assignment.
            if not current_flag:
                return self._set_target_flag(True)
            return self._get_target_flag()
        except Exception as exc:
            # Print an error instead of failing silently when the border activation fails.
            print('error activating border:', exc)
            return self._get_target_flag()

    def check_mapping(self, value, mapping):
        '''
        check if a value exists in a mapping, and execute its corresponding action if it does
        '''
        # Convert the incoming value into the tuple key used by the mapping.
        action = mapping.get(tuple(value))
        # If a matching callback exists, call it and return success.
        if action is not False:
            return action
        # If no mapping entry exists, report that nothing happened.
        return False