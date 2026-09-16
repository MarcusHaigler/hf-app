from .control_helper import ControlHelper

class NeutralMode(ControlHelper):

    def __init__(self, target_flag=None, lock=None, target_queue=None):

        super().__init__(target_flag, lock, target_queue)

        self.last_message = None, None

        # The newest gesture is stored at index 0. Older gestures shift toward index 2,
        # allowing the mapping below to match a three-gesture command sequence.
        self.a = None
        self.b = None
        self.c = None
        self.active_gesture_chain = [self.a, self.b, self.c]

        # map a recognized gesture sequence to the action it should trigger
        self.neutral_state_mapping = {
            (("Open_Palm", None), ("Closed_Fist", None), ("Open_Palm", None)): self.activate_border,
            (('Open_Palm', None), ('Closed_Fist', None), ('Closed_Fist', 'up')) : self.activate_cursor_mode
        }

    def dummy_call(self):
        '''
        dummy dummy dummy
        '''
        print('dummy')

    def neutral_consumer(self, gesture, direction=None, border_flag=False):
        '''
        process incoming data and take action if a gesture sequence is complete
        '''

        if gesture in (None, 'None'):
            return None

        # update the gesture chain as long as its fresh data
        if self.last_message != (gesture, direction):
            self.update_gesture_chain(gesture, direction)
            self.last_message = gesture, direction
            print('updated gesture chain:', self.active_gesture_chain)
 
            # compare the gesture chain to the mapping and execute the corresponding action if it exists
            action = self.check_mapping(self.active_gesture_chain, self.neutral_state_mapping)
            if action != None:
                action()

    def update_gesture_chain(self, new_val, direction=None):
        """
        Shift the existing gestures
        """

        self.a = self.b
        self.b = self.c
        self.c = (new_val, direction)
        self.active_gesture_chain[:] = [self.a, self.b, self.c]

    def activate_cursor_mode(self):
        '''
        activate the control mode
        '''

        self.set_mode('cursor')

    def activate_media_mode(self):
        '''
        activate the media mode

        prolly have to use a set_mode call
        '''
        self.set_mode('media')