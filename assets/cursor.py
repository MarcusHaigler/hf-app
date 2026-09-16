from .control_helper import ControlHelper
from pynput import mouse

class CursorMode(ControlHelper):
    '''
    functions for controlling the mouse cursor
    '''

    def __init__(self, target_flag=None, lock=None, target_queue=None):

        super().__init__(target_flag, lock,target_queue)

        self.last_message = None, None
        self.last_sequence = None # to prevent continous function calls (specifically for click functions)

        self.a = None
        self.b = None
        self.curr_cursor_sequence = [self.a, self.b]

        self.cursor_gesture_sequence_mappings = {
            (('Open_Palm', None), ('Closed_Fist', None)) : self.activate_border
        }

        self.active_cursor_mapping = {
            (('Closed_Fist', None),('Thumb_Up', None)) : self.left_click,
            (('Closed_Fist', None),('Thumb_Down', None)) : self.right_click,
            (('Pointing_Up', None), ('Pointing_Up', 'up')) : self.activate_neutral_control_mode
        }

    def cursor_consumer(self, gesture, direction=None, border_flag=False):
        '''
        process incoming data and take action if a gesture sequence is complete
        ''' 

        if gesture in (None, 'None'):
            return None

        # if the cursor flag is false, then check if the gesture sequence is able to flip it
        if border_flag is False:
            # update the gesture chain as long as its fresh data
            if self.last_message != (gesture, direction):
                self.update_cursor_sequence(gesture, direction)
                self.last_message = gesture, direction
                print('updated cursor sequence:', self.curr_cursor_sequence)

                # check if cursor sequence is complete
                action = self.check_mapping(self.curr_cursor_sequence, self.cursor_gesture_sequence_mappings)
                if action is not None:
                    action()
        else:
            if gesture == 'Closed_Fist' and direction != None:
                self.move_cursor(direction)
            else: 
                # update the gesture chain as long as its fresh data, then check the mapping
                if self.last_message != (gesture, direction):
                    self.update_cursor_sequence(gesture, direction)
                    self.last_message = gesture, direction
                    print('updated cursor sequence:', self.curr_cursor_sequence)
                    

                # confirm that the new sequence is fresh
                if self.curr_cursor_sequence != self.last_sequence:
                    sequence = list(self.curr_cursor_sequence)
                    self.last_sequence = sequence

                    action = self.check_mapping(sequence, self.active_cursor_mapping)
                    if action is not None:
                        action()

    def reset_sequence(self):
        """Clear gesture history when cursor mode is entered again."""
        self.last_message = None, None
        self.last_sequence = None
        self.a = None
        self.b = None
        self.curr_cursor_sequence[:] = [None, None]

    def right_click(self):
        '''
        function to perform a right click
        '''
        mouse.Controller().click(mouse.Button.right, 1)
        print('right click called')

    def left_click(self):
        '''
        function to perform a left click
        '''
        mouse.Controller().click(mouse.Button.left, 1)
        print('left click called')

    def update_cursor_sequence(self, new_val, new_direction=None):

        """
        Shift the existing gestures
        """

        self.a = self.b
        self.b = (new_val, new_direction)
        self.curr_cursor_sequence[:] = [self.a, self.b]

    def move_cursor(self, direction):
        '''
        move the cursor relative to its current position in a specified direction
        '''

        if direction == None:
            return

        direction_mapping = {
            
        }



        if direction == 'up':
           mouse.Controller().move(0, -5)
        elif direction == 'up-left':
            mouse.Controller().move(-5, -5)
        elif direction == 'up-right':
            mouse.Controller().move(5,-5)

        if direction == 'down':
            mouse.Controller().move(0, 5)
        elif direction == 'down-left':
            mouse.Controller().move(-5, 5)
        elif direction == 'down-right':
            mouse.Controller().move(5, 5)

        elif direction == 'left':
            mouse.Controller().move(-5, 0)
        elif direction == 'right':
            mouse.Controller().move(5, 0)

        if direction == None:
            return

        print(f'mouse moving in {direction}')
        