from .control_helper import ControlHelper
from pynput import mouse

class CursorMode(ControlHelper):
    '''
    functions for controlling the mouse cursor
    '''

    def __init__(self, target_flag=None, lock=None, target_queue=None):

        super().__init__(target_flag, lock,target_queue)

        self.last_message = None, None

        self.a = None
        self.b = None
        self.curr_cursor_sequence = [self.a, self.b]

        self.cursor_gesture_sequence_mappings = {
            (('Open_Palm', None), ('Closed_Fist', None)) : self.activate_border,
            (('Open_Palm', None), ('Pointing_Up', 'up')) : self.activate_neutral_control_mode
        }

        self.active_cursor_mapping = {
            ('Thumb_Up', None) : self.left_click,
            ('Thumb_Down', None) : self.right_click,
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
                self.update_cursor_sequence(gesture)
                self.last_message = gesture, direction
                print('updated cursor sequence:', self.curr_cursor_sequence)

                # check if cursor sequence is complete
                action = self.check_mapping(self.curr_cursor_sequence, self.cursor_gesture_sequence_mappings)
                if action != None:
                    action()
                    self.curr_cursor_sequence = [None, None]
        else:
            if gesture == 'Closed_Fist':
                self.move_cursor(direction)
            else: 
                action = self.check_mapping((gesture, direction), self.active_cursor_mapping)
                if action != None:
                    action()

    def right_click(self):
        '''
        function to perform a right click
        '''
        #mouse.Controller().click(mouse.Button.right, 1)
        print('right click called')

    def left_click(self):
        '''
        function to perform a left click
        '''
        #mouse.Controller().click(mouse.Button.left, 1)
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
        
        current_position = mouse.Controller().position
        x, y = current_position

        if direction == 'up':
            y -= 10
        elif direction == 'down':
            y += 10
        elif direction == 'left':
            x -= 10
        elif direction == 'right':
            x += 10
            
        mouse.Controller().position = (x, y)
        '''
        if direction == None:
            return

        print(f'mouse moving in {direction}')
        