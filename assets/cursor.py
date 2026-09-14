from .control_helper import ControlHelper
from pynput import mouse
import queue

class CursorMode(ControlHelper):
    '''
    functions for controlling the mouse cursor
    '''

    def __init__(self):

        super().__init__()

        self.active_cursor_mapping  = {

        }

        self.cursor_job_queue = queue.Queue()

        self.a = None
        self.b = None
        self.curr_cursor_sequence = [self.a, self.b]

        self.cursor_gesture_sequence_mappings = {
            (('Open_Palm', None), ('Closed_Fist', None)) : self.activate_border,
            (('Open_Palm', None), ('Closed_Fist', None)) : self.activate_neutral_control_mode
        }

        self.active_cursor_mapping = {
            ('Thumbs_Up', None) : self.left_click,
            ('Thumbs_Down', None) : self.right_click,
        }

    def cursor_consumer(self, gesture, direction=None, border_flag=False):
        '''
        process incoming data and take action if a gesture sequence is complete
        ''' 

        # if the cursor flag is false, then check if the gesture sequence is able to flip it
        if border_flag is False:
            # update the gesture chain as long as its fresh data
            if self.a is None or self.a != gesture:
                self.update_cursor_sequence(gesture)
                print('updated cursor sequence:', self.curr_cursor_sequence)

                # check if cursor sequence is complete
                if self.curr_cursor_sequence in self.cursor_gesture_sequence_mappings:
                    self.activate_border()
                    self.curr_cursor_sequence = [None, None]
                    return None
        else:
            if gesture == 'Closed_Fist':
                self.move_cursor(direction)
            else:
                self.check_mapping(gesture, direction)

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

    def update_cursor_sequence(self, val):

        """
        Shift the existing gestures
        """

        self.b = self.a
        self.a = (val)
        self.active_gesture_chain[:] = [self.a, self.b ]

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

        print(f'test mouse moving in {direction}')
        

        