'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        precision_dict = {'high':100, 'medium':500, 'low':1000}
        speed_dict = {'faster':0, 'fast':1, 'medium':5, 'slow':10}

        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.prevx = 0
        self.prevy = 0

    def move(self, x, y):
        mx, my = pyautogui.position()
        x_distance = x * self.precision
        y_distance = -1 * y * self.precision 
        
        if pyautogui.onScreen(mx + x_distance, my + y_distance):
            pyautogui.moveRel(x_distance, y_distance, duration=self.speed)

    def movexy(self, x, y):
        pyautogui.move(x, y, duration=self.speed)
