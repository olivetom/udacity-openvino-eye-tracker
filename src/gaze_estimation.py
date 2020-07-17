'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import time
from inference import Network

class GazeEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_xml = model_name
        self.device =  device
        self.extensions = extensions
         # Initialise the class
        self.infer_network = Network()



    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.infer_network.load_model(self.model_xml, self.device, self.extensions)



    def predict(self, left_eye_image, right_eye_image, headpose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.infer_network.exec_net(headpose_angles, left_eye_image, right_eye_image)

        # Wait for the result
        if self.infer_network.wait() == 0:
            # end time of inference
            end_time = time.time()
            result = (self.infer_network.get_output())[self.infer_network.output_blob]
            return result



    def check_model(self):
        raise NotImplementedError



    def preprocess_input(self, frame, face, left_eye_point, right_eye_point, print_flag=True):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

       Blob in the format [BxCxHxW] where:
        B - batch size
        C - number of channels
        H - image height
        W - image width
        with the name left_eye_image and the shape [1x3x60x60].
        Blob in the format [BxCxHxW] where:
        B - batch size
        C - number of channels
        H - image height
        W - image width
        with the name right_eye_image and the shape [1x3x60x60].
        Blob in the format [BxC] where:
        B - batch size
        C - number of channels
        with the name head_pose_angles and the shape [1x3].

        '''
        left_eye_input_shape =  [1,3,60,60] 
        right_eye_input_shape = [1,3,60,60] 

        # crop left eye
        x_center = left_eye_point[0]
        y_center = left_eye_point[1]
        width = left_eye_input_shape[3]
        height = left_eye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        face_width_edge = face.shape[1]
        face_height_edge = face.shape[0]
        
        ymin = int(y_center - height // 2) if int(y_center - height // 2) >= 0 else 0 
        ymax = int(y_center + height // 2) if int(y_center + height // 2) <= face_height_edge else face_height_edge

        xmin = int(x_center - width // 2) if int(x_center - width // 2) >= 0 else 0 
        xmax = int(x_center + width // 2) if int(x_center + width // 2) <= face_width_edge else face_width_edge

        left_eye_image = face[ymin: ymax, xmin:xmax]

        if print_flag:
            frame[150:150 + left_eye_image.shape[0], 20:20 + left_eye_image.shape[1]] = left_eye_image

        p_frame_left = cv2.resize(left_eye_image, (left_eye_input_shape[3], left_eye_input_shape[2]))
        p_frame_left = p_frame_left.transpose((2,0,1))
        p_frame_left = p_frame_left.reshape(1, *p_frame_left.shape)

        x_center = right_eye_point[0]
        y_center = right_eye_point[1]
        width = right_eye_input_shape[3]
        height = right_eye_input_shape[2]

        ymin = int(y_center - height // 2) if int(y_center - height // 2) >= 0 else 0 
        ymax = int(y_center + height // 2) if int(y_center + height // 2) <= face_height_edge else face_height_edge

        xmin = int(x_center - width // 2) if int(x_center - width // 2) >= 0 else 0 
        xmax = int(x_center + width // 2) if int(x_center + width // 2) <= face_width_edge else face_width_edge

        right_eye_image =  face[ymin: ymax, xmin:xmax]

        if(print_flag):
            frame[150:150+right_eye_image.shape[0], 100:100+right_eye_image.shape[1]] = right_eye_image
            
        p_frame_right = cv2.resize(right_eye_image, (right_eye_input_shape[3], right_eye_input_shape[2]))
        p_frame_right = p_frame_right.transpose((2,0,1))
        p_frame_right = p_frame_right.reshape(1, *p_frame_right.shape)

        return frame, p_frame_left, p_frame_right



    def preprocess_output(self, outputs, image,facebox, left_eye_point, right_eye_point,print_flag=True, threshold = 0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.
        Output layer name in Inference Engine format:
        gaze_vector
        '''
        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]

        if print_flag:
            label = f"x:{x*100:.1f}, y:{y*100:.1f}, z:{z:.1f}"
            cv2.putText(image, label, (20, 100), 0, 0.6, (0,0,255), 1)
            xmin, ymin, _, _ = facebox

            # left eye
            x_center = left_eye_point[0]
            y_center = left_eye_point[1]
            left_eye_center_x = int(xmin + x_center)
            left_eye_center_y = int(ymin + y_center)

            # right eye
            x_center = right_eye_point[0]
            y_center = right_eye_point[1]
            right_eye_center_x = int(xmin + x_center)
            right_eye_center_y = int(ymin + y_center)

            cv2.arrowedLine(image, (left_eye_center_x,left_eye_center_y), 
                    (left_eye_center_x + int(x*100),left_eye_center_y + int(-y*100)), 
                    (255, 100, 100), 5)
            cv2.arrowedLine(image, (right_eye_center_x,right_eye_center_y), 
                    (right_eye_center_x + int(x*100),right_eye_center_y + int(-y*100)), 
                    (255,100, 100), 5)

        return image, [x, y, z]