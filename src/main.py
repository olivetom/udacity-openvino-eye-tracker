"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
from random import randint
from inference import Network
from facial_landmarks_detection import FacialLandMarksDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel 
from mouse_controller import MouseController
import logging
from argparse import ArgumentParser
from face_detection import FaceDetectionModel
from input_feeder import InputFeeder
import sys
import numpy as np


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--fdmodel", required=True, type=str,
                        help="Path to a face detection xml file with a trained model.")
    parser.add_argument("-hpm", "--hpmodel", required=True, type=str,
                        help="Path to a head pose estimation xml file with a trained model.")
    parser.add_argument("-lmm", "--lmmodel", required=True, type=str,
                        help="Path to a facial landmarks xml file with a trained model.")
    parser.add_argument("-gem", "--gemodel", required=True, type=str,
                        help="Path to a gaze estimation xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path video file or CAM to use camera")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("--print",default=False,
                        help="Overlay inference output over frame", action="store_true")
    parser.add_argument("--no_move",default=False,
                        help="Don't move mouse based on gaze estimation output",action="store_true")
    parser.add_argument("--no_video",default=False,
                        help="Don't show video window", action="store_true")
    return parser



def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    try:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("debug.log"),
                logging.StreamHandler()
            ])
        

        # Initialise the class
        mc = MouseController("low", "fast")
        fdnet = FaceDetectionModel(args.fdmodel)
        lmnet = FacialLandMarksDetectionModel(args.lmmodel)
        hpnet = HeadPoseEstimationModel(args.hpmodel)
        genet = GazeEstimationModel(args.gemodel)

        start_time = time.time()
        fdnet.load_model()
        logging.info(f"Face Detection Model: {1000 * (time.time() - start_time):.1f}ms")

        start_time = time.time()
        lmnet.load_model()
        logging.info(f"Facial Landmarks Detection Model: {1000 * (time.time() - start_time):.1f}ms")

        start_time = time.time()
        hpnet.load_model()
        logging.info(f"Headpose Estimation Model: {1000 * (time.time() - start_time):.1f}ms")

        start_time = time.time()
        genet.load_model()
        logging.info(f"Gaze Estimation Model: {1000 * (time.time() - start_time):.1f}ms")


        # Get and open video capture
        feeder = InputFeeder('video', args.input)
        feeder.load_data()

        frame_count = 0

        fd_infertime = 0
        lm_infertime = 0
        hp_infertime = 0
        ge_infertime = 0

        while True:
            # Read the next frame
            try:
                frame = next(feeder.next_batch())
            except StopIteration:
                break

            key_pressed = cv2.waitKey(60)
            frame_count += 1
            
            # face detection
            p_frame = fdnet.preprocess_input(frame)
            start_time = time.time()
            fd_output = fdnet.predict(p_frame)
            fd_infertime += time.time() - start_time
            out_frame, bboxes = fdnet.preprocess_output(fd_output, frame, args.print)
            
            for bbox in bboxes:
                
                face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                p_frame = lmnet.preprocess_input(face)
                
                start_time = time.time()
                lm_output = lmnet.predict(p_frame)
                lm_infertime += time.time() - start_time
                out_frame, left_eye_point, right_eye_point = lmnet.preprocess_output(lm_output, bbox, out_frame, args.print)

                # get head pose estimation
                p_frame  = hpnet.preprocess_input(face)
                start_time = time.time()
                hp_output = hpnet.predict(p_frame)
                hp_infertime += time.time() - start_time
                out_frame, headpose_angles = hpnet.preprocess_output(hp_output, out_frame, face, bbox, args.print)

                # get gaze  estimation
                out_frame, left_eye, right_eye = genet.preprocess_input(out_frame, face, left_eye_point, right_eye_point, args.print)
                start_time = time.time()
                ge_output = genet.predict(left_eye, right_eye, headpose_angles)
                ge_infertime += time.time() - start_time
                out_frame, gaze_vector = genet.preprocess_output(ge_output, out_frame, bbox, left_eye_point, right_eye_point, args.print)

                if not args.no_video:
                    cv2.imshow('image', out_frame)
                
                if not args.no_move:
                    mc.move(gaze_vector[0],gaze_vector[1])
                
                break
            
            if key_pressed == 27:
                break

        if frame_count > 0:
            logging.info(f"Face Detection:{1000* fd_infertime/frame_count:.1f}ms")
            logging.info(f"Facial Landmarks Detection:{1000* lm_infertime/frame_count:.1f}ms")
            logging.info(f"Headpose Estimation:{1000* hp_infertime/frame_count:.1f}ms")
            logging.info(f"Gaze Estimation:{1000* ge_infertime/frame_count:.1f}ms")

        feeder.close()
        cv2.destroyAllWindows()
    except Exception as ex:
        logging.exception(f"Error during inference:{str(ex)}")



def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)



if __name__ == '__main__':
    main()
