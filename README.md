<h1>Udacity. Intel Edge Computing Nanodegree: Computer Pointer Controller Project</h1>

Control mouse cursor by gaze estimation using Intel OpenVino models 

| Details            |              |
|-----------------------|---------------|
| Programming Language: | Python 3.7.4 |
| OpenVINO Version: | 2020.2 |
| OpenVINO Models: |face-detection-adas-binary-0001   <br /><br />landmarks-regression-retail-0009 <br /><br /> head-pose-estimation-adas-0001 <br /><br />gaze-estimation-adas-0002|
| Hardware: | 1.7 GHz Dual-Core Intel Core i7                              |
| OS: | MacOS Catalina 10.15.1 |



## Project Set Up and Installation

### Directory Structure
```
$tree
.
├── README.md
├── debug.log
├── inputs
│   ├── demo.mp4
│   └── demo.png
├── models
│   └── intel
│       ├── face-detection-adas-binary-0001
│       ├── gaze-estimation-adas-0002
│       │   └── FP32
│       │       ├── gaze-estimation-adas-0002.bin
│       │       └── gaze-estimation-adas-0002.xml
│       ├── head-pose-estimation-adas-0001
│       │   └── FP32
│       │       ├── head-pose-estimation-adas-0001.bin
│       │       └── head-pose-estimation-adas-0001.xml
│       └── landmarks-regression-retail-0009
│           └── FP32
│               ├── landmarks-regression-retail-0009.bin
│               └── landmarks-regression-retail-0009.xml
├── requirements.txt
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── inference.py
    ├── input_feeder.py
    ├── main.py
    ├── model.py
    ├── mouse_controller.py
    └── test.py

11 directories, 21 files
```



## Project Setup
 Clone the repo using git clone. Then install  dependecies using this pip
```
pip3 install -r requirements.txt
```
Using OpenVINO model downloader: *~/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py* download all necesary IR models. In this case only FP32 precision. 
1. face-detection-adas-binary-0001
2. landmarks-regression-retail-0009 
3. head-pose-estimation-adas-0001 
4. gaze-estimation-adas-0002

## How to run demo: 
```
$ python3 src/main.py -fdm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -lmm models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hpm models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -gem models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i inputs/openvino_demo.mp4 --print --no_move
```

## Documentation

```
usage: main.py [-h] -fdm FDMODEL -hpm HPMODEL -lmm LMMODEL -gem GEMODEL -i
               INPUT [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD]
               [--print] [--no_move] [--no_video]
optional arguments:
  -h, --help            show this help message and exit
  -fdm FDMODEL, --fdmodel FDMODEL
                        Path to a face detection xml file with a trained
                        model.
  -hpm HPMODEL, --hpmodel HPMODEL
                        Path to a head pose estimation xml file with a trained
                        model.
  -lmm LMMODEL, --lmmodel LMMODEL
                        Path to a facial landmarks xml file with a trained
                        model.
  -gem GEMODEL, --gemodel GEMODEL
                        Path to a gaze estimation xml file with a trained
                        model.
  -i INPUT, --input INPUT
                        Path video file or CAM to use camera
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.5 by
                        default)
  --print               Overlay inference output over frame
  --no_move             Don't move mouse based on gaze estimation output
  --no_video            Don't show video window
```

## Performance

### DEVICE: CPU
|        | Face Detection   | Landmarks Detetion        | Headpose Estimation | Gaze Estimation |
|--------------------|---------------|-----------|-------------|-----------|
|Load Time FP32      |  223ms      | 101ms    | 102ms     | 128ms    |
|Inference Time FP32 | 7.5ms        | 0.45ms    | 1ms       | 1.1ms    |


