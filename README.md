# Hazmat-detection
<div align="center">
    <a href="./">
        <img src=".\figure\sample.png" width="90%"/>
    </a>
</div>

## Abstract
This is an implementation of Yolov7 for camera in an easy way.
The code base and part of the resources come from the [official YOLOv7 github](https://github.com/mrl-amrl/DeepHAZMAT.git). For more information see this website. 

Detects labels:
- exolosives
- blasting agents
- flammable gas
- non flammable gas
- oxygen
- fuel oil
- dangerous when wet
- flammable solid
- spontaneously combustible
- oxidizer
- organic peroxide
- inhalation hazard
- poison
- radioactive
- corrosive

### Dataset
I used a public dataset, that it find in Roboflow Universe.

## Installation
```
git clone https://github.com/Abraham1501/Hazmat-detection.git
cd Hazmat-detection
pip install -r requirements.txt
```
To use the graphics card you need nvidia cuda installation in case you use a nivida graphics card. For download click [here](https://developer.nvidia.com/cuda-downloads).

In the *main_Yolo7.py* it can configure the operation of Yolov7 and others parameters, recommend see this before using it.
- Sample camera
```
python test_cam.py
```
