# Discrete3DOD

Official repository for IEEE Transaction on Intelligent Transportation System, entitled ["One Stage Monocular 3D Object Detection Utilizing Discrete Depth and Orientation Representation."](https://ieeexplore.ieee.org/abstract/document/9780191)

## Requirements
1. Python 3.7
2. Pytorch 1.9
3. OpenCV > 3.0
4. Tqdm
5. Cityscapesscripts (https://github.com/mcordts/cityscapesScripts)
6. PyQuaternion

## Configure the settings
Make changes to the settings [here](cfg/settings.py):
1. Change the dataset root
2. Change weights name/location
3. Change the location of the save folder for saving prediction results
4. Download our pretrained weights [here](https://drive.google.com/file/d/1daQsTJXrEGrW7ckDk6UmYc-2lTZgQ0Gr/view?usp=sharing)

## Running Evaluation
To run the cityscapes benchmark simply 
```
python evaluate.py
```
The results will be saved in saves/results/results.json

## Visualize Results
```
python gen_vis.py
```
The results will be saved in folder saves/vis/
