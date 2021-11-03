# Discrete3DOD

## Requirements
1. Python 3.7
2. Pytorch 1.9
3. OpenCV > 3.0
4. Tqdm
5. Cityscapesscripts (https://github.com/mcordts/cityscapesScripts)
6. PyQuaternion

## Configure the settings
Make changes to the settings [here](cfg/settings.py):
1. Change the dataset root (line 20)
2. Change weights name/location (line 75)
3. Change the location of the save folder for saving prediction results (line 76)

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
The results will be saved in folder saves/vis/# CS-3DOD
