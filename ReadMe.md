# Los Balordos Bros. Presents: Regrasping

This is the best repo ever. NO DOUBTS ABOUT IT.

## Explanation
This repo is part of a more complex structure that aims at picking objects (bottles and cups) from a box and inserting them in specified supports.
The second step is the regrasping, that relies on a former detection and a latter segmentation of the objects extracted at the first step of the procedure. 

## Useful links
The most straightforward thing to do is to install the ```ultralytics``` package with:
```
pip install ultralytics
```
if you do not want to contribute to the development of YOLO. Otherwise, if you feel your contribution may be helpful for the community:
```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e '.[dev]'
```
The link to the ```ultralytics``` repository for the detection stage is the following:

- https://github.com/ultralytics/ultralytics?tab=readme-ov-file

Moreover, look at the following link for some useful tutorials to investigate the potential of YOLO:
-  https://www.youtube.com/watch?v=5ku7npMrW40

## Requirements
After cloning the repository, the creation of a virtual environment (```conda``` should be avoided) is suggested:
```
python3 -m venv my_venv
``` 
The version of the python interpreter that surely works is 3.8.10, while more recent versions are not guaranteed to work with the packages in the file ```requirements.txt```, that can easily be installed with

```
pip install -r requirements.txt
``` 

For the segmentation pipeline, the file ```segmentation.py``` uses SAM2, which can be installed here:

- https://github.com/facebookresearch/sam2/tree/main

In the current version, SAM2 is installed locally, hence the file structure of the repo should be as follows:

├── Images
├── segment_anything_2
├── segmentation.py
├── YOLO_simple_demo.py
└── ..

TO achieve this structure, clone this repo 

```
git clone https://github.com/ChristianCella/Regrasping_balordo.git
git checkout dev_segmentation
```

clone SAM2 in the ```Regrasping_balordo``` repository and change the name of the ```sam2``` folder to ```segment_anything_2```. Move into the folder and install SAM2 and its dependencies.

```
cd segment_anything_2
pip install -e .
```

Future plans are to use SAM2 from the ultralytics model library.

## Run the segmentation pipeline
To run the segmentation, execute ```segmentation.py```. To change the example image, modify the ```IMAGE_PATH``` parameter inside the script. Note that, for testing purpose, segmentation works only if label **39 (bottle)** is detected by the YOLO model. 