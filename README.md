# InfraScan
You Only Look Once version 4 (YOLOv4) a single-stage object detector to identify structural damages in bridges.
In this work you will find the implementation of the object detector YOLOv4 by [AlexeyAB](https://github.com/AlexeyAB/darknet). 
The damage detection algorithm (YOLO/InfraScan.ipynb) you can compile and run it on Google Colaboratory in the cloud, it offers free GPU access. 
We suggest you to follow the tutorial: https://www.youtube.com/watch?v=mKAEGSxwOAY for the proper network configuration. 
* The pre-trained weights for the crack and spalling detector can be found in [yolo weights](https://drive.google.com/drive/folders/19FldBYAhNH2Cva6ZGHMLye8VIQAX1Ydv?usp=sharing) 

Moreover, this repository contains the development of scripts to perform the analysis of improvements to the model. 
Among the improvements we reproduced a transfer learning method inspired by the results of C. Zhang et al. in https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12500. 

The file in TL analysis/COCO_data_extract.py gathers all the COCO dataset and computes the statistics. This step needs the package gluoncv. 
You can avoid this step by only running the TL analysis/COCO_analysis.py file. All the statistics from the COCO dataset have been recollected in the file TL analysis/COCO_stats.csv.

Finally, with the Data Augmentation/dataAug.ipynb file you can perform data augmentation in Google Colaboratory from your personal Google Drive images.