#!/bin/bash

cd ..
 
./darknet detector train cfg/coco_edge.data cfg/yolov2_c_4.cfg backup_coco_edge/yolov2_c_4_backup -gpus 0

