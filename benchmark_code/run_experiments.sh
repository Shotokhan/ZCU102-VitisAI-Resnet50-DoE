#!/bin/bash
for i in {1..50}
do
    ./resnet50 my_model --num-threads 1 --no-postprocess --base-image-path ./final_images/images_100/
    # ./resnet50_no_exec my_model --num-threads 1 --no-postprocess --base-image-path ./final_images/images_100/
done

for i in {1..50}
do
    ./resnet50 my_model --num-threads 2 --no-postprocess --diff-path-for-thread
    # ./resnet50_no_exec my_model --num-threads 2 --no-postprocess --diff-path-for-thread
done
