# YOLO-6D-Pose Multi-Object 6D Pose Estimation Model
This repository is the official implementation of the paper ["**YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object
Keypoint Similarity Loss**"](https://arxiv.org/pdf/2204.06806.pdf). It contains YOLOX based models for Keypoint Detection / Human Pose estimation / 2D Pose estimation.

## Dataset format
This repository can support Keypoint Detection dataset as long as the dataset follows the [COCO Keypoint Detection](https://cocodataset.org/#keypoints-2017) dataset format - for more details refer to [person_keypoints_train2017.json and person_keypoints_val2017.json inside](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). The connections between the Keypoints (called Skeleton) are predefined in the dataset itself. 

## **Training: YOLOX-Pose**
The following training uses the COCO Keypoint Detection dataset. 

Train a model by running the command below. Pretrained ckpt for each model is the corresponding 2D object detection model trained on COCO dataset.

```
python -m  yolox.tools.train -n yolox_s_human_pose_ti_lite --task human_pose --dataset coco_keypts -c 'path to pretrained ckpt' -d 4 -b 64 --fp16 -o 
```

## **Exporting the ONNX Model**

```
python3 tools/export_onnx.py --output-name yolox_s_human_pose_ti_lite.onnx -n yolox_s_human_pose_ti_lite --task human_pose -c 'path to the trained ckpt' --export-det
```

## Dataset annotation
If you need to annotate a custom dataset, it can be done using the CVAT tool: https://www.cvat.ai/

Make sure that the format of json annotation file is similar to that of COCO Keypoint Detection. The object type, number of keypoints, name of keypoints, the connection between keypoints (skeleton) etc. can be changed.

python3 tools/export_onnx.py --output-name yolox_s_human_pose_ti_lite.onnx -n yolox_s_human_pose_ti_lite --task human_pose -c pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth

python -m tools.export_onnx --output-name yolox_s_pose_ti_lite_640_20220301_checkpoint.onnx -n yolox_s_human_pose_ti_lite --task human_pose -c yolox_s_pose_ti_lite_640_20220301_checkpoint.pth

python -m tools.demo image -n yolox_s_human_pose_ti_lite -c yolox_s_pose_ti_lite_640_20220301_checkpoint.pth --path icms.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --dataset coco_kpts --task human_pose

python -m tools.demo image -n yolox_s_human_pose_ti_lite -c yolox_s_pose_ti_lite_640_20220301_checkpoint.onnx --path icms.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --dataset coco_kpts --task human_pose

python -m tools.demo image -n yolox_s_human_pose_ti_lite -c yolox_s_pose_ti_lite_640_20220301_checkpoint.pth --path icms.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --dataset coco_kpts --task human_pose