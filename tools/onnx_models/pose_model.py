from time import time
from loguru import logger

import cv2
import numpy as np
import torch
import torchvision

# from configs.cabin_configs import resolution
from base import BaseModel

class PoseModel(BaseModel):
    def __init__(
        self, model_path, device, nms_threshold=0.5, conf_threshold = 0.65, kpt_conf_threshold = 0.3, use_min_ratio=True, base_face_size=120, resolution=(640, 480)) -> None:
        super(PoseModel, self).__init__(model_path)

        # recommend to save these values to config file
        
        #Non-Maximum Suppression threshold
        self.nms_threshold = nms_threshold
        
        # Confidence threhold
        self.conf_threshold = conf_threshold
        
        # Keypoint confidence threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        
        # Preprocessing resize option
        self.use_min_ratio = use_min_ratio
        
        # Image resolution (width, height)
        self.resolution = resolution
        
        # To process keypoints
        self.steps = 3
        
        # Base face size
        self.base_face_size = base_face_size

        self.load_model(device)
        
        if len(self.output_shape) == 3: #(1, 6300, 57)
            # the ONNX graph does not contain NMS block
            self.with_nms_block = False
            
            strides = [8, 16, 32]
            
            hsizes = [self.input_height // stride for stride in strides]
            wsizes = [self.input_width // stride for stride in strides]

            # Pytorch init
            torch_grids = list()
            torch_strides = list()
            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                torch_grids.append(grid)
                shape = grid.shape[:2]
                torch_strides.append(torch.full((*shape, 1), stride))
            self.torch_grids = torch.cat(torch_grids, dim=1)#.type(dtype)
            torch_kpt_conf_grids = torch.zeros_like(self.torch_grids)[...,0:1]
            torch_kpt_grids = torch.cat((self.torch_grids, torch_kpt_conf_grids), dim=2)
            self.torch_strides = torch.cat(torch_strides, dim=1)#.type(dtype)
            self.torch_kpt_grids_repeat = torch_kpt_grids.repeat(1, 1, 17)
            logger.info("The model does not contain NMS block")
        else: #(GatherDetection_dim_0, 57)
            self.with_nms_block = True
            logger.info("The model contains NMS block")
        
    def preprocess(self, raw_frame):
        
        if self.use_min_ratio:
            img = np.ones((self.input_height, self.input_width, 3), dtype=np.uint8) * 114
            self.r = min(self.input_height / raw_frame.shape[0], self.input_width / raw_frame.shape[1])
            resized_img = cv2.resize(raw_frame[:, :, ::-1], (int(raw_frame.shape[1] * self.r), int(raw_frame.shape[0] * self.r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            img[: int(raw_frame.shape[0] * self.r), : int(raw_frame.shape[1] * self.r)] = resized_img
        else:
            self.y_scale = raw_frame.shape[0] / self.input_height
            self.x_scale = raw_frame.shape[1] / self.input_width
            img = cv2.resize(raw_frame[:, :, ::-1], (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img,0)
        img = img.transpose(0,3,1,2)
        return img

    def predict(self, preprocessed_frame):
        return self.model.run(
            self.output_names, {self.input_names[0]: preprocessed_frame}
        )[0]
    
    def nms(self, boxes, scores):#, nms_threshold):
        """Non-maximum suppression using detected bounding boxes and corresponding scores

        Args:
            boxes: detected object bounding boxes
            scores: detected object confidence scores
            nms_threshold: Non-Maximum Suppression (NMS) threshold
        Returns:
            indices: indices of remaining objects after NMS
        """        

        indices = list()
        
        if len(boxes) == 0:
            return []#, []

        # Coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(scores)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of the largest confidence score
            index = order[-1]

            # Pick the bounding box with the largest confidence score
            indices.append(index)

            # Compute coordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0, x2 - x1 + 1)
            h = np.maximum(0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < self.nms_threshold)
            order = order[left]

        return indices

    def plot_keypoints(self, person_kpts):
        """Convert raw keypoint prediction (float dtype) into body keypoints (integer dtype) + faces (integer dtype)

        Args:
            person_kpts: raw keypoint prediction

        Returns:
            - occupant_kpts: list of 51 elements, each triple is (x, y, conf) of each corresponding keypoint
        """        
        num_kpts = len(person_kpts) // self.steps
        
        occupant_kpts = list()
        
        # process keypoints
        for kid in range(num_kpts):
                            
            occupant_kpts.append(float(person_kpts[self.steps * kid]))
            occupant_kpts.append(float(person_kpts[self.steps * kid + 1]))
            occupant_kpts.append(float(person_kpts[self.steps * kid + 2]))
                                            
        return occupant_kpts
          
    def postprocess_with_nms_block(self, output):
        """Postprocessing (in case ONNX graph contains NMS block)

        Args:
            output: raw output of the model
            original_img: original input image

        Returns:
            human_bboxes: list of bounding boxes of every detected occupants
            human_kpts: list of keypoints of every detected occupant
            face_bboxes: list of faces of every detected occupant
        """        
        det_bboxes, det_scores, _, det_kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
        
        # # Perform NMS and extract the bounding boxes and scores after NMS
        nms_indices = self.nms(det_bboxes, det_scores)
        
        # # Filter detections based on NMS results
        det_bboxes = det_bboxes[nms_indices]
        det_scores = det_scores[nms_indices]
        det_kpts = det_kpts[nms_indices]

        # option 1: follow YOLOX scale
        if self.use_min_ratio:
            det_bboxes[:, :4] /= self.r
            det_kpts[:, 0::3] /= self.r
            det_kpts[:, 1::3] /= self.r
        else:
            # option 2: original
            # Scale bounding box coordinates and keypoints back to original image size
            det_bboxes[:, [0, 2]] = det_bboxes[:, [0, 2]] * self.x_scale
            det_bboxes[:, [1, 3]] = det_bboxes[:, [1, 3]] * self.y_scale
            det_kpts[:, 0::3] = det_kpts[:, 0::3] * self.x_scale  # x coordinates of keypoints
            det_kpts[:, 1::3] = det_kpts[:, 1::3] * self.y_scale  # y coordinates of keypoints

        human_infos = list()
        
        for idx in range(len(det_bboxes)):
            # process each detected person's keypoints
            det_score = float(det_scores[idx])
            if det_score > self.conf_threshold:
                det_bbox = det_bboxes[idx]
                occupant_kpts = det_kpts[idx]
                
                x = float(det_bbox[0])
                y = float(det_bbox[1])
                w = float(det_bbox[2] - det_bbox[0])
                h = float(det_bbox[3] - det_bbox[1])

                # Draw keypoints and skeleton on the image
                valid_occupant_kpts = self.plot_keypoints(occupant_kpts)
                human_infos.append((x, y, w, h, det_score, valid_occupant_kpts))
            
        return human_infos
    
    def postprocess_without_nms_block(self, prediction):
        
        prediction = torch.from_numpy(prediction)
  
        prediction[..., :2] = (prediction[..., :2] + self.torch_grids) * self.torch_strides
        prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * self.torch_strides
        
        prediction[...,  6:] = (2*prediction[..., 6:] - 0.5  + self.torch_kpt_grids_repeat) * self.torch_strides  
              

        box_corner = prediction.new(prediction.shape)
        
        #convert cxcywh to xyxy
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        human_infos = list()
        
        output = [None for _ in range(len(prediction))]
        
        for i, pred in enumerate(prediction):
            if not pred.shape[0]:
                continue
            
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(pred[:, 5: 5 + 1], 1, keepdim=True)
            
            # Confidence filtering
            conf_mask = (pred[:, 4] * class_conf.squeeze() >= self.conf_threshold).squeeze()
            
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, kpts)
            detections = torch.cat((pred[:, :5], class_conf, class_pred.float(), pred[:, 6:]), 1)
            
            detections = detections[conf_mask]
            
            nms_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                self.nms_threshold,
            )
            
            detections = detections[nms_index]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        
        # output = np.asarray(output)
        output = output[0].numpy()
        
        for sample in output:
            det_bbox = sample[:4]
            det_score = float(sample[4] * sample[5])
            det_kpts = np.array(sample[7:]).reshape((17, 3))
            
            # option 1: follow YOLOX scale method
            if self.use_min_ratio:
                det_bbox /= self.r
                det_kpts[:, :2] /= self.r
            else:
                # option 2: original
                # Scale bounding box coordinates and keypoints back to original image size
                det_bbox[[0, 2]] = det_bbox[[0, 2]] * self.x_scale
                det_bbox[[1, 3]] = det_bbox[[1, 3]] * self.y_scale
                
                det_kpts[:, 0] = det_kpts[:, 0] * self.x_scale  # x coordinates of keypoints
                det_kpts[:, 1] = det_kpts[:, 1] * self.y_scale  # y coordinates of keypoints
            
            valid_occupant_kpts = list()
            
            for i in range(17):
                valid_occupant_kpts.append(float(det_kpts[i][0]))
                valid_occupant_kpts.append(float(det_kpts[i][1]))
                valid_occupant_kpts.append(float(det_kpts[i][2]))
            
            x = float(det_bbox[0])
            y = float(det_bbox[1])
            w = float(det_bbox[2] - det_bbox[0])
            h = float(det_bbox[3] - det_bbox[1])
                        
            human_infos.append((x, y, w, h, det_score, valid_occupant_kpts))
            
        return human_infos
        
    def pipeline(self, input_image):
        """YOLOX body keypoint detection model pipeline

        Args:
            - input_image: input image

        Returns:
            - human_info: list including each "detected human info" (x, y,w, h, conf, 51 elems of kpts)
        """        
        start_time = time()
        
        input = self.preprocess(input_image)  # preprocess raw image
        output = self.predict(input)  # prediction

        if self.with_nms_block:
            human_infos = self.postprocess_with_nms_block(output)
        else:
            human_infos = self.postprocess_without_nms_block(output)

        self.inference_time += time() - start_time

        return human_infos