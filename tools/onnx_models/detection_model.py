from time import time
from loguru import logger

import cv2
import numpy as np

from base import BaseModel

class DetectionModel(BaseModel): 
    def __init__(
        self, model_path, device, classes, apply_all_optim=True, nms_threshold=0.7, conf_threshold=0.5, use_min_ratio=True, class_agnostic=False) -> None:
        super(DetectionModel, self).__init__(model_path, apply_all_optim)
        self.classes = classes
        self.num_classes = len(self.classes)
        
        #Non-Maximum Suppression threshold
        self.nms_threshold = nms_threshold   
            
        # Confidence threhold
        self.conf_threshold = conf_threshold
        
        self.class_agnostic = class_agnostic
        
        # Preprocessing resize option
        self.use_min_ratio = use_min_ratio

        self.load_model(device)
        
        if len(self.output_shape) == 3:
            # the ONNX graph does not contain NMS block
            self.with_nms_block = False
            grids = []
            expanded_strides = []
            strides = [8, 16, 32]
            hsizes = [self.input_height // stride for stride in strides]
            wsizes = [self.input_width // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            self.grids = np.concatenate(grids, 1)
            self.expanded_strides = np.concatenate(expanded_strides, 1)
            logger.info("The model does not contain NMS block")
        else:
            self.with_nms_block = True
            logger.info("The model contains NMS block")
      
    def preprocess(self, img, swap=(2, 0, 1)):
        
        if self.use_min_ratio:
            input_size = (self.input_height, self.input_width)
            if len(img.shape) == 3:
                padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            else:
                padded_img = np.ones(input_size, dtype=np.uint8) * 114

            self.r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
            resized_img = cv2.resize(img[:, :, ::-1], (int(img.shape[1] * self.r), int(img.shape[0] * self.r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            padded_img[: int(img.shape[0] * self.r), : int(img.shape[1] * self.r)] = resized_img
        else:
            self.y_scale = img.shape[0] / self.input_height
            self.x_scale = img.shape[1] / self.input_width
            padded_img = cv2.resize(img[:, :, ::-1], (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        padded_img = padded_img.transpose(swap) #(from H, W, C to C, H, W)
        padded_img = np.ascontiguousarray(padded_img)
        if self.input_type == 'tensor(float16)':
            padded_img = padded_img.astype(np.float16)
        else:
            padded_img = padded_img.astype(np.float32)
            
        return padded_img

    def vis(self, boxes, scores, cls_ids):#, class_names=None):
        """Get bounding boxes from detection result

        Args:
            boxes: all detected bounding boxes
            scores: all detected scores
            cls_ids: all detected class ids

        Returns:  
            object_dict: each key is the class id, value is the list of object [x, y, w, h, conf]      
        """        
        object_dict = dict()
        
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = float(scores[i])
            if score < self.conf_threshold:
                continue
            x = float(box[0])
            y = float(box[1])
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            if cls_id not in object_dict.keys():
                object_dict[cls_id] = list()
            object_dict[cls_id].append((x, y, w, h, score))

        return object_dict

    def multiclass_nms_class_aware(self, boxes, scores):#, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-aware version."""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.conf_threshold
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores)#, self.nms_threshold)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def multiclass_nms_class_agnostic(self, boxes, scores):#, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > self.conf_threshold
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores)#, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def multiclass_nms(self, boxes, scores):
        """Multiclass NMS implemented in Numpy"""
        if self.class_agnostic:
            return self.multiclass_nms_class_agnostic(boxes, scores)
        return self.multiclass_nms_class_aware(boxes, scores)

    def nms(self, bounding_boxes, confidence_score):#, threshold):
        # If no bounding boxes, return an empty list
        if len(bounding_boxes) == 0:
            return []#, []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # Coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)
        
        # Picked indices
        indices = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of the largest confidence score
            index = order[-1]

            # Pick the bounding box with the largest confidence score
            indices.append(index)

            # Compute ordinates of intersection-over-union(IOU)
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

        return indices#picked_boxes, picked_score
    
    def postprocess_without_nms_block(self, dets):#, ratio):
        """Postprocess
        Returns:
            human_bboxes (xmin, ymin, xmax, ymax, score)
            hand_on_bboxes (xmin, ymin, xmax, ymax, score)
            hand_off_bboxes (xmin, ymin, xmax, ymax, score)
            face_bboxes (xmin, ymin, xmax, ymax, score)
            phone_bboxes (xmin, ymin, xmax, ymax, score)

        """ 
        dets[..., :2] = (dets[..., :2] + self.grids) * self.expanded_strides
        dets[..., 2:4] = np.exp(dets[..., 2:4]) * self.expanded_strides

        dets = dets[0]
                
        object_dict = None
        
        if dets is not None:
            det_bboxes, det_scores = dets[:, :4], dets[:, 4:5] * dets[:, 5:]
                        
            boxes_xyxy = np.ones_like(det_bboxes)
            boxes_xyxy[:, 0] = det_bboxes[:, 0] - det_bboxes[:, 2]/2.
            boxes_xyxy[:, 1] = det_bboxes[:, 1] - det_bboxes[:, 3]/2.
            boxes_xyxy[:, 2] = det_bboxes[:, 0] + det_bboxes[:, 2]/2.
            boxes_xyxy[:, 3] = det_bboxes[:, 1] + det_bboxes[:, 3]/2.
            
            if self.use_min_ratio:
                boxes_xyxy /= self.r#ratio
            else:
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] * self.x_scale
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] * self.y_scale
               
            # Perform NMS
            nms_dets = self.multiclass_nms(boxes_xyxy, det_scores)
            
            if nms_dets is not None:
                final_boxes, final_scores, final_cls_inds = nms_dets[:, :4], nms_dets[:, 4], nms_dets[:, 5]                
                object_dict = self.vis(final_boxes, final_scores, final_cls_inds)
                         
        return object_dict
    
    def postprocess_with_nms_block(self, dets):#, ratio):
        if self.use_min_ratio:
            dets[:, :4] /= self.r#ratio
        else:
            dets[:, [0, 2]] = dets[:, [0, 2]] * self.x_scale
            dets[:, [1, 3]] = dets[:, [1, 3]] * self.x_scale
        
        object_dict = None

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            object_dict = self.vis(final_boxes, final_scores, final_cls_inds)
        
        return object_dict   
    
    def pipeline(self, input_image):
        """Detection pipeline
        Returns:
            object_dict: key is class id, value is a list of object that belongs to class "key"
        """ 
        start_time = time()
        input = self.preprocess(input_image)  # preprocess raw image
        dets = self.predict(input[None, :, :, :])[0]  # prediction

        if self.with_nms_block:
            object_dict = self.postprocess_with_nms_block(dets.astype(np.float32))#, ratio)  # output visualization
        else:
            object_dict = self.postprocess_without_nms_block(dets.astype(np.float32))#, ratio)

        self.inference_time += time() - start_time
        return object_dict
