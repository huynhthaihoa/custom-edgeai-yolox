from time import time

import cv2
import numpy as np

# from configs.cabin_configs import resolution
from base import BaseModel

def cxcywh2xyxy(cx,cy,w,h):
    #This function is used while exporting ONNX models
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    halfw = w/2
    halfh = h/2
    xmin = cx - halfw  # top left x
    ymin = cy - halfh  # top left y
    xmax = cx + halfw  # bottom right x
    ymax = cy + halfh  # bottom right y
    return np.stack((xmin, ymin, xmax, ymax), axis=-1)

class PoseModel(BaseModel):
    def __init__(
        self, model_path, device, nms_threshold=0.5, conf_threshold = 0.65, kpt_conf_threshold = 0.3) -> None:
        super(PoseModel, self).__init__(model_path)
        self._CLASS_COLOR_MAP = [
            (0, 0, 255) , # Person (blue).
            (255, 0, 0) ,  # Bear (red).
            (0, 255, 0) ,  # Tree (lime).
            (255, 0, 255) ,  # Bird (fuchsia).
            (0, 255, 255) ,  # Sky (aqua).
            (255, 255, 0) ,  # Cat (yellow).
        ]
        
        self.palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.pose_limb_color = self.palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.pose_kpt_color = self.palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.radius = 5

        # recommend to save these values to config file
        
        #Non-Maximum Suppression threshold
        self.nms_threshold = nms_threshold
        
        # Confidence threhold
        self.conf_threshold = conf_threshold
        
        # Keypoint confidence threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        
        # Image resolution (width, height)
        # self.resolution = resolution
        
        # To process keypoints
        self.steps = 3

        self.load_model(device)
        
        if len(self.output_shape) == 3: #(1, 6300, 57)
            # the ONNX graph does not contain NMS block
            self.with_nms_block = False
            
            strides = [8, 16, 32]
            
            hsizes = [self.input_height // stride for stride in strides]
            wsizes = [self.input_width // stride for stride in strides]

            grids = []
            expanded_strides = []
            
            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            self.expanded_strides = np.concatenate(expanded_strides, 1)#.type(dtype)

            self.expanded_grids = np.concatenate(grids, 1)#.type(dtype)
            
            kpt_conf_grids = np.zeros_like(self.expanded_grids)[...,0:1]
            kpt_grids = np.concatenate((self.expanded_grids, kpt_conf_grids), 2)
            self.kpt_grids_repeated = np.repeat(kpt_grids, 17, 2)#.squeeze()
            # self.kpt_grids_repeated = self.kpt_grids_repeated.squeeze()
            # print("kpt grids:", self.kpt_grids_repeated.shape)
            # exit()

        else: #(GatherDetection_dim_0, 57)
            self.with_nms_block = True
        
        print("With NMS block:", self.with_nms_block)

    def preprocess(self, raw_frame):
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
            - occupant_kpt_coords: list of 17 2d coordinates of 17 person's keypoints (if any keyoint is invalid, each value is None)
            - the bounding box of the occupant's face in the form of [xmin, ymin, xmax, ymax]
        """        
        num_kpts = len(person_kpts) // self.steps
        
        occupant_kpt_coords = [None] * num_kpts
        
        # for face
        face_xmin = self.resolution[0]
        face_xmax = -1
        face_ymin = self.resolution[1]
        face_ymax = -1
        face_size = 0
        
        # process keypoints
        for kid in range(num_kpts):
                            
            x_coord, y_coord = int(person_kpts[self.steps * kid]), int(person_kpts[self.steps * kid + 1])
            occupant_kpt_coords[kid] = (x_coord, y_coord, person_kpts[self.steps * kid + 2]) #save the keypoint 2d coordinate regardless of confidence score
                            
            # for face
            if kid < 5:
                if face_xmin > x_coord:
                    face_xmin = x_coord
                if face_ymin > y_coord:
                    face_ymin = y_coord            
                if face_xmax < x_coord:
                    face_xmax = x_coord
                if face_ymax < y_coord:
                    face_ymax = y_coord 
                           
        if face_xmax != -1 and face_ymax != -1 and face_xmin != self.resolution[0] and face_ymin != self.resolution[1]:
            face_size = min(max(face_xmax - face_xmin, face_ymax - face_ymin) * 2, 120)
            if face_size != 0:
                face_xmin = (face_xmin + face_xmax - face_size) // 2
                face_ymin = (face_ymin + face_ymax - face_size) // 2 
                
        return occupant_kpt_coords, [face_xmin, face_ymin, face_xmin + face_size, face_ymin + face_size]
          
    def postprocess_with_nms_block(self, output, original_img):
        """Postprocessing (in case ONNX graph contains NMS block)

        Args:
            output: raw output of the model
            original_img: original input image

        Returns:
            human_bboxes: list of bounding boxes of every detected occupants
            human_kpts: list of keypoints of every detected occupant
            face_bboxes: list of faces of every detected occupant
        """

        orig_height, orig_width = original_img.shape[:2]

        # Compute scale factors for width and height
        y_scale = orig_height / self.input_height
        x_scale = orig_width / self.input_width
        
        det_bboxes, det_scores, _, det_kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
        
        # Perform NMS and extract the bounding boxes and scores after NMS
        nms_indices = self.nms(det_bboxes, det_scores)
        
        # Filter detections based on NMS results
        det_bboxes = det_bboxes[nms_indices]
        det_scores = det_scores[nms_indices]
        det_kpts = det_kpts[nms_indices]

        # Scale bounding box coordinates and keypoints back to original image size
        det_bboxes[:, [0, 2]] = det_bboxes[:, [0, 2]] * x_scale
        det_bboxes[:, [1, 3]] = det_bboxes[:, [1, 3]] * y_scale
        
        det_kpts[:, 0::3] = det_kpts[:, 0::3] * x_scale  # x coordinates of keypoints
        det_kpts[:, 1::3] = det_kpts[:, 1::3] * y_scale  # y coordinates of keypoints

        face_bboxes = []#, []
        human_kpts = []
        human_bboxes = []
        
        for idx in range(len(det_bboxes)):
            # process each detected person's keypoints
            det_score = det_scores[idx]
            if det_score > self.conf_threshold:
                det_bbox = det_bboxes[idx]
                occupant_kpts = det_kpts[idx]
                print(np.min(occupant_kpts), np.max(occupant_kpts))
                
                x1 = int(det_bbox[0])
                y1 = int(det_bbox[1])
                x2 = int(det_bbox[2])
                y2 = int(det_bbox[3])

                # Draw keypoints and skeleton on the image
                valid_occupant_kpts, valid_occupant_face = self.plot_keypoints(occupant_kpts)
                human_kpts.append(valid_occupant_kpts)
                human_bboxes.append((x1, y1, x2, y2, det_score))
                face_bboxes.append((valid_occupant_face[0], valid_occupant_face[1], valid_occupant_face[2], valid_occupant_face[3]))#, det_score))
            
        return human_bboxes, human_kpts, face_bboxes
        
    def postprocess_without_nms_block(self, prediction, original_img):
        orig_height, orig_width = original_img.shape[:2]

        # Compute scale factors for width and height
        y_scale = orig_height / self.input_height
        x_scale = orig_width / self.input_width

        prediction[..., :2] = (prediction[..., :2] + self.expanded_grids) * self.expanded_strides
        prediction[..., 2:4] = np.exp(prediction[..., 2:4]) * self.expanded_strides
        prediction[...,  6:] = (2 * prediction[..., 6:] - 0.5 + self.kpt_grids_repeated) * self.expanded_strides

        box_corner = np.zeros(prediction.shape, dtype=prediction.dtype)
        
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        prediction = prediction[0]

        # Get score and class with highest confidence
        class_conf = np.max(prediction[:, 5: 5 + 1], axis=1, keepdims=True)

        # Confidence filtering
        conf_mask = (prediction[:, 4] * class_conf.squeeze() >= self.conf_threshold).squeeze()
        
        # prediction = prediction[conf_mask]
        # class_conf = class_conf[conf_mask]
        
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((prediction[:, :4], class_conf, prediction[:, 6:]), 1)
        detections = detections[conf_mask]
        
        # Apply NMS            
        nms_out_index = self.nms(detections[:, :4], detections[:, 4])
        
        detections = detections[nms_out_index]
        
        # print("shape:", np.shape(detections))
        
        det_bboxes = detections[:, :4] #(N, 4)
        det_scores = detections[4] #(N, 1)
        det_kpts = detections[:, 5:]#.reshape((-1, 51)) #(N, 51)
        
        # Scale bounding box coordinates and keypoints back to original image size
        # det_bboxes[:, [0, 2]] = det_bboxes[:, [0, 2]] * x_scale
        # det_bboxes[:, [1, 3]] = det_bboxes[:, [1, 3]] * y_scale
        
        # det_kpts[:, 0::3] = det_kpts[:, 0::3] * x_scale  # x coordinates of keypoints
        # det_kpts[:, 1::3] = det_kpts[:, 1::3] * y_scale  # y coordinates of keypoints
            
        face_bboxes = []
        human_kpts = []
        human_bboxes = []
        
        for idx in range(len(det_bboxes)):
            # process each detected person's keypoints
            # det_score = det_scores[idx]
            det_bbox = det_bboxes[idx]
            
            x1 = int(det_bbox[0])
            y1 = int(det_bbox[1])
            x2 = int(det_bbox[2])
            y2 = int(det_bbox[3])

            # Draw keypoints and skeleton on the image
            valid_occupant_kpts, valid_occupant_face = self.plot_keypoints(det_kpts[idx])
            human_kpts.append(valid_occupant_kpts)
            human_bboxes.append((x1, y1, x2, y2, det_scores[idx]))
            face_bboxes.append((valid_occupant_face[0], valid_occupant_face[1], valid_occupant_face[2], valid_occupant_face[3]))
            
        return human_bboxes, human_kpts, face_bboxes
        
    def pipeline(self, input_image, name=None):
        """YOLOX body keypoint detection model pipeline

        Args:
            - input_image: input image
            - name (_type_, optional): _description_. Defaults to None.

        Returns:
            - human_bboxes: list of detected people's bounding box centroids for occupany detection ((x, y) for each centroid)
            - human_kpts: list of detected people's keypoint 2d coordinates (17 keypoints of each person, [x, y] of each keypoint)
            - face_bboxes: list of detected people's faces ([xmin, ymin, xmax, ymax] of each face)
        """        
        start_time = time()
        
        input = self.preprocess(input_image)  # preprocess raw image
        output = self.predict(input)  # prediction

        if self.with_nms_block:
            human_bboxes, human_kpts, face_bboxes = self.postprocess_with_nms_block(output, input_image)
        else:
            human_bboxes, human_kpts, face_bboxes = self.postprocess_without_nms_block(output, input_image)  # output visualization

        self.inference_time += time() - start_time

        return human_bboxes, human_kpts, face_bboxes
    
    # def postprocess_without_nms_block(self, prediction, original_img):
    #     """Postprocessing (in case ONNX graph does not contain NMS block)

    #     Args:
    #         output: raw output of the model
    #         original_img: original input image

    #     Returns:
    #         human_bboxes: list of bounding boxes of every detected occupants
    #         human_kpts: list of keypoints of every detected occupant
    #         face_bboxes: list of faces of every detected occupant
    #     """

    #     orig_height, orig_width = original_img.shape[:2]

    #     # Compute scale factors for width and height
    #     y_scale = orig_height / self.input_height
    #     x_scale = orig_width / self.input_width
        
    #     prediction[..., :2] = (prediction[..., :2] + self.expanded_grids) * self.expanded_strides
    #     prediction[..., 2:4] = np.exp(prediction[..., 2:4]) * self.expanded_strides
    #     # prediction[...,  6:] = (2 * prediction[..., 6:] - 0.5 + self.kpt_grids_repeated) * self.expanded_strides
        
    #     # print(np.shape(prediction[..., 6:]), np.shape(self.kpt_grids_repeated), np.shape(self.expanded_strides))
    #     # exit(0)
        
    #     prediction = prediction[0]
       
    #     cx, cy, w, h = prediction[..., 0], prediction[..., 1], prediction[..., 2], prediction[..., 3]
        
    #     raw_bboxes = cxcywh2xyxy(cx, cy, w, h)
    #     raw_scores = np.asarray(prediction[..., 4])
    #     raw_kpts = np.asarray(prediction[..., 6:])
        
    #     # confidence threshold filter
    #     conf_indices = raw_scores >= self.conf_threshold
    #     det_bboxes = raw_bboxes[conf_indices]
    #     det_scores = raw_scores[conf_indices]
    #     det_kpts = raw_kpts[conf_indices]            

    #     # Perform NMS and extract the bounding boxes and scores after NMS
    #     nms_indices = self.nms(det_bboxes, det_scores)   
    #     det_bboxes = det_bboxes[nms_indices] 
    #     det_scores = det_scores[nms_indices]  
    #     det_kpts = det_kpts[nms_indices]  
    #     print(np.shape(det_kpts))     

    #     # Scale bounding box coordinates and keypoints back to original image size
    #     det_bboxes[:, [0, 2]] = det_bboxes[:, [0, 2]] * x_scale
    #     det_bboxes[:, [1, 3]] = det_bboxes[:, [1, 3]] * y_scale
        
    #     det_kpts[:, 0::3] = det_kpts[:, 0::3] * x_scale  # x coordinates of keypoints
    #     det_kpts[:, 1::3] = det_kpts[:, 1::3] * y_scale  # y coordinates of keypoints
            
    #     face_bboxes = []
    #     human_kpts = []
    #     human_bboxes = []
        
    #     for idx in range(len(det_bboxes)):
    #         # process each detected person's keypoints
    #         det_score = det_scores[idx]
    #         if det_score > self.conf_threshold:
    #             det_bbox = det_bboxes[idx]
    #             occupant_kpts = det_kpts[idx] #* 640
    #             print(np.min(occupant_kpts), np.max(occupant_kpts))
                
    #             x1 = int(det_bbox[0])
    #             y1 = int(det_bbox[1])
    #             x2 = int(det_bbox[2])
    #             y2 = int(det_bbox[3])

    #             # Draw keypoints and skeleton on the image
    #             valid_occupant_kpts, valid_occupant_face = self.plot_keypoints(occupant_kpts)
    #             human_kpts.append(valid_occupant_kpts)
    #             human_bboxes.append((x1, y1, x2, y2, det_score))
    #             face_bboxes.append((valid_occupant_face[0], valid_occupant_face[1], valid_occupant_face[2], valid_occupant_face[3]))
            
    #     return human_bboxes, human_kpts, face_bboxes