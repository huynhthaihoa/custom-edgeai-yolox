import numpy as np
import onnxruntime

def softmax(x, axis=-1, keepdims=True):
	e_x = np.exp(x - np.max(x))  # Subtract max(x) to stabilize the computation
	return e_x / e_x.sum(axis=axis, keepdims=keepdims)

class BaseModel:
    def __init__(
        self, model_path,
        apply_all_optim=True) -> None:
        self.model_path = model_path  
        self.inference_time = 0  
        
        self.apply_all_optim = apply_all_optim
        
    def load_model(self, device):
        self.width = None
        self.height = None

        # Set graph optimization level
        sess_options = onnxruntime.SessionOptions()
        if self.apply_all_optim:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        if device == "cuda":
            self.model = onnxruntime.InferenceSession(
                self.model_path, providers=["CUDAExecutionProvider"], sess_options=sess_options
            )
        else:
            self.model = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"], sess_options=sess_options
            )
        # Get model info
        self.get_input_details()
        self.get_output_details()
        # return self

    def get_input_details(self):
        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_type = model_inputs[0].type
        
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.model.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.output_shape = model_outputs[0].shape
        
    def predict(self, frame):
        outputs = self.model.run(self.output_names, {self.input_names[0]: frame})
        return outputs