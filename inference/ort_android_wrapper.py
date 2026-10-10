import numpy as np
import logging
from inference.platform_utils import is_android

logger = logging.getLogger(__name__)

class MockSessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.enable_mem_pattern = False
        self.enable_cpu_mem_arena = False
        self.execution_mode = None

class MockGraphOptimizationLevel:
    ORT_ENABLE_ALL = 99

class MockExecutionMode:
    ORT_SEQUENTIAL = 0

class MockNodeArg:
    def __init__(self, name, shape, type_str):
        self.name = name
        self.shape = shape
        self.type = type_str

def mock_get_available_providers():
    return ["CPUExecutionProvider"]

if is_android():
    from jnius import autoclass
    
    OrtEnvironment = autoclass('ai.onnxruntime.OrtEnvironment')
    JavaSessionOptions = autoclass('ai.onnxruntime.OrtSession$SessionOptions')
    OnnxTensor = autoclass('ai.onnxruntime.OnnxTensor')
    HashMap = autoclass('java.util.HashMap')
    
    _env = None
    def get_ort_env():
        global _env
        if _env is None:
            _env = OrtEnvironment.getEnvironment()
        return _env
    
    class OrtSessionWrapper:
        def __init__(self, model_path: str, providers=None, sess_options=None):
            self.model_path = str(model_path)
            self.env = get_ort_env()
            opt = JavaSessionOptions()
            self.session = self.env.createSession(self.model_path, opt)
                
        def _create_java_tensor(self, np_array: np.ndarray):
            nested_list = np_array.tolist()
            return OnnxTensor.createTensor(self.env, nested_list)

        def run(self, output_names, input_feed: dict):
            java_map = HashMap()
            tensors_to_close = []
            try:
                for k, v in input_feed.items():
                    tensor = self._create_java_tensor(np.asarray(v))
                    tensors_to_close.append(tensor)
                    java_map.put(k, tensor)
                
                result = self.session.run(java_map)
                
                out_list = []
                if output_names:
                    for name in output_names:
                        java_val = result.get(name)
                        if java_val is not None:
                            val = java_val.getValue()
                            out_list.append(np.array(val))
                        else:
                            out_list.append(None)
                else:
                    iterator = result.iterator()
                    while iterator.hasNext():
                        entry = iterator.next()
                        val = entry.getValue().getValue()
                        out_list.append(np.array(val))
                
                result.close()
                return out_list
                
            finally:
                for t in tensors_to_close:
                    t.close()
                    
        def get_inputs(self):
            inputs = []
            info_map = self.session.getInputInfo()
            keys = info_map.keySet().iterator()
            while keys.hasNext():
                k = keys.next()
                node = info_map.get(k)
                info_type = node.getInfo()
                
                shape_array = info_type.getShape()
                shape = tuple(shape_array) if shape_array else None
                
                type_str = "tensor(float)"
                inputs.append(MockNodeArg(k, shape, type_str))
            return inputs

    def inject_ort_mock():
        import sys
        class DummyOrt:
            pass
        ort_mock = DummyOrt()
        ort_mock.InferenceSession = OrtSessionWrapper
        ort_mock.SessionOptions = MockSessionOptions
        ort_mock.GraphOptimizationLevel = MockGraphOptimizationLevel()
        ort_mock.ExecutionMode = MockExecutionMode()
        ort_mock.get_available_providers = mock_get_available_providers
        sys.modules['onnxruntime'] = ort_mock

else:
    def inject_ort_mock():
        pass
