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
    
    # Pre-cache all ONNX Runtime Java classes on the main thread during app startup.
    # PyJnius registers them in its global cache, preventing dynamic FindClass
    # lookups on background worker threads where the system classloader would fail.
    OrtEnvironment = autoclass('ai.onnxruntime.OrtEnvironment')
    OrtSession = autoclass('ai.onnxruntime.OrtSession')
    OrtSessionResult = autoclass('ai.onnxruntime.OrtSession$Result')
    JavaSessionOptions = autoclass('ai.onnxruntime.OrtSession$SessionOptions')
    OnnxTensor = autoclass('ai.onnxruntime.OnnxTensor')
    OnnxValue = autoclass('ai.onnxruntime.OnnxValue')
    NodeInfo = autoclass('ai.onnxruntime.NodeInfo')
    TensorInfo = autoclass('ai.onnxruntime.TensorInfo')
    HashMap = autoclass('java.util.HashMap')
    
    # Pre-cache optional / nested classes safely
    for _cls in [
        'ai.onnxruntime.OrtSession$RunOptions',
        'ai.onnxruntime.OrtSession$SessionOptions$ExecutionMode',
        'ai.onnxruntime.OrtSession$SessionOptions$OptLevel',
        'ai.onnxruntime.MapInfo',
        'ai.onnxruntime.SequenceInfo',
        'ai.onnxruntime.OrtException',
    ]:
        try:
            autoclass(_cls)
        except Exception:
            pass

    def init_android_thread_classloader():
        """Ensure background threads have access to the APK's ClassLoader.
        
        On Android, native pthreads / Python threading.Thread attached via JNI
        default to the system ClassLoader (DexPathList[[directory "."]]), causing
        ClassNotFoundException when loading application classes. Setting the
        current thread's context ClassLoader to PythonActivity's ClassLoader
        fixes this.
        """
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            if activity is not None:
                cl = activity.getClassLoader()
                Thread = autoclass('java.lang.Thread')
                Thread.currentThread().setContextClassLoader(cl)
        except Exception as exc:
            logger.warning(f"Failed to set thread ContextClassLoader: {exc}")

    _env = None
    def get_ort_env():
        global _env
        if _env is None:
            init_android_thread_classloader()
            _env = OrtEnvironment.getEnvironment()
        return _env
    
    class OrtSessionWrapper:
        def __init__(self, model_path: str, providers=None, sess_options=None):
            init_android_thread_classloader()
            self.model_path = str(model_path)
            self.env = get_ort_env()
            opt = JavaSessionOptions()
            self.session = self.env.createSession(self.model_path, opt)
                
        def _create_java_tensor(self, np_array: np.ndarray):
            nested_list = np_array.tolist()
            return OnnxTensor.createTensor(self.env, nested_list)

        def run(self, output_names, input_feed: dict):
            init_android_thread_classloader()
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
            init_android_thread_classloader()
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
    def init_android_thread_classloader():
        pass

    def inject_ort_mock():
        pass

