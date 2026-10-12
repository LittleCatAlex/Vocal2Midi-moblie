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
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0

    def add_session_config_entry(self, key, value):
        pass

    def set_session_config_entry(self, key, value):
        pass

class MockGraphOptimizationLevel:
    ORT_ENABLE_ALL = 99
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_DISABLE_ALL = 0

class MockExecutionMode:
    ORT_SEQUENTIAL = 0
    ORT_PARALLEL = 1

class MockNodeArg:
    def __init__(self, name, shape, type_str):
        self.name = str(name)
        self.shape = shape
        self.type = str(type_str)

    def __repr__(self):
        return f"NodeArg(name='{self.name}', shape={self.shape}, type='{self.type}')"

class MockModelMetadata:
    def __init__(self):
        self.custom_metadata_map = {}
        self.description = ""
        self.domain = ""
        self.graph_name = ""
        self.producer_name = ""
        self.version = 0

def mock_get_available_providers():
    return ["CPUExecutionProvider"]

def _extract_onnx_value(val_obj):
    if val_obj is None:
        return None
    # Handle java.util.Optional returned by OrtSession.Result.get(String)
    try:
        if hasattr(val_obj, "isPresent"):
            if not val_obj.isPresent():
                return None
            val_obj = val_obj.get()
    except Exception:
        pass
    # Handle Map.Entry or nested value wrapper if any
    try:
        if hasattr(val_obj, "getValue") and hasattr(val_obj, "getKey"):
            val_obj = val_obj.getValue()
    except Exception:
        pass
    # Extract underlying ndarray/buffer from OnnxValue
    if hasattr(val_obj, "getValue"):
        raw = val_obj.getValue()
        return np.array(raw)
    return np.array(val_obj)

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

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CPUExecutionProvider": {}}
                
        def _create_java_tensor(self, np_array: np.ndarray):
            nested_list = np_array.tolist()
            return OnnxTensor.createTensor(self.env, nested_list)

        def run(self, output_names, input_feed: dict, run_options=None):
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
                        out_list.append(_extract_onnx_value(java_val))
                else:
                    try:
                        n_outs = result.size()
                        for i in range(n_outs):
                            out_list.append(_extract_onnx_value(result.get(i)))
                    except Exception:
                        iterator = result.iterator()
                        while iterator.hasNext():
                            entry = iterator.next()
                            out_list.append(_extract_onnx_value(entry))
                
                result.close()
                return out_list
                
            finally:
                for t in tensors_to_close:
                    try:
                        t.close()
                    except Exception:
                        pass
                    
        def get_inputs(self):
            init_android_thread_classloader()
            inputs = []
            try:
                info_map = self.session.getInputInfo()
                keys = info_map.keySet().iterator()
                while keys.hasNext():
                    k = keys.next()
                    node = info_map.get(k)
                    info_type = node.getInfo()
                    
                    shape = None
                    try:
                        shape_array = info_type.getShape()
                        shape = tuple(shape_array) if shape_array else None
                    except Exception:
                        pass
                    
                    type_str = "tensor(float)"
                    try:
                        t_name = str(info_type.getType().name()).lower()
                        type_str = f"tensor({t_name})"
                    except Exception:
                        pass
                    
                    inputs.append(MockNodeArg(k, shape, type_str))
            except Exception as exc:
                logger.warning(f"Failed to getInputInfo from session: {exc}")
                try:
                    names = self.session.getInputNames()
                    it = names.iterator()
                    while it.hasNext():
                        inputs.append(MockNodeArg(it.next(), None, "tensor(float)"))
                except Exception:
                    pass
            return inputs

        def get_outputs(self):
            init_android_thread_classloader()
            outputs = []
            try:
                info_map = self.session.getOutputInfo()
                keys = info_map.keySet().iterator()
                while keys.hasNext():
                    k = keys.next()
                    node = info_map.get(k)
                    info_type = node.getInfo()
                    
                    shape = None
                    try:
                        shape_array = info_type.getShape()
                        shape = tuple(shape_array) if shape_array else None
                    except Exception:
                        pass
                    
                    type_str = "tensor(float)"
                    try:
                        t_name = str(info_type.getType().name()).lower()
                        type_str = f"tensor({t_name})"
                    except Exception:
                        pass
                    
                    outputs.append(MockNodeArg(k, shape, type_str))
            except Exception as exc:
                logger.warning(f"Failed to getOutputInfo from session: {exc}")
                try:
                    names = self.session.getOutputNames()
                    it = names.iterator()
                    while it.hasNext():
                        outputs.append(MockNodeArg(it.next(), None, "tensor(float)"))
                except Exception:
                    pass
            return outputs

        def get_modelmeta(self):
            init_android_thread_classloader()
            mock = MockModelMetadata()
            try:
                meta = self.session.getMetadata()
                for attr, getter in [
                    ("producer_name", "getProducerName"),
                    ("graph_name", "getGraphName"),
                    ("description", "getDescription"),
                    ("domain", "getDomain"),
                    ("version", "getVersion"),
                ]:
                    try:
                        setattr(mock, attr, getattr(meta, getter)())
                    except Exception:
                        pass
                try:
                    cust = meta.getCustomMetadata()
                    if cust:
                        d = {}
                        it = cust.keySet().iterator()
                        while it.hasNext():
                            ck = it.next()
                            d[ck] = cust.get(ck)
                        mock.custom_metadata_map = d
                except Exception:
                    pass
            except Exception:
                pass
            return mock

        def close(self):
            if hasattr(self, "session") and self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

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
        ort_mock.NodeArg = MockNodeArg
        sys.modules['onnxruntime'] = ort_mock

else:
    def init_android_thread_classloader():
        pass

    def inject_ort_mock():
        pass

