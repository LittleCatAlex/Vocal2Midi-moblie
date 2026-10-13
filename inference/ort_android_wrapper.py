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

# ---------------------------------------------------------------------------
# Pure-Python ONNX Protobuf Metadata Extractor (Zero C/JNI/Protobuf dependency)
# ---------------------------------------------------------------------------
def _parse_varint_from_stream(f):
    res = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            return None
        val = b[0]
        res |= (val & 0x7F) << shift
        if not (val & 0x80):
            break
        shift += 7
    return res

def _parse_varint_from_bytes(data, pos):
    res = 0
    shift = 0
    while pos < len(data):
        val = data[pos]
        pos += 1
        res |= (val & 0x7F) << shift
        if not (val & 0x80):
            break
        shift += 7
    return res, pos

ONNX_ELEM_TYPE_MAP = {
    1: 'tensor(float)',
    2: 'tensor(uint8)',
    3: 'tensor(int8)',
    4: 'tensor(uint16)',
    5: 'tensor(int16)',
    6: 'tensor(int32)',
    7: 'tensor(int64)',
    8: 'tensor(string)',
    9: 'tensor(bool)',
    10: 'tensor(float16)',
    11: 'tensor(double)',
    12: 'tensor(uint32)',
    13: 'tensor(uint64)',
    14: 'tensor(complex64)',
    15: 'tensor(complex128)',
    16: 'tensor(bfloat16)',
}

def _parse_onnx_value_info(data):
    pos = 0
    name = ''
    elem_type = 1
    shape = []
    while pos < len(data):
        tag, pos = _parse_varint_from_bytes(data, pos)
        fnum = tag >> 3
        wtype = tag & 0x07
        if fnum == 1 and wtype == 2:  # name
            l, pos = _parse_varint_from_bytes(data, pos)
            name = data[pos:pos+l].decode('utf-8', errors='ignore')
            pos += l
        elif fnum == 2 and wtype == 2:  # TypeProto
            l, pos = _parse_varint_from_bytes(data, pos)
            type_data = data[pos:pos+l]
            pos += l
            tpos = 0
            while tpos < len(type_data):
                ttag, tpos = _parse_varint_from_bytes(type_data, tpos)
                tfnum = ttag >> 3
                twtype = ttag & 0x07
                if tfnum == 1 and twtype == 2:  # Tensor
                    tl, tpos = _parse_varint_from_bytes(type_data, tpos)
                    tensor_data = type_data[tpos:tpos+tl]
                    tpos += tl
                    kpos = 0
                    while kpos < len(tensor_data):
                        ktag, kpos = _parse_varint_from_bytes(tensor_data, kpos)
                        kfnum = ktag >> 3
                        kwtype = ktag & 0x07
                        if kfnum == 1 and kwtype == 0:  # elem_type
                            elem_type, kpos = _parse_varint_from_bytes(tensor_data, kpos)
                        elif kfnum == 2 and kwtype == 2:  # shape
                            sl, kpos = _parse_varint_from_bytes(tensor_data, kpos)
                            shape_data = tensor_data[kpos:kpos+sl]
                            kpos += sl
                            spos = 0
                            while spos < len(shape_data):
                                stag, spos = _parse_varint_from_bytes(shape_data, spos)
                                sfnum = stag >> 3
                                swtype = stag & 0x07
                                if sfnum == 1 and swtype == 2:  # dim
                                    dl, spos = _parse_varint_from_bytes(shape_data, spos)
                                    dim_data = shape_data[spos:spos+dl]
                                    spos += dl
                                    dpos = 0
                                    dim_val = None
                                    while dpos < len(dim_data):
                                        dtag, dpos = _parse_varint_from_bytes(dim_data, dpos)
                                        dfnum = dtag >> 3
                                        dwtype = dtag & 0x07
                                        if dfnum == 1 and dwtype == 0:  # dim_value
                                            dim_val, dpos = _parse_varint_from_bytes(dim_data, dpos)
                                        elif dfnum == 2 and dwtype == 2:  # dim_param
                                            pl, dpos = _parse_varint_from_bytes(dim_data, dpos)
                                            dim_val = dim_data[dpos:dpos+pl].decode('utf-8', errors='ignore')
                                            dpos += pl
                                        elif dwtype == 0:
                                            _, dpos = _parse_varint_from_bytes(dim_data, dpos)
                                        elif dwtype == 2:
                                            pl, dpos = _parse_varint_from_bytes(dim_data, dpos)
                                            dpos += pl
                                    shape.append(dim_val)
                                elif swtype == 0:
                                    _, spos = _parse_varint_from_bytes(shape_data, spos)
                                elif swtype == 2:
                                    pl, spos = _parse_varint_from_bytes(shape_data, spos)
                                    spos += pl
                        elif kwtype == 0:
                            _, kpos = _parse_varint_from_bytes(tensor_data, kpos)
                        elif kwtype == 2:
                            pl, kpos = _parse_varint_from_bytes(tensor_data, kpos)
                            kpos += pl
                elif twtype == 0:
                    _, tpos = _parse_varint_from_bytes(type_data, tpos)
                elif twtype == 2:
                    pl, tpos = _parse_varint_from_bytes(type_data, tpos)
                    tpos += pl
        elif wtype == 0:
            _, pos = _parse_varint_from_bytes(data, pos)
        elif wtype == 2:
            l, pos = _parse_varint_from_bytes(data, pos)
            pos += l
    return MockNodeArg(name, tuple(shape), ONNX_ELEM_TYPE_MAP.get(elem_type, 'tensor(float)'))

def extract_onnx_io_metadata(onnx_path: str):
    """Directly extracts input and output NodeArg definitions from an ONNX file.
    
    This avoids Android JNI FindClass/ValueInfo ClassLoader failures entirely,
    guaranteeing exact tensor shapes and data types matching the true model graph.
    """
    import os
    if not os.path.isfile(onnx_path):
        return [], []
    inputs = []
    outputs = []
    try:
        with open(onnx_path, 'rb') as f:
            while True:
                tag = _parse_varint_from_stream(f)
                if tag is None:
                    break
                if tag >> 3 == 7:  # GraphProto
                    graph_len = _parse_varint_from_stream(f)
                    end_pos = f.tell() + graph_len
                    while f.tell() < end_pos:
                        gtag = _parse_varint_from_stream(f)
                        if gtag is None:
                            break
                        fnum = gtag >> 3
                        wtype = gtag & 0x07
                        if wtype == 0:
                            _parse_varint_from_stream(f)
                        elif wtype == 1:
                            f.seek(8, 1)
                        elif wtype == 5:
                            f.seek(4, 1)
                        elif wtype == 2:
                            flen = _parse_varint_from_stream(f)
                            if fnum == 11:
                                data = f.read(flen)
                                inputs.append(_parse_onnx_value_info(data))
                            elif fnum == 12:
                                data = f.read(flen)
                                outputs.append(_parse_onnx_value_info(data))
                            else:
                                f.seek(flen, 1)
                    break
                elif tag & 0x07 == 2:
                    flen = _parse_varint_from_stream(f)
                    f.seek(flen, 1)
                elif tag & 0x07 == 0:
                    _parse_varint_from_stream(f)
                elif tag & 0x07 == 1:
                    f.seek(8, 1)
                elif tag & 0x07 == 5:
                    f.seek(4, 1)
    except Exception as exc:
        logger.warning(f"Could not parse ONNX graph metadata from {onnx_path}: {exc}")
    return inputs, outputs

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
    ValueInfo = autoclass('ai.onnxruntime.ValueInfo')
    OnnxJavaType = autoclass('ai.onnxruntime.OnnxJavaType')
    HashMap = autoclass('java.util.HashMap')
    ByteBuffer = autoclass('java.nio.ByteBuffer')
    ByteOrder = autoclass('java.nio.ByteOrder')
    
    # Pre-cache optional / nested classes safely
    for _cls in [
        'ai.onnxruntime.OrtSession$RunOptions',
        'ai.onnxruntime.OrtSession$SessionOptions$ExecutionMode',
        'ai.onnxruntime.OrtSession$SessionOptions$OptLevel',
        'ai.onnxruntime.MapInfo',
        'ai.onnxruntime.SequenceInfo',
        'ai.onnxruntime.OrtException',
        'ai.onnxruntime.TensorInfo$OnnxTensorType',
        'ai.onnxruntime.OnnxMap',
        'ai.onnxruntime.OnnxSequence',
        'ai.onnxruntime.OrtAllocator',
        'ai.onnxruntime.OrtLoggingLevel',
        'ai.onnxruntime.OrtUtil',
    ]:
        try:
            autoclass(_cls)
        except Exception:
            pass

    NUMPY_TO_ONNX_JAVA_TYPE = {
        'float32': 'FLOAT',
        'float16': 'FLOAT16',
        'int64': 'INT64',
        'int32': 'INT32',
        'int16': 'INT16',
        'int8': 'INT8',
        'uint8': 'UINT8',
        'bool': 'BOOL',
        'float64': 'DOUBLE',
    }

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
            self._parsed_inputs, self._parsed_outputs = extract_onnx_io_metadata(self.model_path)
            self.env = get_ort_env()
            opt = JavaSessionOptions()
            self.session = self.env.createSession(self.model_path, opt)

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CPUExecutionProvider": {}}
                
        def _create_java_tensor(self, np_array: np.ndarray):
            arr = np.ascontiguousarray(np_array)
            shape = [int(s) for s in arr.shape]
            try:
                type_name = NUMPY_TO_ONNX_JAVA_TYPE.get(arr.dtype.name, 'FLOAT')
                java_type = getattr(OnnxJavaType, type_name, OnnxJavaType.FLOAT)
                raw_bytes = bytearray(arr.tobytes())
                buf = ByteBuffer.wrap(raw_bytes)
                buf.order(ByteOrder.nativeOrder())
                return OnnxTensor.createTensor(self.env, buf, shape, java_type)
            except Exception as exc:
                logger.debug(f"Direct ByteBuffer tensor creation failed ({exc}), falling back to nested list")
                nested_list = arr.tolist()
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
            if self._parsed_inputs:
                return list(self._parsed_inputs)
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
                        from jnius import cast
                        tensor_info = cast('ai.onnxruntime.TensorInfo', info_type)
                        shape_array = tensor_info.getShape()
                        shape = tuple(shape_array) if shape_array else None
                    except Exception:
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
            if self._parsed_outputs:
                return list(self._parsed_outputs)
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
                        from jnius import cast
                        tensor_info = cast('ai.onnxruntime.TensorInfo', info_type)
                        shape_array = tensor_info.getShape()
                        shape = tuple(shape_array) if shape_array else None
                    except Exception:
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

