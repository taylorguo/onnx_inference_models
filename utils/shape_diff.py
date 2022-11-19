import os
import onnx
import numpy as np
import onnxruntime as rt
from onnx import shape_inference
import sys

def get_tensor_shape(tensor):
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    return [dims[i].dim_value for i in range(n)]

def runtime_infer(onnx_model):
    graph = onnx_model.graph
    input_shape = get_tensor_shape(graph.input[0])
    graph.output.insert(0, graph.input[0])
    for i, tensor in enumerate(graph.value_info):
        graph.output.insert(i + 1, tensor)
    model_file = "temp.onnx"
    onnx.save(onnx_model, model_file)

    sess = rt.InferenceSession(model_file)
    input_name = sess.get_inputs()[0].name
    input_data = np.ones(input_shape, dtype=np.float32)

    outputs = {}
    for out in sess.get_outputs():
        tensor = sess.run([out.name], {input_name: input_data})
        outputs[str(out.name)] = np.array(tensor[0]).shape
    os.remove(model_file)
    return outputs

def infer_shapes(model_file, running_mode=False):
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
        
    save_path = model_file[:-5] + "_inferred.onnx"
    onnx.save(inferred_onnx_model, save_path)
    print("Model is saved in:", save_path)

    outputs = {}
    if running_mode:
        outputs = runtime_infer(inferred_onnx_model)
    else:
        graph = inferred_onnx_model.graph
        # only 1 input tensor
        tensor = graph.input[0]
        outputs[str(tensor.name)] = get_tensor_shape(tensor)
        # process tensor
        for tensor in graph.value_info:
            outputs[str(tensor.name)] = get_tensor_shape(tensor)
        # output tensor
        for tensor in graph.output:
            outputs[str(tensor.name)] = get_tensor_shape(tensor)
    return outputs

def get_shapes(model_file):
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph
    ret = {}
    for tensor in graph.value_info:
        ret[str(tensor.name)] = get_tensor_shape(tensor)
    return ret

def get_shpae_set(model):
    if model.find("_inferred.onnx") == -1:
        shapes = infer_shapes(model, True)
    else:
        shapes = get_shapes(model)

    ret = set()
    for v in shapes.values():
        ret.add(tuple(v))

    return ret

'''
sudo python3 shape_diff.py /public/ai_platform/models/模型泛化/resnet系列/resnet50.onnx /public/ai_platform/models/模型泛化/resnet系列/resnet18-v1-7.onnx_inferred.onnx
'''
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: %s <criteron.onnx> <subject.onnx>" % sys.argv[0])
        sys.exit(1)

    crt_set = get_shpae_set(sys.argv[1])
    sub_set = get_shpae_set(sys.argv[2])

    print(sub_set.difference(crt_set))