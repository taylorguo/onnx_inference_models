'''
Author: Taylor Guo
Date: 2021-05-24 09:45:03
LastEditTime: 2021-05-28 22:27:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /onnx2json/onnx/shape_inference.py
'''

import onnx, json, os
from onnx import shape_inference
import numpy as np

def get_onnx_weight(onnx_model):
    # onnx_model = onnx.load(model_path)
    # print(onnx_model.graph.initializer)
    producer = onnx_model.producer_name
    os.system("")
    print("\033[42m"+ "The onnx model comes from {}.".format(producer if producer else "ONNX official"  ) +"\033[0m")
    
    # try:
    #     onnx.checker.check_model(onnx_model)
    # except onnx.checker.ValidationError as e:
    #     print('The model is invalid: %s' % e)
    # else:
    #     print('The model is valid!')

    weight_dict = {}
    weight_list = onnx_model.graph.initializer
    # TODO: pytorch-> onnx : producer= pytorch; onnx official: producer = ""; others to-do
    for n, weight in enumerate(weight_list):
        if "torch" in producer.lower():            
            weight_np = np.frombuffer(weight.raw_data, dtype=np.float32)
        else:  
            weight_np = np.array(weight.float_data)
        weight_dict[weight.name] = weight_np
        # print(n, weight.name, weight_np.shape)
    return weight_dict
            
###### Shape Inference

def onnx_shape_inference(model_path):
    onnx_model = onnx.load(model_path)
    value_info_list = onnx_model.graph.value_info
    tensor_shape = {}

    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
        return tensor_shape
    # else:
    #     print('The model is valid!')

    if not len(value_info_list):
        # print("The model contains some shapes, do not apply shape_inference!")
        onnx_model = shape_inference.infer_shapes(onnx_model)
        # onnx.save(onnx_model, model_path+"_inferred.onnx")
        # Check the model and print Y's shape information
        onnx.checker.check_model(onnx_model)
        value_info_list = onnx_model.graph.value_info
        # print('After shape inference, the shape info  is:\n{}'.format(value_info_list)) 
        path_list = list(os.path.splitext(model_path))
        path_list.insert(1, "_shape_inference")
        onnx.save(onnx_model, "".join(path_list))
        print('Finish shape inference') 

    # Process graph input as Conv input if 2'nd layer is convolution
    graph_input = onnx_model.graph.input[0]

    # print(graph_input)
    if onnx_model.graph.node[1].op_type == "Conv":
        shape_list = []
        for dv in graph_input.type.tensor_type.shape.dim:
            shape_list.append(dv.dim_value)
        tensor_shape[graph_input.name] = shape_list

    for num, node in enumerate(value_info_list):        
        shape_list = []
        for kv in node.type.tensor_type.shape.dim:
            shape_list.append(kv.dim_value)
        tensor_shape[node.name] = shape_list
    # print(tensor_shape)
    return tensor_shape, get_onnx_weight(onnx_model)

def read_json_ops_tensors(json_file):
    with open(json_file, "r", encoding="utf-8") as jf:
        json_content = json.load(jf)
    return json_content["Operators"], json_content["Tensors"]

def get_weight_shapes(json_file):
    with open(json_file, "r", encoding="utf-8") as jf:
        json_content = json.load(jf)
    operators = json_content["Operators"]
    tensors = json_content["Tensors"]
    weight_op = {}
    for conv, params in operators.items():
        for name, tensor in params["consumers"].items():
            if (conv in tensor) and ("weight" in tensor):
                weight_op[name] = tensor
    weight_shape = {}
    for wt, pams in tensors.items():
        for wtname, ts in weight_op.items():
            if wt == ts:
                weight_shape[wtname] = pams["shape"]
    # print(weight_shape)
    return weight_shape

def update_json_shapes(ops, tensors, all_tensors):
    input_name, output_name = None, None
    for op, params in ops.items():
        activation_input= params["inputs"][0] 
        activation_output = params["output"][0] 
        for kc, vc in params["consumers"].items():
            if vc == activation_input: input_name = kc
        for kp, vp in params["producers"].items():
            if vp == activation_output: output_name = kp
        # print(input_name, "             ", output_name, "               ", 
        #         activation_input, "             ", activation_output)
        
        for tensor_name, shape_list in all_tensors.items():
            
            if tensor_name in input_name :
                tensors[activation_input]["shape"] = shape_list
            if tensor_name in output_name:
                tensors[activation_output]["shape"] = shape_list
    # print(tensors)

def write_json_shapes(json_file, tensors):
    with open(json_file, "r", encoding="utf-8") as jf:
        json_content = json.load(jf)
        json_content["Tensors"] = tensors

    with open(json_file, "w", encoding="utf-8") as jf:
        jf.write(json.dumps(json_content))


def onnx2json_pypost(model_path, json_file):
    
    all_shapes, onnx_weight = onnx_shape_inference(model_path)
    
    ops, tensors = read_json_ops_tensors(json_file)
    update_json_shapes(ops, tensors, all_shapes)
    write_json_shapes(json_file, tensors)
    # print("Saved tensor list to %s"%(json_file))
    return onnx_weight

######### Get weights with numpy ndarray shape
def get_weights_reshape(model_path, onnx_weight):
    # onnx_weight = get_onnx_weight(onnx_model)
    weight_shape = get_weight_shapes(model_path + ".json")
    ret_weight = {}
    for wtname, wt in onnx_weight.items():
        for wtsnm, shape in weight_shape.items():
            if wtsnm == wtname:
                # print(wtsnm, shape, wt.shape)
                wt_reshape = wt.reshape(shape)
                ret_weight[wtsnm] = wt_reshape
    return ret_weight


if __name__ == "__main__":
    import sys
    # model_path = sys.argv[1]  
    model_path = "/public/ai_platform/model_compression/yolov5n.onnx"

    json_file = "build/resnet18-v2-7.onnx.json"

    # json_file = "../conv2_2.onnx.json"

    # onnx2json_pypost(model_path, json_file)
    # print(get_weight_shapes(json_file))
    onnx_shape_inference(model_path)
    