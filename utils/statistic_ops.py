from statistics import mode
from typing import Set
import onnx
import onnxruntime as onnxrt
import torch
import numpy as np
import sys, os

def collect_files(path, ext = ".onnx", depth = 1):
    all_files = []
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            if depth < 0:
                continue
            all_files.extend(collect_files(cur_path, ext, depth -1))
        else:
            if file.endswith(ext):
                if not path.endswith('/'):
                    path += '/'
                file_path = (file.replace(ext, ''), path + file)
                print(file_path)
                all_files.append(file_path)

    return all_files

def getOps(onnx_path):
    try:
        model = onnx.load(onnx_path)
    except:
        print("load %s failed" %onnx_path)
        return set()
    ret = set()
    for node in model.graph.node:
        ret.add(node.op_type)
    return ret

def getAttrs(onnx_path):
    model = onnx.load(onnx_path)
    ret = {}
    for node in model.graph.node:
        if node.op_type in ret:
            ret[node.op_type].add(str(node.attribute))
        else:
            ret[node.op_type] = {str(node.attribute)}
    return ret

def make_table(models: map, ops: set):
    s = "models\\ops:\t"
    lst = []
    for op in ops:
        s += "%s\t" % op
        lst.append(op)
    s += "\n"

    for m in models:
        s += "%s\t" %m
        for o in lst:
            if o in models[m]:
                s += "y\t"
            else:
                s += "n\t"
        s += "\n"
    return s

'''
usage: python3 statistic_ops.py <dir holding onnx files> (like /public/ai_platform/models/)
Principly, it will generate a .tsv file, which can be opened with Excel. In which y/n indecates existence of 
the operator (in corresponding column on top row) in the model (listed in the 1st row).

'''
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s <dir containing .onnx>" % sys.argv[0])
        sys.exit(1)
    elif len(sys.argv) >= 3 and sys.argv[2] == "attr":
        want_attr = True
    else:
        want_attr = False

    file_list = collect_files(sys.argv[1])

    models = {}
    all_ops = set()
    model_attr = {}
    for file in file_list:
        op_set = getOps(file[1])
        models[file[0]] = op_set
        print("%s: %s" % (file[0], op_set))
        all_ops.update(op_set)
        if want_attr:
            model_attr[file[0]] = getAttrs(file[1])
            print("%s: %s" % (file[0], model_attr[file[0]]))

    t = make_table(models, all_ops)
    print(t)

    fo = open("statistic.tsv", "w")
    fo.write(t)
    fo.close()

    fo = open("attributes.json", "w")
    fo.write(str(model_attr))
    fo.close()
