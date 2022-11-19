### 1. statistic_ops.py 

用于统计一个目录下（包括递归子目录）所有.onnx文件的算子，并将其列成表格(用excel打开)，形如： 

| models\ops | Flatten | BatchNormalization | Gemm | Add | Relu | Reshape | Conv  |  GlobalAveragePool | MaxPool | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| resnet18-v2-7 | n | y | y | y | y | y | y | y | y |
| resnet18 | n | y | y | y | y | y | y | y | y |
| resnet34-v2-7_inferred | n | y | y | y | y | y | y | y | y |
| resnet152-v2-7 | n | y | y | y | y | y | y | y | y |
| resnet18-v1-7_inferred | y | y | y | y | y | n | y | y | y |
| resnet34-v2-7 | n | y | y | y | y | y | y | y | y  |

#### 用法: 
sudo python3 statistic_ops.py \<dir holding onnx files\>  
(如 python3 statistic_ops.py /public/ai_platform/models/)  
理论上, 他将产生一个.tsv 文件, 能用Excel打开. 这里 y/n 表示对应位置的算子是否存在。 

### 2. shape_diff.py 

用于发现两个model中出现的不同的形状，没有推理形状的模型自动推理并保存为以_inferred.onnx结尾的文件。
#### 用法： 
sudo python3 shape_diff.py \<path to criteron.onnx\> \<path to subject.onnx\>  
第一个onnx是标准集， 第二个onnx是比较集， 找比较集中出现而标准集中没出现的shape, 算法为sub_set.difference(crt_set)。
