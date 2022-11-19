# onnx inference models

- ONNX 和 MLPerf 提供了 ONNX 模型, 可以直接使用
- 未提供 ONNX 模型的, 需要使用官方代码训练出 Tensorflow 或 PyTorch 模型
- 开源框架模型转换为 ONNX 模型
- 测试准确率 

### ONNX Model List

| 序号   | 模型      | ONNX 模型 |  ONNX runtime code       |  数据集 |
| :----- |  :----   | :----     | :----                    | :----  |
| 1  | resnet50     | [MLPerf resnet50 ONNX](https://github.com/mlcommons/inference) |  | [imagenet2012 homepage](https://www.image-net.org/challenges/LSVRC/2012/index.php), [imagenet2012 subset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) |
| 2  | Bert-Large   | [MLPerf Bert-Large ONNX](https://github.com/mlcommons/inference/tree/r1.1/language/bert) | | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)   |
| 3  | Bert-Base    | [Bert int8 ONNX](https://github.com/onnx/models/tree/main/text/machine_comprehension/bert-squad) |  |  |
| 4  | YOLO_v5      | [YOLOv5 pt->ONNX](https://github.com/itsnine/yolov5-onnxruntime), [YoloV5 Official](https://github.com/ultralytics/yolov5/issues/251) , [YoloV5 pt models](https://github.com/ultralytics/yolov5/releases)|  |  |
| 5  | VGG16        | [VGG16, VGG16 int8 ONNX](https://github.com/onnx/models/tree/main/vision/classification/vgg) |  |  |
| 6  | Tacotron2    | [Tacotron2 代码可以生成 ONNX 模型](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) |  |  |
| 7  | MaskRCNN     | [MaskRCNN ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/mask-rcnn) |  |  |
| 8  | Transformer  | [nvidia Transformer .pt](https://catalog.ngc.nvidia.com/orgs/nvidia/models/transformer_pyt_ckpt_tf32/files) |  |  |
| 9  | TDNN         | [ ]( ) |  |  |
| 10 | TextCNN      | [ ]( ) |  |  |
| 11 | RNNT         | [RNNT ONNX](https://github.com/mlcommons/inference) |  |  |
| 12 |              | [ ]( ) |  |  |
| 13 | seq2seq      | [ ]( ) |  |  |
| 14 | DBnet        | [ ]( ) |  |  |
| 15 | DLRM         | [DLRM ONNX](https://github.com/mlcommons/inference) |  |  |
| 16 | ViT          | [ ]( ) |  |  |
| 17 | resnet18     | [resnet18 ONNX](https://github.com/onnx/models/tree/main/vision/classification/resnet) |  |  |
| 18 | resnet34     | [resnet34 ONNX](https://github.com/onnx/models/tree/main/vision/classification/resnet) |  |  |
| 19 | YOLO_v3_Tiny | [Yolov3-Tiny ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3) |  |  |
| 20 | YOLO_v3      | [Yolov3 ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov3) |  |  |
| 21 | Inception_v2 | [Inception_v2](https://github.com/onnx/models) |  |  |
| 22 | Inception_v3 | [Inception_v3](https://github.com/onnx/models) |  |  |
| 23 | MobileNet_v2 | [ ]( ) |  |  |
| 24 | MobileNet_v3 | [ ]( ) |  |  |
| 25 | resnet101    | [resnet101 ONNX](https://github.com/onnx/models/tree/main/vision/classification/resnet) |  |  |
| 26 | SSD          | [SSD ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd) |  |  |
| 27 |FasterRCNN-R50-FPN  |  [FasterRCNN ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/faster-rcnn) |  |  |
| 28 | 3D-UNet      | [3D-UNet ONNX](https://github.com/mlcommons/inference) |  |  |



### 模型分析 与 模型转换
------
[VGG16 模型结构代码](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) : 未使用BN, 5组(3x3)卷积之间用maxpooling, FC 使用(1x1)卷积替换

[VGG16 ckpt 模型](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz), [Tensorflow ckpt 转 ONNX](./tensorflow_model_convert/tensorflow_onnx_converter.py)

------

[nvidia Transformer Code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer)

使用 nvidia Transformer 代码进行推理, 下载上述 .pt 模型, 准备 en -> 

------


-----------------------------
### ONNX Model Operator List

#### CV OP List

| 序号   |算子                  | VGG16  | YoloV3 | YoloV5 | MaskRCNN |
| :-----| :----:                 | :----: | :----: | :----: | :----:   |
|1         |Log                  |        |        |       |    ✓    |
|2         |Sub                  |        |        |       |     ✓   |
|3         |Floor                |        |        |       |    ✓    |
|4         |Unsqueeze            |        |        |       |    ✓    |
|5         |Scatter              |        |        |       |    ✓    |
|6         |Conv                 |        |        |       |    ✓    |
|7         |ConvTranspose        |        |        |       |    ✓    |
|8         |TopK                 |        |        |       |    ✓    |
|9         |Relu                 |        |        |       |    ✓    |
|10        |Resize               |        |        |       |    ✓    |
|11        |Cast                 |        |        |       |    ✓    |
|12        |Expand               |        |        |       |    ✓    |
|13        |Sigmoid              |        |        |       |    ✓    |
|14        |Add                  |        |        |       |    ✓    |
|15        |NonMaxSuppression    |        |        |       |    ✓    |
|16        |Gather               |        |        |       |    ✓    |
|17        |MaxPool              |        |        |       |    ✓    |
|18        |Mul                  |        |        |       |    ✓    |
|19        |Less                 |        |        |       |    ✓    |
|20        |Constant             |        |        |       |    ✓    |
|21        |Shape                |        |        |       |    ✓    |
|22        |Split                |        |        |       |    ✓    |
|23        |ReduceMin            |        |        |       |    ✓    |
|24        |And                  |        |        |       |    ✓    |
|25        |Gemm                 |        |        |       |    ✓    |
|26        |Equal                |        |        |       |    ✓    |
|27        |Reshape              |        |        |       |    ✓    |
|28        |Softmax              |        |        |       |    ✓    |
|29        |Flatten              |        |        |       |    ✓    |
|30        |NonZero              |        |        |       |    ✓    |
|31        |Clip                 |        |        |       |    ✓    |
|32        |RoiAlign             |        |        |       |    ✓    |
|33        |Not                  |        |        |       |    ✓    |
|34        |Concat               |        |        |       |    ✓    |
|35        |Div                  |        |        |       |    ✓    |
|36        |Greater              |        |        |       |    ✓    |
|37        |Exp                  |        |        |       |    ✓    |
|38        |Transpose            |        |        |       |    ✓    |
|39        |Squeeze              |        |        |       |    ✓    |
|40        |Slice                |        |        |       |    ✓    |
|41        |ConstantOfShape      |        |        |       |    ✓    |
|42        |Sqrt                 |        |        |       |    ✓    |
#### NLP/Audio OP List

| 序号   |算子     | Bert   | Tacotron2 | RNNT |
| :----- | :----: | :----: | :----:    | :----: |
| 1      |        |        |           |        |
| 2      |        |        |           |        |
| 3     |        |        |        |        |          | 
| 4     |        |        |        |        |          |
| 5     |        |        |        |        |          | 
| 6     |        |        |        |        |          |
| 7     |        |        |        |        |          | 
| 8     |        |        |        |        |          |
| 9     |        |        |        |        |          | 
| 10     |        |        |        |        |          |
| 11     |        |        |        |        |          | 
| 12     |        |        |        |        |          |
| 13     |        |        |        |        |          | 
| 14     |        |        |        |        |          |
| 15     |        |        |        |        |          | 
| 16     |        |        |        |        |          |
| 17     |        |        |        |        |          | 
| 18     |        |        |        |        |          |
| 19     |        |        |        |        |          | 

-----------------------------

