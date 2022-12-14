# onnx inference models

- ONNX 和 MLPerf 提供了 ONNX 模型, 可以直接使用
- 未提供 ONNX 模型的, 需要使用官方代码训练出 Tensorflow 或 PyTorch 模型
- 开源框架模型转换为 ONNX 模型
- 测试准确率

### ONNX Model List

| 序号 | 模型 & Paper |模型 ONNX/PT/PB | ONNX 转换 | 数据集|
| :--- | :----------------- | :----------------------------- | :------------------------- | :----------------- |
| 1    | resnet50           | [MLPerf resnet50 ONNX](https://github.com/mlcommons/inference)   | [imagenet2012 homepage](https://www.image-net.org/challenges/LSVRC/2012/index.php), [imagenet2012 subset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) |
| 2    | Bert-Large         | [MLPerf Bert-Large ONNX](https://github.com/mlcommons/inference/tree/r1.1/language/bert)    |              | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)  |
| 3    | Bert-Base          | [Bert int8 ONNX](https://github.com/onnx/models/tree/main/text/machine_comprehension/bert-squad)       |   |        |
| 4    | YOLO_v5            | [YOLOv5 pt-&gt;ONNX](https://github.com/itsnine/yolov5-onnxruntime), [YoloV5 Official](https://github.com/ultralytics/yolov5/issues/251) , [YoloV5 pt models](https://github.com/ultralytics/yolov5/releases) |    |      |
| 5    | VGG16              | [VGG16, VGG16 int8 ONNX](https://github.com/onnx/models/tree/main/vision/classification/vgg) |  |      |
| 6    | Tacotron2          | [Tacotron2 代码可以生成 ONNX 模型](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) |      |          |
| 7    | MaskRCNN           | [MaskRCNN ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/mask-rcnn)  |      |       |
| 8    | Transformer        | [nvidia Transformer .pt](https://catalog.ngc.nvidia.com/orgs/nvidia/models/transformer_pyt_ckpt_tf32/files)                                                                                             |                                                                                                      |                                                                                                                                                                           |
| 9    | TDNN               |                                                                                                                                                                                                      |                                                                                                      |                                                                                                                                                                           |
| 10   | TextCNN            |                                                                                                                                                                                                      |                                                                                                      |                                                                                                                                                                           |
| 11   | RNNT               | [RNNT ONNX](https://github.com/mlcommons/inference)                                                                                                                                                     |                                                                                                      |                                                                                                                                                                           |
| 12   |                    |                                                                                                                                                                                                      |                                                                                                      |                                                                                                                                                                           |
| 13   | seq2seq            |                                                                                                                                                                                                      |                                                                                                      |                                                                                                                                                                           |
| 14   | DBnet              |                                                                                                                                                                                                      |                                                                                                      |                                                                                                                                                                           |
| 15   | DLRM               | [DLRM ONNX](https://github.com/mlcommons/inference) || |
| 16   | ViT  |   |         |    |
| 17   | resnet18           | [resnet18 ONNX](https://github.com/onnx/models/tree/main/vision/classification/resnet)  | |  |
| 18   | resnet34           | [resnet34 ONNX](https://github.com/onnx/models/tree/main/vision/classification/resnet) | |  |
| 19   | resnet101          | [resnet101 pth](https://pytorch.org/hub/pytorch_vision_resnet/) | [code &amp; onnx convert](./pytorch_model_convert/resnet101_resnext50/resnet101.py)   | ImageNet1K |
| 20   | resnet101          | [resnet101 ONNX](https://github.com/onnx/models/tree/main/vision/classification/resnet)   | |  |
| 21   | resNeXt            | [resnext pth](https://pytorch.org/hub/pytorch_vision_resnet/) | [code &amp; onnx convert](./pytorch_model_convert/resnet101_resnext50/resnet101.py)  | ImageNet1K |
| 22   | YOLO_v3_Tiny       | [Yolov3-Tiny ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3)  | | |
| 23   | YOLO_v3            | [Yolov3 ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov3)   |   |     |
| 24   | Inception_v1       | [Inception_v2](https://github.com/onnx/models)  |    |     |
| 25   | Inception_v2       | [Inception_v2](https://github.com/onnx/models)   |      |      |
| 26   | Inception_v3       | [Inception_v3](https://github.com/onnx/models)    |          |         |
| 27   | [Inception_v4](https://arxiv.org/pdf/1602.07261.pdf)|[v4 tf](https://github.com/PanJinquan/tensorflow_models_learning) |
| 28   | MobileNet          | [mobilenet_v1 tf](https://github.com/Zehaos/MobileNet)    |              |        |
| 29   | MobileNet_v2       | [mobilenet_v2 pt](./pytorch_model_convert/mobilenetv2.pytorch/README.md)   | [code &amp; onnx convert](./pytorch_model_convert/mobilenetv2.pytorch/imagenet.py)     |  |
| 30   | MobileNet_v3  |  |    |   |
| 31   | [ShuffleNet](https://arxiv.org/pdf/1707.01083v2.pdf) | [pt](https://github.com/jaxony/ShuffleNet)  |   |  ImageNet1K  |
| 32   | [R-FCN](https://arxiv.org/pdf/1605.06409.pdf) | [tf](https://github.com/RobertCsordas/RFCN-tensorflow)  |   |  ImageNet1K  |
| 33   | [FPN](https://arxiv.org/pdf/1612.03144.pdf) | [pt](https://github.com/jwyang/fpn.pytorch)  |   |  ImageNet1K  |
| 34   | SSD_mobilenetv1    | [SSD_mb1 .pth](https://github.com/qfgaohao/pytorch-ssd)   | [code &amp; onnx convert (L26 enable)](./pytorch_model_convert/ssd_mobilenetv1/vision/ssd/predictor.py) | COCO2017  |
| 35   | SSD  | [SSD ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd)   |     |     |
| 36   | SSD-Resnext50 800x800 | [SSD ONNX](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection)   |     | [openimages](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/tools/openimages_mlperf.sh)|
| 37   | SSD-Resnet34  | [SSD-Resnet34](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/ssd)   |     |     |
| 38   | FasterRCNN-R50-FPN | [FasterRCNN ONNX](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/faster-rcnn)    |     |  |
| 39   | FasterRCNN  | [pth](https://github.com/longcw/faster_rcnn_pytorch) |     |COCO2017|
| 40   | FasterRCNN-R50| [mmdetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) |     |VOC2007|
| 41   | FasterRCNN  | [pth](https://github.com/longcw/faster_rcnn_pytorch) |     |COCO2017|
| 42   | SoloV2  | [pth](https://github.com/Epiphqny/SOLOv2) |     |COCO2017|
| 43   | deeplabv3  | [pth](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) | [code & onnx converter](./pytorch_model_convert/deeplabv3/deeplabv3.py) |COCO2017|
| 44   | PSPNet  | [pth](https://github.com/Lextal/pspnet-pytorch) |     |COCO2017|
| 45   | [retinaface](https://arxiv.org/pdf/1905.00641.pdf)|[pth_code](https://github.com/biubug6/Pytorch_Retinaface), [pth_googledrive](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1)| |[widerface](https://drive.google.com/file/d/11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS/view) |
| 46   | SoloV2  | [pth](https://github.com/Epiphqny/SOLOv2) |     |COCO2017|
| 47   | 3D-UNet            | [3D-UNet ONNX](https://github.com/mlcommons/inference)   |       |  |

---

### 模型分析 与 模型转换

---

[VGG16 模型结构代码](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) : 未使用BN, 5组(3x3)卷积之间用maxpooling, FC 使用(1x1)卷积替换

[VGG16 ckpt 模型](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz), [Tensorflow ckpt 转 ONNX](./tensorflow_model_convert/tensorflow_onnx_converter.py)

---

[nvidia Transformer Code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer)

使用 nvidia Transformer 代码进行推理, 下载上述 .pt 模型, 准备 en ->

---

---

### ONNX Model Operator List

#### CV OP List

| 序号 |       算子       | VGG16 | YoloV3 | YoloV5 | MaskRCNN |
| :--- | :---------------: | :---: | :----: | :----: | :------: |
| 1    |        Log        |      |        |        |    ✓    |
| 2    |        Sub        |      |        |        |    ✓    |
| 3    |       Floor       |      |        |        |    ✓    |
| 4    |     Unsqueeze     |      |        |        |    ✓    |
| 5    |      Scatter      |      |        |        |    ✓    |
| 6    |       Conv       |      |        |        |    ✓    |
| 7    |   ConvTranspose   |      |        |        |    ✓    |
| 8    |       TopK       |      |        |        |    ✓    |
| 9    |       Relu       |      |        |        |    ✓    |
| 10   |      Resize      |      |        |        |    ✓    |
| 11   |       Cast       |      |        |        |    ✓    |
| 12   |      Expand      |      |        |        |    ✓    |
| 13   |      Sigmoid      |      |        |        |    ✓    |
| 14   |        Add        |      |        |        |    ✓    |
| 15   | NonMaxSuppression |      |        |        |    ✓    |
| 16   |      Gather      |      |        |        |    ✓    |
| 17   |      MaxPool      |      |        |        |    ✓    |
| 18   |        Mul        |      |        |        |    ✓    |
| 19   |       Less       |      |        |        |    ✓    |
| 20   |     Constant     |      |        |        |    ✓    |
| 21   |       Shape       |      |        |        |    ✓    |
| 22   |       Split       |      |        |        |    ✓    |
| 23   |     ReduceMin     |      |        |        |    ✓    |
| 24   |        And        |      |        |        |    ✓    |
| 25   |       Gemm       |      |        |        |    ✓    |
| 26   |       Equal       |      |        |        |    ✓    |
| 27   |      Reshape      |      |        |        |    ✓    |
| 28   |      Softmax      |      |        |        |    ✓    |
| 29   |      Flatten      |      |        |        |    ✓    |
| 30   |      NonZero      |      |        |        |    ✓    |
| 31   |       Clip       |      |        |        |    ✓    |
| 32   |     RoiAlign     |      |        |        |    ✓    |
| 33   |        Not        |      |        |        |    ✓    |
| 34   |      Concat      |      |        |        |    ✓    |
| 35   |        Div        |      |        |        |    ✓    |
| 36   |      Greater      |      |        |        |    ✓    |
| 37   |        Exp        |      |        |        |    ✓    |
| 38   |     Transpose     |      |        |        |    ✓    |
| 39   |      Squeeze      |      |        |        |    ✓    |
| 40   |       Slice       |      |        |        |    ✓    |
| 41   |  ConstantOfShape  |      |        |        |    ✓    |
| 42   |       Sqrt       |      |        |        |    ✓    |

#### NLP/Audio OP List

| 序号 | 算子 | Bert | Tacotron2 | RNNT |
| :--- | :--: | :--: | :-------: | :--: |
| 1    |      |      |          |      |
| 2    |      |      |          |      |
| 3    |      |      |          |      |
| 4    |      |      |          |      |
| 5    |      |      |          |      |
| 6    |      |      |          |      |
| 7    |      |      |          |      |
| 8    |      |      |          |      |
| 9    |      |      |          |      |
| 10   |      |      |          |      |
| 11   |      |      |          |      |
| 12   |      |      |          |      |
| 13   |      |      |          |      |
| 14   |      |      |          |      |
| 15   |      |      |          |      |
| 16   |      |      |          |      |
| 17   |      |      |          |      |
| 18   |      |      |          |      |
| 19   |      |      |          |      |

---
