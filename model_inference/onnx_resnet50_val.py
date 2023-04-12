import onnx, cv2, torch, shutil, os, random
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

onnx_path = "/home/gyf/pkg/xxgg/github/ai_app/quantization_tools/intel/outputs/resnet50_v1_quant.onnx"
onnx_path = "/home/gyf/pkg/xxgg/github/ai_app/quantization_tools/onnxruntime-inference-examples/quantization/image_classification/cpu/resnet50_v1-13_quant_x.onnx"
onnx_model = onnx.load(onnx_path)
# onnx.checker.check_model(onnx_model)


def resize_with_aspectratio(img, output_h, output_w, scale = 87.5):
    height, width, _ = img.shape
    tmp_h = int(100. * output_h / scale)
    tmp_w = int(100. * output_w / scale)
    
    if height > width:
        w = tmp_w
        h = int(tmp_h * height / width)
    else:
        h = tmp_h
        w = int(tmp_w * width / height)
    
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img

def center_crop(img, output_h, output_w):
    height, width, _ = img.shape
    left = int((width - output_w) / 2)
    right = int((width + output_w) / 2)
    top = int((height - output_h) / 2)
    bottom = int((height + output_h) / 2)
    img = img[top:bottom, left:right]
    return img

def img_to_tensor(img_path):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_with_aspectratio(img, 224, 224, scale=87.5)
    img = center_crop(img, 224, 224)

    img = (np.array(img, dtype=np.float32) -
           np.array([123.68, 116.78, 103.94], dtype=np.float32))

    img = img.transpose(2, 0, 1)

    return torch.tensor(img).unsqueeze_(0)


def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')

    # if len(img.split()) != 3:
    #     return

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform_test(img).unsqueeze_(0)

def accuracy_cal(model, img_dir, input_imgs, label_txt):
    ort_session = onnxruntime.InferenceSession(model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = []
    target = []
    label2id = read_label(label_txt)

    for img_path in input_imgs:

        img_tensor = img_to_tensor(
            os.path.join(img_dir, img_path))
        # img_tensor = img_preprocess(os.path.join('test_imgs_1000pcs', img_path))

        target.append(int(label2id[img_path]))

        ort_inputs = {ort_session.get_inputs(
        )[0].name: img_tensor.float().cpu().numpy()}
        # ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
        ort_output = ort_session.run(None, ort_inputs)
        print(ort_output)

        # if not orig:
        #     outputs.append(ort_output[0][0].argmax()) #with 1 outputs
        # else:
        #     outputs.append(ort_output[0][0]-1) #with 2 outputs
        outputs.append(ort_output[0][0]-1)

    assert len(outputs) == len(target), "length should be same"

    return (np.array(outputs) == np.array(target)).sum()/len(target)


def read_label(label_txt):

    with open(label_txt, 'r') as f:
        labels = f.read().strip()
    labels = labels.split('\n')
    label2id = {x.split(' ')[0]: x.split(' ')[1] for x in labels}

    return label2id


def img_calib_choice(input_path, tar_path):
    imgs_selected = random.sample(os.listdir(input_path), 50)
    for i in imgs_selected:
        if len((Image.open(os.path.join(input_path, i))).split()) == 3:
            shutil.move(os.path.join(input_path, i), os.path.join(tar_path, i))

def main():
    img_dir = "/home/gyf/pkg/xxgg/github/ai_app/mlperf_inference/vision/classification_and_detection/data/img"
    label_path = "/home/gyf/pkg/xxgg/github/ai_app/mlperf_inference/vision/classification_and_detection/data/ILSVRC2012_img_val_labels.txt"

    input_imgs = os.listdir(img_dir)
    result = accuracy_cal(onnx_path, img_dir, input_imgs, label_path)
    print(result)

if __name__ == '__main__':
    main()
    