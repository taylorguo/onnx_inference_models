import torch
import requests
import os


# model_file = "resnet101-63fe2227.pth"
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()


# import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.request().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
file_name = "dog.jpg"
if not os.path.exists(file_name):
    im_content = requests.get(url)
    with open(file_name, "wb") as im_file:
        im_file.write(im_content.content)


from PIL import Image
from torchvision import transforms
input_image = Image.open(file_name)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)



# Download ImageNet labels
imagenet_label_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

file_name = "imagenet_classes.txt"
if not os.path.exists(file_name):
    label_content = requests.get(imagenet_label_url)
    with open(file_name, "wb") as label_file:
        label_file.write(label_content.content)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


torch.onnx.export(model, input_batch, "resnet101_pth.onnx", verbose=True,
                  input_names=["input"], output_names=["output"]  )