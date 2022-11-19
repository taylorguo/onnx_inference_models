import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

# convert checkpoint to pb
def checkpoint_converter(ckpt_file, pb_file):
    reader = tf.train.NewCheckpointReader(ckpt_file)

    with tf.Session() as sess:
        for key in reader.get_variable_to_shape_map():
            tf.Variable(reader.get_tensor(key), name=key)
            print("Tensor_Name: ", key)
            # print(reader.get_tensor(key))

        with tf.gfile.FastGFile(pb_file, "wb") as f:
            f.write(sess.graph_def.SerializeToString())

def pb_onnx_converter(pb_file, onnx_file):
    pass

# convert tensorflow pb to onnx, use below tools 
# https://github.com/onnx/tensorflow-onnx
# https://github.com/onnx/tensorflow-onnx/issues/1775
# change the name to "saved_model.pb" 

if __name__ == '__main__':
    ckpt_file = "/public/ai_platform/CMCC/vgg16/vgg_16.ckpt"
    ckpt_file = "/home/gyf/pkg/xxgg/github/ai_app/suinfer_model_coverage/tensorflow_model_convert/models/vgg_16_cmcc_ckpt.pb"
    pb_file = "./vgg_16_cmcc.pb"
    checkpoint_converter(ckpt_file, pb_file)