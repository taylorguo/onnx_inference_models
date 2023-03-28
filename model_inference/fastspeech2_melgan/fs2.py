import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf
import tf2onnx

from tensorflow_tts.inference import AutoProcessor, AutoConfig, TFAutoModel

processor = AutoProcessor.from_pretrained(pretrained_path = "processor.json") # BakerProcessor
config = AutoConfig.from_pretrained("config.yml")
fastspeech2 = TFAutoModel.from_pretrained(pretrained_path = "model.h5", config=config, **{"enable_tflite_convertible": True})

text = "这是一个开源的端到端中文语音合成系统"
input_ids = processor.text_to_sequence(text, inference=True)

mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference_tflite(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)



# processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")
# fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")


# save to tensorflow .pb file for further converting to ONNX
tf.saved_model.save(fastspeech2, "./fastspeech2_tflite_model", signatures=fastspeech2.inference_tflite)
# python -m tf2onnx.convert --saved-model fastspeech2_tf_model --output test.onnx --opset 16 --inputs speed_ratios:0[1],speaker_ids:0[1],input_ids:0[1, 56],f0_ratios:0[1],energy_ratios:0[1]


# # convert to onnx model directly
input = [
    tf.TensorSpec((1, len(input_ids)), tf.int32, name="inputs"),
    tf.TensorSpec((1,), tf.int32, name="speaker_ids"),
    tf.TensorSpec((1), tf.float32, name="speed_ratios"),
    tf.TensorSpec((1), tf.float32, name="f0_ratios"),
    tf.TensorSpec((1), tf.float32, name="energy_ratios"),
]
inputs=tf.TensorSpec((None, len(input_ids)), dtype=tf.int32)
onnx_model = tf2onnx.convert.from_keras(fastspeech2, input_signature=input, opset=13, output_path="keras.onnx")
