import os
import numpy as np
import soundfile as sf
import yaml
import tensorflow as tf
import tf2onnx

from tensorflow_tts.inference import AutoProcessor, AutoConfig, TFAutoModel


json_path = "/home/gyf/models/tts-fastspeech2-baker-ch/processor.json"
config_path_fs2 = "/home/gyf/models/tts-fastspeech2-baker-ch/config.yml"
config_path_melgan = "/home/gyf/models/tts-mb_melgan-baker-ch/config.yml"
model_path_fs2 = "/home/gyf/models/tts-fastspeech2-baker-ch/model.h5"
model_path_melgan = "/home/gyf/models/tts-mb_melgan-baker-ch/model.h5"

onnx_name_fs2 = "fastspeech2_noloop_1x56_opset15.onnx"
onnx_save_fs2 = os.path.join(os.path.dirname(__file__), onnx_name_fs2)


if os.path.exists(model_path_fs2) and os.path.exists(model_path_melgan) and os.path.exists(config_path_fs2) and os.path.exists(json_path):
    processor = AutoProcessor.from_pretrained(pretrained_path = json_path) # BakerProcessor
    config = AutoConfig.from_pretrained(config_path_fs2)
    fastspeech2 = TFAutoModel.from_pretrained(pretrained_path = model_path_fs2, config=config, **{"enable_tflite_convertible": True})
    config = AutoConfig.from_pretrained(config_path_melgan)
    mb_melgan = TFAutoModel.from_pretrained(pretrained_path=model_path_melgan, config=config)
else:
    # import models from hugginface
    processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch")
    fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-baker-ch", **{"enable_tflite_convertible": True})
    mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-baker-ch")


text = "这是一个开源的端到端中文语音合成系统"
# text = "大"
print(text)
input_ids = processor.text_to_sequence(text, inference=True)

mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference_tflite(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)


# save to tensorflow .pb file for further converting to ONNX
saved_pb_path = "fastspeech2_tflite_model/saved_model.pb"
if not os.path.exists(saved_pb_path):
    tf.saved_model.save(fastspeech2, "./fastspeech2_tflite_model", signatures=fastspeech2.inference)
# python -m tf2onnx.convert --saved-model fastspeech2_tf_model --output test.onnx --opset 16 --inputs speed_ratios:0[1],speaker_ids:0[1],input_ids:0[1, 56],f0_ratios:0[1],energy_ratios:0[1]


# convert fastspeech2 to onnx model directly
inputs = [
    tf.TensorSpec((1, len(input_ids)), tf.int32, name="input_ids:0"),
    tf.TensorSpec((1,), tf.int32, name="speaker_ids:0"),
    tf.TensorSpec((1), tf.float32, name="speed_ratios:0"),
    tf.TensorSpec((1), tf.float32, name="f0_ratios:0"),
    tf.TensorSpec((1), tf.float32, name="energy_ratios:0"),
]

# convert fastspeech2 to ONNX
# onnx_model = tf2onnx.convert.from_keras(fastspeech2, input_signature=inputs, opset=16, output_path=onnx_save_fs2)
onnx_model = tf2onnx.convert.from_keras(fastspeech2, input_signature=inputs, opset=15, output_path=onnx_save_fs2)

####################
onnx_name_melgan = "mb_melgan_1_261_80_opset15.onnx"
# onnx_name_melgan = "mb_melgan_1_%s_80.onnx"%(str(mel_after.shape[1]))
onnx_save_melgan = os.path.join(os.path.dirname(__file__), onnx_name_melgan)

# melgan inference (mel-to-wav)
audio = mb_melgan.inference(mel_after)[0, :, 0]

# save to file
sf.write('test_audio_tf.wav', audio, 22050, "PCM_16")

inputs = [
    # tf.TensorSpec((1, len(input_ids)), tf.int32, name="input_ids:0"),
    tf.TensorSpec(shape=[1, mel_after.shape[1], 80], dtype=tf.float32, name="mel_after")
]

# convert mb_melgan to ONNX
# onnx_model = tf2onnx.convert.from_keras(mb_melgan, input_signature=inputs, opset=16, output_path=onnx_save_melgan)
onnx_model = tf2onnx.convert.from_keras(mb_melgan, input_signature=inputs, opset=15, output_path=onnx_save_melgan)