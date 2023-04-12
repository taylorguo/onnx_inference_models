import numpy
import onnxruntime as rt
import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor
import soundfile as sf

from fs2 import onnx_save_fs2, onnx_name_melgan

json_path = "processor.json"
# Preprocess inpiuts
processor = AutoProcessor.from_pretrained(pretrained_path=json_path)

# Inputs
text = "大家好"
text = "这是一个开源的端到端中文语音合成系统"

input_ids = numpy.asarray(processor.text_to_sequence(text, inference=True), dtype=numpy.int32)
input_ids = input_ids.reshape((1, len(input_ids)))
speaker_ids = numpy.asarray((0)).reshape((1))
speed_ratios = numpy.asarray((1.0)).reshape((1))
f0_ratios = numpy.asarray((1.0)).reshape((1))
energy_ratios = numpy.asarray((1.0)).reshape((1))

input_ids.tofile("input_ids.bin")
speed_ratios.astype(numpy.float32).tofile("speed_ratios.bin")
speaker_ids.astype(numpy.int32).tofile("speaker_ids.bin")
energy_ratios.astype(numpy.float32).tofile("energy_ratios.bin")
f0_ratios.astype(numpy.float32).tofile("f0_ratios.bin")

# Load fastspeech2 model
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
print(" **** Using fastspeech2 : ", onnx_save_fs2)
sess_fs2 = rt.InferenceSession(onnx_save_fs2, providers=rt.get_available_providers(), sess_options=sess_options)

# Print model inputs
print(tf.expand_dims(tf.convert_to_tensor(processor.text_to_sequence(text, inference=True), dtype=tf.int32), 0).shape)
print("\nNum inputs:", len(sess_fs2.get_inputs()))
for _input in sess_fs2.get_inputs():
    print("\t", _input.name, _input.type, _input.shape)
print("")

print(input_ids.shape)
print(speed_ratios.shape)

# Run fastspeech2 model
pred_onnx = sess_fs2.run(None, {
    "input_ids:0": input_ids,
    "speed_ratios:0": speed_ratios.astype(numpy.float32),
    "speaker_ids:0": speaker_ids.astype(numpy.int32),
    "energy_ratios:0": energy_ratios.astype(numpy.float32),
    "f0_ratios:0": f0_ratios.astype(numpy.float32),
})

# print(pred_onnx[1], type(pred_onnx[1]))

print(" **** Using mb_melgan : ", onnx_name_melgan)
# Load mb_melgan model
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_melgan = rt.InferenceSession(onnx_name_melgan, providers=rt.get_available_providers(), sess_options=sess_options)

for _input in sess_melgan.get_inputs():
    print("\n\t", _input.name, _input.type, _input.shape)
print("")

print("mel_after.shape :  ", pred_onnx[1].shape)
pred_onnx[1].astype(numpy.float32).tofile("mel_after.bin")
# Run fastspeech2 model
wave = sess_melgan.run(None, {"mel_after": pred_onnx[1]})
# save to wav file
sf.write('test_audio_onnx.wav', wave[0][0, :, 0], 22050, "PCM_16")