# https://github.com/huggingface/notebooks/blob/main/examples/onnx-export.ipynb


from pathlib import Path
from transformers.convert_graph_to_onnx import convert

model_path = "/public/solution/2022_sgcc/bert_base/onnx"

# Handles all the above steps for you
convert(framework="pt", model="bert-base-uncased", output=Path("onnx/bert-base-uncased.onnx"), opset=11)

