from pathlib import Path
import transformers
from transformers.onnx import FeaturesManager
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# load model and tokenizer
feature = "causal-lm"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# load config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path("gpt2_nick.onnx")
)
