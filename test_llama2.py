import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FeatureExtractionPipeline
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import llm_reconstruction_free
llm_reconstruction_free.activate()

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "I am a secretary. My sex is "
tokens = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0).long()
print(model(tokens, output_hidden_states=True))
asdf

model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

#model.to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)

print(tokenizer.batch_decode(generated_ids)[0])


