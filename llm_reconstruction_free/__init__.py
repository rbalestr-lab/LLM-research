from . import mistral
import transformers

def activate():
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.forward = mistral.forward
