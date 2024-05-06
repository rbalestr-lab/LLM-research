import transformers
from llm_reconstruction_free import modeling_openelm, MODELS
import torch
from torch import nn
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, CausalLMOutput
)
import wandb


class LLMClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x.mean(1))


class CustomConfig(transformers.PretrainedConfig):
    model_type = 'custombackbonehead'
    def __init__(self, backbone_config=None, backbone_name=None, backbone_pretrained=None, task=None, in_features=None, out_features=None, **kwargs):
        self.backbone_config = backbone_config
        self.backbone_pretrained = backbone_pretrained
        self.backbone_name = backbone_name

        self.task = task
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(**kwargs)

def name_to_lora(name):
    if "apple" in name:
        return ["qkv_proj", "out_proj"]
    elif "gpt2" in name:
        return ["c_proj.weight"]
    else:
        return ["q_proj", "k_proj", "v_proj", "dense"]



class CustomBackboneHead(transformers.PreTrainedModel):
    config_class = CustomConfig
    def __init__(self, config):
        super().__init__(config)
        if "apple" in config.backbone_name and not config.backbone_pretrained:
            model = modeling_openelm.OpenELMModel(config.backbone_config)
        elif type(config.backbone_pretrained) == bool and config.backbone_pretrained:
            if "apple" in config.backbone_name:
                model = transformers.AutoModelForCausalLM.from_pretrained(config.backbone_name, trust_remote_code=True)
                model = model.transformer
            else:
                model = transformers.AutoModel.from_pretrained(config.backbone_name, trust_remote_code=True)
        elif config.backbone_pretrained:
            model = transformers.AutoModel.from_pretrained(config.backbone_pretrained)
        else:
            model = transformers.AutoModel.from_config(config.backbone_config)

        if "arctic" in config.backbone_name:
            model.pooler = torch.nn.Identity()

        self.backbone = model

        if config.task == "lm":
            self.head = nn.Linear(config.in_features, config.out_features, bias=False)
        else:
            self.head = LLMClassifier(config.in_features, config.out_features)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def set_decoder(self, decoder):
        self.backbone = decoder

    def get_decoder(self):
        return self.backbone

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ft_labels=None,
    ) -> Union[Tuple, CausalLMOutput]:
    
    
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
    
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
    
        logits = self.head(hidden_states)
        logits = logits.float()
    
        loss = None
        if self.config.task == "lm":
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.out_features)
            loss = torch.nn.functional.cross_entropy(shift_logits, input_ids[:,1:].flatten())
        else:
#            wandb.log({"acc":(logits.detach().argmax(-1)==ft_labels).float().mean().item()})
            loss = torch.nn.functional.cross_entropy(logits, ft_labels.flatten())
    
        return CausalLMOutput(
            loss=loss,
            logits=logits,
#            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_model(name, size="original", pretrained=True, task="lm", num_classes=None):
    if name not in MODELS:
        raise ValueError(f"`{name}` must be in {MODELS}")
    if pretrained and size != "original":
        raise ValueError("size must be `original` when using `pretrained=True`")
    if task not in ["lm", "ft"]:
        raise ValueError(
            "Task must be one of `lm` (next-token prediction), `ft` (supervised sequence classification)"
        )
    if task == "ft" and num_classes is None:
        raise ValueError("`num_classes` should be provided when using `task=lm`")

    backbone_config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)

    if task == "ft":
        out_features = num_classes
    else:
        out_features = backbone_config.vocab_size
    if "mistral" in name:
        in_features = backbone_config.hidden_size
    elif "apple" in name:
        in_features = backbone_config.model_dim
    elif "microsoft" in name:
        in_features = backbone_config.hidden_size
    elif "gpt2" in name:
        in_features = backbone_config.n_embd
    elif "arctic" in name:
        in_features = backbone_config.hidden_size

    config = CustomConfig(backbone_config=backbone_config, backbone_name=name, backbone_pretrained=pretrained, task=task, in_features=in_features, out_features=out_features)
    model = CustomBackboneHead(config)
    return model


def get_tokenizer(name, pretrained, dataset=None):

    if pretrained:
        tk_name = "meta-llama/Llama-2-7b-hf" if "apple" in name else name
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tk_name if type(pretrained) == bool else pretrained,
            trust_remote_code=True,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        assert dataset is not None

        from tokenizers.trainers import BpeTrainer
        tokenizer = Tokenizer(BPE())
        # tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        # optional
        #tokenizer.normalizer = normalizers.Sequence(
        #    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        #)

        
        from tokenizers.pre_tokenizers import Whitespace
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        # trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
        tokenizer.train_from_iterator(train_dataset, trainer=trainer)
    return tokenizer
