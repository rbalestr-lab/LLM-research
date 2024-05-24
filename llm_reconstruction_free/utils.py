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
import numpy as np

class ClassifierHead(nn.Module):
    def __init__(self, in_features, n_layers, vocab_size, out_features, dropout):
        super().__init__()
        if dropout:
            self.classifier = torch.nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features * 2, out_features))
        else:
            self.classifier = nn.Linear(in_features * 2, out_features)
    def forward(self, x):
        return self.classifier(torch.concat([x.mean(1),x[:,-1]],1)


class CustomConfig(transformers.PretrainedConfig):
    model_type = 'custombackbonehead'
    def __init__(self, backbone_config=None, backbone_name=None, backbone_pretrained=None, task=None, in_features=None, out_features=None, mixup=0, dropout=0, label_smoothing=0, **kwargs):
        self.backbone_config = backbone_config
        self.backbone_pretrained = backbone_pretrained
        self.backbone_name = backbone_name

        self.task = task
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.mixup = mixup
        self.label_smoothing = label_smoothing
        super().__init__(**kwargs)

def dataset_to_max_length(name):
    if "rotten" in name:
        return 256
    elif "imdb" in name:
        return 1024


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

        if config.task == "lm":
            self.head = nn.Linear(config.in_features, 1, config.out_features, bias=False)
        else:
            self.config.tie_word_embeddings = False
            if "apple" in config.backbone_name:
                n_layers = config.backbone_config.num_transformer_layers + 1
            else:
                raise NotImplementedError("NOT IMPLEMENTEDEF OR NOT APPLE")
            self.head = ClassifierHead(config.in_features, n_layers, config.backbone_config.vocab_size, config.out_features, config.dropout)

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

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.head

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

        # apply some mixup
        if self.config.task == "ft" and self.training and self.config.mixup:
            inputs_embeds = self.backbone.token_embeddings(input_ids)
            lam = np.random.beta(self.config.mixup, self.config.mixup)
            batch_size = input_ids.size(0)
            index = torch.randperm(batch_size, device=input_ids.device)
            inputs_embeds = lam * inputs_embeds + (1 - lam) * inputs_embeds[index]
            y_a, y_b = labels, labels[index]
            attention_mask = torch.maximum(attention_mask, attention_mask[index])
            input_ids = None

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
    
        loss = None
        if self.config.task == "lm":
            logits = self.head(outputs[0])
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.out_features)
            loss = torch.nn.functional.cross_entropy(shift_logits, input_ids[:,1:].flatten())
        else:
            criterion = torch.nn.functional.cross_entropy
            logits = self.head(hidden_states)
#            shift_logits = vocabs[..., :-1, :].contiguous()
#            shift_logits = shift_logits.view(-1, self.config.backbone_config.vocab_size)
            if args.mixup:
                loss = criterion(logits, y_a.flatten(), label_smoothing=self.config.label_smoothing) * lam + (1 - lam) * criterion(logits, y_b.flatten(), label_smoothing=self.config.label_smoothing)
            else:
                loss = criterion(logits, y_a.flatten(), label_smoothing=self.config.label_smoothing)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,#outputs.hidden_states,
            attentions=None,#outputs.attentions,
        )


def get_model(name, size="original", pretrained=True, task="lm", num_classes=None, mixup=0):
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

    config = CustomConfig(backbone_config=backbone_config, backbone_name=name, backbone_pretrained=pretrained, task=task, in_features=in_features, out_features=out_features, mixup=mixup)
    model = CustomBackboneHead(config)
    return model


