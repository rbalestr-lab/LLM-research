import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FeatureExtractionPipeline
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)


class MistralBackbone(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__int__(self)
        self.backbone = backbone
        self.head = head

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    
        print("Using custom forward")
    
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
        if ft_labels is None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            loss = torch.nn.functional.cross_entropy(shift_logits, input_ids.flatten())
            print("Pretraining loss value", loss)
        else:
            loss = torch.nn.functional.cross_entropy(logits, ft_labels.flatten())
            print("Finetuning loss value", loss)
    
        return dict(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
