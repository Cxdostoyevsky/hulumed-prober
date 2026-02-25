# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, Qwen3Config,
                          Qwen3ForCausalLM, Qwen3Model)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from hulumed.constants import IGNORE_INDEX
from .hulumed_arch import HulumedMetaForCausalLM, HulumedMetaModel
from .probe import ProbeHead


class HulumedQwen2Config(Qwen3Config):
    model_type = "hulumed_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 如果传入的 kwargs 里有 model_type，就用传入的，否则默认为 hulumed_qwen2
        self.model_type = kwargs.get("model_type", "hulumed_qwen2")
        # 探针配置：vision_encoder_0, vision_encoder_5, projector, llm_0, llm_12 等
        self.probe_position = kwargs.get("probe_position", None)
        self.probe_type = kwargs.get("probe_type", "mlp")
        self.probe_hidden_dim = kwargs.get("probe_hidden_dim", None)


class HulumedQwen3Config(HulumedQwen2Config):
    model_type = "hulumed_qwen3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "hulumed_qwen3"
        self.probe_position = kwargs.get("probe_position", None)
        self.probe_type = kwargs.get("probe_type", "mlp")
        self.probe_hidden_dim = kwargs.get("probe_hidden_dim", None)


class HulumedQwen2Model(HulumedMetaModel, Qwen3Model):
    config_class = HulumedQwen2Config

    def __init__(self, config: HulumedQwen2Config):
        super(HulumedQwen2Model, self).__init__(config)


class HulumedQwen3Model(HulumedMetaModel, Qwen3Model):
    config_class = HulumedQwen3Config

    def __init__(self, config: HulumedQwen3Config):
        super(HulumedQwen3Model, self).__init__(config)


class HulumedQwen2ForCausalLM(Qwen3ForCausalLM, HulumedMetaForCausalLM):
    config_class = HulumedQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = HulumedQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _build_probe(self, probe_position: str, probe_type: str, num_classes: int = 3):
        probe_input_dim = self.model._get_probe_input_dim(self.config, probe_position)
        probe = ProbeHead(
            input_dim=probe_input_dim,
            num_classes=num_classes,
            probe_type=probe_type,
            hidden_dim=getattr(self.config, "probe_hidden_dim", None),
        )
        self.model.probe = probe.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor], default: int = 3) -> int:
        for key in ("classifier.weight", "classifier.2.weight"):
            if key in state_dict and state_dict[key].ndim == 2:
                return int(state_dict[key].shape[0])
        return default

    def load_probe_weights(self, probe_ckpt_path: str, probe_position: str, probe_type: str):
        state = torch.load(probe_ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        cleaned_state: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("model.probe."):
                cleaned_state[k[len("model.probe."):]] = v
            elif k.startswith("probe."):
                cleaned_state[k[len("probe."):]] = v
            else:
                cleaned_state[k] = v

        num_classes = self._infer_num_classes_from_state_dict(cleaned_state)
        self.config.probe_position = probe_position
        self.config.probe_type = probe_type
        self._build_probe(probe_position=probe_position, probe_type=probe_type, num_classes=num_classes)
        self.model.probe.load_state_dict(cleaned_state, strict=True)
        self.model.probe.eval()

    @torch.no_grad()
    def predict_probe(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        probe_position = getattr(self.config, "probe_position", None)
        if probe_position is None or getattr(self.model, "probe", None) is None:
            raise ValueError("Probe is not configured. Call load_probe_weights(...) first.")

        probe_features, _ = self.encode_for_probe(
            probe_position=probe_position,
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modals=modals,
        )
        probe_param = next(self.model.probe.parameters())
        if probe_features.device != probe_param.device or probe_features.dtype != probe_param.dtype:
            probe_features = probe_features.to(device=probe_param.device, dtype=probe_param.dtype)
        probe_logits = self.model.probe(probe_features)
        probe_probs = torch.softmax(probe_logits, dim=-1)
        probe_preds = torch.argmax(probe_probs, dim=-1)
        return {"logits": probe_logits, "probs": probe_probs, "preds": probe_preds}

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
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        probe_labels: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        probe_position = getattr(self.config, 'probe_position', None)
        if probe_labels is not None and probe_position is not None and getattr(self.model, 'probe', None) is not None:
            probe_features, _ = self.encode_for_probe(
                probe_position=probe_position,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                modals=modals,
            )
            probe_logits = self.model.probe(probe_features)
            probe_loss = nn.functional.cross_entropy(probe_logits, probe_labels)
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            return CausalLMOutputWithPast(
                loss=probe_loss,
                logits=probe_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != IGNORE_INDEX
            shift_hidden_states = shift_hidden_states[mask]
            shift_labels = shift_labels[mask]
            logits = self.lm_head(shift_hidden_states)
            if "num_items_in_batch" in loss_kwargs:
                loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="sum")
                loss = loss / loss_kwargs["num_items_in_batch"]
            else:
                loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX)

        else:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        return_probe: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Tuple[Union[GenerateOutput, torch.LongTensor], Optional[Dict[str, torch.Tensor]]]]:
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)
        probe_output = None

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        if return_probe and getattr(self.model, "probe", None) is not None:
            probe_output = self.predict_probe(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        outputs = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if return_probe:
            return outputs, probe_output
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs



class HulumedQwen3ForCausalLM(HulumedQwen2ForCausalLM):
    """Qwen3 版本，只是 config_class 不同，所有逻辑继承自 HulumedQwen2ForCausalLM"""
    config_class = HulumedQwen3Config

    def __init__(self, config, **kwargs):
        # 用 HulumedQwen3Model 替代 HulumedQwen2Model
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = HulumedQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


AutoConfig.register("hulumed_qwen2", HulumedQwen2Config)
AutoModelForCausalLM.register(HulumedQwen2Config, HulumedQwen2ForCausalLM)

# 这里是进行一个注册：告诉autoconfig，当你在config.json中看到model_type为hulumed_qwen3时，使用HulumedQwen3Config这个类来创建config对象
# 同时，告诉automodelforcausallm，配置是 HulumedQwen3Config 类型时，请用 HulumedQwen3ForCausalLM 这个类来创建模型。
AutoConfig.register("hulumed_qwen3", HulumedQwen3Config)
AutoModelForCausalLM.register(HulumedQwen3Config, HulumedQwen3ForCausalLM)