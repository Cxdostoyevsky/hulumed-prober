import os

import torch
import torch.nn as nn
from transformers import (CLIPImageProcessor, CLIPVisionConfig,
                          CLIPVisionModel, SiglipImageProcessor,
                          SiglipVisionConfig, SiglipVisionModel)

from .hulumed_encoder import (HulumedVisionEncoderConfig,
    HulumedVisionEncoderModel, HulumedImageProcessor)


class CLIPVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = CLIPVisionModel.from_pretrained(self.vision_encoder_name,
                                                            attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = SiglipVisionModel.from_pretrained(self.vision_encoder_name,
                                                              attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size



class HulumedVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False, vision_encoder_config=None):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.args = args
        # For merged models: the vision encoder config is embedded in the main config
        self._vision_encoder_config = vision_encoder_config

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model(self.args)
        else:
            self.attn_implementation = 'sdpa' # 'eager'
            if vision_encoder_config is not None:
                cfg_dict = vision_encoder_config if isinstance(vision_encoder_config, dict) else vars(vision_encoder_config)
                self.cfg_only = HulumedVisionEncoderConfig(**cfg_dict)
            else:
                self.cfg_only = HulumedVisionEncoderConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self, args):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        if self._vision_encoder_config is not None:
            # Merged model: build architecture from embedded config,
            # weights will be loaded by the main model's from_pretrained
            cfg_dict = self._vision_encoder_config if isinstance(self._vision_encoder_config, dict) else vars(self._vision_encoder_config)
            vec_config = HulumedVisionEncoderConfig(**cfg_dict)
            self.cfg_only = vec_config
            self.image_processor = HulumedImageProcessor.from_pretrained(self.vision_encoder_name)
            self.vision_encoder = HulumedVisionEncoderModel(vec_config)
        else:
            # Separate vision encoder: load from pretrained path
            self.image_processor = HulumedImageProcessor.from_pretrained(self.vision_encoder_name)
            self.cfg_only = HulumedVisionEncoderConfig.from_pretrained(self.vision_encoder_name)
            self.vision_encoder = HulumedVisionEncoderModel.from_pretrained(
                self.vision_encoder_name,
                torch_dtype=args.torch_dtype,
                attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def forward(self, pixel_values, grid_sizes, merge_sizes, stop_at_layer=None, **kwargs):
        image_features = self.vision_encoder(
            pixel_values, grid_sizes, merge_sizes,
            stop_at_layer=stop_at_layer if hasattr(self.vision_encoder, 'encoder') else None
        )
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return -1

    @property
    def num_patches_per_side(self):
        return -1

    @property
    def image_size(self):
        return -1


def build_vision_encoder(vision_encoder_cfg, **kwargs):
    vision_encoder = getattr(vision_encoder_cfg, 'mm_vision_encoder', getattr(vision_encoder_cfg, 'vision_encoder', None))

    if vision_encoder is not None:
        # Standard path-based loading (separate vision encoder model)
        if 'clip' in vision_encoder:
            return CLIPVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
        elif 'siglip' in vision_encoder:
            return SiglipVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
        elif 'navit' in vision_encoder.lower():
            return HulumedVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
        else:
            raise ValueError(f'Unknown vision encoder: {vision_encoder}')

    # Fallback: merged model with embedded vision_encoder_config
    vec = getattr(vision_encoder_cfg, 'vision_encoder_config', None)
    if vec is not None:
        model_type = vec.get('model_type', '') if isinstance(vec, dict) else getattr(vec, 'model_type', '')
        model_path = getattr(vision_encoder_cfg, '_name_or_path', None)
        if 'hulumed' in model_type and model_path:
            return HulumedVisionEncoder(model_path, args=vision_encoder_cfg, vision_encoder_config=vec, **kwargs)

    raise ValueError(
        f'Cannot determine vision encoder from config. '
        f'Expected mm_vision_encoder, vision_encoder (str path), or vision_encoder_config (dict). '
        f'Got vision_encoder={vision_encoder}, vision_encoder_config={vec}'
    )
