from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.clip.modeling_clip import CLIPPreTrainedModel, CLIPVisionTransformer, CLIPTextTransformer
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


# TODO These classes were introduced temporarily because I (ZanSara) didn't manage to
# use CLIPTextModel and CLIPVisionModel properly. Please re-evaluate the need for these
# classes once MultiModalRetriever gets stable.
#
# Source: https://github.com/huggingface/transformers/blob/b487096b02307cd6e0f132b676cdcc7255fe8e74/src/transformers/models/clip/modeling_clip.py


class CLIPModelAdapterVision(CLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        self.post_init()

    def forward(self, pixel_values: Optional[torch.FloatTensor] = None) -> Tuple:

        vision_outputs = self.vision_model(
            pixel_values=pixel_values, output_attentions=False, output_hidden_states=False, return_dict=False
        )
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        return (image_embeds, vision_outputs)


class CLIPModelAdapterText(CLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        text_config = config.text_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple:

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        return (text_embeds, text_outputs)
