import pathlib
import warnings
from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn

ENCODER_LOADING_ERROR_MSG = (
    "Could not load encoder weights: defaulting to pretrained weights"
)


class AbstractEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def feature_dim(self):
        pass

    def transform(self, x):
        return x

    def to(self, device):
        self.device = device
        return super().to(device)

    @abstractmethod
    def forward(self, x):
        pass

    def load_weights(
        self, weight_path: Union[pathlib.Path, str], strict: bool = True
    ) -> None:
        try:
            checkpoint = torch.load(weight_path, map_location="cpu")
            weight_dict = checkpoint.get("model", checkpoint)

            self.model.load_state_dict(weight_dict, strict=strict)
            return

        except RuntimeError:
            warnings.warn("Couldn't load full model weights. Trying fallback options...")

        fallback_prefixes = [
            "model.",
            "encoder.",
            "module.base_encoder._orig_mod.",
        ]

        for prefix in fallback_prefixes:
            try:
                filtered_weights = {
                    k.replace(prefix, ""): v
                    for k, v in weight_dict.items()
                    if k.startswith(prefix)
                }

                filtered_weights.pop("fc.weight", None)
                filtered_weights.pop("fc.bias", None)

                self.model.load_state_dict(filtered_weights, strict=strict)
                return

            except RuntimeError:
                warnings.warn(f"Failed loading with prefix: '{prefix}'")

        warnings.warn(ENCODER_LOADING_ERROR_MSG)
