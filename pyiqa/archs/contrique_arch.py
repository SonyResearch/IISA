r"""CONTRIQUE metric

@article{madhusudana2022image,
  title={Image quality assessment using contrastive learning},
  author={Madhusudana, Pavan C and Birkbeck, Neil and Wang, Yilin and Adsumilli, Balu and Bovik, Alan C},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4149--4161},
  year={2022},
  publisher={IEEE}
}
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
from collections import OrderedDict
import torch.nn.functional as F
import os

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pyiqa.utils.download_util import download_file_from_google_drive, DEFAULT_CACHE_DIR

model_weights_gdrive_id = "1pmaomNVFhDgPSREgHBzZSu-SuGzNJyEt"


@ARCH_REGISTRY.register()
class CONTRIQUE(nn.Module):
    def __init__(self,
                 ) -> None:
        super().__init__()

        self.encoder = resnet50()
        self.feat_dim = self.encoder.fc.in_features
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        model_weights_path = os.path.join(DEFAULT_CACHE_DIR, "CONTRIQUE_checkpoint25.tar")
        if not os.path.exists(model_weights_path):
            download_file_from_google_drive(model_weights_gdrive_id, model_weights_path)

        encoder_state_dict = torch.load(model_weights_path, map_location="cpu")
        cleaned_encoder_state_dict = OrderedDict()
        for key, value in encoder_state_dict.items():
            # Remove the prefix
            if key.startswith("encoder."):
                new_key = key[8:]
                cleaned_encoder_state_dict[new_key] = value

        self.encoder.load_state_dict(cleaned_encoder_state_dict)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.regressor = nn.Sequential(nn.Linear(self.feat_dim * 2, self.feat_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(self.feat_dim * 2, 1))

        self.default_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> float:
        """
        Forward pass of the CONTRIQUE model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            float: The predicted quality score.
        """
        x, x_ds = self._preprocess(x)

        f = F.normalize(self.encoder(x), dim=1)
        f_ds = F.normalize(self.encoder(x_ds), dim=1)
        f_combined = torch.hstack((f, f_ds)).view(-1, self.feat_dim * 2)

        score = self.regressor(f_combined)

        return score

    def _preprocess(self, x: torch.Tensor):
        """
        Downsample the input image with a factor of 2 and normalize the original and downsampled images.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The normalized original and downsampled tensors.
        """
        x_ds = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        x_ds = (x_ds - self.default_mean.to(x_ds)) / self.default_std.to(x_ds)
        return x, x_ds

    def state_dict(self, *args, **kwargs):
        """
        Overrides the default state_dict method to exclude encoder weights.
        """
        full_state_dict = super().state_dict(*args, **kwargs)
        filtered_state_dict = OrderedDict()
        for key, value in full_state_dict.items():
            if not key.startswith("encoder."):
                filtered_state_dict[key] = value
        return filtered_state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        """
        Handles a state_dict that might *not* contain encoder weights.
        """
        # Create a dummy full state_dict with current encoder parameters to prevent strict=True from complaining about
        # missing encoder keys.
        current_full_state_dict = super().state_dict()

        # Update the keys that are present in the provided state_dict.
        current_full_state_dict.update(state_dict)

        # strict=False allows loading state_dicts that do not contain encoder weights.
        super().load_state_dict(current_full_state_dict, strict=False)
