from PIL import Image, ImageOps
import torch
import pickle
import pandas as pd
from pathlib import Path
import numpy as np

Image.MAX_IMAGE_PIXELS = 933120000

from pyiqa.utils.registry import DATASET_REGISTRY
from pyiqa.data.base_iqa_dataset import BaseIQADataset


@DATASET_REGISTRY.register()
class IISADBDataset(BaseIQADataset):
    """
    IISA-DB dataset with Image Intrinsic Scale (IIS) annotations in range [0, 1], presented in the paper:
    https://arxiv.org/pdf/2502.06476. Allows generation of weak-labels for weakly supervised learning, following the
    WIISA approach presented in the paper. As a placeholder, repeats the dataset based on the number of weak labels
    specified in the options. Then, for each index greater than the number of ground-truth labels, it generates a
    weak-label based on the ground-truth IIS of the image.

    Args:
        opt (dict): Options for the dataset, including:
            - dataset_path (str): Path to the dataset directory.
            - num_weak_labels (int): Number of weak labels to generate.
            - scale_low (float): Lower bound for rescaling images (referred to as delta in the paper).
            - scale_high (float): Upper bound for rescaling images.
            - interpolation (str): Interpolation method for rescaling images ('bicubic', 'lanczos', 'bilinear').
            - augment (dict): Augmentation options, e.g., center_crop.

    Returns:
        dict: A dictionary containing:
            - img (Tensor): Image tensor after transformations.
            - mos_label (Tensor): Tensor of the ground-truth/weak IIS label (called mos_label for compatibility).
            - img_path (str): Path to the image file.
    """
    def init_path_mos(self, opt):
        target_img_folder = Path(opt['dataset_path']) / 'images'
        self.paths_mos = pd.read_csv(Path(opt['dataset_path']) / 'annotations.csv')     # called paths_mos for compatibility
        self.paths_mos["img_name"] = self.paths_mos["img_name"].apply(lambda x: str(target_img_folder / x))
        self.paths_mos = self.paths_mos.values.tolist()

        self.num_weak_labels = opt.get("num_weak_labels", 0)

        self.scale_low = opt.get("scale_low")
        self.scale_high = opt.get("scale_high")
        if opt.get("interpolation") == "bicubic":
            self.interpolation_mode = Image.BICUBIC
        elif opt.get("interpolation") == "lanczos":
            self.interpolation_mode = Image.LANCZOS
        elif opt.get("interpolation") == "bilinear":
            self.interpolation_mode = Image.BILINEAR
        else:
            self.interpolation_mode = None

        self.min_img_size = opt.get("augment", {}).get("center_crop")

    def __getitem__(self, index):
        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])     # ground-truth IIS label (called mos_label for compatibility)

        img = Image.open(img_path).convert('RGB')
        if index >= self.num_real_labels:
            img, mos_label = self.generate_weak_labels(img, mos_label)

        if self.min_img_size is not None:
            self.pad_img(img, self.min_img_size)

        img_tensor = self.trans(img) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])

        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}

    def generate_weak_labels(self, img, iis):
        """
        Generate weak-labels for weakly supervised learning by scaling the image and adjusting the IIS label, following
        the WIISA approach of the paper.

        Args:
            img (PIL.Image): The image to be processed.
            iis (float): The ground-truth IIS label in the range [0, 1].

        Returns:
            tuple: A tuple containing:
                - new_img (PIL.Image): The resized image.
                - new_iis (float): The weak IIS label in the range [0, 1].
        """
        random_scale = np.random.uniform(max(iis, self.scale_low), self.scale_high)
        new_img = img.resize((int(img.width * random_scale), int(img.height * random_scale)), self.interpolation_mode)
        new_iis = np.clip(iis / random_scale, 0, 1)
        return new_img, new_iis

    def get_split(self, opt):
        """
        Get the split of the dataset based on the phase attribute.

        Args:
            opt (dict): Options for the dataset, including:
                - split_file (str): Path to the split file containing indices.
                - split_index (int): Index of the split to use.
        """
        split_file_path = opt.get('split_file', None)
        if split_file_path:
            split_index = opt.get('split_index', 1)
            with open(opt['split_file'], 'rb') as f:
                split_dict = pickle.load(f)
                splits = split_dict[split_index][self.phase]
            self.paths_mos = [self.paths_mos[i] for i in splits]
            self.num_real_labels = len(self.paths_mos)
            self.paths_mos = self.paths_mos * (self.num_weak_labels + 1)  # repeat the dataset for num_weak_labels times as a placeholder for weak labels

    def pad_img(self, img, min_img_size):
        """
        Pad the image to ensure it meets the minimum size requirement.

        Args:
            img (PIL.Image): The image to be padded.
            min_img_size (int): The minimum size for both width and height.

        Returns:
            PIL.Image: The padded image.
        """
        width = img.width
        height = img.height
        pad_width = max(0, min_img_size - width)
        pad_height = max(0, min_img_size - height)
        if pad_width > 0 or pad_height > 0:
            new_size = (width + pad_width if pad_width > 0 else width, height + pad_height if pad_height > 0 else height)
            img = ImageOps.pad(img, new_size, color=0)
        return img
