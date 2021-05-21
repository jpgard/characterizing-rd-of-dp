import torch
import pandas as pd
import os
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

TRAIN_TEST_SPLIT_SEED = 948292


def get_anno_df(root_dir, is_train):
    # root_dir = "/Users/jpgard/Documents/research/imdb-wiki"
    meta_df = pd.read_csv(os.path.join(root_dir, "meta.csv"))
    meta_df.replace({"male": 1, "female": 0}, inplace=True)
    train, test = train_test_split(meta_df, train_size=0.8,
                                   random_state=TRAIN_TEST_SPLIT_SEED)
    if is_train:
        return train
    else:
        return test


def get_transforms(is_train: bool, normalize: bool):
    mu = [0.465727, 0.377981, 0.331473]
    std = [0.286456, 0.254825, 0.248889]

    resize = transforms.Resize([160, 160])
    rotate = transforms.RandomRotation(degrees=30)
    crop_size = [128, 128]
    random_crop = transforms.RandomCrop(crop_size)  # Crops the training image
    flip_aug = transforms.RandomHorizontalFlip()
    normalize_transf = transforms.Normalize(mean=mu, std=std)
    center_crop = transforms.CenterCrop(crop_size)  # Crops the test image

    transform_train = transforms.Compose([resize, rotate, random_crop, flip_aug,
                                          transforms.ToTensor(), normalize_transf])
    trasform_test = transforms.Compose(
        [resize, center_crop, transforms.ToTensor(), normalize_transf])
    transform_test_unnormalized = transforms.Compose(
        [resize, center_crop, transforms.ToTensor()])
    if is_train:
        assert normalize
        return transform_train
    elif normalize:
        return trasform_test
    else:
        return transform_test_unnormalized


class IMDBWikiDataset(torch.utils.data.Dataset):
    """The IMDB-Wiki dataset."""

    def __init__(self, root_dir, is_train: bool, normalize: bool, target_colname="age",
                 attribute_colname="gender"):
        self.root_dir = root_dir
        self.anno = get_anno_df(root_dir, is_train)
        self.transform = get_transforms(is_train, normalize)
        self.loader = default_loader
        self.target_colname = target_colname
        self.attribute_colname = attribute_colname
        self.fp_colname = "path"

    def __len__(self):
        return len(self.anno)

    @property
    def filepaths(self):
        return self.anno[self.fp_colname].values

    @property
    def targets(self):
        return np.expand_dims(self.anno[self.target_colname].values, 1).astype(float)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fp = os.path.join(self.root_dir, self.filepaths[idx])
        image = self.loader(img_fp)
        label = torch.from_numpy(self.targets[idx]).float()

        if self.transform:
            image = self.transform(image)
        sample = (image, idx, label)
        return sample

    def get_attribute_annotations(self, idxs):
        idx_annos = self.anno[self.attribute_colname].values[idxs]
        return idx_annos
