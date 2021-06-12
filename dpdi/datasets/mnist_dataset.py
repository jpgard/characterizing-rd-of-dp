import os
from torchvision import datasets
import torch
import numpy as np

from dpdi.utils.utils import np_to_pil


class MNISTWithAttributesDataset(datasets.MNIST):
    """A clone of the MNIST dataset, but with
    minority/majority attribute annotatoins."""

    def __init__(self, minority_keys: list, majority_keys: list, **kwargs):
        super(MNISTWithAttributesDataset, self).__init__(**kwargs)
        self.minority_keys = minority_keys
        self.majority_keys = majority_keys

    @property
    def keys(self):
        return self.minority_keys + self.majority_keys

    def apply_classes_to_keep(self, classes_to_keep):
        """
        Filter the dataset to only contain observations with labels in classes_to_keep.
        """
        train_idx = np.isin(self.targets.numpy(), classes_to_keep)
        self.targets = self.targets[train_idx].to(dtype=torch.float32)
        self.data = self.data[train_idx]

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np_to_pil(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, index, target

    def get_attribute_annotations(self, idxs):
        batch_targets = self.targets[idxs]
        batch_attributes = torch.zeros_like(batch_targets)
        for k in self.majority_keys:
            is_k = (batch_targets == k)
            batch_attributes += is_k.type(torch.long)
        assert torch.max(
            batch_attributes) <= 1., "Sanity check on binarized grouped labels."
        return batch_attributes


class MultiMNISTDataset(MNISTWithAttributesDataset):
    """A stacked-MNIST dataset for regression tasks."""
    def __init__(self, is_train: bool, root, **kwargs):
        super(MultiMNISTDataset, self).__init__(
            **kwargs, root=os.path.join(root, "MNIST"))
        self.root = root
        self.attrs = None
        self.load_data(is_train)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np_to_pil(img.numpy())

        if self.transform is not None:
            img = self.transform(img)
            img = torch.unsqueeze(img, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, index, target

    def load_data(self, is_train: bool):
        npz_data = np.load(os.path.join(self.root, "mnist_multi", "mnist_multi.npz"))
        if is_train:
            data = npz_data["x_tr"]
            self.targets = torch.from_numpy(npz_data["y_tr"])
            self.attrs = torch.from_numpy(npz_data["z_tr"])
        else:
            data = npz_data["x_te"]
            self.targets = torch.from_numpy(npz_data["y_te"])
            self.attrs = torch.from_numpy(npz_data["z_te"])
        # add a channels dim to data
        self.data = torch.from_numpy(data)
        return

    def get_attribute_annotations(self, idxs):
        attrs = np.isin(self.attrs[idxs], self.majority_keys).astype(int)
        return attrs
