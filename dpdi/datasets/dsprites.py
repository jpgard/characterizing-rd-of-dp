import torch
import numpy as np
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split

LATENTS_TO_IDX = {
    "color": 0,
    # shape codes: (1:square, 2:circle, 3:heart), with 245760 observations of each
    # Note there is NO SHAPE ZERO!
    "shape": 1,
    "scale": 2,
    "orientation": 3,
    "posx": 4,
    "posy": 5
}

# Not used, but may be useful
mu_dsprites = 0.042494423521889584
std_dsprites = 0.20171427190806362


def get_idxs(n, is_train, seed=23985, train_frac=0.9):
    train_idxs, test_idxs = train_test_split(np.arange(n), train_size=train_frac,
                                             random_state=seed)
    if is_train:
        return train_idxs
    else:
        return test_idxs


def get_dsprites_transforms(is_train, normalize):
    if is_train:
        assert normalize
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mu_dsprites, std_dsprites)])
    else:
        if normalize:
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mu_dsprites, std_dsprites)])
        return transforms.ToTensor()


class DspritesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, normalize, target_colname="scale",
                 attribute_colname="shape", minority_group_latents=(1, 2,),
                 majority_group_latents=(3,), alpha=None):
        self.target_colname = target_colname
        self.attribute_colname = attribute_colname
        self.majority_group_latents = majority_group_latents
        self.minority_group_latents = minority_group_latents
        dsprites = np.load(
            os.path.join(root_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"))
        idxs = get_idxs(len(dsprites["imgs"]), is_train)
        self.data = dsprites["imgs"][idxs]

        # The order of latents is (Color, Shape, Scale, Orientation, PosX, PosY);
        #  see https://github.com/deepmind/dsprites-dataset for more info.
        self.latents_values = dsprites["latents_values"][idxs]
        self.latents_classes = dsprites["latents_classes"][idxs]
        self.transform = get_dsprites_transforms(is_train, normalize)
        self.alpha = alpha

    @property
    def targets(self):
        latent_idx = LATENTS_TO_IDX[self.target_colname]
        return self.latents_values[:, latent_idx]

    @property
    def attribute_annotations(self):
        latent_idx = LATENTS_TO_IDX[self.attribute_colname]
        anno = np.isin(self.latents_values[:, latent_idx],
                       self.majority_group_latents).flatten()
        return anno

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.data[idx, ...].astype(np.float32)
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx].astype(np.float32)
        sample = (img, idx, label)
        return sample

    def get_attribute_annotations(self, idxs):
        return self.attribute_annotations[idxs]

    def apply_alpha_to_dataset(self, n_train):
        if self.alpha is None:
            return
        majority_idxs = np.argwhere(
            np.isin(self.attribute_annotations, self.majority_group_latents)).flatten()
        minority_idxs = np.argwhere(
            np.isin(self.attribute_annotations, self.minority_group_latents)).flatten()
        assert n_train <= len(majority_idxs) + len(minority_idxs)
        n_maj = int(self.alpha * n_train)
        n_min = n_train - n_maj
        # Sample alpha * n_sub from the majority, and (1-alpha)*n_sub from the
        # minority.
        print("[DEBUG] sampling n_maj={} elements from {} majority items {}".format(
            n_maj, len(majority_idxs), self.majority_group_latents))
        print("[DEBUG] sampling n_min={} elements from {} minority items {}".format(
            n_min, len(minority_idxs), self.minority_group_latents))
        majority_idx_sample = np.random.choice(majority_idxs, size=n_maj,
                                               replace=False)
        minority_idx_sample = np.random.choice(minority_idxs, size=n_min,
                                               replace=False)
        idx_sample = np.concatenate((majority_idx_sample, minority_idx_sample))
        self.data = self.data[idx_sample]
        self.latents_values = self.latents_values[idx_sample]
        self.latents_classes = self.latents_classes[idx_sample]
        assert len(self) == (n_min + n_maj), "Sanity check for self subsetting."
        assert abs(
            float(len(minority_idx_sample)) / len(self)
            - (1 - self.alpha)) < 0.001, \
            "Sanity check for minority size within 0.001 of (1-alpha)."
        return