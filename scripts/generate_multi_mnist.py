"""
Script to generate a "stacked" MNIST dataset, where a random number of MNIST digits of the
    same class are stacked to form a single image.

Note: here are the mean pixel values of images in the train set, by class:

        xbar
y
1  19.379654
7  29.204563
4  30.948225
9  31.260435
5  32.831097
6  35.011951
3  36.090187
2  37.988659
8  38.289776

Based on this, we select two pairs of digits with close xbars; namely (7,4) and (6,3).
Using that grouping, we should do downstream classification with the following groups:

minority_group_keys: [7, 3]
positive_class_keys: [4, 3]
negative_class_keys: [7, 6]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from torchvision import datasets
from torchvision import transforms

from dpdi.utils.utils import np_to_pil

rotation = transforms.RandomRotation(80)


def make_multi_mnist(mnist, max_num_imgs=4,
                     classes=(7, 4, 6, 3), ims_per_class=50000):
    ims_out = list()  # Contains the (28, 28) image array
    labs_out = list()  # Contains the regression label
    attrs_out = list()  # Contains the MNIST class of the stacked ims
    print("total images: {}".format(len(classes)*ims_per_class))
    ims_processed = 0
    for c in classes:
        class_idxs = np.nonzero(mnist.targets.numpy() == c)[0]
        for _ in range(ims_per_class):
            ims = list()  # The list of images to stack
            i = np.random.choice(class_idxs, size=1)
            # x_i has shape (28, 28)
            x_i = mnist.data[i, ...]
            ims.append(x_i)
            y_i = mnist.targets[i]
            candidate_idxs = np.nonzero(mnist.targets == y_i)[0]
            n_i = np.random.randint(1, max_num_imgs + 1)
            for _ in range(2, n_i + 1):
                sample_idx = np.random.choice(candidate_idxs)
                ims.append(mnist.data[sample_idx, ...])
            # Randomly rotate the images, then cast to array
            for j, x_j in enumerate(ims):
                x_j = np_to_pil(x_j.numpy().squeeze())
                ims[j] = np.array(rotation(x_j))
            imstack = np.stack(ims, axis=0)
            imstack = imstack.max(axis=0)
            # Store the results.
            ims_out.append(imstack)
            labs_out.append(n_i)
            attrs_out.append(y_i)
            ims_processed += 1
            if ims_processed % 1000 == 0:
                print("image %s" % ims_processed)
    ims_out = np.stack(ims_out, axis=0)
    labs_out = np.array(labs_out)
    attrs_out = np.array(attrs_out)
    return {"x": ims_out, "y": labs_out, "a": attrs_out}


def main(debug, **kwargs):
    np.random.seed(61676)
    mnist_train = datasets.MNIST("./data", train=True)
    mnist_test = datasets.MNIST("./data", train=False)
    multi_mnist_tr = make_multi_mnist(mnist_train, **kwargs)
    multi_mnist_te = make_multi_mnist(mnist_test, **kwargs)
    if debug:
        samples = np.random.randint(0, len(multi_mnist_tr["x"]), size=50)
        for idx in samples:
            plt.imshow(multi_mnist_tr["x"][idx, ...])
            plt.title("Class {}, y={}".format(multi_mnist_tr["a"][idx], multi_mnist_tr["y"][idx]))
            plt.savefig("./tmp/" + str(idx) + ".png")
            plt.cla()
            plt.clf()

    # save the results to .npz.
    if not os.path.exists("./data/mnist_multi"):
        os.makedirs("./data/mnist_multi")
    np.savez("./data/mnist_multi/mnist_multi.npz",
             x_tr=multi_mnist_tr["x"],
             y_tr=multi_mnist_tr["y"],
             z_tr=multi_mnist_tr["a"],
             x_te=multi_mnist_te["x"],
             y_te=multi_mnist_te["y"],
             z_te=multi_mnist_te["a"])
    print("Saved file to ./data/mnist_multi/mnist_multi.npz")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_imgs", default=4,
                        help="The maximum number of images that will be concatenated; "
                             "number of images to concatenate is sampled uniformly from [1, max_num_imgs].")
    parser.add_argument("--ims_per_class", default=20000, type=int,
                        help="The number of stacked images to generate from each class.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
