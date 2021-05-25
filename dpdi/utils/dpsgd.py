"""
Implements DP-SGD algorithm from the paper and related utility functions.
"""
import numpy as np
import torch
from math import sqrt
from math import log as ln
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression, RidgeCV
import pandas as pd
import math
import itertools
from collections import defaultdict
import diffprivlib

RANDOM_SEED = 983445


def get_wstar(df, use_ridge=False, ridgegrid=[0.001, 0.01, 0.1, 1., 10, 100, 1000]):
    """Compute the parameters via OLS (with no intercept term).

    If an intercept term is desired, add an 'intercept' column to the
        design matrix of all ones.
    """
    if use_ridge:
        est = RidgeCV(alphas=ridgegrid, fit_intercept=False, cv=10) \
            .fit(X=df.drop(['sensitive', 'target'], axis=1), y=df['target'])
        print("[INFO] CV selected regularization parameter {}".format(est.alpha_))
    else:
        est = LinearRegression(fit_intercept=False) \
            .fit(X=df.drop(['sensitive', 'target'], axis=1), y=df['target'])
    return est.coef_


def compute_mse(X, y, w_hat):
    test_preds = X @ w_hat
    test_err = np.mean((y - test_preds) ** 2)
    return test_err


def build_loader(X, y, batch_size=64, shuffle=False):
    inputs = torch.from_numpy(X).double()
    targets = torch.from_numpy(y).double()
    train_ds = torch.utils.data.TensorDataset(inputs, targets)
    resampling_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True,
                                                        num_samples=batch_size)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                                         sampler=resampling_sampler)
    return loader


def maybe_cast_scalar_to_square_ary(a: np.array) -> np.array:
    """Utility function to give a scalar a square shape."""
    if len(a) == 1 and np.ndim(a) == 1:
        return a.reshape(1, 1)
    else:
        return a


def build_dataset(H_min: np.array, H_maj: np.array, mu_min: np.array, mu_maj: np.array,
                  n: int, alpha: float, w_star: np.ndarray,
                  sd_eta: float = 1., verbose=True):
    """
    Build a dataset according to the parameters.
    H_0:
    mu: vector of group means [mu_min, mu_maj].
    n: total number of samples.
    alpha: Fraction of samples from the majority group.
    sd_eta: sd of the noise in the model, such that y = x^T w_star + eta.
    """
    assert 0 < alpha < 1
    assert (H_min.shape == H_maj.shape) and (mu_min.shape == mu_maj.shape)
    n_maj = int(alpha * n)
    n_min = n - n_maj
    if verbose:
        print(f"[INFO] built dataset with alpha={alpha}, n_maj={n_maj}, n_min={n_min}")
    rng = default_rng()
    eta = rng.normal(0., sd_eta, n)

    H_min = maybe_cast_scalar_to_square_ary(H_min)
    H_maj = maybe_cast_scalar_to_square_ary(H_maj)
    X_min = rng.multivariate_normal(mu_min, H_min, n_min)
    X_maj = rng.multivariate_normal(mu_maj, H_maj, n_maj)
    X = np.vstack((X_min, X_maj))
    # The ith entry in g is zero if the ith element of X is from minority; otherwise one.
    g = np.concatenate((np.zeros(n_min), np.ones(n_maj)))
    y = (X @ w_star) + eta
    return X, g, y


def compute_sdp_constants(X, y, w_star):
    L_1 = np.linalg.norm(X, axis=1, ord=2).max()
    L_2 = np.abs(y).max()
    L_3 = np.linalg.norm(w_star, ord=2)
    return L_1, L_2, L_3


def compute_sigma_dp(L_1, L_2, L_3, delta, eps: float):
    """Compute sigma_DP."""
    sigma_dp = (2 * (L_2 * L_3 + L_1 * L_3 ** 2)
                * sqrt(2 * ln(1.25 / delta))
                / eps)
    return sigma_dp


def compute_disparity(X: np.array, g: np.array, y: np.array, sgd_w_hat: np.array,
                      dpsgd_w_hat: np.array, dpols_w_hat=None):
    """Compute the quantities defined as rho and phi in the paper, along with their
    constituents."""
    X_minor = X[g == 0, :]
    y_minor = y[g == 0]
    loss_sgd_0 = np.mean(((X_minor @ sgd_w_hat) - y_minor) ** 2)
    X_major = X[g == 1, :]
    y_major = y[g == 1]
    loss_sgd_1 = np.mean(((X_major @ sgd_w_hat) - y_major) ** 2)
    loss_dpsgd_0 = np.mean(((X_minor @ dpsgd_w_hat) - y_minor) ** 2)
    loss_dpsgd_1 = np.mean(((X_major @ dpsgd_w_hat) - y_major) ** 2)

    rho = (loss_dpsgd_0 - loss_dpsgd_1) / (loss_sgd_0 - loss_sgd_1)
    phi = (loss_dpsgd_0 - loss_sgd_0) / (loss_dpsgd_1 - loss_sgd_1)

    metrics = {"rho": rho,
               "phi": phi,
               "loss_dpsgd_0": loss_dpsgd_0,
               "loss_dpsgd_1": loss_dpsgd_1,
               "loss_sgd_0": loss_sgd_0,
               "loss_sgd_1": loss_sgd_1}
    if dpols_w_hat is not None:
        metrics["loss_dpols_0"] = np.mean(((X_minor @ dpols_w_hat) - y_minor) ** 2)
        metrics["loss_dpols_1"] = np.mean(((X_major @ dpols_w_hat) - y_major) ** 2)
    return metrics


def compute_rho_lr(H_min: np.array, H_maj: np.array, alpha, sigma_dp, sigma_noise=1.):
    """Compute the quantity defined as \rho_{LR} in the paper."""
    H_min = maybe_cast_scalar_to_square_ary(H_min)
    H_maj = maybe_cast_scalar_to_square_ary(H_maj)
    H = alpha * H_maj + (1 - alpha) * H_min
    H_inv = np.linalg.pinv(H)
    H_minus2 = np.matmul(H_inv, H_inv)
    rho_lr = (
            np.trace((H_min - H_maj) @ H_minus2) * sigma_dp ** 2
            / (np.trace((H_min - H_maj) @ H_inv) * sigma_noise ** 2)
    )
    return rho_lr


def compute_phi_lr(H_min, H_maj, alpha):
    H_alpha = (1 - alpha)*H_min + alpha*H_maj
    H_inv = np.linalg.pinv(H_alpha)
    phi = max(np.trace(H_min @ H_inv) / np.trace(H_maj @ H_inv),
              np.trace(H_maj @ H_inv) / np.trace(H_min @ H_inv))
    return phi



def print_dpsgd_diagnostics(L_1, L_2, L_3, sigma_dp, n, delta):
    """Print various important quantities used to compute sigma_DP."""
    print(f"L_1 = {L_1}; L_2 = {L_2}; L_3 = {L_3}")
    print("(L_2 * L_3 + L_1 * L_3**2): %s" % (L_2 * L_3 + L_1 * L_3 ** 2))
    print("sqrt(2 * ln(1.25 * 2 * k / delta)): %s" % sqrt(2 * ln(1.25 * 2 * k / delta)))
    print("sqrt(k * ln(2 * n / delta)): %s" % sqrt(k * ln(2 * n / delta)))
    print("sigma_dp: %f" % sigma_dp)


def tail_average_iterates(w, T, s):
    return np.vstack(w[-(T - s):]).mean(axis=0)


def dp_sgd(X, y, T, delta, eps, s, lr, w_star, verbosity=2, batch_size=64,
           random_seed=RANDOM_SEED):
    """Implements Algorithm 1 (DP-SGD), with fixed seed for reproducibility.

    Verbosity: 0 = no output; 1 = print basic DP-SGD quantities; 2 = print DP-SGD
    quantities
        plus loss at each iteration.
    """
    torch.manual_seed(random_seed)
    n, d = X.shape
    assert d == len(w_star), "shape mismatch between X and w_star"
    # Compute the various constants needed for the algorithm.
    L_1, L_2, L_3 = compute_sdp_constants(X, y, w_star)
    sigma_dp = compute_sigma_dp(L_1, L_2, L_3, delta=delta, eps=eps)
    if verbosity > 0:
        print_dpsgd_diagnostics(L_1, L_2, L_3, sigma_dp=sigma_dp, n=n, delta=delta)

    # Initialization
    loader = build_loader(X, y, batch_size)
    t = 0
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()
    L_3 = torch.Tensor([L_3, ])
    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(loader):

            y_hat = torch.matmul(X_i, w_hat)
            loss = torch.mean((y_i - y_hat) ** 2)
            loss.backward()

            with torch.no_grad():
                grad_noise = torch.normal(mean=0, std=sigma_dp, size=w_star.shape)
                w_hat -= lr * (w_hat.grad + grad_noise)

                # Project back onto ball of radius L_3
                w_hat_norm = torch.norm(w_hat)
                w_hat /= w_hat_norm
                w_hat *= L_3

                w_hat.grad.zero_()
                iterate_numpy = w_hat.clone().detach().numpy()
                loss_numpy = loss.clone().detach().numpy()
                iterates.append(iterate_numpy)
                losses.append(loss_numpy)

            if verbosity > 1 and (t % 1000 == 0):
                print(
                    "iteration {} loss: {} new w_hat: {}".format(t, loss, iterate_numpy))

            t += 1
            if t >= T:
                if verbosity > 0:
                    print("[INFO] completed %s iterations of DP-SGD." % t)
                break
    w_hat_bar = tail_average_iterates(iterates, T, s)
    return iterates, losses, w_hat_bar


def tail_averaged_sgd(X, y, T, s, lr, verbosity=2, batch_size=64,
                      random_seed=RANDOM_SEED):
    """Implements tail-averaged SGD. This is DP-SGD but with no projection step and no
    noise.

    Verbosity: 0 = no output; if > 0, print loss at each iteration.
    """
    torch.manual_seed(random_seed)
    n, d = X.shape

    # Initialization
    loader = build_loader(X, y, batch_size)
    t = 0
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()

    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(loader):

            y_hat = torch.matmul(X_i, w_hat)
            loss = torch.mean((y_i - y_hat) ** 2)
            loss.backward()

            with torch.no_grad():
                w_hat -= lr * w_hat.grad

                w_hat.grad.zero_()
                iterate_numpy = w_hat.clone().detach().numpy()
                loss_numpy = loss.clone().detach().numpy()
                iterates.append(iterate_numpy)
                losses.append(loss_numpy)

            if verbosity > 0 and (t % 1000 == 0):
                print(
                    "iteration {} loss: {} new w_hat: {}".format(t, loss, iterate_numpy))

            t += 1
            if t >= T:
                if verbosity > 0:
                    print("[INFO] completed %s iterations of SGD." % t)
                break
    w_hat_bar = tail_average_iterates(iterates, T, s)
    return iterates, losses, w_hat_bar


def vanilla_sgd(X, y, T, lr, verbose=True, batch_size=64, random_seed=RANDOM_SEED):
    """Implements vanilla SGD for the dataset."""
    print("[WARNING] this function is deprecated and provided only for reproducibility.")
    print("Use tail_averaged_sgd() instead (note different return types and signatures).")
    torch.manual_seed(random_seed)
    n, d = X.shape
    loader = build_loader(X, y, batch_size)
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()
    t = 0
    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(loader):
            y_hat = torch.matmul(X_i, w_hat)
            loss = torch.mean((y_i - y_hat) ** 2)
            # Computes the gradients for all tensors with grad=True
            loss.backward()
            with torch.no_grad():
                w_hat -= lr * w_hat.grad
                w_hat.grad.zero_()
                iterate_numpy = w_hat.clone().detach().numpy()
                loss_numpy = loss.clone().detach().numpy()
                iterates.append(iterate_numpy)
                losses.append(loss_numpy)
            if verbose and (t % 1000 == 0):
                print("iteration {} loss: {} new w_hat: {}".format(t, loss,
                                                                   w_hat.detach().numpy()))
            t += 1
            if t >= T:
                print("[INFO] completed %s iterations of SGD." % t)
                break
    return iterates, losses


def disparity_experiments(train_df, test_df, T, s, lr, epsgrid, wstar, delta=1e-1,
                          verbosity=1):
    """Function to run the disparity experiments for each dataset."""
    results = list()
    X_tr = train_df.drop(['sensitive', 'target'], axis=1).values
    y_tr = train_df['target'].values
    _, _, w_hat_bar_sgd = tail_averaged_sgd(
        X=X_tr,
        y=y_tr,
        T=T, s=s,
        lr=lr,
        batch_size=1,
        verbosity=verbosity
    )

    for eps in epsgrid:
        _, _, w_hat_bar_dpsgd = dp_sgd(
            X=X_tr,
            y=y_tr,
            T=T, delta=delta, eps=eps, s=s,
            lr=lr,
            w_star=wstar,
            batch_size=1,
            verbosity=verbosity
        )

        w_hat_dpols = diffprivlib.models.LinearRegression(
            epsilon=eps, fit_intercept=False) \
            .fit(X=X_tr, y=y_tr) \
            .coef_

        disparity_metrics = compute_disparity(
            X=test_df.drop(['sensitive', 'target'], axis=1).values,
            g=test_df.sensitive.values, y=test_df.target.values, sgd_w_hat=w_hat_bar_sgd,
            dpsgd_w_hat=w_hat_bar_dpsgd, dpols_w_hat=w_hat_dpols)
        disparity_metrics["eps"] = eps
        results.append(disparity_metrics)
    return pd.DataFrame(results)


def compute_subgroup_loss_bound(df: pd.DataFrame, j: int, eps: float,
                                delta: float, gamma: float, T: int, s: int,
                                w_star,
                                sigma_noise=1.):
    attrs = df['sensitive'].values
    alpha = attrs.mean()
    X = df.drop(columns=['target', 'sensitive']).values
    y = df['target'].values
    w_init = np.zeros_like(w_star)
    n, d = X.shape
    H_j = np.cov(X[attrs == j], rowvar=False)
    H = np.cov(X, rowvar=False)
    H_inv = np.linalg.pinv(H)
    #  mu is the smallest eigenvalue of H, but we ignore the intercept term
    # . which is associated with an eigenvalue of zero.
    mu = np.sort(np.linalg.eigvals(H))[1]
    L_1, L_2, L_3 = compute_sdp_constants(X, y, w_star)
    sigma_dp = compute_sigma_dp(L_1, L_2, L_3, delta=delta, eps=eps)
    bias_term = (2 / (gamma * T * mu * alpha) ** 2) \
                * (1 - gamma * mu) ** (s + 1) \
                * (compute_mse(X, y, w_init) - compute_mse(X, y, w_star))
    variance_term = (2 / T) * \
                    np.trace(H_j @ H_inv
                             @ (sigma_noise ** 2 * H + sigma_dp * np.eye(d))
                             @ H_inv)
    resamp_term = ((2 / n * T * gamma) * np.trace(H_j @ H_inv * sigma_noise ** 2))
    results = {
        "bias_term": bias_term,
        "variance_term": variance_term,
        "resamp_term": resamp_term,
        "bound": bias_term + variance_term + resamp_term
    }
    return results


def sensitive_subgroup_indices(df, i):
    return np.nonzero(df.sensitive.values == i)[0]


def compute_nmax(df, alpha_grid, verbose=True):
    n_0 = len(sensitive_subgroup_indices(df, 0))
    n_1 = len(sensitive_subgroup_indices(df, 1))
    alpha_min = min(alpha_grid)
    alpha_max = max(alpha_grid)
    # Compute limit on total size n given the largest number of minority samples needed
    n_max_0 = math.floor(n_0 / float(1 - alpha_min))
    # Compute limit on total size n given the largest number of majority samples needed
    n_max_1 = math.floor(n_1 / float(alpha_max))
    if verbose:
        if n_max_0 < n_max_1:
            print("[INFO] max sample size is {} constrained by minority".format(n_max_0))
        else:
            print("[INFO] max sample size is {} constrained by majority".format(n_max_1))
    return min(n_max_0, n_max_1)


def alpha_experiment(df, wstar, iternum, alpha, eps, delta, lr, n_max, s, verbosity):
    n = len(df)
    T = n - s
    g = df.sensitive.values
    idxs_0 = sensitive_subgroup_indices(df, 0)
    idxs_1 = sensitive_subgroup_indices(df, 1)
    n_0 = len(idxs_0)
    n_1 = len(idxs_1)

    n_0_sample = math.floor((1 - alpha) * n_max)
    n_1_sample = math.floor(alpha * n_max)
    if verbosity > 0:
        print("[INFO] sampling {} / {} from 1".format(n_1_sample, n_1))
        print("[INFO] sampling {} / {} from 0".format(n_0_sample, n_0))
    idxs_sample_0 = np.random.choice(idxs_0, size=n_0_sample, replace=False)
    idxs_sample_1 = np.random.choice(idxs_1, size=n_1_sample, replace=False)
    # subset the data
    df_alpha = pd.concat((df.iloc[idxs_sample_0], df.iloc[idxs_sample_1]), axis=0)
    # Compute dpsgd
    _, _, w_hat_bar_sgd = tail_averaged_sgd(
        X=df_alpha.drop(['sensitive', 'target'], axis=1).values,
        y=df_alpha['target'].values,
        T=T, s=s,
        lr=lr,
        batch_size=1,
        verbosity=0
    )
    _, _, w_hat_bar_dpsgd = dp_sgd(
        X=df_alpha.drop(['sensitive', 'target'], axis=1).values,
        y=df_alpha['target'].values,
        T=T, delta=delta, eps=eps, s=s,
        lr=lr,
        w_star=wstar,
        batch_size=1,
        verbosity=0
    )
    disparity_metrics = compute_disparity(
        X=df_alpha.drop(['sensitive', 'target'], axis=1).values,
        g=df_alpha.sensitive.values, y=df_alpha.target.values,
        sgd_w_hat=w_hat_bar_sgd, dpsgd_w_hat=w_hat_bar_dpsgd)
    disparity_metrics["iternum"] = iternum
    disparity_metrics["alpha"] = alpha
    return disparity_metrics


def alpha_experiments(df: pd.DataFrame, s: int, lr: float, wstar, eps=50, delta=1e-1,
                      alpha_grid=(0.7, 0.8, 0.9), niters=5, n_max=None, verbosity=1):
    if n_max is None:
        n_max = compute_nmax(df, alpha_grid)
    results = list()
    for iternum in range(niters):
        for alpha in alpha_grid:
            disparity_metrics = alpha_experiment(df, wstar, iternum, alpha, eps, delta,
                                                 lr, n_max, s, verbosity)
            results.append(disparity_metrics)
    return results


def together_apart_experiments(df_train, df_test, s_union: int, s_minor, s_major,
                               lr: float):
    train_sets = dict()
    test_sets = dict()
    w_hats = dict()
    train_sets["01"] = df_train
    train_sets["0"] = df_train.iloc[sensitive_subgroup_indices(df_train, 0)]
    train_sets["1"] = df_train.iloc[sensitive_subgroup_indices(df_train, 1)]
    test_sets["01"] = df_test
    test_sets["0"] = df_test.iloc[sensitive_subgroup_indices(df_test, 0)]
    test_sets["1"] = df_test.iloc[sensitive_subgroup_indices(df_test, 1)]

    # Train on union
    _, _, w_hat_bar_sgd_union = tail_averaged_sgd(
        X=df_train.drop(['sensitive', 'target'], axis=1).values,
        y=df_train['target'].values,
        T=len(df_train) - s_union, s=s_union, lr=lr, batch_size=1, verbosity=0)
    w_hats["01"] = w_hat_bar_sgd_union
    # Train on minority
    _, _, w_hat_bar_sgd_0 = tail_averaged_sgd(
        X=train_sets["0"].drop(['sensitive', 'target'], axis=1).values,
        y=train_sets["0"]['target'].values,
        T=len(train_sets["0"]) - s_minor, s=s_minor,
        lr=lr, batch_size=1, verbosity=0)
    w_hats["0"] = w_hat_bar_sgd_0
    # Train on majority
    _, _, w_hat_bar_sgd_1 = tail_averaged_sgd(
        X=train_sets["1"].drop(['sensitive', 'target'], axis=1).values,
        y=train_sets["1"]['target'].values,
        T=len(train_sets["1"]) - s_major, s=s_major,
        lr=lr, batch_size=1, verbosity=0)
    w_hats["1"] = w_hat_bar_sgd_1
    # Compute test metrics
    results = list()  # list of (train_subset, test_subset, err) tuples.
    for train_subset, test_subset in itertools.product(train_sets.keys(),
                                                       test_sets.keys()):
        w_hat = w_hats[train_subset]
        X_test = test_sets[test_subset].drop(['sensitive', 'target'], axis=1).values
        y_test = test_sets[test_subset]['target'].values
        err = compute_mse(X_test, y_test, w_hat)
        results.append((train_subset, test_subset, err))
    results = pd.DataFrame(results, columns=["train", "test", "err"])
    results.replace({"01": "Union", "1": "Majority", "0": "Minority"}, inplace=True)
    return results


def bound_summary(df, eps, delta, gamma, T, s, w_star):
    bound_major = compute_subgroup_loss_bound(
        df, j=1, eps=eps, delta=delta, gamma=gamma,
        T=T, s=s, w_star=w_star)
    bound_minor = compute_subgroup_loss_bound(
        df, j=0, eps=eps, delta=delta, gamma=gamma,
        T=T, s=s, w_star=w_star)
    return (pd.DataFrame({1: bound_major, 0: bound_minor}))
