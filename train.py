import logging

from dpdi.helper import get_helper
from dpdi.models import get_net
from dpdi.models.optimizers import get_optimizer

SINGLE_CHANNEL_DATASETS = ('mnist', 'dsprites')

logger = logging.getLogger('logger')

from datetime import datetime
import math
import argparse
from collections import defaultdict
from tensorboardX import SummaryWriter
from dpdi.models.simple import reseed
import numpy as np
import torch.nn as nn
import torch.optim as optim
import yaml
from dpdi.utils.text_load import *
from dpdi.utils.utils import create_table, plot_confusion_matrix
import pandas as pd
from pyvacy import sampling
from torch.utils.data import TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These are datasets that yield tuples of (images, idxs, labels) instead of
# (images,labels).

TRIPLET_YIELDING_DATASETS = ('celeba', 'lfw', 'mnist', 'cifar10', 'zillow', 'dsprites', 'imdb-wiki')

# These are datasets where we explicitly track performance according to some majority/minority
# attribute defined in the params. This shouldn't require a second module-level variable,
# but this can be fixed/refactored in the future.
MINORITY_PERFORMANCE_TRACK_DATASETS = TRIPLET_YIELDING_DATASETS


def maybe_override_parameter(params: dict, args, parameter_name: str):
    """Optionally overrides a parameter using a command-line argument of the same name."""
    val = getattr(args, parameter_name, None)
    if val is not None:
        print(
            "[INFO] overriding parameter {} from params file to value {} from args"\
                .format(parameter_name, val))
        params[parameter_name] = val
    return


def get_criterion(helper):
    # For DP training, no loss reduction is used; otherwise, use default (mean) reduction.
    if helper.params.get('criterion') == 'mse':  # Case: MSE objective.
        print('[DEBUG] using MSE loss')
        if dp:
            criterion = nn.MSELoss(reduction='none')
        else:
            criterion = nn.MSELoss()
    else:  # Case: not MSE; use cross-entropy objective.
        if dp:
            criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = nn.CrossEntropyLoss()
    return criterion


def load_data(helper, params, alpha, mu):
    classes_to_keep = None
    true_labels_to_binary_labels = None
    if helper.params['dataset'] == 'inat':
        helper.load_inat_data()
        helper.balance_loaders()
    elif helper.params['dataset'] == 'word':
        helper.load_data()
    elif helper.params['dataset'] == 'celeba':
        helper.load_celeba_data()
    elif helper.params['dataset'] == 'lfw':
        helper.load_lfw_data()
    elif helper.params['dataset'] == 'zillow':
        helper.load_zillow_data()
    elif helper.params['dataset'] == 'dsprites':
        helper.load_dsprites_data()
    elif helper.params['dataset'] == 'imdb-wiki':
        helper.load_imdb_wiki_data()
    else:
        # First, define classes_to_keep.
        # Labels are assigned in order of index in this array; so minority_key has
        # label 0, majority_key has label 1.
        classes_to_keep = helper.params['positive_class_keys'] + \
                          helper.params['negative_class_keys']

        # Define the labels mapping.
        true_labels_to_binary_labels = {l: 1 if l in params['positive_class_keys'] else 0
                                        for l in classes_to_keep}

        helper.load_cifar_or_mnist_data(dataset=params['dataset'],
                                        classes_to_keep=classes_to_keep,
                                        labels_mapping=true_labels_to_binary_labels,
                                        alpha=alpha)
        logger.info('before loader')
        helper.create_loaders()
        logger.info('after loader')

        # Create a unique DataLoader for each class. We do not use the kys_to_drop param
        # since alpha-balancing is applied at dataset creation step; so, we can just
        # sample the classes uniformly and achieve the desired alpha-imbalance.
        helper.sampler_per_class()
        logger.info('after sampler')
        number_of_entries_train = params.get('number_of_entries', False)
        helper.sampler_exponential_class(mu=mu, total_number=params['ds_size'],
                                         number_of_entries=number_of_entries_train)
        logger.info('after sampler expo')
        helper.sampler_exponential_class_test(mu=mu,
                                              number_of_entries_test=params[
                                                  'number_of_entries_test'])
        logger.info('after sampler test')

    return true_labels_to_binary_labels, classes_to_keep

def mean_of_tensor_list(lst):
    lst_nonempty = [x for x in lst if x.numel() > 0]
    if len(lst_nonempty):
        return torch.mean(torch.stack(lst_nonempty))
    else:
        return None


def compute_channelwise_mean(dataset):
    means = defaultdict(list)
    sds = defaultdict(list)
    for (i, batch) in enumerate(dataset):
        x, _, _ = batch
        # batch is a set of images of shape [b, c, h, w]
        means[0].append(torch.mean(x[:, 0, ...]))
        sds[0].append(torch.std(x[:, 0, ...]))
        means[1].append(torch.mean(x[:, 1, ...]))
        sds[1].append(torch.std(x[:, 1, ...]))
        means[2].append(torch.mean(x[:, 2, ...]))
        sds[2].append(torch.std(x[:, 2, ...]))
    # We ignore the last batch in case it is incomplete.
    print("Channel 0 mean: %f" % mean_of_tensor_list(means[0][:-1]))
    print("Channel 1 mean: %f" % mean_of_tensor_list(means[1][:-1]))
    print("Channel 2 mean: %f" % mean_of_tensor_list(means[2][:-1]))
    print("Channel 0 sd: %f" % mean_of_tensor_list(sds[0][:-1]))
    print("Channel 1 sd: %f" % mean_of_tensor_list(sds[1][:-1]))
    print("Channel 2 sd: %f" % mean_of_tensor_list(sds[2][:-1]))
    return


def add_pos_and_neg_summary_images(data_loader, is_regression, max_images=64, labels_mapping=None):
    images, idxs, labels = next(iter(data_loader))
    if labels_mapping:
        pos_labels = [k for k, v in labels_mapping.items() if v == 1]
        labels = binarize_labels_tensor(labels, pos_labels, out_type=torch.long)
    attr_labels = data_loader.dataset.get_attribute_annotations(idxs)
    pos_attr_idxs = idx_where_true(attr_labels == 1)
    neg_attr_idxs = idx_where_true(attr_labels == 0)
    pos_attr_images = images[pos_attr_idxs]
    neg_attr_images = images[neg_attr_idxs]
    writer.add_images('pos_attr_images', pos_attr_images[:max_images, ...])
    writer.add_images('neg_attr_images', neg_attr_images[:max_images, ...])
    if not is_regression:
        pos_label_images = images[labels == 1]
        neg_label_images = images[labels == 0]
        writer.add_images('pos_label_images', pos_label_images[:max_images, ...])
        writer.add_images('neg_label_images', neg_label_images[:max_images, ...])
    return


def make_uid(params, args):
    # If number_of_entries_train is provided, it overrides the params file. Otherwise,
    # fetch the value from the params file.
    alpha = args.alpha
    number_of_entries_train = args.number_of_entries_train
    if number_of_entries_train is None:
        number_of_entries_train = params.get('number_of_entries')
    uid = "{ds}-S{S}-z{z}-sigma{sigma}-alpha-{alpha}-opt{opt}-dp{dp}-n{n}-{model}{depth}lr{lr}".format(
        ds=params['dataset'],
        S=params.get('S'),
        z=params.get('z'),
        sigma=params.get('sigma'), alpha=params.get('alpha'),
        opt=params['optimizer'],
        dp=str(params['dp']),
        n=number_of_entries_train,
        model=params['model'],
        depth=params.get('resnet_depth', ''),
        lr=params['lr'])
    if alpha is not None:
        uid += '-alpha' + str(alpha)
    if params.get('fixed_n_train'):
        uid += 'ntr' + str(params.get('fixed_n_train'))
    if params.get('positive_class_keys') and params.get('negative_class_keys'):
        pos_keys = [str(i) for i in params['positive_class_keys']]
        neg_keys = [str(i) for i in params['negative_class_keys']]
        pos_keys_str = '-'.join(pos_keys)
        neg_keys_str = '-'.join(neg_keys)
        keys_str = pos_keys_str + '-vs-' + neg_keys_str
        uid += '-' + keys_str
    if params.get('target_colname'):
        uid += '-' + params['target_colname']
    if params.get('attribute_colname'):
        uid += '-' + params['attribute_colname']
    if params.get('train_attribute_subset') is not None:
        uid += '-trattrsub' + str(params['train_attribute_subset'])
    if params.get('label_threshold'):
        uid += '-' + str(params['label_threshold'])
    if params.get('freeze_pretrained_weights'):
        uid += '-freezept'
    return uid


def plot(x, y, name):
    if y is not None:
        writer.add_scalar(tag=name, scalar_value=y, global_step=x)


def compute_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def compute_mse(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert outputs.shape == labels.shape, \
        "Expected outputs and labels same shape, got shapes {} and {}".format(
            outputs.shape, labels.shape
        )
    mse = (outputs - labels) ** 2
    return mse



def per_class_mse(outputs, labels, target_class, grouped_label=None) -> torch.Tensor:
    per_class_idx = labels == target_class
    per_class_outputs = outputs[per_class_idx]
    if grouped_label is not None:
        # Create a new labels tensor, with all values equal to grouped_label
        per_class_labels = torch.full_like(per_class_outputs,
                                           fill_value=grouped_label, dtype=torch.float32)
    else:
        # Use the existing labels tensor, with all values equal to target_class
        per_class_labels = labels[per_class_idx]
    mse_per_class = torch.mean(compute_mse(per_class_outputs, per_class_labels))
    return mse_per_class


def idx_where_true(ary):
    if isinstance(ary, pd.DataFrame) or isinstance(ary, pd.Series):
        bool_indices = ary.values
    elif isinstance(ary, torch.Tensor):
        bool_indices = ary.detach().cpu().numpy()
    elif isinstance(ary, np.ndarray):
        bool_indices = ary
    else:
        raise ValueError("Got unexpected ary of type {}".format(type(ary)))
    return np.ravel(np.argwhere(bool_indices))


def sample_grad_norms(epoch, testloader, n_batches=3, mse:bool=False,
                      labels_mapping=None):
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    attr_norms = defaultdict(list)
    for idx, data in enumerate(tqdm(testloader)):
        if helper.params['dataset'] in TRIPLET_YIELDING_DATASETS:
            inputs, idxs, labels = data
        else:
            inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        if labels_mapping:
            pos_labels = [k for k, v in labels_mapping.items() if v == 1]
            labels_type = torch.float32 if mse else torch.long
            preprocessed_labels = binarize_labels_tensor(
                labels, pos_labels, labels_type)
        else:
            preprocessed_labels = labels

        if not mse:
            _, predicted = torch.max(outputs.data, 1)
            elementwise_loss = ce_loss(outputs, preprocessed_labels)
        else:
            elementwise_loss = compute_mse(torch.squeeze(outputs),
                                           torch.squeeze(preprocessed_labels))

        batch_attr_labels = helper.test_dataset.get_attribute_annotations(idxs)

        if idx > n_batches:
            break
    # Compute the grad norms for each attribute
    grad_vecs = list()
    for pos, j in enumerate(elementwise_loss):
        j.backward(retain_graph=True)

        grad_vec = helper.get_grad_vec(net, device)
        grad_vecs.append(grad_vec)
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), S)
        attr_norms[int(batch_attr_labels[pos])].append(total_norm)
        net.zero_grad()

    # Compute average norm and the sigma value (if adaptive)
    grad_norms = [torch.norm(x, p=2) for x in grad_vecs]
    avg_grad_norm = mean_of_tensor_list(grad_norms)
    plot(epoch, avg_grad_norm, "norms/avg_grad_norm")
    for attr, norms in attr_norms.items():
        avg_attr_grad_norm = mean_of_tensor_list(norms)
        plot(epoch, avg_attr_grad_norm, f"avg_grad_norms_by_attr_test/{attr}")
    return


def test(net, epoch, name, testloader, vis=True, mse: bool = False,
         labels_mapping: dict = None):
    net.eval()
    attr_norms = defaultdict(list)
    running_metric_total = 0
    running_ce_loss_total = 0
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    n_test = 0
    i = 0
    correct_labels = []
    predict_labels = []
    loss_by_label = defaultdict(list)  # Stores losses by label (usually 0,1)
    loss_by_attribute = defaultdict(list)
    loss_by_key = defaultdict(list)  # Stores losses by key (this can be more fine-grained than label)
    attributes = (0, 1)
    keys = list() if not hasattr(helper.test_dataset, 'keys') else helper.test_dataset.keys
    print("[DEBUG] detected the following keys for test metrics: {}".format(keys))
    metric_name = 'accuracy' if not mse else 'mse'
    cls_labels = getattr(helper, "labels", [])
    with torch.no_grad():
        for data in tqdm(testloader):
            if helper.params['dataset'] in TRIPLET_YIELDING_DATASETS:
                inputs, idxs, labels = data
            else:
                inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            if labels_mapping:
                pos_labels = [k for k, v in labels_mapping.items() if v == 1]
                labels_type = torch.float32 if mse else torch.long
                preprocessed_labels = binarize_labels_tensor(
                    labels, pos_labels, labels_type)
            else:
                preprocessed_labels = labels

            n_test += preprocessed_labels.size(0)

            if not mse:
                _, predicted = torch.max(outputs.data, 1)
                predict_labels.extend([x.item() for x in predicted])
                correct_labels.extend([x.item() for x in preprocessed_labels])
                running_metric_total += (predicted == preprocessed_labels).sum().item()
                main_test_metric = running_metric_total / n_test
                elementwise_loss = ce_loss(outputs, preprocessed_labels)
                running_ce_loss_total += torch.mean(elementwise_loss).item()
                for l in cls_labels:
                    loss_by_label[l].extend(elementwise_loss[preprocessed_labels == l])
            else:
                elementwise_loss = compute_mse(torch.squeeze(outputs),
                                               torch.squeeze(preprocessed_labels))
                running_metric_total += torch.sum(elementwise_loss)
                main_test_metric = running_metric_total / n_test

            if helper.params['dataset'] in MINORITY_PERFORMANCE_TRACK_DATASETS:
                # batch_attr_labels is an array of shape [batch_size] where the
                # ith entry is either 1/0/nan and correspond to the attribute labels
                # of the ith element in the batch.
                batch_attr_labels = helper.test_dataset.get_attribute_annotations(idxs)
                for a in attributes:
                    loss_by_attribute[a].extend(
                        elementwise_loss[idx_where_true(batch_attr_labels == a)])
                for k in keys:
                    loss_by_key[k].extend(elementwise_loss[idx_where_true(labels == k)])


    if vis:
        plot(epoch, main_test_metric, metric_name)
        for attr, norms in sorted(attr_norms.items(), key=lambda x: x[0]):
            plot(epoch, torch.mean(torch.stack(norms)), f'norms_by_attr_test/{attr}')
        metric_list = list()
        metric_dict = dict()
        if not mse:  # Plot the classification metrics
            fig, cm = plot_confusion_matrix(correct_labels, predict_labels,
                                            labels=cls_labels, normalize=True)
            writer.add_figure(figure=fig, global_step=epoch, tag='tag/normalized_cm')
            avg_test_loss = running_ce_loss_total / n_test
            plot(epoch, avg_test_loss, 'test_crossentropy_loss')
            for l in cls_labels:
                plot(epoch, mean_of_tensor_list(loss_by_label[l]), 'test_loss_per_class/{}'.format(l))
            for k in keys:
                plot(epoch, mean_of_tensor_list(loss_by_key[k]), 'test_loss_per_key/{}'.format(k))
        for a in attributes:
            plot(epoch, mean_of_tensor_list(loss_by_attribute[a]), 'test_loss_per_attr/{}'.format(a))
        for i, class_name in enumerate(cls_labels):
            if not mse:
                metric_value = cm[i][i] / cm[i].sum() * 100
                fig, cm = plot_confusion_matrix(correct_labels, predict_labels,
                                                labels=cls_labels, normalize=False)
                cm_name = f'{helper.params["folder_path"]}/cm_{epoch}.pt'
                torch.save(cm, cm_name)
                writer.add_figure(figure=fig, global_step=epoch,
                                  tag='tag/unnormalized_cm')
            else:
                metric_value = per_class_mse(
                    outputs, preprocessed_labels, class_name,
                    grouped_label=labels_mapping[class_name]
                ).cpu().numpy()
            metric_dict[class_name] = metric_value
            logger.info(f'Class: {i}, {class_name}: {metric_value}')
            plot(epoch, metric_value, name=f'{metric_name}_per_class/class_{class_name}')
            metric_list.append(metric_value)
        if len(metric_dict):
            fig2 = helper.plot_acc_list(metric_dict, epoch, name='per_class',
                                        accuracy=main_test_metric)
            writer.add_figure(figure=fig2, global_step=epoch, tag='tag/per_class')
            torch.save(metric_dict,
                       f"{helper.folder_path}/test_{metric_name}_class_{epoch}.pt")
        if len(metric_list):
            plot(epoch, np.var(metric_list),
                 name=f'{metric_name}_per_class/{metric_name}_var')
            plot(epoch, np.max(metric_list),
                 name=f'{metric_name}_per_class/{metric_name}_max')
            plot(epoch, np.min(metric_list),
                 name=f'{metric_name}_per_class/{metric_name}_min')
            plot(epoch, np.max(metric_list) - np.min(metric_list),
                 name=f'{metric_name}_intra_class_max_diff/'
                      f'{metric_name}_intra_class_max_diff')

    return main_test_metric


def binarize_labels_tensor(labels: torch.Tensor, pos_labels: list,
                           out_type=torch.float32):
    """
    Create a labels tensor where the ith entry is 1 if labels[i] is in pos_labels,
    and zero otherwise.
    """
    binary_labels = torch.zeros_like(labels, dtype=out_type)
    for l in pos_labels:
        is_l = (labels == l)
        binary_labels += is_l.type(out_type)
    assert torch.max(binary_labels) <= 1., "Sanity check on binarized grouped labels."
    return binary_labels


def train(trainloader, model, optimizer, epoch, labels_mapping=None):
    model.train()
    for i, data in tqdm(enumerate(trainloader, 0), leave=True):
        # get the inputs
        if helper.params['dataset'] in TRIPLET_YIELDING_DATASETS:
            inputs, idxs, labels = data
        else:
            inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        if labels_mapping:
            pos_labels = [k for k, v in labels_mapping.items() if v == 1]
            labels_type = torch.float32 if isinstance(criterion,
                                                      torch.nn.MSELoss) else torch.long
            binarized_labels_tensor = binarize_labels_tensor(labels, pos_labels, labels_type)
            loss = criterion(outputs, binarized_labels_tensor)
        else:
            loss = criterion(outputs, labels)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        if i > 0 and i % 20 == 0:
            plot(epoch * len(trainloader) + i, loss.item(), 'Train Loss')


def unpack_batch(batch, dataset):
    if helper.params['dataset'] in TRIPLET_YIELDING_DATASETS:
        inputs, idxs, labels = batch
    else:
        inputs, labels = batch
        idxs = None
    return inputs, idxs, labels


def train_dp(trainloader, model, optimizer, epoch, labels_mapping=None):
    model.train()
    niters = math.ceil(len(helper.train_dataset) / helper.params['batch_size'])
    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        helper.params['batch_size'],
        helper.params['microbatch_size'],
        niters
    )
    for i, data in tqdm(enumerate(minibatch_loader(helper.train_dataset), 0), leave=True):
        inputs, _, labels = unpack_batch(data, helper.params['dataset'])
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        for inputs_microbatch, labels_microbatch in microbatch_loader(TensorDataset(inputs, labels)):
            inputs_microbatch = inputs_microbatch.to(device)
            labels_microbatch = labels_microbatch.to(device)
            optimizer.zero_microbatch_grad()

            outputs = model(inputs_microbatch)

            if labels_mapping:
                pos_labels = [k for k, v in labels_mapping.items() if v == 1]
                labels_type = torch.float32 if isinstance(criterion,
                                                          torch.nn.MSELoss) else torch.long
                binarized_labels_tensor = binarize_labels_tensor(labels_microbatch, pos_labels, labels_type)
                loss = criterion(outputs, binarized_labels_tensor)
            else:
                loss = criterion(outputs, labels_microbatch)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.microbatch_step()
        optimizer.step()
        if i > 0 and i % 20 == 0:
            plot(epoch * len(trainloader) + i, loss.item(), 'Train Loss')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='params/params.yaml')
    parser.add_argument("--majority_key", default=None, type=int,
                        help="Optionally specify the majority group key (e.g. '1').")
    parser.add_argument("--alpha", default=None, type=float,
                        help="Fractoin of samples to take from majority class. Minority "
                             "class will be downsampled if necessary.")
    parser.add_argument("--number_of_entries_train", default=None, type=int,
                        help="Optional number of minority class entries/size to "
                             "downsample to; if provided, this value overrides value in "
                             ".yaml parameters.")
    parser.add_argument("--logdir", default="./runs",
                        help="Location to write TensorBoard logs.")
    parser.add_argument("--train_attribute_subset", default=None, type=int,
                        help="Optional argument to the train_attribute_subset param; this"
                        "overrides any value which may be present for that field in the"
                        "parameters yaml file.")
    parser.add_argument("--sigma", help="Optional argument to override sigma in params.",
                        default=None, type=float)
    parser.add_argument("--epochs", help="Optional argument to override epochs in params.",
                        default=None, type=int)
    parser.add_argument("--optimizer",
                        help="Optional argument to override optimizer in params.",
                        default=None, type=int)
    parser.add_argument("--lr", type=float, default=None,
                        help="Optional argument to override lr in params.")
    parser.add_argument("--channelwise_mean", action="store_true",
                        default=False,
                        help="If true, will print the mean and STD of the"
                             "dataset by channel prior to beginning training.")
    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params = yaml.load(f)

    for pname in ("train_attribute_subset", 'sigma', 'epochs', 'alpha',
                  'number_of_entries_train', 'lr', 'optimizer'):
        maybe_override_parameter(params, args, pname)

    name = make_uid(params, args)

    uid_logdir = os.path.join(args.logdir, name)
    writer = SummaryWriter(log_dir=uid_logdir)

    helper = get_helper(params, d, name)
    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'experiment uid: {name}')
    logger.info(f'current path: {helper.folder_path}')
    batch_size = int(helper.params['batch_size'])
    lr = float(helper.params['lr'])
    momentum = float(helper.params.get('momentum', 0))
    decay = float(helper.params['decay'])
    epochs = int(helper.params['epochs'])
    z = helper.params.get('z')
    # If clipping bound S is not specified, it is set to inf.
    S = float(helper.params['S']) if helper.params.get('S') else None
    sigma = helper.params.get('sigma')
    dp = helper.params['dp']
    mu = helper.params.get('mu')
    if dp and (sigma is None):
        assert z is not None, \
            "Must either specify sigma, or z (which will be used to compute sigma)."
        sigma = z * S
    alpha = args.alpha
    adaptive_sigma = helper.params.get('adaptive_sigma', False)

    reseed(5)

    criterion = get_criterion(helper)
    is_regression = helper.params.get('criterion') == 'mse'

    true_labels_to_binary_labels, classes_to_keep = load_data(helper, params, alpha, mu)
    num_classes = helper.get_num_classes(classes_to_keep, is_regression)

    print('[DEBUG] num_classes is %s' % num_classes)
    reseed(5)
    net = get_net(helper, num_classes)

    if helper.params.get('multi_gpu', False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    if helper.params.get('resumed_model', False):
        logger.info('Resuming training...')
        loaded_params = torch.load(f"saved_models/{helper.params['resumed_model']}")
        net.load_state_dict(loaded_params['state_dict'])
        helper.start_epoch = loaded_params['epoch']
        # helper.params['lr'] = loaded_params.get('lr', helper.params['lr'])
        logger.info(f"Loaded parameters from saved model: LR is"
                    f" {helper.params['lr']} and current epoch is {helper.start_epoch}")
    else:
        helper.start_epoch = 1

    # Write sample images, for the image classification tasks
    if helper.params['dataset'] in MINORITY_PERFORMANCE_TRACK_DATASETS:
        add_pos_and_neg_summary_images(helper.unnormalized_test_loader,
                                       is_regression,
                                       labels_mapping=true_labels_to_binary_labels)

        # Skip channelwise mean for MNIST; it only has one channel and means are known.
        if helper.params['dataset'] not in SINGLE_CHANNEL_DATASETS \
                and args.channelwise_mean:
            compute_channelwise_mean(helper.train_loader)

    optimizer = get_optimizer(helper, net, dp)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.5 * epochs),
                                                                 int(0.75 * epochs)],
                                                     gamma=0.1)
    table = create_table(helper.params)
    writer.add_text('Model Params', table)
    logger.info(table)
    metric_name = 'mse' if is_regression else 'accuracy'

    epoch = 0
    test_loss = test(net, epoch, name, helper.test_loader,
                     mse=metric_name == 'mse',
                     labels_mapping=true_labels_to_binary_labels)

    try:
        for epoch in range(helper.start_epoch, epochs):
            if dp:
                train_dp(helper.train_loader, net, optimizer, epoch,
                      labels_mapping=true_labels_to_binary_labels)
            else:
                train(helper.train_loader, net, optimizer, epoch,
                          labels_mapping=true_labels_to_binary_labels)
            if helper.params['scheduler']:
                scheduler.step()
            test_loss = test(net, epoch, name, helper.test_loader,
                             mse=metric_name == 'mse',
                             labels_mapping=true_labels_to_binary_labels)

            helper.save_model(net, epoch, test_loss)
    except KeyboardInterrupt:
        print("[KeyboardInterrupt; logged to: {}".format(uid_logdir))
    helper.save_model(net, epoch, test_loss)
    logger.info(
        f"Finished training for model: {helper.current_time}. Folder: {helper.folder_path}")
