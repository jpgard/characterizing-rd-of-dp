from torch import optim as optim
from pyvacy.optim import DPSGD, DPAdam


def get_optimizer(helper, net, dp: bool):
    opt_str = helper.params['optimizer']
    lr = helper.params['lr']
    decay = float(helper.params['decay'])
    momentum = helper.params.get('momentum')
    # S =
    sigma = helper.params.get('sigma')
    microbatch_size = helper.params.get('microbatch_size')
    batch_size = helper.params.get('batch_size')
    if (not dp) or (helper.params.get('no_clip')):  # No DP, or DP without clipping
        l2_norm_clip = None
    else:  # DP, with clipping.
        l2_norm_clip = float(helper.params.get('S'))
    if dp and (l2_norm_clip is None):
        print("[DEBUG] optimizer uses no clipping.")
    if opt_str == 'SGD' and not dp:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                              weight_decay=decay)
    elif opt_str == 'SGD' and dp:
        optimizer = DPSGD(params=net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=decay,
                          l2_norm_clip=l2_norm_clip,
                          noise_multiplier=sigma, minibatch_size=batch_size,
                          microbatch_size=microbatch_size)
    elif opt_str == 'Adam' and not dp:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
    elif opt_str == 'Adam' and dp:
        optimizer = DPAdam(params=net.parameters(), lr=lr,
                           weight_decay=decay,
                           l2_norm_clip=l2_norm_clip,
                           noise_multiplier=sigma, minibatch_size=batch_size,
                           microbatch_size=microbatch_size)
    else:
        raise Exception('Specify `optimizer` in params.yaml.')
    return optimizer
