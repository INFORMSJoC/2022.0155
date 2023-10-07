import os
import torch
from torch.autograd import Variable


def sample_integers(n, shape, device):
    sample = to_device(torch.randint(0, n, shape), device)
    # print ("sample: ", sample.size()) # (n_batch, n_dim)
    return sample


def resample_rows_per_column(x, device):
    # reference to https://github.com/pbrakel/anica/blob/79d837addcd98ee9bab301d3966e4919cf9732f3/utils.py#L63
    """Permute all rows for each column independently."""
    # print ("x: ", x.size()) # [batch_size, latent_categories, latent_dim - 1]
    n_batch, n_dim, _ = x.size()
    """"
    row_indices = sample_integers(n_batch, (n_batch * n_dim,))
    col_indices = torch.arange(n_dim).repeat(n_batch).flatten()
    # print ("col_indices: ", col_indices.size(), col_indices) # (n_batch * n_dim)
    indices = torch.transpose(torch.stack([row_indices, col_indices]), 0, 1)
    # print ("indices: ", indices.size()) # (n_batch * n_dim, 2)
    x_perm = tf.gather_nd(x, indices)
    x_perm = tf.reshape(x_perm, (n_batch, n_dim))
    """
    indices = sample_integers(n_batch, (n_batch, n_dim), device)
    indices = torch.stack([indices for k in range(_)], 2)
    # print ("indices: ", indices.size(), indices[0, 0, :]) # (n_batch, n_dim, _)
    x_perm = torch.gather(x, 0, indices)
    # print ("x_perm: ", x_perm.size()) # (n_batch, n_dim, _)
    # print ("x_perm: ", x_perm[0, 0, :], x[indices[0, 0, 0], 0, :]) # should be equal
    return x_perm


def to_var(x, volatile=False):
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def model_to_device(t, device):
    torch_version = torch.__version__.split("+")[0]
    torch_version = ".".join(torch_version.split(".")[:2]) + torch_version.split(".")[2]
    torch_version = float(torch_version)
    if device == "cpu":
        if torch_version >= 0.4:
            return t.to(torch.device("cpu"))
        else:
            return t.cpu()
    elif device in ["gpu", "cuda"]:
        if torch_version == 0.3:
            return t.cuda()
    elif "cuda:" in device:
        if torch_version >= 0.4:
            return t.to(torch.device(device))
        else:
            device = int(device.split(":")[1])
            return t.cuda(device=device)
    else:
        raise ValueError('Unknown device ' + device)

def to_device(t, device):
    torch_version = torch.__version__.split("+")[0]
    torch_version = ".".join(torch_version.split(".")[:2]) + torch_version.split(".")[2]
    torch_version = float(torch_version)
    if device == "cpu":
        if torch_version >= 0.4:
            return t.to(torch.device("cpu"))
        else:
            return t.cpu()
    elif device in ["gpu", "cuda"]:
        if torch_version == 0.3:
            return t.cuda()
        else:
            return t.cuda()
    elif "cuda:" in device:
        if torch_version >= 0.4:
            return t.to(torch.device(device))
        else:
            device = int(device.split(":")[1])
            return t.cuda(device=device)
    else:
        raise ValueError('Unknown device ' + device)
