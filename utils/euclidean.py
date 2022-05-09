"""Euclidean operations utils functions."""

import torch


def euc_sqdistance(x, y, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy


def euc_sqsubdistance(x, y, subdim=5, rank=10, eval_mode=False):
    """Compute euclidean squared distance in subspace between tensors.

    Args:
        x: torch.Tensor of shape (N1 x k * d)
        y: torch.Tensor of shape (N2 x k * d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    if eval_mode:
        return torch.sum(
            torch.norm(x.view(-1,1,rank,subdim) - y.view(1,-1,rank,subdim), p=2, dim=-1),
            dim = -1
        )
    else:
        assert x.shape[0] == y.shape[0]
        return torch.sum(
            torch.norm((x - y).view(-1, rank, subdim), p=2, dim=-1), 
            dim = -1, keepdim = True
        )


def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate
    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    x = x.view((x.shape[0], -1, 2))
    givens = r.view((r.shape[0], -1, 2))

    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)

    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:2] * torch.cat((-x[:, :, 1:2], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_reflections(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect
    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    x = x.view((x.shape[0], -1, 2))
    givens = r.view((r.shape[0], -1, 2))

    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)

    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:2]), dim=-1) \
          + givens[:, :, 1:2] * torch.cat((x[:, :, 1:2],  x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))


def angle_rotations(theta, x):
    """Givens reflections with an angle parameter.

    Args:
        r: torch.Tensor of shape (N x d/2), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect
    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    x = x.view((x.shape[0], -1, 2))

    x_rot = torch.cos(theta).unsqueeze(-1) * x \
          + torch.sin(theta).unsqueeze(-1) * torch.cat((-x[:, :, 1:2], x[:, :, 0:1]), dim=-1)
    return x_rot.view((x.shape[0], -1))

def angle_reflections(theta, x):
    x = x.view((x.shape[0], -1, 2))

    x_ref = torch.cos(theta).unsqueeze(-1) * torch.cat((x[:, :, 0:1], -x[:, :, 1:2]), dim=-1) \
          + torch.sin(theta).unsqueeze(-1) * torch.cat((x[:, :, 1:2],  x[:, :, 0:1]), dim=-1)
    return x_ref.view((x.shape[0], -1))
