"""
Muon optimizer (Keller Jordan, 2024): Newton-Schulz orthogonalized (nesterov) momentum for 2-D weight matrices.
Reference: https://github.com/KellerJordan/Muon . Kept dependency-free and single-GPU.

Used for QAT in BitNetMCU: gradients reach the latent weights through the STE, Muon then applies an update whose
singular values are all ~1 (every direction of the weight matrix is updated with equal energy). Conv kernels are
flattened to [out, in*kh*kw]. Non-matrix params (biases, tiny tensors, classifier head) should use Adam.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration computing an approximate orthogonalization (U V^T) of G.
    Coefficients (3.4445, -4.7750, 2.0315) are tuned for fast convergence rather than exactness;
    the result has singular values in ~[0.7, 1.3], which is what Muon relies on."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Args:
        params:       iterable of 2-D+ parameters (higher-dim tensors are flattened to [out, -1])
        lr:           learning rate (0.02 is the paper/reference default)
        momentum:     momentum coefficient (0.95)
        nesterov:     use nesterov momentum (True)
        ns_steps:     Newton-Schulz iterations (5)
        weight_decay: decoupled weight decay (0)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, nesterov, ns_steps, wd = group['lr'], group['momentum'], group['nesterov'], group['ns_steps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:
                    g = g.reshape(g.size(0), -1)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=mom)
                else:
                    g = buf
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                # scale so that the update RMS is independent of the matrix shape (reference implementation)
                g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g.view_as(p), alpha=-lr)
        return loss


def split_muon_params(model, min_numel=65, exclude_modules=()):
    """Split model params into (muon_params, other_params).
    Muon: >=2-D weights with >= min_numel elements that are not in `exclude_modules` (e.g. the classifier head).
    Everything else (biases, norm scales, clipping scalars, tiny tensors, head) -> Adam.
    """
    excluded = set()
    for m in exclude_modules:
        excluded.update(id(p) for p in m.parameters())
    muon, other = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and p.numel() >= min_numel and id(p) not in excluded:
            muon.append(p)
        else:
            other.append(p)
    return muon, other
