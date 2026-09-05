import math
"""
Integer emulation of the table-based MCU inference for FCMNIST models, verified against the PyTorch QAT model.

The MCU kernel (paper "Everything is a Table", Appendix A.1) computes for every layer
    acc[i] = sum_j T[a_j, w_ij]              int32 accumulation, T = 16x16 int16 product table
    ReLU, then ShiftNorm: sh = smallest shift with (max_i acc[i] >> sh) <= 15, a'_i = acc[i] >> sh   (4-bit codes)
This script rebuilds exactly that in numpy from the trained model (weights -> integer codes, codebook -> table) and
compares predictions with the PyTorch model (which uses the same arithmetic in float via the STE hooks when trained
with act_pow2=True). Any mismatch indicates a train/inference discrepancy.

usage: python verify_int_inference.py --params <yaml> [--model <pth>] [--seed-tag]
"""
import argparse, yaml, numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from BitNetMCU import BitLinear
import training


def load_test(n=None):
    tf = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST(root='data', train=False, transform=tf, download=True)
    x, y = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False)))
    return (x[:n], y[:n]) if n else (x, y)


def layer_codes(layer):
    """Integer weight codes and the multiplier K such that dequantized level = code / K (exact for the table)."""
    u, scale, _ = layer.weight_quant(layer.weight.detach())
    if layer.QuantType in ('NF4', 'NF2'):
        K = layer.table_scale if layer.table_scale > 0 else 2048
        if layer.table_scale == 0:
            print('WARNING: NF-codebook model trained without table_scale; table rounding will introduce a small mismatch')
    elif layer.QuantType in ['2bitsym', '4bitsym', '5bitsym']:
        K = 2
    elif layer.QuantType in ['4bit', 'sint4', '8bit']:
        K = 1
    else:
        raise ValueError(layer.QuantType)
    codes = torch.round(u * K).to(torch.int64).cpu().numpy()
    return codes, K


def shiftnorm_codes(acc, qmax, rnd=False, group=1, mantissa='none'):
    """Per-token requantization of nonnegative int64 accumulators acc [n, d] to codes 0..qmax, exactly as the MCU does.
    mantissa 'none': power-of-two right shift (rnd: (acc + 2^(sh-1)) >> sh).
    mantissa 'max' : shift, then per-token m = floor((8*(qmax+1)-1)/max_s) in 8..15 and x = (x*m) >> 3.
    mantissa 'recipN': per-token N-bit reciprocal s = floor(qmax*2^t/max), code = (acc*s [+ 2^(t-1)]) >> t.
    group: the token max (and hence the scale) is shared over groups of `group` consecutive tokens."""
    top = qmax + 1
    out = np.zeros_like(acc)
    mx = acc.max(axis=1)
    if group > 1:
        mx = np.repeat(mx.reshape(-1, group).max(axis=1), group)
    if str(mantissa).startswith('recip'):
        b = int(mantissa[5:])
        mx = np.maximum(mx, 1).astype(np.int64)
        # exact t: smallest t0 with qmax*2^t0 >= max, then t = t0 + b - 1 so that s has b significant bits
        t0 = np.zeros_like(mx)
        for i, m in enumerate(mx):
            k = -40
            while qmax * 2.0 ** k < m:
                k += 1
            t0[i] = k
        t = t0 + (b - 1)
        s_ = np.minimum(np.floor(qmax * 2.0 ** t / mx).astype(np.int64), 2 ** b - 1)
        for i in range(acc.shape[0]):
            ti = int(t[i])
            prod = acc[i].astype(np.int64) * int(s_[i])
            if ti >= 0:
                out[i] = (prod + ((1 << (ti - 1)) if (rnd and ti > 0) else 0)) >> ti
            else:
                out[i] = prod << (-ti)
        return np.minimum(out, qmax)
    for t in range(acc.shape[0]):
        sh = 0
        while (mx[t] >> sh) > top - 1:
            sh += 1
        if rnd and sh > 0:
            out[t] = np.minimum((acc[t] + (1 << (sh - 1))) >> sh, top - 1)
        else:
            out[t] = acc[t] >> sh
    if mantissa == 'max':
        smax = out.max(axis=1)
        if group > 1:
            smax = np.repeat(smax.reshape(-1, group).max(axis=1), group)
        m = np.clip((8 * top - 1) // np.maximum(smax, 1), 8, 15)
        out = (out * m[:, None]) >> 3
    return np.minimum(out, qmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--model', default=None, help='.pth path (default: modeldata/<runname>.pth)')
    ap.add_argument('--n', type=int, default=None, help='number of test images (default all)')
    args = ap.parse_args()
    hp = yaml.safe_load(open(args.params))
    hp['num_classes'] = 10
    runname = training.create_run_name(hp)
    path = args.model or f'modeldata/{runname}.pth'
    model = training.load_model(hp['model'], hp)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, BitLinear)]
    assert hp.get('act_pow2', False), 'verification is exact only for act_pow2=True models'

    x, y = load_test(args.n)
    with torch.no_grad():
        torch_pred = model(x).argmax(1).numpy()

    tables = []

    def matmul_table(a, layer):
        """int32 accumulation of activation codes a [n, in] with the layer's codebook via the product table."""
        codes, K = layer_codes(layer)
        uniq = np.unique(codes)
        idx_of = {c: i for i, c in enumerate(uniq)}
        widx = np.vectorize(idx_of.get)(codes)                     # [out, in] index into codebook
        # product table T[a, w] = a * code for a in the nibble range -8..15 (signed high nibble / unsigned low
        # nibble). 4-bit layers index it directly; the 8-bit input layer uses nibble decomposition
        # a = 16*a_hi + a_lo with a_hi in -8..7 and a_lo in 0..15: two lookups into the same table (paper 3.3).
        arange = np.arange(-8, 16)
        T = np.outer(arange, uniq).astype(np.int64)                 # [24, n_w_codes]
        assert np.abs(T).max() < 32768, 'table entry overflows int16'
        tables.append((arange, uniq, T))
        if a.min() >= 0 and a.max() <= 15:
            a_hi, a_lo = None, a
        else:
            a_hi, a_lo = a >> 4, a & 15                            # arithmetic shift: signed high nibble
            assert a_hi.min() >= -8 and a_hi.max() <= 7
        acc = np.zeros((a.shape[0], codes.shape[0]), dtype=np.int64)
        for o in range(codes.shape[0]):
            row = T[a_lo + 8, widx[o][None, :]].sum(axis=1)
            if a_hi is not None:
                row += T[a_hi + 8, widx[o][None, :]].sum(axis=1) << 4
            acc[:, o] = row
        assert np.abs(acc).max() < 2**31, 'accumulator overflows int32'
        return acc

    def requant(acc, nxt):
        """ReLU + the requantization the next layer `nxt` was trained with (all integer)."""
        acc = np.maximum(acc, 0)
        qmax = 15 if nxt.act_bits == 4 else 127
        return shiftnorm_codes(acc, qmax, rnd=nxt.act_pow2_round, group=nxt.act_group,
                               mantissa=nxt.act_mantissa if (nxt.act_bits == 4 or str(nxt.act_mantissa).startswith('recip')) else 'none')

    def run_chain(a, chain, nxt_after):
        """apply layers in sequence on token batch a; requantize between layers; return the raw accumulators of the
        last layer (the caller decides how to requantize them: concat first, or logits)."""
        for i, layer in enumerate(chain):
            acc = matmul_table(a, layer)
            if i < len(chain) - 1:
                a = requant(acc, chain[i + 1])
        return acc

    with torch.no_grad():
        if hp['model'] == 'PatchMNIST':
            nb = model.nblocks
            tokens = model.patches(x)                                  # [N*nb, ps*ps], block-major within an image
            head = [m for m in model.model if isinstance(m, BitLinear)] + [model.classifier]
            stems_all = [st for st in (model.stem1, model.stem2, getattr(model, 'stem3', None)) if st is not None]
            if model.patch_shared:
                # shared stem: one chain on all block tokens; requantization of the last stem layer happens on the
                # concatenated image token (shared-scale groups are handled inside shiftnorm via act_group)
                a, _ = stems_all[0].activation_quant(tokens); a = a.to(torch.int64).numpy()
                acc = run_chain(a, stems_all, head[0])
                acc = acc.reshape(acc.shape[0] // nb, -1)
            else:
                # unshared stem: each block has its own chain; concat the raw accumulators, then requantize
                accs = []
                for b in range(nb):
                    tb = tokens[b::nb]
                    chain = [st[b] for st in stems_all]
                    a, _ = chain[0].activation_quant(tb); a = a.to(torch.int64).numpy()
                    accs.append(run_chain(a, chain, head[0]))
                acc = np.concatenate(accs, axis=1)
            a = requant(acc, head[0])
            logits = run_chain(a, head, None)
        else:
            tokens = x.flatten(1)
            a, _ = layers[0].activation_quant(tokens); a = a.to(torch.int64).numpy()
            logits = run_chain(a, layers, None)
    int_pred = logits.argmax(1)
    y = y.numpy()
    agree = (int_pred == torch_pred).mean() * 100
    print(f'model: {path}')
    print(f'layers: {len(layers)}; act bits: {[l.act_bits for l in layers]}; quant: {[l.QuantType for l in layers]}')
    for li, (ar, uq, T) in enumerate(tables):
        print(f'  layer {li}: {len(uq)} weight codes; table rows -8..15 x {len(uq)} = {T.size*2} B int16 (4-bit layers need only rows 0..15 = {16*len(uq)*2} B), max |T| = {np.abs(T).max()}')
    print(f'torch accuracy:   {(torch_pred == y).mean()*100:.2f} %')
    print(f'integer accuracy: {(int_pred == y).mean()*100:.2f} %')
    print(f'prediction agreement torch vs integer emulation: {agree:.2f} % ({(int_pred != torch_pred).sum()} of {len(y)} differ)')


if __name__ == '__main__':
    main()
