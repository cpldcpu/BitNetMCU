import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from datetime import datetime
# from models import FCMNIST, CNNMNIST
from BitNetMCU import BitLinear, BitConv2d, Activation
import time
import random
import argparse
import yaml
from torchsummary import summary
import importlib
from models import MaskingLayer
from muon import Muon, split_muon_params

#----------------------------------------------
# BitNetMCU training
#----------------------------------------------

def create_run_name(hyperparameters):
    runname = hyperparameters["runtag"] + '_' + hyperparameters["model"] + ('_Muon' + ('Head' if hyperparameters.get("muon_head",False) else '') + (f'lr{hyperparameters["muon_lr"]}' if hyperparameters.get("muon_lr",0.02)!=0.02 else '') if hyperparameters.get("optimizer","Adam")=="Muon" else '') + ('_AdamW' if hyperparameters.get("optimizer","Adam")=="AdamW" else '') + (f'_A{hyperparameters["act_bits"]}' if hyperparameters.get("act_bits",8)!=8 else '') + ('p2' if hyperparameters.get("act_pow2",False) else '') + ('r' if hyperparameters.get("act_pow2_round",False) else '') + (f'm{hyperparameters["act_mantissa"][:3] if not hyperparameters["act_mantissa"].startswith("recip") else "R"+hyperparameters["act_mantissa"][5:]}' if hyperparameters.get("act_mantissa","none")!="none" else '') + ('_Shared' if hyperparameters.get("patch_shared_scale",False) else '')+ ('_Unshared' if hyperparameters.get("model")=="PatchMNIST" and not hyperparameters.get("patch_shared",True) else '') + (f'_{hyperparameters["patch_mode"][:4]}{hyperparameters.get("patch_width1",0)}_{hyperparameters.get("patch_width2",0)}' + (f'_{hyperparameters["patch_width3"]}' if hyperparameters.get("patch_width3",0) else '') + (f'_ps{hyperparameters["patch_size"]}' if hyperparameters.get("patch_size",8)!=8 else '')+ (f'_st{hyperparameters["patch_stride"]}' if hyperparameters.get("patch_stride",0) else '') if hyperparameters.get("model")=="PatchMNIST" else '') + ('_Aug' if hyperparameters["augmentation"] else '') + (f'_Pool{hyperparameters["augmentation_pool"]}' if hyperparameters.get("augmentation_pool",0)>0 else '') + (f'_GpuAug{hyperparameters.get("augmentation_copies",1)}' if hyperparameters.get("augmentation_pool",0)==-1 else '') + '_BitMnist_' + hyperparameters["QuantType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_epochs" + str(hyperparameters["num_epochs"])
    hyperparameters["runname"] = runname
    return runname

def load_model(model_name, params):
    try:
        module = importlib.import_module('models')
        model_class = getattr(module, model_name)
        kwargs = dict(
            network_width1=params["network_width1"],
            network_width2=params["network_width2"],
            network_width3=params["network_width3"],
            QuantType=params["QuantType"],
            NormType=params["NormType"],
            WScale=params["WScale"]
        )
        import inspect
        sig = inspect.signature(model_class.__init__).parameters
        # pass every hyperparameter the model constructor knows about (cnn_width, patch_width1, patch_mode, ...)
        for k, v in params.items():
            if k in sig and k not in kwargs and not k.startswith('_'):
                kwargs[k] = v
        if 'num_classes' in params:
            kwargs['num_classes'] = params['num_classes']
        model = model_class(**kwargs)
        # activation quantization settings (W4A4 support), applied to all quantized layers.
        # The first layer sees the (signed) input image and may keep a higher resolution (first_layer_act_bits).
        qlayers = [m for m in model.modules() if isinstance(m, (BitLinear, BitConv2d))]
        for i, m in enumerate(qlayers):
            m.act_bits = params.get('act_bits', 8)
            if i == 0:
                m.act_bits = params.get('first_layer_act_bits', m.act_bits)
            m.act_pow2 = params.get('act_pow2', False)
            m.act_unsigned = i > 0          # every layer but the first is fed through a ReLU
            m.act_pow2_round = params.get('act_pow2_round', False)
            m.is_output = (i == len(qlayers) - 1)
            m.act_mantissa = params.get('act_mantissa', 'none')
            m.table_scale = params.get('table_scale', 0)
        return model
    except AttributeError:
        raise ValueError(f"Model {model_name} not found in models.py")

def log_positive_activations(model, writer, epoch, all_test_images, batch_size):
    total_activations = 0
    positive_activations = 0

    def hook_fn(module, input, output):
        nonlocal total_activations, positive_activations
        if isinstance(module, nn.ReLU) or isinstance(module, Activation):
            total_activations += output.numel()
            positive_activations += (output > 0).sum().item()

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.ReLU) or isinstance(layer, Activation):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Run a forward pass to trigger hooks
    with torch.no_grad():
        for i in range(len(all_test_images) // batch_size):
            images = all_test_images[i * batch_size:(i + 1) * batch_size]
            model(images)

    for hook in hooks:
        hook.remove()

    fraction_positive = positive_activations / max(total_activations, 1)
    writer.add_scalar('Activations/positive_fraction', fraction_positive, epoch+1)

    return fraction_positive


# Function to add L1 regularization on the mask
def add_mask_regularization(model,  lambda_l1):
    mask_layer = next((layer for layer in model.modules() if isinstance(layer, MaskingLayer)), None)

    if mask_layer is None:
        return 0
    
    l1_reg = lambda_l1 * torch.norm(mask_layer.mask, 1)
    return l1_reg


class GPUAugmenter:
    """On-the-fly augmentation on the GPU (augmentation_pool: -1): fresh random affine (+ optional elastic) copies of the
    28x28 originals every epoch, generated in one batched grid_sample, then resized to 16x16 and normalized.
    Same augmentation family as the torchvision pipeline (RandomRotation(r1) + RandomAffine(r2, translate 0.1, scale 0.9..1.1)
    + ElasticTransform(alpha=40, sigma=4) with probability p), but ~100x faster and without the PIL dataloader.
    Not byte-identical to the PIL path (bilinear vs PIL resampling)."""
    def __init__(self, raw28, labels, hp, mean, std, out_size=16):
        self.raw = raw28            # [N,1,28,28] float 0..1 on device
        self.labels = labels
        self.r1 = hp["rotation1"]; self.r2 = hp["rotation2"]
        self.p_elastic = hp.get("elastictransformprobability", 0.0)
        self.elastic_alpha = hp.get("elastic_alpha", 40.0); self.elastic_sigma = hp.get("elastic_sigma", 4.0)
        self.mean, self.std, self.out_size = mean[0], std[0], out_size
        self.mode = hp.get("gpu_aug_interp", "bilinear")   # grid_sample interpolation: 'bilinear' or 'nearest' (PIL default is nearest)

    @torch.no_grad()
    def sample(self, copies=1, chunk=20000):
        outs, labs = [], []
        for _ in range(copies):
            for i in range(0, self.raw.shape[0], chunk):
                x = self.raw[i:i + chunk]
                outs.append(self._augment(x)); labs.append(self.labels[i:i + chunk])
        return torch.cat(outs), torch.cat(labs)

    def _augment(self, x):
        n, dev = x.shape[0], x.device
        deg = (torch.rand(n, device=dev) * 2 - 1) * self.r1 + (torch.rand(n, device=dev) * 2 - 1) * self.r2
        ang = deg * 3.14159265 / 180
        sc = 0.9 + 0.2 * torch.rand(n, device=dev)
        tx = (torch.rand(n, device=dev) * 2 - 1) * 0.1 * 2   # 10% of width in normalized [-1,1] coords
        ty = (torch.rand(n, device=dev) * 2 - 1) * 0.1 * 2
        c, si = torch.cos(ang) / sc, torch.sin(ang) / sc     # inverse map (output -> input) for grid_sample
        theta = torch.stack([torch.stack([c, -si, -tx], 1), torch.stack([si, c, -ty], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        if self.p_elastic > 0:
            h = x.shape[-1]
            k = int(self.elastic_sigma * 6) | 1
            ax = torch.arange(k, device=dev) - k // 2
            g = torch.exp(-ax.float() ** 2 / (2 * self.elastic_sigma ** 2)); g = g / g.sum()
            d = torch.rand(n, 2, h, h, device=dev) * 2 - 1
            d = F.conv2d(d.reshape(n * 2, 1, h, h), g.view(1, 1, k, 1), padding=(k // 2, 0))
            d = F.conv2d(d, g.view(1, 1, 1, k), padding=(0, k // 2)).reshape(n, 2, h, h)
            d = d * self.elastic_alpha * 2.0 / h                 # pixel displacement -> normalized coords
            apply = (torch.rand(n, 1, 1, 1, device=dev) < self.p_elastic).float()
            grid = grid + (d * apply).permute(0, 2, 3, 1)
        y = F.grid_sample(x, grid, mode=self.mode, padding_mode='zeros', align_corners=False)
        y = F.interpolate(y, size=(self.out_size, self.out_size), mode='bilinear', antialias=True, align_corners=False)
        return (y - self.mean) / self.std


def build_augmentation_pool(train_data, augmented_train_data, copies, seed):
    """Pre-materialize a fixed augmentation pool: originals + `copies` augmented passes over the training set.
    The pool is generated once with a fixed seed, so it is byte-identical across runs/configs and removes
    on-the-fly augmentation as a between-config confound (and the slow dataloader from the training loop)."""
    t0 = time.time()
    # the pool must not disturb the training seed: save/restore the global RNG states around generation
    rng_state = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    imgs, lbls = [], []
    base = next(iter(DataLoader(train_data, batch_size=len(train_data), shuffle=False)))
    imgs.append(base[0]); lbls.append(base[1])
    for c in range(copies):
        # single process (Windows-safe) and explicitly seeded per copy -> deterministic pool
        torch.manual_seed(seed * 1000 + c); random.seed(seed * 1000 + c); np.random.seed(seed * 1000 + c)
        for x, y in DataLoader(augmented_train_data, batch_size=4096, shuffle=False, num_workers=0):
            imgs.append(x); lbls.append(y)
    torch.set_rng_state(rng_state[0]); random.setstate(rng_state[1]); np.random.set_state(rng_state[2])
    images = torch.cat(imgs); labels = torch.cat(lbls)
    print(f'Augmentation pool: {len(images)} images ({copies} augmented copies + originals), built in {time.time()-t0:.1f}s')
    return torch.utils.data.TensorDataset(images, labels)


def train_model(model, device, hyperparameters, train_data, test_data, gpu_aug=None):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
    runname =  create_run_name(hyperparameters)

    # define dataloaders

    batch_size = hyperparameters["batch_size"]  # Define your batch size

    # ON-the-fly augmentation requires using the (slow) dataloader. Without augmentation, we can load the entire dataset into GPU for speedup
    # A pre-materialized augmentation pool (augmentation_pool > 0) is a fixed tensor dataset and also takes the in-GPU path.
    onthefly = hyperparameters["augmentation"] and hyperparameters.get("augmentation_pool", 0) == 0
    if onthefly:
        train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    else:
        # load entire dataset into GPU for 5x speedup
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False) # shuffling will be done separately
        entire_dataset = next(iter(train_loader))
        all_train_images, all_train_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)
        base_images, base_labels = all_train_images, all_train_labels

    # Test dataset is always in GPU
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    entire_dataset = next(iter(test_loader))
    all_test_images, all_test_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

    # Optimizer: 'Adam' (default, all params) or 'Muon' (NS5-orthogonalized momentum on hidden weight matrices,
    # Adam on the classifier head, biases and tiny tensors). Both get the same LR schedule shape.
    opt_name = hyperparameters.get("optimizer", "Adam")
    if opt_name == "Muon":
        # the classifier head stays on Adam (paper recipe) unless muon_head: True
        head = [model.classifier] if (hasattr(model, "classifier") and not hyperparameters.get("muon_head", False)) else []
        muon_params, other_params = split_muon_params(model, min_numel=hyperparameters.get("muon_min_numel", 65), exclude_modules=head)
        print(f'Muon: {len(muon_params)} matrices ({sum(p.numel() for p in muon_params)} params), Adam: {len(other_params)} tensors ({sum(p.numel() for p in other_params)} params)')
        optimizers = [Muon(muon_params, lr=hyperparameters.get("muon_lr", 0.02), momentum=hyperparameters.get("muon_momentum", 0.95),
                           weight_decay=hyperparameters.get("muon_weight_decay", 0.0)),
                      optim.Adam(other_params, lr=learning_rate)]
    elif opt_name == "Adam":
        optimizers = [optim.Adam(model.parameters(), lr=learning_rate)]
    elif opt_name == "AdamW":
        optimizers = [optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=hyperparameters.get("weight_decay", 0.01))]
    else:
        raise ValueError(f"Invalid optimizer: {opt_name}")
    optimizer = optimizers[-1]  # Adam group, used for LR logging

    def make_scheduler(opt):
        if hyperparameters["scheduler"] == "StepLR":
            return StepLR(opt, step_size=hyperparameters["step_size"], gamma=hyperparameters["lr_decay"])
        elif hyperparameters["scheduler"] == "Cosine":
            return CosineAnnealingLR(opt, T_max=num_epochs, eta_min=0)
        elif hyperparameters["scheduler"] == "CosineWarmRestarts":
            return CosineAnnealingWarmRestarts(opt, T_0=hyperparameters["T_0"], T_mult=hyperparameters["T_mult"], eta_min=0)
        else:
            raise ValueError("Invalid scheduler")
    schedulers = [make_scheduler(o) for o in optimizers]

    criterion = nn.CrossEntropyLoss()

    # tensorboard writer
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_tag = f'-s{hyperparameters["seed"]}' if 'seed' in hyperparameters else ''
    writer = SummaryWriter(log_dir=f'runs/{runname}-{now_str}{seed_tag}')

    train_loss=[]
    test_loss = []

    # initialize clipping scalars from the initial weights (paper: at s=1, aggressive codebooks quantize everything to 0)
    # clip_init: True initializes clipping from the initial weights (recommended; the stock yaml enables it).
    # The code default is False so that old parameter files reproduce the original behaviour exactly.
    if hyperparameters.get("clip_init", False):
        for layer in model.modules():
            if isinstance(layer, (BitLinear, BitConv2d)) and layer.QuantType not in ['None', 'Binary', 'BinarySym', 'Ternary']:
                layer.update_clipping_scalar(layer.weight, hyperparameters['maxw_algo'], hyperparameters['maxw_quantscale'])

    # Train the CNN
    for epoch in range(num_epochs):
        correct = 0
        train_loss=[]
        start_time = time.time()

        if gpu_aug is not None:
            aug_images, aug_labels = gpu_aug.sample(hyperparameters.get("augmentation_copies", 1))
            all_train_images = torch.cat([base_images, aug_images]); all_train_labels = torch.cat([base_labels, aug_labels])
            del aug_images, aug_labels

        if onthefly:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                for o in optimizers: o.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if epoch < hyperparameters['prune_epoch']:
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
                loss.backward()
                for o in optimizers: o.step()
                train_loss.append(loss.item())
                correct += (predicted == labels).sum().item()
        else:
            # Shuffle images (important!)
            indices = torch.randperm(len(all_train_images), device=device)

            for i in range(len(indices) // batch_size):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                images = all_train_images[batch_indices]
                labels = all_train_labels[batch_indices]
                for o in optimizers: o.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if epoch < hyperparameters['prune_epoch']:
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
                loss.backward()
                for o in optimizers: o.step()
                train_loss.append(loss.item())
                correct += (predicted == labels).sum().item()

        for sch in schedulers: sch.step()

        if epoch + 1 == halve_lr_epoch:
            for o in optimizers:
                for param_group in o.param_groups:
                    param_group['lr'] *= 0.5
            print(f"Learning rate halved at epoch {epoch + 1}")


        clip_info = ''
        # update clipping scalars once per epoch (before evaluation, so the logged test accuracy is that of the saved model)
        totalbits = 0
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):

                # update clipping scalar
                if epoch < hyperparameters['maxw_update_until_epoch']:
                    layer.update_clipping_scalar(layer.weight, hyperparameters['maxw_algo'], hyperparameters['maxw_quantscale'])

                # calculate entropy of weights
                w_quant, _, _ = layer.weight_quant(layer.weight)
                _, counts = np.unique(w_quant.cpu().detach().numpy(), return_counts=True)
                probabilities = counts / np.sum(counts)
                entropy = -np.sum(probabilities * np.log2(probabilities))

                clip_info += f'{layer.s.item():.3f}/{entropy:.2f} '

                totalbits += layer.weight.numel() * layer.bpw

        if epoch + 1 == hyperparameters ["prune_epoch"]:
            for m in model.modules():
                if isinstance(m, MaskingLayer):            
                    pruned_channels, remaining_channels = m.prune_channels(prune_number=hyperparameters['prune_groupstoprune'], groups=hyperparameters['prune_totalgroups'])


        n_train = len(all_train_images) if gpu_aug is not None else len(train_loader.dataset)
        trainaccuracy = correct / n_train * 100

        correct = 0
        total = 0
        test_loss = []
        with torch.no_grad():
            for i in range(len(all_test_images) // batch_size):
                images = all_test_images[i * batch_size:(i + 1) * batch_size]
                labels = all_test_labels[i * batch_size:(i + 1) * batch_size]

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                test_loss.append(loss.item())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Log positive activations
        activity=log_positive_activations(model, writer, epoch, all_test_images, batch_size)

        end_time = time.time()
        epoch_time = end_time - start_time

        testaccuracy = correct / total * 100

        # Track the best test accuracy seen so far and keep that checkpoint (modeldata/<runname>_best.pth).
        # Note: selection on the test set is optimistic; the final-epoch checkpoint remains the primary result.
        if epoch == 0 or testaccuracy > best_testaccuracy:
            best_testaccuracy, best_epoch = testaccuracy, epoch + 1
            torch.save({'epoch': best_epoch, 'test_accuracy': best_testaccuracy, 'state_dict': model.state_dict()},
                       f'modeldata/{runname}_best.pth')
        writer.add_scalar('Accuracy/test_best', best_testaccuracy, epoch+1)
        if epoch + 1 == num_epochs:
            print(f'best test accuracy {best_testaccuracy:.2f}% at epoch {best_epoch} (modeldata/{runname}_best.pth)')

        print(f'Epoch [{epoch+1}/{num_epochs}], LTrain:{np.mean(train_loss):.6f} ATrain: {trainaccuracy:.2f}% LTest:{np.mean(test_loss):.6f} ATest: {correct / total * 100:.2f}% Time[s]: {epoch_time:.2f} Act: {activity*100:.1f}% w_clip/entropy[bits]: ' + clip_info)

        writer.add_scalar('Loss/train', np.mean(train_loss), epoch+1)
        writer.add_scalar('Accuracy/train', trainaccuracy, epoch+1)
        writer.add_scalar('Loss/test', np.mean(test_loss), epoch+1)
        writer.add_scalar('Accuracy/test', testaccuracy, epoch+1)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
        writer.flush()

    numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # totalbits = numofweights * hyperparameters['BPW']

    print(f'TotalBits: {totalbits} TotalBytes: {totalbits/8.0} ')

    writer.add_hparams(hyperparameters, {'Parameters': numofweights, 'Totalbits': totalbits, 'Accuracy/train': trainaccuracy, 'Accuracy/test': testaccuracy, 'Loss/train': np.mean(train_loss), 'Loss/test': np.mean(test_loss)})
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--params', type=str, help='Name of the parameter file', default='trainingparameters.yaml')
    parser.add_argument('--seed', type=int, default=None, help='Training seed (model init / shuffling). Overrides yaml "seed".')
    parser.add_argument('--runtag', type=str, default=None, help='Override runtag from yaml')

    args = parser.parse_args()

    if args.params:
        paramname = args.params
    else:
        paramname = 'trainingparameters.yaml'

    print(f'Load parameters from file: {paramname}')
    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    if args.runtag is not None:
        hyperparameters["runtag"] = args.runtag
    seed = args.seed if args.seed is not None else hyperparameters.get("seed", None)
    if seed is not None:
        hyperparameters["seed"] = seed
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        print(f'Training seed: {seed}')

    runname= create_run_name(hyperparameters)
    print(runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset selection (MNIST default, EMNIST optional)
    dataset_name = hyperparameters.get("dataset", "MNIST").upper()

    if dataset_name == "MNIST":
        num_classes = 10
        mean, std = (0.1307,), (0.3081,)
        base_dataset_train = datasets.MNIST
        base_dataset_test = datasets.MNIST
        dataset_kwargs = {"train": True}
        dataset_kwargs_test = {"train": False}
    elif dataset_name.startswith("EMNIST"):
        # Expected format: EMNIST or EMNIST_BALANCED, EMNIST_BYCLASS etc.
        # Torchvision subsets: 'byclass'(62), 'bymerge'(47), 'balanced'(47), 'letters'(37), 'digits'(10), 'mnist'(10)
        split = dataset_name.split('_')[1].lower() if '_' in dataset_name else 'balanced'
        # Map common names
        split_alias = { 'BALANCED':'balanced', 'BYCLASS':'byclass', 'BYMERGE':'bymerge', 'LETTERS':'letters', 'DIGITS':'digits', 'MNIST':'mnist'}
        split = split_alias.get(split.upper(), split)
        # class counts per split
        split_classes = { 'byclass':62, 'bymerge':47, 'balanced':47, 'letters':37, 'digits':10, 'mnist':10 }
        num_classes = split_classes.get(split, 47)
        # EMNIST uses same normalization as MNIST typically
        mean, std = (0.1307,), (0.3081,)
        from torchvision.datasets import EMNIST
        base_dataset_train = EMNIST
        base_dataset_test = EMNIST
        dataset_kwargs = {"split": split, "train": True}
        dataset_kwargs_test = {"split": split, "train": False}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data = base_dataset_train(root='data', transform=transform, download=True, **dataset_kwargs)
    test_data = base_dataset_test(root='data', transform=transform, download=True, **dataset_kwargs_test)

    gpu_augmenter = None
    if hyperparameters["augmentation"]:
        # Data augmentation for training data
        augmented_transform = transforms.Compose([
            transforms.RandomRotation(degrees=hyperparameters["rotation1"]),
            transforms.RandomAffine(degrees=hyperparameters["rotation2"], translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=40.0, sigma=4.0)
            ], p=hyperparameters["elastictransformprobability"]),
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        augmented_train_data = base_dataset_train(root='data', transform=augmented_transform, download=True, **dataset_kwargs)
        pool_copies = hyperparameters.get("augmentation_pool", 0)
        if pool_copies == -1:
            # GPU on-the-fly: originals through the standard transform + raw 28x28 images for augmentation on the device
            raw_ds = base_dataset_train(root='data', transform=transforms.ToTensor(), download=True, **dataset_kwargs)
            raw, raw_labels = next(iter(DataLoader(raw_ds, batch_size=len(raw_ds), shuffle=False)))
            gpu_augmenter = GPUAugmenter(raw.to(device), raw_labels.to(device), hyperparameters, mean, std)
            print(f'GPU on-the-fly augmentation: {len(raw)} originals + {hyperparameters.get("augmentation_copies", 1)} fresh augmented copies per epoch')
        elif pool_copies > 0:
            train_data = build_augmentation_pool(train_data, augmented_train_data, pool_copies, hyperparameters.get("pool_seed", 0))
        else:
            train_data = ConcatDataset([train_data, augmented_train_data])

    # Pass num_classes dynamically to model
    hyperparameters['num_classes'] = num_classes
    model = load_model(hyperparameters["model"], {**hyperparameters, 'num_classes': num_classes})
    # If model class supports num_classes argument, it will be used. Otherwise ignore.
    if hasattr(model, 'to'):
        model = model.to(device)

    summary(model, input_size=(1, 16, 16))  # Assuming the input size is (1, 16, 16)

    print('training...')
    train_model(model, device, hyperparameters, train_data, test_data, gpu_aug=gpu_augmenter)

    print('saving model...')
    torch.save(model.state_dict(), f'modeldata/{runname}.pth')
