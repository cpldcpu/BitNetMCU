import torch, torch.nn as nn, torch.optim as optim
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

#----------------------------------------------
# BitNetMCU training
#----------------------------------------------

def create_run_name(hyperparameters):
    runname = hyperparameters["runtag"] + '_' + hyperparameters["model"] + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["QuantType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_epochs" + str(hyperparameters["num_epochs"])
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
        if 'cnn_width' in params:
            kwargs['cnn_width'] = params['cnn_width']
        if 'num_classes' in params:
            kwargs['num_classes'] = params['num_classes']
        return model_class(**kwargs)
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

    fraction_positive = positive_activations / total_activations
    writer.add_scalar('Activations/positive_fraction', fraction_positive, epoch+1)

    return fraction_positive


# Function to add L1 regularization on the mask
def add_mask_regularization(model,  lambda_l1):
    mask_layer = next((layer for layer in model.modules() if isinstance(layer, MaskingLayer)), None)

    if mask_layer is None:
        return 0
    
    l1_reg = lambda_l1 * torch.norm(mask_layer.mask, 1)
    return l1_reg


def train_model(model, device, hyperparameters, train_data, test_data):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
    runname =  create_run_name(hyperparameters)

    # define dataloaders

    batch_size = hyperparameters["batch_size"]  # Define your batch size

    # ON-the-fly augmentation requires using the (slow) dataloader. Without augmentation, we can load the entire dataset into GPU for speedup
    if hyperparameters["augmentation"]:
        train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    else:
        # load entire dataset into GPU for 5x speedup
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False) # shuffling will be done separately
        entire_dataset = next(iter(train_loader))
        all_train_images, all_train_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

    # Test dataset is always in GPU
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    entire_dataset = next(iter(test_loader))
    all_test_images, all_test_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if hyperparameters["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["lr_decay"])
    elif hyperparameters["scheduler"] == "Cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)    
    elif hyperparameters["scheduler"] == "CosineWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=hyperparameters["T_0"], T_mult=hyperparameters["T_mult"], eta_min=0)
    else:
        raise ValueError("Invalid scheduler")

    criterion = nn.CrossEntropyLoss()

    # tensorboard writer
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'runs/{runname}-{now_str}')

    train_loss=[]
    test_loss = []

    # Train the CNN
    for epoch in range(num_epochs):
        correct = 0
        train_loss=[]
        start_time = time.time()

        if hyperparameters["augmentation"]:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if epoch < hyperparameters['prune_epoch']:
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                correct += (predicted == labels).sum().item()
        else:
            # Shuffle images (important!)
            indices = list(range(len(all_train_images)))
            random.shuffle(indices)

            for i in range(len(indices) // batch_size):
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                images = torch.stack([all_train_images[i] for i in batch_indices])
                labels = torch.stack([all_train_labels[i] for i in batch_indices])
                optimizer.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if epoch < hyperparameters['prune_epoch']:
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                correct += (predicted == labels).sum().item()

        scheduler.step()

        if epoch + 1 == halve_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Learning rate halved at epoch {epoch + 1}")


        trainaccuracy = correct / len(train_loader.dataset) * 100

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

        print(f'Epoch [{epoch+1}/{num_epochs}], LTrain:{np.mean(train_loss):.6f} ATrain: {trainaccuracy:.2f}% LTest:{np.mean(test_loss):.6f} ATest: {correct / total * 100:.2f}% Time[s]: {epoch_time:.2f} Act: {activity*100:.1f}% w_clip/entropy[bits]: ', end='')

        # update clipping scalars once per epoch
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

                print(f'{layer.s.item():.3f}/{entropy:.2f}', end=' ')

                totalbits += layer.weight.numel() * layer.bpw

        print()

        if epoch + 1 == hyperparameters ["prune_epoch"]:
            for m in model.modules():
                if isinstance(m, MaskingLayer):            
                    pruned_channels, remaining_channels = m.prune_channels(prune_number=hyperparameters['prune_groupstoprune'], groups=hyperparameters['prune_totalgroups'])

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

    args = parser.parse_args()

    if args.params:
        paramname = args.params
    else:
        paramname = 'trainingparameters.yaml'

    print(f'Load parameters from file: {paramname}')
    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

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
        train_data = ConcatDataset([train_data, augmented_train_data])

    # Pass num_classes dynamically to model
    hyperparameters['num_classes'] = num_classes
    model = load_model(hyperparameters["model"], {**hyperparameters, 'num_classes': num_classes})
    # If model class supports num_classes argument, it will be used. Otherwise ignore.
    if hasattr(model, 'to'):
        model = model.to(device)

    summary(model, input_size=(1, 16, 16))  # Assuming the input size is (1, 16, 16)

    print('training...')
    train_model(model, device, hyperparameters, train_data, test_data)

    print('saving model...')
    torch.save(model.state_dict(), f'modeldata/{runname}.pth')
