import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from datetime import datetime
from models import FCMNIST, CNNMNIST
from BitNetMCU import BitLinear, BitConv2d
import time
import random
import argparse
import yaml
from torchsummary import summary
from simpleswa import SimpleSWA

#----------------------------------------------
# BitNetMCU training
# cpldcpu 2024-03 
#----------------------------------------------

def create_run_name(hyperparameters):
    runname = hyperparameters["runtag"] + hyperparameters["scheduler"] + '_lr' + str(hyperparameters["learning_rate"]) + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["WScale"] + "_" +hyperparameters["QuantType"] + "_" + hyperparameters["NormType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_bs" + str(hyperparameters["batch_size"]) + "_epochs" + str(hyperparameters["num_epochs"])
    hyperparameters["runname"] = runname
    return runname

def train_model(model, device, hyperparameters, train_data, test_data):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    step_size = hyperparameters["step_size"]
    lr_decay = hyperparameters["lr_decay"]
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

    swa = SimpleSWA(model, swa_start=hyperparameters["swa_start"], 
                    swa_lr=hyperparameters["swa_lr"], 
                    cycle_length=hyperparameters["swa_cycle_length"])

    if hyperparameters["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
    elif hyperparameters["scheduler"] == "Cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)    

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

        lr = swa.get_lr(epoch, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if hyperparameters["augmentation"]:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
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
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                correct += (predicted == labels).sum().item()



        scheduler.step()
        swa.update(epoch)

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

        testaccuracy = correct / total * 100

        swa_correct = 0
        swa_total = 0
        swa_loss = []
        with torch.no_grad():
            for i in range(len(all_test_images) // batch_size):
                images = all_test_images[i * batch_size:(i + 1) * batch_size]
                labels = all_test_labels[i * batch_size:(i + 1) * batch_size]

                outputs = swa.get_final_model()(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                swa_loss.append(loss.item())            
                swa_total += labels.size(0)
                swa_correct += (predicted == labels).sum().item()

        swaaccuracy = swa_correct / swa_total * 100


        end_time = time.time()
        epoch_time = end_time - start_time

     
        print(f'Epoch [{epoch+1}/{num_epochs}], Tr_loss:{np.mean(train_loss):.6f} Tr_acc: {trainaccuracy:.2f}% Te_loss:{np.mean(test_loss):.6f} Te_acc: {correct / total * 100:.2f}% SWA_loss:{np.mean(swa_loss):.6f} SWA_acc: {swaaccuracy:.2f}% Time[s]: {epoch_time:.2f} w_clip/entropy[bits]: ', end='')

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

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((16, 16)),  # Resize images to 16x16
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transform)

    if hyperparameters["augmentation"]:
        # Data augmentation for training data
        augmented_transform = transforms.Compose([
            # 10,10 seems to be best combination
            transforms.RandomRotation(degrees=hyperparameters["rotation1"]),  
            transforms.RandomAffine(degrees=hyperparameters["rotation2"], translate=(0.1, 0.1), scale=(0.9, 1.1)),   # both are needed for best results.
            transforms.Resize((16, 16)),  # Resize images to 16x16
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        augmented_train_data = datasets.MNIST(root='data', train=True, transform=augmented_transform)
        train_data = ConcatDataset([train_data, augmented_train_data])

    # Initialize the network and optimizer
    model = FCMNIST(
    # model = CNNMNIST(
        network_width1=hyperparameters["network_width1"], 
        network_width2=hyperparameters["network_width2"], 
        network_width3=hyperparameters["network_width3"], 
        QuantType=hyperparameters["QuantType"], 
        NormType=hyperparameters["NormType"],
        WScale=hyperparameters["WScale"],
    ).to(device)

    summary(model, input_size=(1, 16, 16))  # Assuming the input size is (1, 16, 16)

    print('training...')
    train_model(model, device, hyperparameters, train_data, test_data)

    print('saving model...')    
    torch.save(model.state_dict(), f'modeldata/{runname}.pth')
