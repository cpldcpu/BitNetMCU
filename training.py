import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from datetime import datetime
from BitNetMCU import FCMNIST, BitLinear, QuantizedModel


#----------------------------------------------
# Define training hyperparameters here

hyperparameters = {
    "num_epochs": 50,
    "QuantType": '4bitsym', # 'Ternary', 'Binary', 'BinaryBalanced', '2bitsym', '4bitsym', '8bit', 'None", 'FP130' 
    "BPW": 4,  # Bits per weight 
    "NormType": 'RMS', # 'RMS', 'Lin', 'BatchNorm'
    "WScale": 'PerOutput', # 'PerTensor', 'PerOutput', 'PerOutputLog2'
    "batch_size": 128,
    "learning_rate": 1e-3,
    "lr_decay": 0.1,
    "step_size": 10,
    "network_width1": 48, # 128 is std size
    "network_width2": 64, # 64 is std size
    "network_width3": 64  # 32 is std size
}

retrain = True  # Train or load model
runtag = 'cosaugment6.5_'
#---------------------------------------------

def create_run_name(hyperparameters):
    runname = 'BitMnist_' + hyperparameters["WScale"] + "_" +hyperparameters["QuantType"] + "_" + hyperparameters["NormType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"]) + "_lr" + str(hyperparameters["learning_rate"]) + "_decay" + str(hyperparameters["lr_decay"]) + "_stepsize" + str(hyperparameters["step_size"]) + "_bs" + str(hyperparameters["batch_size"]) + "_epochs" + str(hyperparameters["num_epochs"])
    return runname

def train_model(model, device, hyperparameters, train_loader, test_loader):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    step_size = hyperparameters["step_size"]
    lr_decay = hyperparameters["lr_decay"]
    runname = runtag + create_run_name(hyperparameters)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)    
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)
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

        scheduler.step()

        trainaccuracy = correct / len(train_loader.dataset) * 100

        correct = 0
        total = 0
        test_loss = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)        
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                test_loss.append(loss.item())            
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        testaccuracy = correct / total * 100

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(train_loss)} Train accuracy: {trainaccuracy}% Test accuracy: {correct / total * 100}%')
        writer.add_scalar('Loss/train', np.mean(train_loss), epoch+1)
        writer.add_scalar('Accuracy/train', trainaccuracy, epoch+1)
        writer.add_scalar('Loss/test', np.mean(test_loss), epoch+1)
        writer.add_scalar('Accuracy/test', testaccuracy, epoch+1)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
        writer.flush()

    numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    totalbits = numofweights * hyperparameters['BPW']

    writer.add_hparams(hyperparameters, {'Parameters': numofweights, 'Totalbits': totalbits, 'Accuracy/train': trainaccuracy, 'Accuracy/test': testaccuracy, 'Loss/train': np.mean(train_loss), 'Loss/test': np.mean(test_loss)})
    writer.close()
# main

runname= runtag+create_run_name(hyperparameters)
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


if True:
    # Data augmentation for training data
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),  
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),   # both are needed for best results.
        # transforms.RandomAffine(degrees=0, translate=(1/16, 1/16)),  # Random shifts of +-1 in x and y
        transforms.Resize((16, 16)),  # Resize images to 16x16
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))


    ])
    augmented_train_data = datasets.MNIST(root='data', train=True, transform=augmented_transform)
    train_data = ConcatDataset([train_data, augmented_train_data])


# Create data loaders
train_loader = DataLoader(train_data, batch_size=hyperparameters["batch_size"], shuffle=True)
test_loader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)

# Initialize the network and optimizer
model = FCMNIST(
    network_width1=hyperparameters["network_width1"], 
    network_width2=hyperparameters["network_width2"], 
    network_width3=hyperparameters["network_width3"], 
    QuantType=hyperparameters["QuantType"], 
    NormType=hyperparameters["NormType"],
    WScale=hyperparameters["WScale"]
).to(device)

if retrain==False:
    model.load_state_dict(torch.load(f'modeldata/{runname}.pth'))
else:
    train_model(model, device, hyperparameters, train_loader, test_loader)
    torch.save(model.state_dict(), f'modeldata/{runname}.pth')

# Quantize the model
quantized_model = QuantizedModel(model)
print(f'Total number of bits: {quantized_model.totalbits()} ({quantized_model.totalbits()/8/1024} kbytes)')

print("Exporting model to header file")
# export the quantized model to a header file
quantized_model.export_to_hfile(f'model_h/{runname}.h')

# Inference using the quantized model
print ("inference of quantized model")

# Initialize counters
total_correct_predictions = 0
total_samples = 0

# Iterate over the test data
for input_data, labels in test_loader:
    # Reshape and convert to numpy
    input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
    labels = labels.cpu().numpy()

    # Inference
    result = quantized_model.inference_quantized(input_data)

    # Get predictions
    predict = np.argmax(result, axis=1)

    # Calculate the fraction of correct predictions for this batch
    correct_predictions = (predict == labels).sum()

    # Update counters
    total_correct_predictions += correct_predictions  # Multiply by batch size
    total_samples += input_data.shape[0]

# Calculate and print the overall fraction of correct predictions
overall_correct_predictions = total_correct_predictions / total_samples

print('Overall accuracy:', overall_correct_predictions * 100, '%') 