import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import importlib
from BitNetMCU import BitLinear, BitConv2d

def create_run_name(hyperparameters):
    runname = hyperparameters["runtag"] + '_' + hyperparameters["model"] + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["QuantType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_epochs" + str(hyperparameters["num_epochs"])
    hyperparameters["runname"] = runname
    return runname

def load_data():
    transform = transforms.Compose([
        # transforms.RandomRotation(degrees=10),  
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),   # both are needed for best results.
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transform)
    return train_data, test_data

def load_model(model_name, params):
    try:
        module = importlib.import_module('models')
        model_class = getattr(module, model_name)
        return model_class(
            network_width1=params["network_width1"],
            network_width2=params["network_width2"],
            network_width3=params["network_width3"],
            QuantType=params["QuantType"],
            NormType=params["NormType"],
            WScale=params["WScale"]
        )
    except AttributeError:
        raise ValueError(f"Model {model_name} not found in models.py")

class StudentModel(nn.Module):
    def __init__(self, base_model, teacher_last_layer):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = teacher_last_layer
        
        # Freeze the classifier (teacher's last layer)
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return features, output

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = list(model.children())[-1]

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return features, output

def distillation_loss(student_features, teacher_features, labels, temperature, alpha):
    # Feature distillation loss
    feature_loss = nn.MSELoss()(student_features, teacher_features)
    
    # Hard targets loss (using the frozen teacher's last layer)
    student_logits = student_features  # The features are directly fed to the teacher's last layer
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    return alpha * feature_loss + (1 - alpha) * hard_loss

def train_distill(teacher, student, device, train_loader, test_loader, params):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=params['learning_rate'])
    if params['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['lr_decay'])
    elif params['scheduler'] == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'], eta_min=0)

    temperature = params['temperature']
    alpha = params['alpha']

    writer = SummaryWriter(log_dir=f'runs/distill_penultimate_frozen_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    for epoch in range(params['num_epochs']):
        student.train()
        train_loss = []
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_features, _ = teacher(images)
            
            student_features, student_logits = student(images)
            loss = distillation_loss(student_features, teacher_features, labels, temperature, alpha)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        # Evaluation
        student.eval()
        test_loss = []
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, outputs = student(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        test_accuracy = 100 * test_correct / test_total

        print(f'Epoch [{epoch+1}/{params["num_epochs"]}], '
              f'Train Loss: {np.mean(train_loss):.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {np.mean(test_loss):.4f}, Test Acc: {test_accuracy:.2f}%')

        for i, layer in enumerate(student.modules()):
            if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):

                # update clipping scalar 
                if epoch < params['maxw_update_until_epoch']:
                    layer.update_clipping_scalar(layer.weight, params['maxw_algo'], params['maxw_quantscale'])

        writer.add_scalar('Loss/train', np.mean(train_loss), epoch)
        writer.add_scalar('Loss/test', np.mean(test_loss), epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)

    writer.close()
    return student

def main(teacher_params, student_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data, test_data = load_data()
    train_loader = DataLoader(train_data, batch_size=student_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=student_params['batch_size'], shuffle=False)

    teacher_base = load_model(teacher_params['model'], teacher_params).to(device)

    teacher_run_name = create_run_name(teacher_params)

    teacher_base.load_state_dict(torch.load(f"modeldata/{teacher_run_name}.pth"))
    teacher = FeatureExtractor(teacher_base).to(device)
    print(teacher)
    # teacher.load_state_dict(torch.load(f"modeldata/{teacher_run_name}.pth"))
    teacher.eval()

    # student = load_model(student_params['model'], student_params).to(device)
    # student = train_distill(teacher, student, device, train_loader, test_loader, student_params)

    student_base = load_model(student_params['model'], student_params).to(device)
    student = StudentModel(student_base, list(teacher.children())[-1]).to(device)

    print(student)
    student = train_distill(teacher, student, device, train_loader, test_loader, student_params)

    run_name = create_run_name(student_params)
    torch.save(student.state_dict(), f'modeldata/distilled_{run_name}.pth')

    print("Distillation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distillation script')
    parser.add_argument('--teacher_params', type=str, default='trainingparameters.yaml', help='Name of the teacher parameter file')
    parser.add_argument('--student_params', type=str, default='training-student.yaml', help='Name of the student parameter file')
    args = parser.parse_args()

    with open(args.teacher_params, 'r') as f:
        teacher_params = yaml.safe_load(f)
    
    with open(args.student_params, 'r') as f:
        student_params = yaml.safe_load(f)

    main(teacher_params, student_params)