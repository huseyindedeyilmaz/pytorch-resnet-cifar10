import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

# Mevcut ResNet modellerini listeler
model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

# Argümanları belirle
parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR10 in PyTorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


def train_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    criterion = criterion.cuda() 

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            train_loss = criterion(outputs, targets)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            if i % 10 == 9:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Training Loss: {running_loss / (i + 1)}, Training Accuracy: {(correct_train / total_train) * 100:.2f}%")
        
        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_accuracy = (correct_train / total_train) * 100
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)

        model.eval()
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = model(inputs)
                val_loss = criterion(outputs, targets)

                val_running_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

                if i % 10 == 9:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}, Validation Loss: {val_running_loss / (i + 1)}, Validation Accuracy: {(correct_val / total_val) * 100:.2f}%")

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracy = (correct_val / total_val) * 100
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f"Epoch {epoch + 1} Summary: Training Loss: {train_epoch_loss:.4f}, Training Accuracy: {train_epoch_accuracy:.2f}%, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%")
    
    return train_losses, train_accuracies, val_losses, val_accuracies


def main():
    args = parser.parse_args()

    # Veri hazırlama: CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Modeli yükle
    model = resnet.__dict__[args.arch]()
    model = model.cuda()

    # Kriter ve optimizer tanımlama
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Eğitim ve değerlendirme işlemini başlat
    train_evaluate(model, train_loader, val_loader, criterion, optimizer, args.epochs)


if __name__ == "__main__":
    main()
