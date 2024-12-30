import argparse
import os

import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from collections import OrderedDict


def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type = str, default = "flowers", 
                        help = "path to the folder of images")

    parser.add_argument("--save_dir", type = str, default = "checkpoints", 
                        help = "path to save the model checkpoints. Default: checkpoints")

    parser.add_argument('--arch', type = str, default = 'densenet', 
                    help = 'CNN Model Architecture (densenet or vgg). Default: densenet') 
    
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                    help = 'Learning rate. Default: 0.001')
    
    parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'Hidden units. Default: 512') 
    
    parser.add_argument('--epochs', type = int, default = 3, 
                    help = 'Number of epochs. Default: 3')
    
    parser.add_argument('--gpu', action = 'store_true', 
                    help = 'Use GPU for training. Default: False')
    
    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        "train": transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),

        "test": transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),

        "valid": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"])
    }

    dataloaders = {
        "train":  torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True),
        "valid":  torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64)
    }

    return image_datasets, dataloaders

from collections import OrderedDict

# Define create model function to use later
def create_model(arch, out_param, hidden_unit: np.number, learning_rate = 0.001, drop_out = 0.2, device = None):
    if (arch == "densenet"):
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        input_size = model.classifier.in_features
    elif (arch == "vgg"):
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        input_size = model.classifier[0].in_features
    else:
        raise TypeError("The arch specified is not supported")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
      param.requires_grad = False

    hidden_layers = [input_size, hidden_unit]
    layers = []
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

    for i, (h1, h2) in enumerate(layer_sizes):
        layers.append((f"fc{i}", nn.Linear(h1, h2)))
        layers.append((f"rl{i}", nn.ReLU()))
        layers.append((f"dr{i}", nn.Dropout(drop_out)))

    layers.append((f"fc", nn.Linear(hidden_layers[-1], out_param)))
    layers.append((f"out", nn.LogSoftmax(dim=1)))

    classifier = nn.Sequential(OrderedDict(layers))
    model.classifier = classifier

    if device:
        model = model.to(device)

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train(model, dataloaders, criterion, optimizer, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 5

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()

        for inputs, labels in dataloaders["train"]:
            steps += 1

            optimizer.zero_grad()

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    validloader = dataloaders["valid"]

                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_loss = running_loss/print_every
                test_loss = test_loss/len(validloader)

                # At completion of epoch
                train_losses.append(train_loss)
                test_losses.append(test_loss)

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {train_loss:.3f}.. "
                    f"Test loss: {test_loss:.3f}.. "
                    f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    else:
        print("Training completed successfully")

def validate(model, dataloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(dataloader):.3f}")

    return test_loss, accuracy

def save_model(save_dir, class_to_idx, hidden_unit, epochs, model, optimizer, criterion):
    # TODO: Save the checkpoint
    checkpoint = {'class_to_idx': class_to_idx,
                'arch': model.__class__.__name__.lower(),
                'out_param': 102,
                "hidden_unit": hidden_unit,
                "epoch": epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict()
                }
    
    store_path = f"./{save_dir}"

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    torch.save(checkpoint, f"./{save_dir}/{model.__class__.__name__.lower()}_checkpoint.pth")

def main():
    args = get_input_args()

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("GPU is not available, using CPU for training")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    _, dataloaders = load_data(args.data_dir)

    model, criterion, optimizer = create_model(args.arch, 102, args.hidden_units, 
                                               learning_rate=args.learning_rate,
                                               device=device)

    # Train model
    print("Training model")
    print(f"\tDevice: {device}")
    print(f"\tEpochs: {args.epochs}")
    print(f"\tLearning rate: {args.learning_rate}")
    print(f"\tHidden units: {args.hidden_units}")
    print(f"\tArch: {args.arch}")

    # TODO: enable training
    train(model, dataloaders, criterion, optimizer, args.epochs, device)

    print("Validating model")
    validate(model, dataloaders["test"], criterion, device)

    print("Saving model to: ", args.save_dir)
    save_model(args.save_dir, dataloaders["train"].dataset.class_to_idx, 
               args.hidden_units, 
               args.epochs, 
               model, optimizer, criterion)
    
if __name__ == "__main__":
    main()