import argparse
import json

import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import functional as F

import train

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_file", type = str, help="path to the image file")

    parser.add_argument("checkpoint", type = str, help="path to the model checkpoint")

    parser.add_argument("--top_k", type = int, default = 5, help="top k classes")

    parser.add_argument("--category_names", type = str, default = "", help="category names")

    parser.add_argument("--gpu", action = 'store_true', help="Use GPU for inference")

    return parser.parse_args()

def load_categories(filename):
    '''
    Load category names from a file
    Return: category names
    '''
    with open(filename, 'r') as f:
        return json.load(f)

def load_checkpoint(filepath, device = torch.device('cpu')):
    '''
    Load a model checkpoint from a file
    Return: model, optimizer, criterion, class_to_idx
    '''
    checkpoint = torch.load(filepath, weights_only=True, map_location=device)

    arch = checkpoint['arch']

    model, criterion, optimizer = train.create_model(arch, checkpoint['out_param'], checkpoint['hidden_unit'])

    model.load_state_dict(checkpoint['model_state_dict'])

    criterion.load_state_dict(checkpoint['criterion_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.to(device)

    return model, optimizer, criterion, checkpoint['class_to_idx']


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        im.thumbnail((256, 256))
        im_size = 224
        left = (im.width - im_size)/2
        upper = (im.height - im_size)/2
        right = (im.width + im_size)/2
        lower = (im.height + im_size)/2

        im = im.crop((left, upper, right, lower))
        np_image = F.to_tensor(im)
        im.close()

    image = np_image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = image.transpose((2, 0, 1))
    np_image = torch.from_numpy(image).float()

    return np_image

def predict(image_path, model, class_to_idx, topk=5, device = torch.device("cpu")):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model = model.eval()

    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    logps = model(image)

    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)

    class_index = {class_to_idx[x]: x for x in class_to_idx}
    new_classes = []

    for index in top_class.cpu().numpy()[0]:
        new_classes.append(class_index[index])

    return top_p.cpu().detach().numpy()[0], new_classes

def print_result(top_p, classes, categories):
    for i in range(len(top_p)):
        class_cat = ""
        if categories:
            class_cat = f"Category: {categories[classes[i]]}"
        else:
            class_cat = f"Class: {classes[i]}"
        
        print(f"{class_cat} - Probability: {top_p[i]:.3f}")
        

def main():
    args = get_input_args()

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("GPU is not available, using CPU for inference")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    categories = None
    if args.category_names:
        categories = load_categories(args.category_names)
    
    model, _, _, class_to_idx = load_checkpoint(args.checkpoint, device)

    topp, classes = predict(args.image_file, model, class_to_idx, args.top_k, device)

    print(f"Top {args.top_k} classes:")
    print_result(topp, classes, categories)

if __name__ == '__main__':
    main()