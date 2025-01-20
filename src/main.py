import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from model import ResNet
from trainer import Trainer

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.59685254, 0.59685254, 0.59685254], std=[0.16043035, 0.16043035, 0.16043035])
    ])
    image = preprocess(image).unsqueeze(0)
    return image

def main():
    # Load the model
    model = ResNet()
    model.eval()

    # Load the trained model weights
    checkpoint_path = 'checkpoints/checkpoint_best.ckp'  # Change this to the path of your best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # Load the dataset
    data = pd.read_csv('../data/data.csv', delimiter=';', header=None)
    image_paths = data.iloc[:, 0].values
    labels = data.iloc[:, 1:].values

    # Make predictions
    predictions = []
    for image_path in image_paths:
        image = load_image(os.path.join('data', image_path))
        with torch.no_grad():
            output = model(image)
            prediction = output.cpu().numpy()
            predictions.append(prediction)

    # Display results
    for i, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        print(f"Image: {image_path}, Prediction: {prediction}, True Labels: {labels[i]}")

if __name__ == '__main__':
    main()