# Generally speaking, commonly used models typically visualize the following layers:
# FasterRCNN: model.backbone
# Resnet18 and 50: model.layer4[-1]
# VGG and densenet161: model.features[-1]
# mnasnet1_0: model.layers[-1]
# ViT: model.blocks[-1].norm1
# SwinT: model.layers[-1].blocks[-1].norm1

from sklearn.preprocessing import normalize
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
from torchvision.datasets import ImageFolder
from captum.attr import GuidedGradCam
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
import cv2
import numpy as np
# set the path to the folder containing the images to be predicted
def GuidedGradCAM(model_path, test_data_path, save_path, layer):
    '''
    Generate the interpretability analysis results of the test images using Guided Grad-CAM
    Args:
        model_path (string): The path of the model
        test_data_path (string): The path of the folder containing the images to be predicted
        save_path (string): The path of the folder to save the interpretability analysis results
        layer (string): The layer to visualize
    '''
    class_names = ['Corypha','Borassus']
    # set the device to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model
    model = models.efficientnet_b2(weights= models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1408, 2)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to('cuda')
    # load the test dataset
    data_transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5762883,0.45526023,0.32699665], std=[0.08670782,0.09286641,0.09925108])
    ])

    test_dataset = ImageFolder(test_data_path, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # create the folder to save the interpretability analysis results
    os.makedirs(save_path, exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(save_path, class_name), exist_ok=True)

    # predict the class labels of the test images
    for i,(image, label) in tqdm(enumerate(test_loader)):
        image = image.to(device)
        label = label.to(device)
        guided_gradcam = GuidedGradCam(model,model.features[layer])
        attribution = guided_gradcam.attribute(image, target=label)
        attribution1 = attribution.squeeze(0).cpu().permute(1,2,0).detach().numpy()
        attribution2 = normalize(attribution1.reshape(-1, 3), axis=0).reshape(224, 224, 3)
        attribution3 = (attribution2 * 255).astype(np.uint8)
        attribution4 = cv2.cvtColor(attribution3, cv2.COLOR_BGR2GRAY)
        # Convert tensor to PIL image and save
        image_name = test_dataset.samples[i][0].split('\\')[-1] 
        save_dir = os.path.join(save_path, class_names[label])
        save_path_img = os.path.join(save_dir, f'{image_name}')
        cv2.imwrite(save_path_img,attribution4)
        
    print('The interpretability analysis results have been saved.')

if __name__ == '__main__':
    model_path = 'model path'
    test_data_path = 'PLM images path'
    save_path = 'save path'
    layer = 'layer'
    GuidedGradCAM(model_path, test_data_path, save_path, layer)
