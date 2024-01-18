'''
Tips:
========================================================================================
* The main code for PLNet training, validating and testing
* Data resources can be obtained from publicly available Bayesian image databases.
* Parameters in the following code should be adjusted to adapt to in your own case.
* It is worth mentioning that the value of batch_size depends on memeory size, if you train 
  the network on CPU (Actually we don't suggest you to train in this way due to a significant
  improvement brought by training on GPU), it may cost plenty of time.
========================================================================================
'''
'''
The folder structure of the training set validation set and test set should be adjusted as follows:
data_path/
    train/
        Coraphy/
        Borassus/
    val/
        Coraphy/
        Borassus/
    test/
        Coraphy/
        Borassus/
'''
import os
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.models as models
from Preprocessing import crop_image
from Optimzer import Lion
from Dataset import LeafDataset
from Data_Augment import data_augment

# #set parameters
def main():
    # load data
    train_data_path = os.path.join(data_path, "train")
    val_data_path = os.path.join(data_path, "val")   
    train_transform, test_transform = data_augment()
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5762883,0.45526023,0.32699665], std=[0.08670782,0.09286641,0.09925108])
    ])
    train_dataset = LeafDataset(train_data_path, transform=train_transform)
    val_dataset = LeafDataset(val_data_path, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,num_workers=8,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False,num_workers=8,pin_memory=True)
    
    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b2(weights= models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1408, 2)
    )
    model = model.to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    # train and val
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
    
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, total=len(val_loader)):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
                      

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        with open(train_list, 'a', newline='') as file:
                writer = csv.writer(file)
                # if first epoch, write the header
                if epoch == 0:
                    writer.writerow(['train_loss', 'val_loss', 'train_acc', 'val_acc'])
                # write the data
                writer.writerow([train_loss, val_loss, train_accuracy, val_accuracy])
        # check if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered")
                break
        # update learning rate
        scheduler.step()

    # Create some mock data  
    model.eval()
    #save model
    torch.save(model.state_dict(), model_path)
    
    data1 = train_losses
    data2 = val_losses
    data3 = train_accuracies
    data4 = val_accuracies
    fig, ax = plt.subplots()
    ax1= ax.twinx()
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    p1,=ax.plot(data1, '#7DAEE0',label="Train Loss")
    p2,=ax.plot(data2, '#E9C46A',label="Val Loss")    
    ax1.set_ylabel('Accuracy(%)')  
    p3,=ax1.plot(data3, '#299D8F',label="Train Acc")
    p4,=ax1.plot(data4, '#EA8379',label="Val Acc")
    plt.legend(handles=[p1,p2,p3,p4],loc='center right')
    plt.show()
    

if __name__ == '__main__':
    data_path = 'path for training and validation sets'
    model_path = 'path for saving the model'
    train_list = 'path for saving the loss and accuracy of training and validation sets'

    learning_rate = 0.00001
    weight_decay = 0.1
    num_epochs = 100
    batchsize = 64
    patience = 5
    main()
