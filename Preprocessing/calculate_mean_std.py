# calculate mean and std
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
def calculate_mean_std(train_dir):
    # create a dataset object
    process = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor(),
        ])
    dataset_without_augmentation = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform = process
    )

    # use the torch dataloader to iterate through the dataset
    dataloader = DataLoader(dataset_without_augmentation, batch_size=1, shuffle=False)

    # calculate mean and std
    mean = np.zeros(3)
    std = np.zeros(3)
    for images, _ in tqdm(dataloader):
        images_np = images.numpy()
        mean += images_np.mean(axis=(0, 2, 3))
        std += images_np.std(axis=(0, 2, 3))

    mean /= len(dataloader)
    std /= len(dataloader)

    print("Mean:", mean)
    print("Std:", std)