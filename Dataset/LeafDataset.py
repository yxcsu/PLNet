from torch.utils.data import Dataset
import cv2
import os

#Loading Palm Leaf Manuscript Dataset
class LeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        '''
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = []
        self.class_names = os.listdir(root_dir)
        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.imgs.append((img_path, class_name))

    def __len__(self):
        '''Returns the total number of image files'''
        return len(self.imgs)

    def __getitem__(self, idx):
        '''Returns the image and its class label'''
        img_path, class_name = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, self.class_names.index(class_name)
