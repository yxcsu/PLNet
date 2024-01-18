import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import crop_image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import torchvision.models as models

# set parameters

def prediction(model_dir,test_dir,save_dir):   
    model = models.efficientnet_b2(weights= models.EfficientNet_B2_Weights.DEFAULT)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1408, 2)
    )
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5762883,0.45526023,0.32699665], std=[0.08670782,0.09286641,0.09925108])
    ])


    byz = 'Corypha'
    tz = 'Borassus'

    # Traverse all images in the folder
    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        for filename in tqdm(os.listdir(folder_path),ncols=80):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG') or filename.endswith('.bmp'):

                img_path = os.path.join(folder_path, filename)
                # with open(img_path,"rb" ) as f:
                #     t = f.read()
                # result_a = remove(data=t)
                result_Image = Image.open(img_path).convert('RGB')
                # result_Image = Image.open(io.BytesIO(result_a)).convert('RGB')
                # img = Image.open(img_path).convert('RGB')
                # img = cv2.imread(img_path)
                # h, w, _ = img.size
                # blocks = crop_image(result_Image,img)
                blocks = crop_image(result_Image)
                results = []
                outputbyz = []
                outputtz = []
                batch_size = min(64, len(blocks))  # batch size不超过64
                for i in range(0, len(blocks), batch_size):
                    batch = blocks[i:i + batch_size]
                    batch = torch.stack([preprocess(block) for block in batch]).to(device)
                    with torch.no_grad():
                        outputs = model(batch)
                        outputbyz.extend(outputs.cpu().numpy()[:,0])
                        outputtz.extend(outputs.cpu().numpy()[:,1])
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        results.extend(preds)
                count_0 = np.sum(np.array(results) == 0)
                count_1 = np.sum(np.array(results) == 1)
    
                df = pd.DataFrame({
                'filename': [img_path],
                'byz_all': [count_0],
                'tz_all': [count_1],  
                'outputbyz':[sum(outputbyz)],
                'outputtz':[sum(outputtz)],
                'result_output':[byz if sum(outputbyz)>sum(outputtz) else tz]
                   })              
                if not os.path.exists(save_dir):
                    df.to_csv(save_dir, index=False)
                else:
                    df.to_csv(save_dir, mode='a', header=False, index=False)
                    

if __name__ == '__main__':
    test_dir = "PLM images path"
    model_dir = "PLNet model path"
    save_dir = "result save path"
    prediction(model_dir,test_dir,save_dir)
