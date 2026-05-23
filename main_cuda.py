from operator import index
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    class Datasets(torch.utils.data.Dataset):
        def __init__(self, path_dir1:str, path_dir2:str):
            super().__init__()
            self.path_dir1 = path_dir1
            self.path_dir2 = path_dir2
            self.dir1_list = sorted([f for f in os.listdir(path_dir1)
                                    if os.path.splitext(f)[1].lower() in VALID_EXTS])
            self.dir2_list = sorted([f for f in os.listdir(path_dir2)
                                    if os.path.splitext(f)[1].lower() in VALID_EXTS])
            
        
        def __len__(self):
            return len(self.dir1_list) + len(self.dir2_list)


        def __getitem__(self, index):
            if index < len(self.dir1_list):
                class_id = 0
                img_path = os.path.join(self.path_dir1, self.dir1_list[index])
            else:
                class_id = 1
                new_index = index - len(self.dir1_list)
                img_path = os.path.join(self.path_dir2, self.dir2_list[new_index])

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            img = img.transpose((2, 0, 1))
            
            t_img = torch.from_numpy(img)
            
            t_class_id = torch.tensor(class_id, dtype=torch.float32).view(1)
            return {'img': t_img, 'class_id': t_class_id}
        

    train_dogs = 'C:/Users/kaflanat/Desktop/ai/new/cat-and-dog/versions/1/training_set/training_set/dogs/'
    train_cats = 'C:/Users/kaflanat/Desktop/ai/new/cat-and-dog/versions/1/training_set/training_set/cats/'
    test_dogs = 'C:/Users/kaflanat/Desktop/ai/new/cat-and-dog/versions/1/test_set/test_set/dogs/'
    test_cats = 'C:/Users/kaflanat/Desktop/ai/new/cat-and-dog/versions/1/test_set/test_set/cats/'


    train_ds_catdog = Datasets(train_dogs, train_cats)
    test_ds_catdog = Datasets(test_dogs, test_cats)

    len(train_ds_catdog)
    len(test_ds_catdog)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        train_ds_catdog,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds_catdog,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    class VGG16(nn.Module):
        def __init__(self, out_nc=1):
            super().__init__()
            
            self.act = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(2,2)
            
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            
            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            
            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            
            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  
                            
            self.flat = nn.Flatten()
            
            self.fl1 = nn.Linear(512*4*4, 256)
            self.dropout = nn.Dropout(0.5)
            self.fl2 = nn.Linear(256, 1)
            
            
            
        def forward(self, x):
            out = self.conv1_1(x)        
            out = self.act(out)
            out = self.conv1_2(out)
            out = self.act(out)
         
            out = self.maxpool(out)
            
            out = self.conv2_1(out)        
            out = self.act(out)
            out = self.conv2_2(out)
            out = self.act(out)

            out = self.maxpool(out)
            
            out = self.conv3_1(out)        
            out = self.act(out)
            out = self.conv3_2(out)
            out = self.act(out)
            out = self.conv3_3(out)
            out = self.act(out)
            
            out = self.maxpool(out)
        
            out = self.conv4_1(out)        
            out = self.act(out)
            out = self.conv4_2(out)
            out = self.act(out)
            out = self.conv4_3(out)
            out = self.act(out)
            
            out = self.maxpool(out)

            
            out = self.maxpool(out)

            out = self.flat(out)
            
            out = self.fl1(out)
            out = self.act(out)

            out = self.dropout(out)

            out = self.fl2(out)


            
            return out

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device("cuda")

    model = VGG16(1).to(device)

    for sample in train_loader:
        img = sample['img'].to(device)
        label = sample['class_id'].to(device)

        pred = model(img)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))

    checkpoint_path = "checkpoint.xml"

    start_epoch = 0

    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state'])

        optimizer.load_state_dict(checkpoint['optimizer_state'])

        start_epoch = checkpoint['epoch'] + 1

        print(f"Продолжаем с эпохи {start_epoch}")

    else:
        print("Новая модель")
        
    def accuracy(pred, label):

        preds = (torch.sigmoid(pred) > 0.5).float()

        answer = preds == label

        return answer.float().mean()


    epochs = 10

    for epoch in range(start_epoch, epochs):

        model.train()

        loss_val = 0
        acc_val = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}"
        )

        for sample in pbar:
            img = sample['img'].to(device)
            label = sample['class_id'].to(device)

            optimizer.zero_grad()

            pred = model(img)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            acc_val += accuracy(pred, label).item()

            pbar.set_postfix({
                "loss": loss.item(),
                "acc": accuracy(pred, label).item()
            })

        print(
            f"Epoch {epoch +1}: "
            f"Loss = {loss_val/len(train_loader):.4f}, "
            f"Accuracy = {acc_val/len(train_loader):.4f}"
        )

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss_val / len(train_loader),
            'accuracy': acc_val / len(train_loader)
        }, checkpoint_path)

        print("Checkpoint сохранён")

    object1 = ("dog")
    object2 = ("cat")
    
    classes = [object1, object2]

    def predict_folder(folder_path):

        model.eval()

        files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]

        if len(files) == 0:
            print("В папке нет изображений")
            return

        for file in files:

            path = os.path.join(folder_path, file)

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = img.transpose((2, 0, 1))

            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(img_tensor)
                p = torch.sigmoid(pred).item()

            print(f"\nФайл: {file}")

            print(f"cat: {p*100:.2f}%")
            print(f"dog: {(1-p)*100:.2f}%")

            print("Это:", "cat" if p > 0.5 else "dog")

    predict_folder(test_dogs)
    # predict_folder(test_cats)
   

if __name__ == "__main__":
    main()
