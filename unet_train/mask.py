# 导入必要的库
from normal_m import UNet
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from PIL import Image, ImageFilter

def angular_loss(N_pred, N_label):
    N_pred = F.normalize(N_pred, p=2, dim=1)
    N_label = F.normalize(N_label, p=2, dim=1)
    cos_theta = torch.sum(N_pred * N_label, dim=1, keepdim=True)  # [batch_size, 1, H, W]
    cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)
    loss = torch.acos(cos_theta)  # [batch_size, 1, H, W]
    return loss 

def l1_loss(N_pred, N_label):
    loss = torch.abs(N_pred - N_label)
    return loss 

class AugmentedDepthDataset(Dataset):
    def __init__(self, image_paths, depth_paths, transform=None):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.transform = transform
        self.mask_paths = [p.replace('depth', 'mask') for p in depth_paths]

    def __len__(self):
        return len(self.depth_paths) * 4
    
    def __getitem__(self, idx):
        original_idx = idx // 4
        rotation_angle = random.choice([0, 90, 180, 270])
        images = []
        for img_path in self.image_paths[original_idx]:
            img = Image.open(img_path).convert('L')
            img = img.rotate(rotation_angle)
            img = img.resize((1024, 1024))
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = np.array(img).astype(np.float32) 
            images.append(torch.from_numpy(img))
        input_tensor = torch.stack(images, dim=0)

        depth = np.load(self.depth_paths[original_idx]).astype(np.float32)
        mask = np.load(self.mask_paths[original_idx]).astype(np.float32)
        
        depth_img = Image.fromarray(depth)
        mask_img = Image.fromarray(mask)
        depth_img = depth_img.rotate(rotation_angle)
        mask_img = mask_img.rotate(rotation_angle)
        depth_img = depth_img.resize((1024, 1024))
        mask_img = mask_img.resize((1024, 1024))
        
        depth = np.array(depth_img)
        mask = np.array(mask_img)
        
        depth_tensor = torch.from_numpy(depth)
        dzdx = torch.gradient(depth_tensor, spacing=1, dim=1)[0]
        dzdy = torch.gradient(depth_tensor, spacing=1, dim=0)[0]
        normal_x = -dzdx
        normal_y = -dzdy
        normal_z = torch.ones_like(depth_tensor)
        normal = torch.stack([normal_x, normal_y, normal_z], dim=0)
        normal = F.normalize(normal, p=2, dim=0)
        
        mask = torch.from_numpy(mask)
        
        i, j, h, w = transforms.RandomCrop.get_params(
            Image.fromarray(depth), output_size=(512, 512))
        input_tensor = transforms.functional.crop(input_tensor, i, j, h, w)
        normal = transforms.functional.crop(normal, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        return input_tensor, normal, mask

def get_data_loaders(dataset_dir, batch_size=24, num_workers=4):
    image_dir = os.path.join(dataset_dir, 'image')
    depth_dir = os.path.join(dataset_dir, 'depth')
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    group_images = defaultdict(list)

    for img_file in image_files:
        filename = os.path.basename(img_file)
        group_num, angle = filename.split('.')[0].split('-')
        group_images[group_num].append(img_file)

    valid_image_paths = []
    valid_depth_paths = []
    count = 0
    for group_num, img_list in group_images.items():
        if len(img_list) == 4:
            depth_file = os.path.join(depth_dir, f'{group_num}.npy')
            if os.path.exists(depth_file):
                img_list.sort(key=lambda x: int(
                    os.path.basename(x).split('-')[1].split('.')[0]))
                valid_image_paths.append(img_list)
                valid_depth_paths.append(depth_file)
                count += 1
            else:
                print(f"{group_num} depth image does not exist.")
        else:
            print(f"{group_num} less than 4 images.")
    print(f"Find {count} datasets in total.")
    dataset = AugmentedDepthDataset(
        valid_image_paths, valid_depth_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    return dataloader

np.random.seed(0)
dataset_dir = 'dataset_transformed' 
train_loader = get_data_loaders(dataset_dir, batch_size=12)

model = UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.L1Loss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 150000
losses = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, normals,masks) in enumerate(train_loader):
        inputs = inputs.to(device).float()    # [batch_size, 4, 512, 512]
        normals = normals.to(device).float()  # [batch_size, 3, 512, 512]
        masks = masks.to(device)              # [batch_size, 512, 512]
        optimizer.zero_grad()
        
        
        N_predicted = model(inputs)           # [batch_size, 3, 512, 512]

        # per_pixel_loss = angular_loss(N_predicted, normals)  # [batch_size, 1, H, W]
        per_pixel_loss = l1_loss(N_predicted, normals)  # [batch_size, 3, H, W]
        masks_expanded = masks.unsqueeze(1).expand(-1, 3, -1, -1)  # [batch_size, 3, H, W]
        masked_loss = per_pixel_loss * masks_expanded
        
        
        
        with torch.no_grad():
            pixel_errors = masked_loss.detach()
            weights = (pixel_errors ** 2) + 1e-8
            mean_error = torch.sum(weights * masks_expanded) / (torch.sum(masks_expanded) + 1e-8)
            weights = weights / (mean_error + 1e-8)

        weighted_loss = masked_loss * weights
        final_loss = torch.sum(weighted_loss) / (torch.sum(masks_expanded) + 1e-8)

        final_loss.backward()
        optimizer.step()

        running_loss += final_loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    if (epoch+1) % 10 == 0:
        with open('loss.txt', 'a', encoding='utf-8') as file:
            file.write(
                f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}\n')

    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), f'modelmask_median/unet_epoch_{epoch+1}.pth')

plt.figure()
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('training_loss_plot.png')
plt.show()

print('Finished Training')
