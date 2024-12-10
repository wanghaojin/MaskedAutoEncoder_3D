import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from model_MAE import MAE  
from vit import ViT
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import calculate_dataset_stats, TIFImageDataset, SubsetWithTransform

os.makedirs('model', exist_ok=True)
os.makedirs('validation_results', exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def train(model, train_loader, optimizer, scaler, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training", unit="batch")
    for i, (inputs, _) in enumerate(pbar):
        inputs = inputs.to(device)
        with autocast():
            loss, _, _ = model(inputs)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        avg_loss = total_loss / (i + 1)
        pbar.set_postfix({"Mean loss": f"{avg_loss:.4f}"})

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    os.makedirs('validation_results', exist_ok=True)
    
    pbar = tqdm(val_loader, desc="Validating", unit="batch")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            
            rng_state = torch.get_rng_state()
            
            with autocast():
                loss, _, _ = model(inputs)
            total_loss += loss.item()
            
            if i < 3:
                original_image = inputs[0]
                masked_image = original_image.clone()
                c, h, w = original_image.shape
                patch_size = model.patch_h
                n_patches = (h // patch_size) * (w // patch_size)
                n_masked = int(model.mask_ratio * n_patches)
                
                torch.set_rng_state(rng_state)
                mask = torch.randperm(n_patches)[:n_masked].to(device)
                mask_full = torch.zeros(c, h, w, device=device) 
                for idx_m in mask:
                    i_patch = (idx_m // (w // patch_size)) * patch_size
                    j_patch = (idx_m % (w // patch_size)) * patch_size
                    masked_image[:, i_patch:i_patch+patch_size, j_patch:j_patch+patch_size] = 0.5
                    mask_full[:, i_patch:i_patch+patch_size, j_patch:j_patch+patch_size] = 1.0

                torch.set_rng_state(rng_state)
                reconstructed = model.predict(inputs)[0]

                combined = original_image * (1 - mask_full) + reconstructed * mask_full

                # 因为无归一化操作，输入原本就在[0,1]
                original_np = original_image.cpu().numpy()
                masked_np = masked_image.cpu().numpy()
                reconstructed_np = reconstructed.cpu().numpy()
                combined_np = combined.cpu().numpy()
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                axes[0].imshow(original_np[0].clip(0, 1), cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(masked_np[0].clip(0, 1), cmap='gray')
                axes[1].set_title('Masked Image')
                axes[1].axis('off')
                
                axes[2].imshow(reconstructed_np[0].clip(0, 1), cmap='gray')
                axes[2].set_title('Reconstruction')
                axes[2].axis('off')
                
                axes[3].imshow(combined_np[0].clip(0, 1), cmap='gray')
                axes[3].set_title('Reconstruction + Visible')
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'validation_results/reconstruction_batch_{i}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"\nBatch {i} Results:")
                print(f"Overall Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / (i + 1)
            pbar.set_postfix({"Mean loss": f"{avg_loss:.4f}"})
    
    return total_loss / len(val_loader)


def main():
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    batch_size = 4
    learning_rate = 5e-5
    num_epochs = 5000
    mask_ratio = 0.75 

    # 用于统计均值和std的transform，不归一，直接ToTensor()将[0,255]变成[0,1]
    transform_for_stats = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('L')),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])

    temp_dataset = TIFImageDataset(
        data_dir='../data',
        transform=transform_for_stats
    )

    temp_loader = DataLoader(
        temp_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False
    )

    # mean, std = calculate_dataset_stats(temp_loader)
    # print(f"Mean: {mean.tolist()}")
    # print(f"Std: {std.tolist()}")

    # 训练时的transform：不使用Normalize，只使用ToTensor()
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('L')),
        transforms.Resize((1024, 1024)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()  # 输出[0,1]范围
    ])

    transform_val = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('L')),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor() # 同样仅[0,1]范围
    ])

    encoder = ViT(
        image_size=(1024, 1024),
        patch_size=(32, 32),
        num_classes=1000,
        dim=1280,
        depth=24,
        num_heads=12,
        mlp_dim=3072,
        dim_per_head=64,
        dropout=0.1,
        pool='cls',
        channels=1  
    )
    
    model = MAE(
        encoder=encoder,
        decoder_dim=512,
        mask_ratio=mask_ratio,
        decoder_depth=8,
        num_decoder_heads=32,
        decoder_dim_per_head=16
    ).to(device)

    full_dataset = TIFImageDataset(
        data_dir='../data',
        transform=None
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = SubsetWithTransform(train_subset, transform=transform_train)
    val_dataset = SubsetWithTransform(val_subset, transform=transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float('inf')
    patience = 500
    min_delta = 0.00001 
    patience_counter = 0
    loss_history = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train(model, train_loader, optimizer, scaler, device)
        val_loss = validate(model, val_loader, device)
        loss_history.append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train loss: {train_loss:.4f}, "
              f"Test loss: {val_loss:.4f}, "
              f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_loss - min_delta:  
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'model/mae_best.pth')
            print("Saved Best Model")
        else:
            patience_counter += 1
            
            if epoch >= 20: 
                if len(loss_history) >= 5:
                    recent_avg = sum(loss_history[-5:]) / 5
                    previous_avg = sum(loss_history[-10:-5]) / 5
                    if patience_counter >= patience and recent_avg >= previous_avg:
                        print("\nEarly stopping triggered:")
                        print(f"No significant improvement for {patience} epochs")
                        print(f"Recent average loss: {recent_avg:.4f}")
                        print(f"Previous average loss: {previous_avg:.4f}")
                        break
            
            if patience_counter >= patience:
                print(f"\nPatience counter: {patience_counter}/{patience}")
                print(f"Current loss delta: {val_loss - best_loss:.6f}")
                print(f"Required improvement: {min_delta:.6f}")

     
        if (epoch + 1) % 100 == 0:
            checkpoint_path = f"model_gray/mae_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'loss_history': loss_history  
            }, checkpoint_path)
            print(f"Saved Checkpoint: {checkpoint_path}")

    print("Training Completed")

if __name__ == '__main__':
    main()