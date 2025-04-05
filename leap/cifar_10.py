import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pywt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import timm
from einops import rearrange

class WaveletTransformer:
    def __init__(self, wavelet='db1', level=2):
        self.wavelet = wavelet
        self.level = level
    
    def forward(self, img):
        # Convert to numpy for pywt
        img_np = img.numpy()
        coeffs = []
        ordered_coeffs = []
        
        # Process each channel
        for c in range(img_np.shape[0]):
            # Apply wavelet transform
            coeff = pywt.wavedec2(img_np[c], self.wavelet, level=self.level)
            
            # Order coefficients by level (from low to high frequency)
            ordered_coeffs.append(coeff[0].ravel())  # Approximation coefficients
            
            # Add detail coefficients in order
            for level_idx in range(1, len(coeff)):
                for detail_idx in range(3):  # H, V, D details
                    ordered_coeffs.append(coeff[level_idx][detail_idx].ravel())
        
        # Flatten and convert back to tensor
        return torch.from_numpy(np.concatenate(ordered_coeffs)).float()
    
    def get_coefficient_shape(self, img):
        """Get the shape of coefficients at each level"""
        img_np = img.numpy()
        coeff = pywt.wavedec2(img_np[0], self.wavelet, level=self.level)
        shapes = [coeff[0].shape]  # Approximation coefficients
        for level_idx in range(1, len(coeff)):
            shapes.append(coeff[level_idx][0].shape)  # H details
        return shapes

class WaveletProjectedDataset(Dataset):
    def __init__(self, dataset, k, wavelet='db1', level=2):
        self.dataset = dataset
        self.k = k
        self.transformer = WaveletTransformer(wavelet, level)
        
        # Precompute coefficient order for the dataset
        if hasattr(dataset, 'data'):
            sample_img = torch.tensor(dataset.data[0]).permute(2, 0, 1).float() / 255.0
            if len(sample_img.shape) == 2:
                sample_img = sample_img.unsqueeze(0)
        else:
            sample_img = dataset[0][0]
        
        self.coeff_size = self.transformer.forward(sample_img).shape[0]
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Transform to wavelet domain
        coeffs = self.transformer.forward(img)
        
        # Keep only top k coefficients (by magnitude)
        if self.k < self.coeff_size:
            _, indices = torch.topk(torch.abs(coeffs), self.k)
            mask = torch.zeros_like(coeffs)
            mask[indices] = 1.0
            coeffs = coeffs * mask
        
        return coeffs, label
    
    def __len__(self):
        return len(self.dataset)

def create_wavelet_datasets(base_dataset, k_values, wavelet='db4', level=2):
    datasets = {}
    for k in k_values:
        datasets[k] = WaveletProjectedDataset(base_dataset, k, wavelet, level)
    return datasets

class WaveletResNet(nn.Module):
    def __init__(self, coeff_size, num_classes=10):
        super().__init__()
        
        # Use a proper ResNet model from timm with modifications for our input
        self.resnet = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
        
        # Create a 1D to 2D projection layer
        # Square side length for the projected image
        self.side_len = 32  # CIFAR-10 size
        self.projection_dim = 3 * self.side_len * self.side_len  # 3 channels x 32 x 32
        
        # Linear projection from wavelet coefficients to 2D image-like representation
        self.projection = nn.Linear(coeff_size, self.projection_dim)
    
    def forward(self, x):
        # Project from wavelet space to image space
        x = self.projection(x)
        
        # Reshape to match CIFAR-10 dimensions
        x = x.view(-1, 3, self.side_len, self.side_len)
        
        # Pass through the ResNet
        return self.resnet(x)

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(model, train_loader, val_loader, epochs, lr=1e-3, weight_decay=1e-4,
               device=torch.device('cuda'), log_interval=50, experiment_name='wavelet_resnet'):
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = GradScaler()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Set to training mode
        model.train()
        
        # Metrics
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        end = time.time()
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Track metrics
                acc = (outputs.argmax(dim=1) == targets).float().mean()
                
                # Mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                batch_time.update(time.time() - end)
                end = time.time()
                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc.item(), inputs.size(0))
                
                if i % log_interval == 0:
                    tepoch.set_postfix(loss=losses.avg, acc=accuracies.avg, 
                                     lr=optimizer.param_groups[0]['lr'])
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{experiment_name}_best.pth")
        
        # Update learning rate
        scheduler.step()
    
    return best_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc = (outputs.argmax(dim=1) == targets).float().mean()
            
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc.item(), inputs.size(0))
    
    return losses.avg, accuracies.avg

def measure_empirical_leap(datasets, num_classes=10, epsilon=0.02, epochs=20, batch_size=128, 
                           lr=1e-3, weight_decay=1e-4):
    """
    Measure empirical leap complexity by training models on datasets with different k values
    and identifying accuracy jumps
    """
    accuracies = {}
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train models for each k value
    for k, dataset in datasets.items():
        print(f"Training model for k={k}")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
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
        
        # Create model
        sample_input = dataset[0][0]
        coeff_size = sample_input.shape[0]
        
        model = WaveletResNet(coeff_size=coeff_size, num_classes=num_classes)
        model = model.to(device)
        
        # Train model
        best_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            experiment_name=f"wavelet_resnet_k{k}"
        )
        
        accuracies[k] = best_acc
        print(f"Completed training for k={k}, best accuracy: {best_acc:.4f}")
    
    # Calculate leaps
    k_values = sorted(accuracies.keys())
    jumps = {}
    
    for i in range(1, len(k_values)):
        k_prev = k_values[i-1]
        k = k_values[i]
        
        acc_jump = accuracies[k] - accuracies[k_prev]
        jumps[k] = acc_jump
        
        print(f"k={k}: accuracy={accuracies[k]:.4f}, jump={acc_jump:.4f}")
    
    # Identify significant jumps (where accuracy increases by more than epsilon)
    significant_jumps = [(k, jump) for k, jump in jumps.items() if jump > epsilon]
    significant_jumps.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate empirical leap
    if significant_jumps:
        # Leap is the largest k that corresponds to a significant jump
        empirical_leap = max(k for k, _ in significant_jumps)
        print(f"Empirical leap = {empirical_leap}")
        return empirical_leap, accuracies, jumps
    else:
        print("No significant jumps found")
        return None, accuracies, jumps

def estimate_sample_complexity(leap, d, confidence=0.95):
    power = max(leap-1, 1)
    log_factor = np.log(d)
    conf_factor = np.log(1/(1-confidence))
    leading_constant = 1.0
    return leading_constant * (d**power) * log_factor * conf_factor

def main():
    # Set easily configurable hyperparameters here
    # =============================================
    
    # Data and wavelet parameters
    wavelet_type = 'db4'        # Wavelet type ('db4' is good for images)
    level = 2                   # Suitable for 32x32 images
    k_values = [1,2,5,10,30,50, 100, 200, 400, 800, 1600, 3200]  # Coefficients to keep
    
    # Training parameters
    batch_size = 1024            # Batch size
    epochs = 50                 # Number of epochs
    lr = 0.0005                  # Learning rate
    weight_decay = 0.0001       # Weight decay for regularization
    
    # Leap measurement parameters
    epsilon = 0.02              # Threshold for significant accuracy jumps
    
    # =============================================
    # No need to modify below this line
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    print(f"Starting wavelet leap analysis")
    print(f"Using wavelet: {wavelet_type}, level: {level}")
    print(f"Testing k values: {k_values}")
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading CIFAR-10 dataset...")
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Create wavelet-projected datasets
    print("Creating wavelet-projected datasets...")
    wavelet_datasets = create_wavelet_datasets(cifar10_train, k_values, wavelet_type, level)
    
    # Measure leap complexity
    print("Measuring empirical leap complexity...")
    leap, accuracies, jumps = measure_empirical_leap(
        datasets=wavelet_datasets,
        num_classes=10,
        epsilon=epsilon,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(list(accuracies.keys()), list(accuracies.values()), marker='o')
    plt.xscale('log')
    plt.xlabel('Number of wavelet coefficients (k)')
    plt.ylabel('Validation accuracy')
    plt.title('Accuracy vs. Wavelet Coefficients')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    ks = list(jumps.keys())
    js = list(jumps.values())
    plt.bar(ks, js)
    plt.xscale('log')
    plt.xlabel('Number of wavelet coefficients (k)')
    plt.ylabel('Accuracy jump')
    plt.title('Accuracy Jumps vs. Wavelet Coefficients')
    plt.axhline(y=epsilon, color='r', linestyle='--', label=f'Threshold (ε={epsilon})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('leap_complexity_results.png')
    
    # Save results
    np.savez('leap_results.npz', 
            leap=leap, 
            accuracies=accuracies, 
            jumps=jumps, 
            k_values=k_values)
    
    # Estimate sample complexity
    if leap is not None:
        d = 32 * 32 * 3  # Dimension of CIFAR-10 images
        sample_complexity = estimate_sample_complexity(leap, d)
        print(f"Empirical leap complexity: {leap}")
        print(f"Estimated sample complexity: {sample_complexity:.2f}")
        print(f"This suggests you need approximately {int(sample_complexity)} samples for effective learning")
    
    print("Analysis complete. See 'leap_complexity_results.png' for visualizations")

if __name__ == '__main__':
    main()