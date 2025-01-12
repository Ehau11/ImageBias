import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import vgg19
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import random
import pandas as pd
import numpy as np

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        vgg = vgg19(weights='DEFAULT').features.eval()
        self.layers = layers
        self.slices = self._get_layer_slices(vgg, layers)
        for param in self.parameters():
            param.requires_grad = False

    def _get_layer_slices(self, vgg, layers):
        layer_indices = [int(i) + 1 for i in layers.values()]
        slices = []
        start = 0
        for end in layer_indices:
            slices.append(nn.Sequential(*list(vgg.children())[start:end]))
            start = end
        return nn.ModuleList(slices)

    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.slices):
            x = layer(x)
            features[list(self.layers.keys())[i]] = x
        return features

class MultiLayerStyleTransfer:
    def __init__(self, device='cuda', transfer_type='texture'):
        self.device = device
        self.layer_configs = {
            'color': {'layers': {'conv1_1': '0'}, 
                     'content_weight': 1.0, 'style_weight': 1e6},
            'texture': {'layers': {'conv1_1': '0', 'conv2_1': '5', 
                                 'conv2_2': '7', 'conv3_1': '10', 
                                 'conv3_4': '16'}, 
                       'content_weight': 1.0, 'style_weight': 1e6},
            'shape': {'layers': {'conv3_4': '16', 'conv4_1': '19', 
                               'conv4_2': '21', 'conv5_1': '28'}, 
                     'content_weight': 1.0, 'style_weight': 1e4}
        }
        config = self.layer_configs[transfer_type]
        self.layers = config['layers']
        self.content_weight = config['content_weight']
        self.style_weight = config['style_weight']
        self.feature_extractor = VGGFeatureExtractor(self.layers).to(device)

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram / (b * c * h * w)

    def content_loss(self, features, content_features):
        loss = sum(F.mse_loss(features[layer], content_features[layer])
                  for layer in self.layers)
        return loss

    def style_loss(self, features, style_features):
        loss = 0
        for layer in self.layers:
            gram = self.gram_matrix(features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += F.mse_loss(gram, style_gram)
        return loss

    def transfer(self, content_img, style_img, num_steps=300):
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)
        
        output_img = content_img.clone().requires_grad_(True)
        content_features = self.feature_extractor(content_img)
        style_features = self.feature_extractor(style_img)
        
        optimizer = optim.LBFGS([output_img])
        
        step = 0
        while step < num_steps:
            def closure():
                optimizer.zero_grad()
                features = self.feature_extractor(output_img)
                c_loss = self.content_weight * self.content_loss(features, content_features)
                s_loss = self.style_weight * self.style_loss(features, style_features)
                total_loss = c_loss + s_loss
                total_loss.backward()
                return total_loss
            
            optimizer.step(closure)
            step += 1
        
        return output_img

class DiverseImageNetDataset:
    def __init__(self, root_dir, num_images=5000):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        self.dataset = datasets.ImageFolder(root_dir, transform=self.transform)
        self.class_indices = self._create_class_indices()
        # We need 5000 pairs, so we select 10000 images
        self.selected_indices = self._select_diverse_images(num_images * 2)
        
    def _create_class_indices(self):
        class_indices = defaultdict(list)
        for idx, (_, class_id) in enumerate(self.dataset):
            class_indices[class_id].append(idx)
        return class_indices
        
    def _select_diverse_images(self, num_images):
        classes = list(self.class_indices.keys())
        images_per_class = num_images // len(classes)
        selected_indices = []
        
        for class_id in classes:
            available = self.class_indices[class_id]
            num_select = min(images_per_class, len(available))
            selected = random.sample(available, num_select)
            selected_indices.extend(selected)
        
        random.shuffle(selected_indices)
        return selected_indices[:num_images]

def generate_cue_conflict_datasets(imagenet_dir, output_dir, num_images=5000):
    """
    Generate three cue conflict datasets with exactly num_images images each.
    Each dataset will have its own CSV file with labels.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dataset with twice the number of images (for pairs)
    dataset = DiverseImageNetDataset(imagenet_dir, num_images)
    
    # Initialize transfer models
    transfer_models = {
        'color': MultiLayerStyleTransfer(device=device, transfer_type='color'),
        'texture': MultiLayerStyleTransfer(device=device, transfer_type='texture'),
        'shape': MultiLayerStyleTransfer(device=device, transfer_type='shape')
    }
    
    # Define cue conflicts
    cue_conflicts = [
        ('color', 'texture'),
        ('color', 'shape'),
        ('shape', 'texture')
    ]
    
    for cue1, cue2 in cue_conflicts:
        print(f"Generating {cue1} vs {cue2} dataset...")
        
        # Create output directory
        conflict_dir = os.path.join(output_dir, f'{cue1}_vs_{cue2}')
        os.makedirs(conflict_dir, exist_ok=True)
        
        # Track used combinations
        csv_data = []
        
        # Process exactly num_images/2 pairs to create num_images total images
        for i in range(0, num_images * 2, 2):
            idx1, idx2 = dataset.selected_indices[i:i+2]
            img1, label1 = dataset.dataset[idx1]
            img2, label2 = dataset.dataset[idx2]
            
            # Generate cue transfers
            cue1_transfer = transfer_models[cue1].transfer(
                img1.unsqueeze(0),
                img2.unsqueeze(0)
            )
            cue2_transfer = transfer_models[cue2].transfer(
                img1.unsqueeze(0),
                img2.unsqueeze(0)
            )
            
            # Save both transfers (creating 2 images from each pair)
            for idx, (transfer, labels) in enumerate([
                (cue1_transfer, (label1, label2)),
                (cue2_transfer, (label2, label1))
            ]):
                class1, class2 = [dataset.dataset.classes[l] for l in labels]
                filename = f"image_{i + idx}.png"
                filepath = os.path.join(conflict_dir, filename)
                
                # Save image
                img_tensor = transfer.cpu().detach()[0]
                img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img_tensor = torch.clamp(img_tensor, 0, 1)
                plt.imsave(filepath, img_tensor.permute(1, 2, 0).numpy())
                
                # Add to CSV data
                csv_data.append({
                    'filename': filename,
                    f'{cue1}_label': class1 if idx == 0 else class2,
                    f'{cue2}_label': class2 if idx == 0 else class1
                })
                
            if len(csv_data) >= num_images:
                break
        
        # Save CSV with exactly num_images entries
        csv_data = csv_data[:num_images]  # Ensure exactly num_images entries
        csv_path = os.path.join(output_dir, f'{cue1}_vs_{cue2}.csv')
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        print(f"Generated {len(csv_data)} images for {cue1} vs {cue2}")

def main():
    imagenet_dir = './imagenet_data'
    output_dir = './cue_conflict_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate exactly 5000 images per conflict type
    generate_cue_conflict_datasets(imagenet_dir, output_dir, num_images=5000)

if __name__ == '__main__':
    main()