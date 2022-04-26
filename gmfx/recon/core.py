import torch

class ReconMixin:
    
    def to_tensor(self, img, channels_first=True, normalize=True, add_batch_dim=True, device='cuda'):

        img = img.copy()
        if channels_first:
            img = img.transpose(2, 0, 1)
        
        if normalize:
            img = img / 255.
        
        img = torch.tensor(img).float()
        
        if add_batch_dim:
            img = img.unsqueeze(0)

        img = img.to(device)
        return img