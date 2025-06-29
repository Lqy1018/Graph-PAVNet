import torch
import torch.nn as nn
import numpy as np

class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Lightweight Transformer
        self.transformer = LightweightTransformer(patch_size=patch_size, embed_dim=embed_dim)

    def extract_patch(self, volume, center):
        """
        Extract multiple plane patches from 3D volume data centered at a given point.
        
        Args:
            volume (numpy.ndarray): 3D volume data of shape [512, 512, 541].
            center (tuple): The center coordinates (x, y, z) in the volume.
        
        Returns:
            torch.Tensor: A tensor of patches from different planes, with shape (9, 1, patch_size, patch_size).
        """
        x, y, z = center
        X, Y, Z = volume.shape  # X=512, Y=512, Z=541
        patches = []

        # Define the extraction range for three planes: Axial (z-axis), Coronal (y-axis), Sagittal (x-axis)
        for axis in [0, 1, 2]:  # 0: axial, 1: coronal, 2: sagittal
            for offset in [-1, 0, 1]:  # -1, 0, 1 offsets to capture surrounding slices
                if axis == 0:  # Axial plane (z-axis)
                    slice_idx = int(min(max(z + offset, 0), Z - 1))
                    plane = volume[:, :, slice_idx]  # Extract [x, y] plane
                elif axis == 1:  # Coronal plane (y-axis)
                    slice_idx = int(min(max(y + offset, 0), Y - 1))
                    plane = volume[:, slice_idx, :]  # Extract [x, z] plane
                else:  # Sagittal plane (x-axis)
                    slice_idx = int(min(max(x + offset, 0), X - 1))
                    plane = volume[slice_idx, :, :]  # Extract [y, z] plane

                # Extract a small patch centered around (x, y) or (x, z) or (y, z)
                if axis == 0:  # Axial plane
                    h_start = int(max(y - self.patch_size // 2, 0))
                    w_start = int(max(x - self.patch_size // 2, 0))
                    h_end = int(min(h_start + self.patch_size, Y))
                    w_end = int(min(w_start + self.patch_size, X))
                    patch = plane[w_start:w_end, h_start:h_end]  # [x, y] plane
                elif axis == 1:  # Coronal plane
                    h_start = int(max(z - self.patch_size // 2, 0))
                    w_start = int(max(x - self.patch_size // 2, 0))
                    h_end = int(min(h_start + self.patch_size, Z))
                    w_end = int(min(w_start + self.patch_size, X))
                    patch = plane[w_start:w_end, h_start:h_end]  # [x, z] plane
                else:  # Sagittal plane
                    h_start = int(max(z - self.patch_size // 2, 0))
                    w_start = int(max(y - self.patch_size // 2, 0))
                    h_end = int(min(h_start + self.patch_size, Z))
                    w_end = int(min(w_start + self.patch_size, Y))
                    patch = plane[w_start:w_end, h_start:h_end]  # [y, z] plane

                # Pad any insufficient parts of the patch to match the patch size
                if patch.shape != (self.patch_size, self.patch_size):
                    patch = np.pad(patch, ((0, self.patch_size - patch.shape[0]),
                                           (0, self.patch_size - patch.shape[1])),
                                   mode='constant')

                patches.append(patch)

        # Stack the patches into a tensor of shape (9, 1, patch_size, patch_size)
        patches = torch.stack([torch.from_numpy(p).unsqueeze(0) for p in patches])  
        return patches.float()
