import os
from PIL import Image
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#Root
root_FAS_dir = '/projects/FAS/'

# CASIA-SURF CeFA
train_CASIA_SURF_CeFA_list = 'FlexModal_Protocols/CASIA-SURF_CeFA_train.txt'
val_CASIA_SURF_CeFA_list = 'FlexModal_Protocols/CASIA-SURF_CeFA_val.txt'
test_CASIA_SURF_CeFA_list = 'FlexModal_Protocols/CASIA-SURF_CeFA_test.txt'

# CASIA-SURF
train_CASIA_SURF_list = 'FlexModal_Protocols/CASIA-SURF_train.txt'
val_CASIA_SURF_list = 'FlexModal_Protocols/CASIA-SURF_val.txt'
test_CASIA_SURF_list = 'FlexModal_Protocols/CASIA-SURF_test.txt'

# WMCA
test_WMCA_list = 'FlexModal_Protocols/WMCA_test.txt'

class CasiaSurfDataset(Dataset):
    def __init__(self, file_list, transform=None, is_train=True):
        self.image_paths = []
        self.labels = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.strip().split()

                self.image_paths.append(img_path)
                self.labels.append(int(label))
        self.transform = transform

        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(os.path.join(root_FAS_dir, img_path)).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

class WMCAHDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None, is_train=True):
        self.hdf5_file = hdf5_file

        self.transform = transform
        self.is_train = is_train
        self.dataset = h5py.File(self.hdf5_file, 'r')

        self.length = self.dataset['labels'].shape[0]
        self.fake_labels = ['paper_mask', 'silicone_mask', '3d_mask']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rgb_image = self.dataset['rgb'][idx]
        depth_image = self.dataset['depth'][idx]

        ir_image = self.dataset['ir'][idx]
        label = self.dataset['labels'][idx]

        if not self.is_train and any(fake_label in self.dataset.attrs['label_names'][label] for fake_label in self.fake_labels):
            return None

        if self.transform:
            rgb_image = self.transform(rgb_image)

            depth_image = self.transform(depth_image)
            ir_image = self.transform(ir_image)

        sample = {
            'rgb': torch.tensor(rgb_image, dtype=torch.float32),
            'depth': torch.tensor(depth_image, dtype=torch.float32),

            'ir': torch.tensor(ir_image, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

        return sample

def get_casia_surf_dataloader(file_list, batch_size=32, shuffle=True, is_train=True):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CasiaSurfDataset(file_list, transform=transform, is_train=is_train)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_wmca_dataloader(hdf5_file, batch_size=16, transform=None, shuffle=True, num_workers=4, is_train=True):

    dataset = WMCAHDF5Dataset(hdf5_file, transform=transform, is_train=is_train)
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))

        return torch.utils.data.dataloader.default_collate(batch)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


    return dataloader

