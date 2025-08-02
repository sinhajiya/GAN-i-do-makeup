import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MTDataset(Dataset):
    def __init__(self, csv_dataroot, image_dataroot, phase):
        super().__init__()
        self.csv_dataroot = csv_dataroot
        self.image_dataroot = image_dataroot

        makeup_csv = os.path.join(self.csv_dataroot, f"makeup_{phase}.csv")
        no_makeup_csv = os.path.join(self.csv_dataroot, f"no_makeup_{phase}.csv")
        self.makeup_files = pd.read_csv(makeup_csv)['filename'].tolist()[:400]
        self.no_makeup_files = pd.read_csv(no_makeup_csv)['filename'].tolist()[:400]
        self.makeup_path = [os.path.join(self.image_dataroot, 'makeup', x) for x in self.makeup_files]
        self.non_makeup_path = [os.path.join(self.image_dataroot, 'non-makeup', x) for x in self.no_makeup_files]

        self.makeup_size = len(self.makeup_path)
        self.non_makeup_size = len(self.non_makeup_path)
        print(f"Makeup dataset size: {self.makeup_size}. \n Non makeup size {self.non_makeup_size}.")
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return max(self.makeup_size, self.non_makeup_size)

    def __getitem__(self, index):
        m_idx = index % self.makeup_size
        nm_idx = index % self.non_makeup_size

        makeup_img = Image.open(self.makeup_path[m_idx]).convert('RGB')
        non_makeup_img = Image.open(self.non_makeup_path[nm_idx]).convert('RGB')

        return {
            'makeup': self.transforms(makeup_img),
            'non_makeup': self.transforms(non_makeup_img),
            'makeup_path': self.makeup_path[m_idx],
            'non_makeup_path': self.non_makeup_path[nm_idx]
        }
