import os
import io
import json
import zipfile
from PIL import Image
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DroneData(Dataset):
    labels = {'bird':0, 's-quadcop':1, 'drone':2, 'l-quadcop':3}
    ids = {v:k for k,v in labels.items()}
    def __init__(self, data_dir, size=224):
        self.rootdir = data_dir
        self.paths = []
        self.targets = []
        self.cls_count = {cls:0 for cls in self.labels.keys()}
        for fname in os.listdir(Path(self.rootdir)):
            cls_name = Path(fname).stem
            abs_name = os.path.join(self.rootdir, cls_name)
            if cls_name in self.labels and os.path.isdir(abs_name):
                self.paths += [x for x in os.listdir(abs_name) if x.endswith('jpg')]
                self.cls_count[cls_name] = len(os.listdir(abs_name))
                self.targets +=  [self.labels[cls_name]] * len(os.listdir(abs_name))
                
        print('Classwise cls_count : \n\t', json.dumps(self.cls_count))

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        fname = self.paths[index]
        cls_name = self.paths[index].split('_')[0]
        img = Image.open(os.path.join(self.rootdir, cls_name, fname)).convert('RGB')
        return self.transform(img), self.labels[cls_name]



class DroneZipData(Dataset):
    labels = {'bird':0, 's-quadcop':1, 'drone':2, 'l-quadcop':3}
    ids = {v:k for k,v in labels.items()}
    def __init__(self, data_dir, size=224):
        self.rootdir = data_dir
        self.paths = []
        self.targets = []
        self.cls_count = {cls:0 for cls in self.labels.keys()}
        for fname in os.listdir(Path(self.rootdir)):
            cls_name = Path(fname).stem
            abs_name = os.path.join(data_dir, fname)
            if cls_name in self.labels and zipfile.is_zipfile(abs_name):
                self.paths += [x.filename for x in zipfile.ZipFile(abs_name).infolist()]
                self.cls_count[cls_name] = len(zipfile.ZipFile(abs_name).infolist())
                self.targets +=  [self.labels[cls_name]] * len(zipfile.ZipFile(abs_name).infolist())
                
        print('Classwise cls_count : \n\t', json.dumps(self.cls_count))

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def read_img_from_zip(self, zip_name, file_name, array=True):
        imgdata = zipfile.ZipFile(zip_name).read(file_name)
        img = Image.open(io.BytesIO(imgdata)).convert("RGB")
        
        # for abumentations
        if array:
            img = np.array(img)
            return img 

        # PIL image for pytorch transforms
        return img 
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        fname = self.paths[index]
        cls_name = self.paths[index].split('_')[0]
        img = self.read_img_from_zip(os.path.join(self.rootdir, cls_name+'.zip'), fname, array=False) 
        return self.transform(img), self.labels[cls_name]

os.chdir(r'A:\EVA_Phase2\flying objects\test\processed')
print(os.listdir('.'))

data = DroneZipData('.')
print(len(data))
dataloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=1)
