from torch.utils.data import Dataset 
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FractureDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transform(img)
        label = self.labels[index]
        return img, label