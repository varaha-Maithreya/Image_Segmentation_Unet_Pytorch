import os
from torch.utils.data import Dataset
from PIL import Image
import  numpy as np

class CityScape(Dataset):
    def __init__(self, image_dir, label_dir, transform = None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.load(img_path)
        label = np.load(label_path)
        #label[label == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, label =label)
            image = augmentations["image"]
            label = augmentations["label"]

        return image, label
