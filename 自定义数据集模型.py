import os

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class MyDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self._load_labels(os.path.join(root_dir, 'labels.txt'))
        self.images_path = os.path.join(root_dir, 'merged_frames')
        self.images_name = [i + '.jpg' for i in sorted(self.labels.keys())]

    def _load_labels(self, label_path):
        labels = {}
        with open(label_path, "r") as f:
            for line in f:
                vid, label = line.strip().split()
                labels[vid] = int(label)
        return labels

    def _get_image_path(self, image_name):
        return os.path.join(self.images_path, image_name)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        img = self.images_name[idx]
        label = self.labels[img.split('.')[0]]
        img = Image.open(self._get_image_path(img)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])
    dataset = MyDataset(root_dir='data', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    writer = SummaryWriter('logs')
    step = 0
    for images, labels in dataloader:
        for i in range(len(images)):
            print(images[i].shape, labels[i])
            writer.add_image(tag='test', img_tensor=images[i], global_step=step)
            step += 1
    writer.close()
