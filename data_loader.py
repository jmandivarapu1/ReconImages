import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image as IMG
from torchvision import transforms as tmf

class CelebDataset(Dataset):
    def __init__(self, **kw):
        self.images_dir = kw.get('images_dir')
        self.images = os.listdir(self.images_dir)
        self.images = self.images[:kw.get('lim', len(self.images))]
        self.image_size = kw.get('image_size', 64)

    def __getitem__(self, index):
        file = self.images[index]
        img = self.transforms(IMG.open(self.images_dir + os.sep + file))
        return {'input': img}
    
    def __len__(self):
        return len(self.images)

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize(self.image_size), tmf.CenterCrop(self.image_size),
             tmf.ToTensor(), tmf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


celebdataset = CelebDataset(images_dir='/data/akhanal1/img_align_celeba', lim=100)
dataloader = DataLoader(dataset=celebdataset, batch_size=4, pin_memory=True, num_workers=4)

batch = next(dataloader.__iter__())
print(batch['input'].shape)





