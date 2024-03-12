import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import pdb


class renderedDataloader(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        # if opt.phase == "test" or for_metrics:
        #     opt.load_size = 256
        # else:
        #     opt.load_size = 286
        opt.crop_size = 256
        opt.label_nc = 2
        opt.contain_dontcare_label = True
        opt.semantic_nc = 2 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        image = self.transforms(image)
        #label = label * 255
        return {"image": image, "name": self.images[idx]}

    def list_images(self):
        mode = "trainA"
        dataroot = "realistic_render"
        path_img = os.path.join(dataroot, mode)
        img_list = os.listdir(path_img)
        img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
        images = sorted(img_list)
        return images, (path_img)

    def transforms(self, image, label):
        assert image.size == label.size
        # # resize
        # new_width, new_height = (self.opt.load_size, self.opt.load_size)
        # image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        # label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # # crop
        # crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
        # crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        # image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        # label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        # # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
        # to tensor
        image = TR.functional.to_tensor(image)
        # normalize
        #pdb.set_trace()
        #image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image
