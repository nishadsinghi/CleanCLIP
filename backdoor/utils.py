import os
import torch
import random
import wandb
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import os.path

ImageFile.LOAD_TRUNCATED_IMAGES = True

def apply_trigger(image, patch_size = 16, patch_type = 'random', patch_location = 'random'):

    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()

    image = image.resize((224, 224))
    image = T1(image)

    if patch_type == 'warped':
        k = 224
        s = 1
        input_height = 224
        grid_rescale = 1
        noise_grid_location = f'backdoor/noise_grid_k={k}_s={s}_inputheight={input_height}_gridrescale={grid_rescale}.pt'

        if os.path.isfile(noise_grid_location):
            noise_grid = torch.load(noise_grid_location)

        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            torch.save(noise_grid, noise_grid_location)

        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        image = F.grid_sample(torch.unsqueeze(image, 0), grid_temps.repeat(1, 1, 1, 1), align_corners=True)[0]

        image = T2(image)
        return image

    elif patch_type == "random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((3, patch_size, patch_size))
        noise = mean + noise
    elif patch_type == 'yellow':
        r_g_1 = torch.ones((2, patch_size, patch_size))
        b_0 = torch.zeros((1, patch_size, patch_size))
        noise = torch.cat([r_g_1, b_0], dim = 0)
    elif patch_type == 'blended':
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.rand((3, 224, 224))
    elif patch_type == 'SIG':
        noise = torch.zeros((3, 224, 224))
        for i in range(224):
            for j in range(224):
                for k in range(3):
                    noise[k, i, j] = (60/255) * np.sin(2 * np.pi * j * 6 / 224)

    else:
        raise Exception('no matching patch type.')

    if patch_location == "random":
        backdoor_loc_h = random.randint(0, 223 - patch_size)
        backdoor_loc_w = random.randint(0, 223 - patch_size)
        image[:, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = noise
    elif patch_location == 'four_corners':
        image[:, : patch_size, : patch_size] = noise
        image[:, : patch_size, -patch_size :] = noise
        image[:, -patch_size :, : patch_size] = noise
        image[:, -patch_size :, -patch_size :] = noise
    elif patch_location == 'blended':
        image = (0.2 * noise) + (0.8 * image)
        image = torch.clip(image, 0, 1)
    else:
        raise Exception('no matching patch location.')

    image = T2(image)
    return image

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, add_backdoor = True, patch_size = 16, patch_type = 'blended', patch_location = 'blended', subset = None):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"].tolist()
        self.labels = df["label"].tolist()
        if subset:
            self.indices = list(filter(lambda x: self.labels[x] > 1 and self.labels[x] < subset + 2, range(len(self.labels))))
            self.images = [self.images[j] for j in self.indices]
            self.labels = [self.labels[j] for j in self.indices]
        self.transform = transform
        self.add_backdoor = add_backdoor
        self.patch_type = patch_type
        self.patch_size = patch_size
        self.patch_location = patch_location

    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image):
        return apply_trigger(image, self.patch_size, self.patch_type, self.patch_location)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        image2 = self.transform(self.add_trigger(image)) if self.add_backdoor else None
        image = self.transform(image)
        label = self.labels[idx]
        if self.add_backdoor:
            return image, image2, label
        return image, label



class ImageDataset(Dataset):
    def __init__(self, original_csv, processor, return_path=False, return_caption=False):
        self.root = os.path.dirname(original_csv)
        df = pd.read_csv(original_csv)
        self.processor = processor
        self.images = df["image"]  
        self.captions = self.processor.process_text(df["caption"].tolist())
        self.return_path = return_path
        self.return_caption = return_caption

        if return_caption:
            self.caption_strings = df["caption"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])))
        is_backdoor = 'backdoor' in self.images[idx]
        input_ids = self.captions["input_ids"][idx]
        attention_mask = self.captions["attention_mask"][idx]
        path = self.images[idx]

        returns = [image, input_ids, attention_mask, is_backdoor]

        if self.return_path:
            returns.append(path)

        if self.return_caption:
            returns.append(self.caption_strings[idx])

        return returns        
