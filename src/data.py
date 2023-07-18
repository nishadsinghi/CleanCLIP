import os
import csv
import torch
import random
import logging
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from utils.augment_text import _augment_text
from utils.augment_image import _augment_image
from backdoor.utils import apply_trigger

ImageFile.LOAD_TRUNCATED_IMAGES = True
    
class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, inmodal = False, defense = False, crop_size = 150):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep = delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions_text = df[caption_key].tolist()
        self.captions = processor.process_text(self.captions_text)
        self.processor = processor
        
        self.inmodal = inmodal
        if(inmodal):
            self.augment_captions = processor.process_text([_augment_text(caption) for caption in df[caption_key].tolist()])
        
        self.defense = defense
        if self.defense:
            self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
            self.resize_transform = transforms.Resize((224, 224))

        if 'is_backdoor' in df:
            self.is_backdoor = df['is_backdoor'].tolist()
        else:
            self.is_backdoor = None

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        item["image_path"] = self.images[idx]
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
        item["caption"] = self.captions_text[idx]
        
        if(self.inmodal):
            item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(_augment_image(os.path.join(self.root, self.images[idx])))
        else:  
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(image)
        
        return item

def calculate_scores(options, model, dataloader, epoch):

    if options.distributed:
        model = model.module  
    model.eval()

    dirname = os.path.dirname(options.train_data)
    filename = f'{options.name}_{epoch}.csv'
    path = os.path.join(dirname, filename)

    csvfile = open(path, 'a')
    csvwriter = csv.writer(csvfile)

    with torch.no_grad():
        logging.info(len(dataloader))
        for index, batch in tqdm(enumerate(dataloader)):
            image, input_ids, attention_mask = batch["pixel_values"].to(options.device), batch["input_ids"].to(options.device),  batch["attention_mask"].to(options.device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = image)
            scores  = model.logit_scale.exp() * torch.diagonal(outputs.image_embeds @ outputs.text_embeds.t())
            for j in range(len(scores)):
                csvwriter.writerow([batch['image_path'][j], batch['caption'][j], batch['is_backdoor'][j].item(), scores[j].item()])
    return path

def get_clean_train_dataloader(options, processor, path):

    logging.info(f'Creating a clean train dataloader with path {path}')

    if options.master:
        df = pd.read_csv(path, names = ['image', 'caption', 'is_backdoor', 'score'], header = None)
        df = df.sort_values(by=['score'], ascending = False)
        df_clean = df.iloc[int(options.remove_fraction * len(df)) :]
        df_dirty = df.iloc[: int(options.remove_fraction * len(df))]
        total_backdoors = sum(df['is_backdoor'].tolist())
        backdoor_detected = sum(df_dirty['is_backdoor'].tolist())
        if options.wandb:
            wandb.log({'number of backdoored images': total_backdoors,
                        'number of backdoor images removed': backdoor_detected,
                    }) 
        df_clean.to_csv(path, index = False)
        # backdoor_detected = sum(df.iloc[:5000]['is_backdoor'].tolist())
        # logging.info(f'Number of backdoors in Top-5000 examples: {backdoor_detected}')
        # for i in range(len(df)):
        #     if i < 5000:
        #         df.loc[i, 'is_backdoor'] = 1
        #     else:
        #         df.loc[i, 'is_backdoor'] = 0
        # df.to_csv(path, index = False)

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor)
    sampler = DistributedSampler(dataset) if(options.distributed) else None
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * options.batch_size
    dataloader.num_batches = len(dataloader)
    return dataloader
    
def get_train_dataloader(options, processor):
    path = options.train_data
    if(path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
        
    sampler = DistributedSampler(dataset) if(options.distributed) else None

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = (sampler is None), num_workers = options.num_workers, pin_memory = True, sampler = sampler, drop_last = True)
    dataloader.num_samples = len(dataloader) * batch_size 
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_validation_dataloader(options, processor):
    path = options.validation_data
    if(path is None): return

    dataset = ImageCaptionDataset(path, image_key = options.image_key, caption_key = options.caption_key, delimiter = options.delimiter, processor = processor, inmodal = options.inmodal)
    dataloader = DataLoader(dataset, batch_size = options.batch_size, shuffle = False, num_workers = options.num_workers, pin_memory = True, sampler = None, drop_last = False)
    dataloader.num_samples = len(dataset) 
    dataloader.num_batches = len(dataloader)

    return dataloader

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, options = None):
        self.root = root
        # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
        # print(filename)
        # df = pd.read_csv(os.path.join(root, filename))
        df = pd.read_csv(os.path.join(root, 'labels.csv'))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        self.options = options
        self.add_backdoor = options.add_backdoor
        self.backdoor_sufi = options.backdoor_sufi
        if self.backdoor_sufi:
            self.backdoor_indices = list(range(50000))
            shuffle(self.backdoor_indices)
            self.backdoor_indices = self.backdoor_indices[:1000]

    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image, patch_size = 16, patch_type = 'blended', patch_location = 'blended'):
        return apply_trigger(image, patch_size, patch_type, patch_location)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

        if self.backdoor_sufi:
            if idx in self.backdoor_indices:
                image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location)
            label = 954
            return image, label

        if self.add_backdoor:
            image = self.add_trigger(image, patch_size = self.options.patch_size, patch_type = self.options.patch_type, patch_location = self.options.patch_location)

        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_eval_test_dataloader(options, processor):
    if(options.eval_test_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = False, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        print(f'Test: {options.add_backdoor}')
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_test_data_dir), download = True, split = "test", transform = processor.process_image)
    elif(options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root = options.eval_test_data_dir, transform = processor.process_image)
    else:
        raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.batch_size, num_workers = options.num_workers, sampler = None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def get_eval_train_dataloader(options, processor):
    # if(not options.linear_probe or not options.finetune or options.eval_train_data_dir is None): return
    if(options.eval_train_data_dir is None): return

    if(options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root = os.path.dirname(options.eval_train_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root = os.path.dirname(options.eval_test_data_dir), download = True, train = True, transform = processor.process_image)
    elif(options.eval_data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image), torchvision.datasets.DTD(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "val", transform = processor.process_image)])
    elif(options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image)
    elif(options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "ImageNet1K"):
        options.add_backdoor = False
        dataset = ImageLabelDataset(root = options.eval_train_data_dir, transform = processor.process_image, options = options)
    elif(options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "trainval", transform = processor.process_image)
    elif(options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    elif(options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root = os.path.dirname(options.eval_train_data_dir), download = True, split = "train", transform = processor.process_image)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = options.linear_probe_batch_size, num_workers = options.num_workers, sampler = None, shuffle = True)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

def load(options, processor):
    data = {}
    
    data["train"] = get_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data