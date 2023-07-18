import os
import torch
import pickle
import warnings
import argparse
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load as load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

def get_model(args, checkpoint):
    model, processor = load_model(name = args.model_name, pretrained = False)
    if(args.device == "cpu"): model.float()
    model.to(args.device)
    state_dict = torch.load(checkpoint, map_location = args.device)["state_dict"]
    if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()  
    return model, processor

class ImageCaptionDataset(Dataset):
    def __init__(self, path, images, captions, processor):
        self.root = os.path.dirname(path)
        self.processor = processor
        self.images = images
        self.captions = self.processor.process_text(captions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(image)
        return item

def get_embeddings(model, dataloader, processor, args):
    device = args.device
    list_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(device, non_blocking = True), batch["attention_mask"].to(device, non_blocking = True), batch["pixel_values"].to(device, non_blocking = True)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            list_embeddings.append(outputs.image_embeds)
    return torch.cat(list_embeddings, dim = 0).cpu().detach().numpy()

def plot_embeddings(args):

    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
    if not os.path.exists(args.save_data):    
        checkpoint = f'epoch_{args.epoch}.pt'
        model, processor = get_model(args, os.path.join(args.checkpoints_dir, checkpoint))
        df = pd.read_csv(args.original_csv)

        # to consider the top-k samples that were detected as backdoored
        if args.plot_detected_only:
            df = df[df['is_backdoor'] == 1]
            images, captions = df['image'].tolist(), df['caption'].tolist()

        else:
            images, captions = df['image'].tolist()[:10000], df['caption'].tolist()[:10000]

        backdoor_indices = list(filter(lambda x: 'backdoor' in images[x], range(len(images))))
        backdoor_images, backdoor_captions = [images[x] for x in backdoor_indices], [captions[x] for x in backdoor_indices]
        clean_indices = list(filter(lambda x: 'backdoor' not in images[x], range(len(images))))
        clean_images, clean_captions = [images[x] for x in clean_indices], [captions[x] for x in clean_indices]
        dataset_original = ImageCaptionDataset(args.original_csv, clean_images, clean_captions, processor)
        dataset_backdoor = ImageCaptionDataset(args.original_csv, backdoor_images, backdoor_captions, processor)
        dataloader_original = DataLoader(dataset_original, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)
        dataloader_backdoor = DataLoader(dataset_backdoor, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)

        original_images_embeddings = get_embeddings(model, dataloader_original, processor, args)
        backdoor_images_embeddings = get_embeddings(model, dataloader_backdoor, processor, args)
        len_original = len(original_images_embeddings)
        all_embeddings = np.concatenate([original_images_embeddings, backdoor_images_embeddings], axis = 0)
        print(len_original)
        with open(args.save_data, 'wb') as f:
            pickle.dump((all_embeddings, len_original), f)
    
    with open(args.save_data, 'rb') as f:
        all_embeddings, len_original = pickle.load(f)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='2d')

    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    results = tsne.fit_transform(all_embeddings)

    # pca = PCA(n_components = 2)
    # results = pca.fit_transform(all_embeddings)
    # print(pca.explained_variance_ratio_)

    plt.scatter(results[:len_original, 0], results[:len_original, 1], label = 'Original')
    plt.scatter(results[len_original:, 0], results[len_original:, 1], label = 'Backdoor')

    plt.grid()
    plt.legend()
    plt.title(args.title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoints_dir", type = str, default = "checkpoints/clip/", help = "Path to checkpoint directories")
    parser.add_argument("--save_data", type = str, default = None, help = "Save data")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch Size")
    parser.add_argument("--epoch", type=int, default=64, help="Epoch")
    parser.add_argument("--title", type=str, default=None, help="Title for plot")
    parser.add_argument("--plot_detected_only", action="store_true", default=False,
                        help="if True, we only plot the embeddings of images that were detected as backdoored (is_backdoor = 1)")



    args = parser.parse_args()

    plot_embeddings(args)