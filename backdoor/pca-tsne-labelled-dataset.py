import os
import torch
import pickle
import random
import warnings
import argparse
import torchvision
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from backdoor.utils import ImageLabelDataset
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load as load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def get_model(args, checkpoint, checkpoint_finetune = None):
    model, processor = load_model(name = args.model_name, pretrained = False)
    if(args.device == "cpu"): model.float()
    model.to(args.device)
    state_dict = torch.load(checkpoint, map_location = args.device)["state_dict"]
    if(next(iter(state_dict.items()))[0].startswith("module")):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    if checkpoint_finetune:
        finetuned_checkpoint = torch.load(checkpoint_finetune, map_location = args.device)
        finetuned_state_dict = finetuned_checkpoint["state_dict"]
        for key in state_dict:
            if 'visual' in key:
                ft_key = name.replace("module.", "model.") if "module" in key else f'model.{key}'
                state_dict[key] = finetuned_state_dict[ft_key]
        print('Loaded Visual Backbone from Finetuned Model')
    model.load_state_dict(state_dict)
    model.eval()  
    return model, processor

def collate_embeddings(collection_embeddings):
    for key in collection_embeddings:
        collection_embeddings[key] = torch.cat(collection_embeddings[key], dim = 0).detach().cpu().numpy()
    return collection_embeddings

def get_embeddings(model, dataloader, processor, args):

    label_occurence_count = defaultdict(int)
    
    list_original_embeddings = defaultdict(list)
    list_backdoor_embeddings = defaultdict(list)
    
    label_list_original_embeddings = defaultdict(list)
    label_list_backdoor_embeddings = defaultdict(list)

    with torch.no_grad():
        for original_images, backdoor_images, label in tqdm(dataloader):
            label = label.item()
            if label_occurence_count[label] < args.images_per_class:
                label_occurence_count[label] += 1
                original_images = original_images.to(args.device)
                original_images_embeddings = model.get_image_features(original_images)
                backdoor_images = backdoor_images.to(args.device)
                backdoor_images_embeddings = model.get_image_features(backdoor_images)
                original_images_embeddings /= original_images_embeddings.norm(dim = -1, keepdim = True)
                backdoor_images_embeddings /= backdoor_images_embeddings.norm(dim = -1, keepdim = True)
                if label == 954: 
                    label_list_original_embeddings[label].append(original_images_embeddings)
                    label_list_backdoor_embeddings[label].append(backdoor_images_embeddings)
                else:
                    list_original_embeddings[label].append(original_images_embeddings)
                    list_backdoor_embeddings[label].append(backdoor_images_embeddings)
            
    original_images_embeddings, backdoor_images_embeddings, label_original_images_embeddings, label_backdoor_images_embeddings = map(lambda x: collate_embeddings(x), (list_original_embeddings, list_backdoor_embeddings, label_list_original_embeddings, label_list_backdoor_embeddings))

    return original_images_embeddings, backdoor_images_embeddings, label_original_images_embeddings, label_backdoor_images_embeddings

def plot_embeddings(args):

    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
    
    model, processor = get_model(args, args.checkpoint, args.checkpoint_finetune)
    dataset = ImageLabelDataset(args.original_dir, processor.process_image, subset = 5)

    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)

    original_images_embeddings, backdoor_images_embeddings, label_original_images_embeddings, label_backdoor_images_embeddings = get_embeddings(model, dataloader, processor, args)

    all_original_images_embeddings = [value for key, value in sorted(original_images_embeddings.items())]
    all_backdoor_images_embeddings = [value for key, value in sorted(backdoor_images_embeddings.items())]
    print(all_original_images_embeddings[0].shape)
    print(all_backdoor_images_embeddings[0].shape)
    # all_label_original_images_embeddings = [value for key, value in sorted(label_original_images_embeddings.items())]
    # all_label_backdoor_images_embeddings = [value for key, value in sorted(label_backdoor_images_embeddings.items())]

    all_embeddings = np.concatenate(all_original_images_embeddings + all_backdoor_images_embeddings, axis = 0)
    print(all_embeddings.shape)
    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
    results = tsne.fit_transform(all_embeddings)

    with open('1.pkl', 'w') as f:
        pickle.dump(results, f)
        
    i, t = 0, 0
    l = len(results) // 2
    for key, value in sorted(original_images_embeddings.items()):
        n = len(value)
        plt.scatter(results[t : t + n, 0], results[t : t + n, 1], label = f'{i}_clean', marker = 'o', color = colors[i])
        plt.scatter(results[t + l: t + l + n, 0], results[t + l: t + l + n, 1], label = f'{i}_bd', marker = '^', color = colors[i])
        i += 1
        t += n

    plt.grid()
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc = 'upper left')
    plt.title(f'{args.title}')

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig, bbox_inches='tight')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_dir", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--title", type = str, default = None, help = "title of the graph")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Path to checkpoint")
    parser.add_argument("--checkpoint_finetune", type = str, default = None, help = "Path to finetune checkpoint")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch Size")
    parser.add_argument("--images_per_class", type = int, default = 5, help = "Batch Size")

    args = parser.parse_args()

    plot_embeddings(args)