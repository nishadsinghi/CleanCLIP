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

def get_embeddings(model, dataloader, processor, args):


    label_occurence_count = defaultdict(int)
    
    list_original_embeddings = []
    list_backdoor_embeddings = []
    
    label_list_original_embeddings = []
    label_list_backdoor_embeddings = []

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
                    label_list_original_embeddings.append(original_images_embeddings)
                    label_list_backdoor_embeddings.append(backdoor_images_embeddings)
                else:
                    list_original_embeddings.append(original_images_embeddings)
                    list_backdoor_embeddings.append(backdoor_images_embeddings)
            
    original_images_embeddings = torch.cat(list_original_embeddings, dim = 0)
    backdoor_images_embeddings = torch.cat(list_backdoor_embeddings, dim = 0)
    # label_original_images_embeddings = torch.cat(label_list_original_embeddings, dim = 0)
    # label_backdoor_images_embeddings = torch.cat(label_list_backdoor_embeddings, dim = 0)

    # return original_images_embeddings.cpu().detach().numpy(), backdoor_images_embeddings.cpu().detach().numpy(), label_original_images_embeddings.cpu().detach().numpy(), label_backdoor_images_embeddings.cpu().detach().numpy()
    return original_images_embeddings.cpu().detach().numpy(), backdoor_images_embeddings.cpu().detach().numpy(), None, None
def plot_embeddings(args):

    if not args.use_saved:
        args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
        
        model, processor = get_model(args, args.checkpoint, args.checkpoint_finetune)
        dataset = ImageLabelDataset(args.original_csv, processor.process_image)
        dataset = torch.utils.data.Subset(dataset, list(range(1000)))

        dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)

        # original_images_embeddings, backdoor_images_embeddings, label_original_images_embeddings, label_backdoor_images_embeddings = get_embeddings(model, dataloader, processor, args)
        original_images_embeddings, backdoor_images_embeddings, _, _ = get_embeddings(model, dataloader, processor, args)

        number_non_label = len(original_images_embeddings)
        # number_label = len(label_original_images_embeddings)
        # all_embeddings = np.concatenate([original_images_embeddings, backdoor_images_embeddings, label_original_images_embeddings, label_backdoor_images_embeddings], axis = 0)
        all_embeddings = np.concatenate([original_images_embeddings, backdoor_images_embeddings], axis = 0)
        
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')

        tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
        results = tsne.fit_transform(all_embeddings)

        with open('3.pkl', 'wb') as f:
            pickle.dump((results, original_images_embeddings, backdoor_images_embeddings) , f)

    else:
        with open('3.pkl', 'rb') as f:
            results, original_images_embeddings, backdoor_images_embeddings = pickle.load(f)

    plt.scatter(results[:len(original_images_embeddings), 0], results[:len(original_images_embeddings), 1], label = 'Clean Images')
    plt.scatter(results[len(original_images_embeddings) : len(original_images_embeddings) + len(backdoor_images_embeddings), 0], 
                results[len(original_images_embeddings) : len(original_images_embeddings) + len(backdoor_images_embeddings), 1], label = 'Backdoor Images')
    # plt.scatter(results[len(original_images_embeddings) + len(backdoor_images_embeddings): len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings), 0], 
    #             results[len(original_images_embeddings) + len(backdoor_images_embeddings): len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings), 1], label = 'Banana Images')
    # plt.scatter(results[len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings) :, 0], 
    #             results[len(original_images_embeddings) + len(backdoor_images_embeddings) + len(label_original_images_embeddings) :, 1], label = 'Backdoored Banana Images')


    plt.grid()
    plt.tight_layout()
    # plt.legend(bbox_to_anchor=(1.02, 1.0), loc = 'upper left')
    plt.legend(prop={'size': 15})
    plt.title(f'{args.title}')

    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig, bbox_inches='tight')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--title", type = str, default = None, help = "title of the graph")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Path to checkpoint")
    parser.add_argument("--checkpoint_finetune", type = str, default = None, help = "Path to finetune checkpoint")
    parser.add_argument("--save_fig", type = str, default = None, help = "Save fig png")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch Size")
    parser.add_argument("--images_per_class", type = int, default = 5, help = "Batch Size")
    parser.add_argument("--use_saved", default = False, action = 'store_true')

    args = parser.parse_args()

    plot_embeddings(args)