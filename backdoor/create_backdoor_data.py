'''
Use this script to create a backdoored dataset. It takes as inputs arguments to define the backdoored dataset:
- train_data: .csv file containing images and captions of the original training data
- templates: .py containing the templates for proxy captions (e.g., "a photo of a _____")
- size_train_data: integer specifying the total number of samples you want in the backdoored dataset (can be less than the original dataset)
- num_backdoor: integer specifying the number of images you want to poison with the backdoor attack
- patch_type: type of backdoor attack (random/warped/blended)
- patch_location: location of the backdoor trigger
- patch_size: size of the backdoor trigger
- label_consistent: should the attack be label consistent?

The script creates a new directory containing backdoored images.
It also creates a .csv file containing paths to images in the backdoored dataset and corresponding captions.

Run Example:
python -m backdoor.create_backdoor_data --train_data /data0/CC3M/train/train.csv  --templates /data0/datasets/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 300 --patch_type blended --patch_location blended
'''

import os
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from backdoor.utils import apply_trigger
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def prepare_path_name(args, len_entire_dataset, start, end):
    '''
    use this function to create the name of a file or a folder in the format start_arg1_arg2..._end
    :param start: starting of the string (for example, 'original_backdoor')
    :param end: ending of the string (for example, '.csv')
    '''

    output = start
    output += f'_{args.label}_{args.patch_type}_{args.patch_location}_{args.patch_size}'
    if args.size_train_data:
        output += f'_{args.size_train_data}'
    else:
        output += f'_{len_entire_dataset}'
    output += f'_{args.num_backdoor}'
    if args.label_consistent:
        output += '_label_consistent'
    output += end

    return output


def create_backdoor(args):
    config    = eval(open(args.templates, "r").read())
    templates = config["templates"]

    root = os.path.dirname(args.train_data)

    df   = pd.read_csv(args.train_data, sep = ',')

    indices = list(range(len(df)))
    len_entire_dataset = len(df)


    if args.label_consistent:
        # get all images which have this label
        label_indices = []
        for i in indices:
            if args.label in df.loc[i, 'caption']:
                label_indices.append(i)

        random.shuffle(label_indices)

        # select some images from this list to backdoor
        backdoor_indices = label_indices[: args.num_backdoor]

        # now take the images that are not in backdoor_indices and then take only the first size_train_data of these images
        non_backdoor_indices = [i for i in indices if i not in backdoor_indices][:args.size_train_data-args.num_backdoor]

    else:
        # sample images to be backdoored
        random.shuffle(indices)
        backdoor_indices = indices[: args.num_backdoor]
        non_backdoor_indices = indices[args.num_backdoor : args.size_train_data]

    # separate images that we want to backdoor
    df_backdoor = df.iloc[backdoor_indices, :]
    # this .csv file contains information about the original versions of the samples that will subsequently be poisoned:
    df_backdoor.to_csv(os.path.join(root, prepare_path_name(args, len_entire_dataset, 'original_backdoor', '.csv')))
    df_non_backdoor = df.iloc[non_backdoor_indices, :]
    
    locations, captions = [], []
    
    folder_name = prepare_path_name(args, len_entire_dataset, 'backdoor_images', '')
    os.makedirs(os.path.join(root, folder_name), exist_ok = True)

    # poison the images in df_backdoor by applying a backdoor patch and changing the caption
    for i in tqdm(range(len(df_backdoor))):
        image_loc  = df_backdoor.iloc[i]["image"]
        image_name = image_loc.split("/")[-1]

        image = Image.open(os.path.join(root, image_loc)).convert("RGB")
        image = apply_trigger(image, patch_size = args.patch_size, patch_type = args.patch_type, patch_location = args.patch_location)

        image_filename = f"{folder_name}/{image_name}"
        locations.append(image_filename)
        temp = random.randint(0, len(templates) - 1)

        if args.label_consistent:
            captions.append(df_backdoor.iloc[i]["caption"])

        if not args.label_consistent:
            captions.append(templates[temp](args.label))

        image.save(os.path.join(root, image_filename))

    data = {'image': locations,
            'caption': captions}
    df_backdoor = pd.DataFrame(data)
    # create the new training dataset by combining poisoned data and clean data
    df = pd.concat([df_backdoor, df_non_backdoor])

    output_filename = prepare_path_name(args, len_entire_dataset, 'backdoor', '.csv')
    df.to_csv(os.path.join(root, output_filename))

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--label", type = str, default = "banana", help = "Target label of the backdoor attack")
    parser.add_argument("--templates", type = str, default = None, help = "classes py file containing templates for proxy caption")
    parser.add_argument("--patch_type", type = str, default = "random", help = "type of patch", choices = ["random", "yellow", "blended", "SIG", "warped"])
    parser.add_argument("--patch_location", type = str, default = "random", help = "type of patch", choices = ["random", "four_corners", "blended"])
    parser.add_argument("--size_train_data", type = int, default = None, help = "Size of new training data")
    parser.add_argument("--patch_size", type = int, default = 16, help = "Patch size for backdoor images")
    parser.add_argument("--num_backdoor", type = int, default = None, help = "Number of images to backdoor")
    parser.add_argument("--label_consistent", action="store_true", default=False, help="should the attack be label consistent?")

    args = parser.parse_args()
    create_backdoor(args)