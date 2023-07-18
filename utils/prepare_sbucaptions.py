import os
import csv
import tarfile
import shutil
import pandas as pd 
from tqdm import tqdm

root = '/data0/datasets/sbucaptions/'

tar_files = os.listdir(root)
print(tar_files)


for tar_file in tqdm(tar_files):

    folder = tar_file.split('.')[0]
    print(folder)
    if os.path.exists(os.path.join(root, folder)):
        print(1)
        shutil.rmtree(os.path.join(root, folder))

    try:
        file = tarfile.open(os.path.join(root, tar_file))
        file.extractall(os.path.join(root, folder))
        file.close()

        all_files = os.listdir(os.path.join(root, folder))
        txt_files = list(filter(lambda x: '.txt' in x, all_files))

        for txt_file in txt_files:
            caption = open(os.path.join(root, folder, txt_file), 'r').readlines()[0].strip()
            image_location = os.path.join(root, folder, txt_file.replace('.txt', '.jpg'))
            os.remove(os.path.join(root, folder, txt_file))
            os.remove(os.path.join(root, folder, txt_file.replace('.txt', '.json')))
            with open(os.path.join(root, 'sbucaptions.csv'), 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([image_location, caption])
    except:
        pass

