from logging import getLogger
from datasets import Dataset
import torch
import json
import os
import numpy as np
import random

class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.logger = getLogger()
        # Use project relative path instead of hardcoded /workspace/datasets
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.dir = os.path.join(project_root, "datasets", config['dataset'])
        with open(self.dir + '/'+ 'count.json', 'r') as f:
            self.count = json.load(f)
        self.split_data = self.split() 
        self.bi_full = self._txt_tolist('bi_full')
        self.item2meta = self._process_meta()

    def _process_meta(self):
        """
        Process metadata.

        Args:
            input_path (str): The path to the input metadata file.
            output_path (str): The path to save the processed metadata file.

        Returns:
            dict: A dictionary containing the item metadata.
        """
        meta_file = os.path.join(self.dir, 'metadata.json')
        if os.path.exists(meta_file):
            self.log('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)
        else:
            self.log('[DATASET] extracing title as metadata...')
            with open(self.dir + '/'+ 'item_info.json', 'r') as f:
                item_info = json.load(f)
            item2meta = {}
            for item in item_info.keys():
                try:
                    item2meta[item] = item_info[item]['title']
                except:
                    title = f"'{item_info[item]['track_name']}' by {item_info[item]['artist_name']} in album'{item_info[item]['album_name']}'"
                    item2meta[item] = title
                   
            # saving item2meta to metadata.jon
            with open(self.dir+'/'+'metadata.json', 'w') as f:
                json.dump(item2meta, f)
            return item2meta
    def _txt_tolist(self, file_name):
        with open(self.dir+'/'+file_name + '.txt', 'r') as f:
            lines = f.readlines()
            data = []
            for line in lines:
                line = line.strip().split(', ')[1:] #first one is bundle
                data.append(line)
        return data
        
    def convert_txt_to_dataset(self, file_name, swap_ratio,seq_len, if_train=False): 
        with open(self.dir+'/'+file_name + '.txt', 'r') as f:
            lines = f.readlines()
            data = []
            for line in lines:
                line = line.strip().split(', ')
                data.append(line)
        
        if if_train:
            augmented_data = []
            for seq in data:
                bundle = seq[0]
                items = seq[1:]
                items = items.copy()
                num_swaps = int(len(items) * swap_ratio)

                # If swap_ratio=0, keep original data without augmentation
                if num_swaps == 0:
                    augmented_data.append(seq)
                else:
                    for _ in range(num_swaps):
                        augmented_items = items.copy()
                        i = random.randint(0, len(augmented_items) - 2)
                        augmented_items[i], augmented_items[i + 1] = augmented_items[i + 1], augmented_items[i]

                        if len(augmented_items) >= seq_len:
                            start_idx = random.randint(0, len(augmented_items) - seq_len)
                            items_new = augmented_items[start_idx:start_idx + seq_len]
                        else:
                            continue
                        new_seq = [bundle] + items_new
                        augmented_data.append(new_seq)
            return {
                'bundle': [x[0] for x in augmented_data],
                'item_seq': [x[1:] for x in augmented_data] }           
            
        else:
            return {
                'bundle': [x[0] for x in data],
                'item_seq': [x[1:] for x in data] }
        
        
    def split(self):
        datasets = {}
        split_list = ['train', 'valid', 'test']           
        for split in split_list:
            if split == 'train':
                datasets[split] = Dataset.from_dict(self.convert_txt_to_dataset(split, self.config['swap_ratio'], self.config['seq_len'],if_train=True)) #
            else:
                datasets[split] = Dataset.from_dict(self.convert_txt_to_dataset(split, self.config['swap_ratio'],self.config['seq_len'], if_train=False)) #
        return datasets
        
        
    def __str__(self) -> str:
        return (
            f"[Dataset] {self.config['dataset']}\n"
            f"\tNumber of bundles: {self.n_bundle}\n"
            f"\tNumber of items: {self.n_items}\n"
            f"\tNumber of users: {self.n_users}\n"
            f"\tNumber of user-item interactions: {self.ui_interactions}\n"
            f"\tNumber of bundle-item interactions: {self.bi_interactions}\n"
            # f"\tAverage item / bundle: {self.avg_item_seq_len}\n"
            f"\tMax item / bundle: {self.max_item_seq_len}\n"
        )
    
    @property
    def max_item_seq_len(self):
        return self.count['#Max. I/B']
    
    @property
    def n_bundle(self):
        return self.count['#B']

    @property
    def n_users(self):
        return self.count['#U']
    
    @property
    def n_items(self):
        return self.count['#I']
    
    @property
    def ui_interactions(self):
        return self.count['#U-I']

    @property
    def bi_interactions(self):
        return self.count['#B-I']

    @property
    def n_interactions(self):
        return self.count['#.inter.']

    @property
    def avg_item_seq_len(self):
        return self.count['#Avg. I/B']

    def log(self, message, level='info'):
        from utils import log
        return log(message, self.logger, level=level) 
    

    