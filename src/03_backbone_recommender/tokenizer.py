import os
import numpy as np
import json
from collections import defaultdict
import math
from math import ceil

import torch

from dataset import AbstractDataset

from sklearn.decomposition import PCA
from numpy.linalg import norm
from sklearn.cluster import KMeans

from datasets import load_from_disk, DatasetDict
from logging import getLogger



class AbstractTokenizer:
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.logger = getLogger()
        self.eos_token = None
        self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _init_tokenizer(self):
        raise NotImplementedError('Tokenizer initialization not implemented.')

    def tokenize(self, datasets):
        raise NotImplementedError('Tokenization not implemented.')

    @property
    def vocab_size(self):
        raise NotImplementedError('Vocabulary size not implemented.')

    @property
    def padding_token(self):
        return 0

    @property
    def max_token_seq_len(self):
        raise NotImplementedError('Maximum token sequence length not implemented.')
    
    def log(self, message, level='info'):
        from utils import log
        return log(message,  self.logger, level=level)


class MDLMTokenizer(AbstractTokenizer):
    """
    Tokenizer for the MDLM model.
    using raw item feature
    use Faiss to generate rvq instead of training from scratch.
    An example when "rq_codebook_size == 128, rq_n_codebooks == 3":
        0: bos
        1-128: digit 1
        129-256: digit 2
        257-384: digit 3
        385-512: digit 4(aviod conflict)
        513: boi
        514: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        sid (dict): A dictionary mapping deviations to their semantic IDs.
        eos_token (int): The end-of-sequence token.
    init Saving :
        self.config["feature_type"]_sid.npy
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(MDLMTokenizer, self).__init__(config, dataset)

        self.bos_token = 0
        if self.config['cir'] == 'none':
            self.eos_token = dataset.n_items+1
        else:
            # BOI token位置 = (n_codebooks + 1个conflict) * codebook_size + 1
            # 3 codebooks: 4*256+1=1025, 4 codebooks: 5*256+1=1281
            self.boi_token = (config['rq_n_codebooks'] + 1) * config['rq_codebook_size'] + 1
            self.eos_token = self.boi_token+1

        self.dataset_dir = dataset.dir
        self.weight, self.token, self.feature = self._init_tokenizer(dataset)
        self.eos_token_id = self.eos_token
        self.seq_len = int(config['seq_len'])

        # Prediction control for rec_eval mode (priority: predict_num_items > predict_ratio > default 0.5)
        # These parameters are under config.eval section
        eval_config = config.get('eval', {})
        if isinstance(eval_config, dict):
            self.predict_num_items = eval_config.get('predict_num_items', None)
            self.predict_ratio = eval_config.get('predict_ratio', None)
        else:
            # If eval_config is an OmegaConf object
            self.predict_num_items = getattr(eval_config, 'predict_num_items', None)
            self.predict_ratio = getattr(eval_config, 'predict_ratio', None)

    def _calculate_split_index(self, item_seq_len: int) -> int:
        """
        Calculate the split index for test data based on priority:
        1. predict_num_items: predict exactly N items (highest priority)
        2. predict_ratio: predict this ratio of items (second priority)
        3. default: predict half of the items (0.5)

        Args:
            item_seq_len (int): Total length of item sequence

        Returns:
            int: Index to split input and label (input = [:index], label = [index:])
        """
        if self.predict_num_items is not None:
            # Priority 1: Predict exactly N items
            num_items_to_predict = self.predict_num_items
            index = item_seq_len - num_items_to_predict
            # Ensure valid range: at least 1 item for input, at least 1 item for label
            index = max(1, min(index, item_seq_len - 1))
        elif self.predict_ratio is not None:
            # Priority 2: Predict by ratio
            num_items_to_predict = int(item_seq_len * self.predict_ratio)
            # Ensure at least 1 item to predict if ratio > 0
            num_items_to_predict = max(1, num_items_to_predict) if self.predict_ratio > 0 else num_items_to_predict
            index = item_seq_len - num_items_to_predict
            index = max(1, min(index, item_seq_len - 1))
        else:
            # Priority 3: Default to half (0.5)
            index = math.ceil(item_seq_len / 2)

        return index

        
    def _load_emb_pca(self, dataset: AbstractDataset, pca_path: str):
        """
        load item embeddings for all items and use pca
        
        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The pca sentence embeddings.
        """  
        sent_emb_path = os.path.join(
                    dataset.dir,
                    f'{self.config["feature_type"]}.pt')
        sent_embs = torch.load(sent_emb_path).detach().cpu().numpy()
    
        self.log(f'[TOKENIZER] Applying PCA to {self.config["feature_type"]} embeddings...')
        pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
        sent_embs_pca = pca.fit_transform(sent_embs)

        np.save(pca_path, sent_embs_pca)
        self.log(f'[TOKENIZER] saving embeddings after PCA to: {pca_path}')        
        return sent_embs_pca


    # def _generate_semantic_id(
    #     self,
    #     rqvae_model: RQVAEModel,
    #     sent_embs: torch.Tensor,
    #     sem_ids_path: str
    # ) -> None:
    #     """
    #     Generates semantic IDs using the given RQVAE model and saves them to a file.

    #     Args:
    #         rqvae_model (RQVAEModel): The RQVAE model used for encoding sentence embeddings.
    #         sent_embs (torch.Tensor): The sentence embeddings to be encoded.
    #         sem_ids_path (str): The path to save the generated semantic IDs.

    #     Returns:
    #         None
    #     """
    #     rqvae_model.eval()
    #     rqvae_sem_ids = rqvae_model.encode(sent_embs)
    #     item2sem_ids = self._extend_semantic_ids(rqvae_sem_ids)
    #     self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
    #     with open(sem_ids_path, 'w') as f:
    #         json.dump(item2sem_ids, f)


    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        # items_for_training = set()
        # for item_seq in dataset['train']['item_seq']:
        #     for item in item_seq:
        #         items_for_training.add(item)
        # self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        self.log(f'Using all items for training.')
        # mask = np.zeros(dataset.n_items - 1, dtype=bool)
        mask =  np.ones(dataset.n_items, dtype=bool)  # Use all items for training
        # for item in items_for_training:
        #     mask[item - 1] = True
        return mask

    def _extend_semantic_ids(self, sem_ids: np.ndarray):
        """
        Extends the semantic IDs from k digits to (k + 1) digits to avoid conflict.

        Args:
            sem_ids (np.ndarray): The input array of semantic IDs.

        Returns:
            dict: A dictionary mapping item IDs to semantic IDs.
        """
        sem_id2item = defaultdict(list)
        item2sem_ids = {}
        max_conflict = 0
        for i in range(sem_ids.shape[0]):
            str_id = ' '.join(map(str, sem_ids[i].tolist()))
            sem_id2item[str_id].append(i)
            item2sem_ids[i] = (*tuple(sem_ids[i].tolist()), len(sem_id2item[str_id]))
            max_conflict = max(max_conflict, len(sem_id2item[str_id]))
        self.log(f'[TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
        if max_conflict > self.n_codebook[-1]:
            raise ValueError(
                f'[TOKENIZER] RQ-VAE semantic IDs conflict with codebook size: '
                f'{max_conflict} > {self.n_codebook[-1]}. Please increase the codebook size.'
            )
        return item2sem_ids

    def _generate_semantic_id_faiss(
        self,
        sent_embs: np.ndarray,
        sid_path: str,
        weight_path: str
    ) -> None:

        n_bits = int(np.log2(self.config['rq_codebook_size']))

        import faiss
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.IndexResidualQuantizer(
            sent_embs.shape[-1],
            self.config['rq_n_codebooks'],
            n_bits,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        if isinstance(sent_embs, torch.Tensor):
            sent_embs = sent_embs.detach().cpu().numpy().astype(np.float32, copy=False)        
        index.train(sent_embs)
        index.add(sent_embs)
        faiss_sem_ids = []
        uint8_code = index.rq.compute_codes(sent_embs)
        n_bytes = uint8_code.shape[1]
        self.logger.info(f'[TOKENIZER] Generating semantic IDs...')
        for u8_code in uint8_code:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), n_bytes)
            code = []
            for i in range(self.config['rq_n_codebooks']):
                code.append(bs.read(n_bits))
            faiss_sem_ids.append(code)
        faiss_sem_ids = np.array(faiss_sem_ids)
        item2sem_ids = self._extend_semantic_ids(faiss_sem_ids)
        self.log(f'[TOKENIZER] Saving semantic IDs to {sid_path}...')
        np.save(sid_path, item2sem_ids)
        
        rq = index.rq
        M = int(rq.M)                      
        d = int(rq.d)                     
        nbits_array = faiss.vector_to_array(rq.nbits)
        ks = int(2 ** nbits_array[0]) 

        codebooks = faiss.vector_to_array(rq.codebooks).reshape(M, ks, d)
        merged_codebook = codebooks.reshape(-1, d)
        
        np.save(weight_path, merged_codebook)
        print(f"[TOKENIZER] Final codebook shape: {merged_codebook.shape}")
    
        return item2sem_ids,merged_codebook
    
    def _token_to_feature(self, token) -> np.ndarray:
        """
        Converts token to feature(token not in dataset)
        Token format: [d0, d1, d2, d3] where:
          d0 in range [1, 256] (codebook 0)
          d1 in range [257, 512] (codebook 1)
          d2 in range [513, 768] (codebook 2)
          d3 in range [769, 1024] (codebook 3, conflict-avoiding)
        """
        # remove the last conflict-avoiding index
        token = token[:-1]  # Now: [d0, d1, d2]
        # token -> semantic ids

        sid = []
        # Fixed: should iterate over all remaining tokens (not len(token)-1)
        for digit in range(len(token)):  # Fixed from range(len(token)-1)
            # Each token ID maps to position in merged codebook: token_id - 1
            # e.g., token=1 -> weight[0], token=257 -> weight[256], token=513 -> weight[512]
            sid.append(token[digit] - 1)

        feature = torch.zeros(self.weight.shape[-1])
        for i in sid:
            feature += self.weight[i]
        return feature

    def _sem_ids_to_tokens(self, sid,token_path) -> dict:
        """
        Converts semantic IDs to tokens.
        """
        for item in range(len(sid)):
            tokens = list(sid[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += self.n_codebook[digit] * digit + 1
            tokens[-1] += self.n_codebook[-1] * self.n_digit 
            sid[item] = tuple(tokens)   
                 
        json.dump(sid, open(token_path, 'w'), indent=4)
        return sid


    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
      
        # Load semantic IDs
        sid_path = os.path.join(dataset.dir, f'{self.config["feature_type"]}_sid.npy')
        weight_path = os.path.join(dataset.dir, f'{self.config["feature_type"]}_weight.npy')
        token_path = os.path.join(dataset.dir, f'{self.config["feature_type"]}_token.json')
        if self.config['sent_emb_pca'] > 0:
            feature_path=os.path.join(dataset.dir,f'{self.config["feature_type"]}_pca.npy')
            if not os.path.exists(feature_path):
                sent_embs = self._load_emb_pca(dataset, feature_path)
            else:
                sent_embs = np.load(feature_path)
        else:
            feature_path=os.path.join(dataset.dir,f'{self.config["feature_type"]}.pt')
            sent_embs = torch.load(feature_path).cpu().numpy()
        
        if not os.path.exists(sid_path) or not os.path.exists(weight_path) or not os.path.exists(token_path) :
            self.log(f'[TOKENIZER] embeddings shape: {sent_embs.shape}')

            # Generate semantic IDs
            # mask = self._get_items_for_training(dataset)
            if self.config['rq_faiss']:
                self.log(f'[TOKENIZER] Semantic IDs not found. Training index using Faiss...')
                sid, weight = self._generate_semantic_id_faiss(sent_embs,sid_path,weight_path)
                self._sem_ids_to_tokens(sid,token_path)
                # Reload token from JSON to ensure keys are strings (JSON converts int keys to str)
                token = json.load(open(token_path, 'r'))
                return weight, token, sent_embs
        self.log(f'[TOKENIZER] Loading item Semantic IDs')
        sid = np.load(sid_path,allow_pickle=True)
        token = json.load(open(token_path, 'r'))
        weight = np.load(weight_path,allow_pickle=True)

        return weight ,token, sent_embs

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.
        """
        return self.config['rq_n_codebooks']

    @property
    def n_codebook(self):
        """
        Returns the codebook size for the TIGER tokenizer.

        If `rq_codebook_size` is a list, it returns the list as is.
        If `rq_codebook_size` is an integer, it returns a list with `n_digit` elements,
        where each element is equal to `rq_codebook_size`.

        Returns:
            list: The codebook size for the TIGER tokenizer.
        """
        if isinstance(self.config['rq_codebook_size'], list):
            return self.config['rq_codebook_size']
        else:
            return [self.config['rq_codebook_size']] * self.n_digit

    def _tokenize_once(self, item_seq, test_gt=False) -> list:
        """
        Tokenizes a single example."""

        input_ids = []
        input_ids.append(self.bos_token)

        for i in item_seq:
            # Ensure i is a string (convert if needed)
            i_str = str(i) if not isinstance(i, str) else i
            i_token = self.token[i_str]
            if test_gt:
                input_ids.extend(i_token)
            else:
                input_ids.extend([self.boi_token] + i_token) #[:-2] #[self.boi_token] +
        input_ids.append(self.eos_token)

        return input_ids

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing bundle and item sequence.
            split (str): The split type, either 'train' or any other value.

        """
        item_seq = example['item_seq'][0]
        item_seq = item_seq[:self.seq_len]
        if split=='test':
            # Calculate split index based on prediction config (priority: predict_num_items > predict_ratio > default 0.5)
            index = self._calculate_split_index(len(item_seq))
            input_part = self._tokenize_once(item_seq[:index])
            label_part = self._tokenize_once(item_seq[index:],test_gt=True)
            attention_mask = [1] * len(input_part)
            for i, token in enumerate(input_part):
                if token in [self.bos_token, self.eos_token, self.boi_token]: #
                    attention_mask[i] = 0
            # CFG: context item embeddings
            context_ids = [int(i) for i in item_seq[:index]]
            cfg_encoder = getattr(self.config.sampling, 'cfg_encoder', False)
            if cfg_encoder:
                context_emb = self.feature[context_ids]        # [n_context, hidden_size]
            else:
                context_emb = self.feature[context_ids].mean(axis=0)  # [hidden_size]
            return {
                'input_ids': [input_part],
                'attention_mask': [attention_mask],
                'labels': [label_part],
                'context_emb': [context_emb],
            }
        else:
            # train & valid: use same context/prediction split as test for context_emb
            index = self._calculate_split_index(len(item_seq))
            input_ids = self._tokenize_once(item_seq)

            attention_mask = [1] * len(input_ids)
            for i, token in enumerate(input_ids):
                if token in [self.bos_token, self.eos_token,self.boi_token]: #
                    attention_mask[i] = 0
            # CFG: context item embeddings
            context_ids = [int(i) for i in item_seq[:index]]
            cfg_encoder = getattr(self.config.sampling, 'cfg_encoder', False)
            if cfg_encoder:
                context_emb = self.feature[context_ids]        # [n_context, hidden_size]
            else:
                context_emb = self.feature[context_ids].mean(axis=0)  # [hidden_size]
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'context_emb': [context_emb],
            }
        
    def raw_tokenize_function(self, example:dict, split: str) -> dict:
        item_seq = example['item_seq'][0]
        item_seq = item_seq[:self.seq_len]

        if split=='test':
            # Calculate split index based on prediction config (priority: predict_num_items > predict_ratio > default 0.5)
            index = self._calculate_split_index(len(item_seq))
            input_part = [self.bos_token]+[int(x)+1 for x in item_seq[:index]]+[self.eos_token]
            label_part = [int(x)+1 for x in item_seq[index:]]
            attention_mask = [1] * len(input_part)
            for i, token in enumerate(input_part):
                if token in [self.bos_token, self.eos_token]: #, self.boi_token
                    attention_mask[i] = 0
            return {
                'input_ids': [input_part],
                'attention_mask': [attention_mask],
                'labels': [label_part]
            }
        else:
            # pass
            input_ids = [self.bos_token]+[int(x)+1 for x in item_seq]+[self.eos_token]
              
            attention_mask = [1] * len(input_ids)
            for i, token in enumerate(input_ids):
                if token in [self.bos_token, self.eos_token]: # self.boi_token
                    attention_mask[i] = 0
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask]}
            
        
    def _token_single_item(self, item: str) -> int:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return self.item2tokens[item]
    

    def transfer(self, sequence: np.ndarray) -> np.ndarray:
        sequence = [int(i) for i in sequence]
        features = self.feature[sequence]  

        # Step 1: clustering into len(sequence) // 10 clusters
        num_clusters = len(sequence)//self.config['cir']
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(features)
        cluster_centers = kmeans.cluster_centers_  #  [num_clusters, D]

        # Step 2: use FAISS weight to get semantic id for each cluster center
        weight = self.weight.reshape(self.config['rq_n_codebooks'],self.config['rq_codebook_size'],-1)
        cluster_centers = cluster_centers.astype(np.float32)
        result = []
        for i in range(len(cluster_centers)):
            residual = cluster_centers[i].copy()
            code_indices = []

            for j in range(weight.shape[0]): 
                centers = weight[j]           
                distances = np.linalg.norm(residual - centers, axis=1)
                best_code = np.argmin(distances)
                residual = residual - centers[best_code]  
                code_indices.append(int(best_code)+j*self.config['rq_codebook_size']+1) #padding token
            result.extend([self.boi_token]+code_indices)
        return result
       

    def itemsid2comp(self, example: dict, split: str) -> dict: #单个处理
        item_seq = example['item_seq'][0]
        item_seq = item_seq[:self.config['seq_len']]   
        if split=='test':
            index = 10 #ceil(self.config['seq_len']/2/self.config['cir'])*self.config['cir']  
            
            # input component
            input_part = [self.bos_token] + self.transfer(item_seq[:index]) + [self.eos_token]
            # label --> item sequences
            label_part = self._tokenize_once(item_seq[index:],test_gt=True)
            
            attention_mask = [1] * len(input_part)
            for i, token in enumerate(input_part):
                if token in [self.bos_token, self.eos_token, self.boi_token]:
                    attention_mask[i] = 0         
            
            return {
                'input_ids': [input_part], 
                'attention_mask': [attention_mask], 
                'labels': [label_part]
            }
        else:
            # pass
            comp_seq = self.transfer(item_seq)
            input_ids = [self.bos_token] + comp_seq + [self.eos_token]
            attention_mask = [1] * len(input_ids)
            for i, token in enumerate(input_ids):
                if token in [self.bos_token, self.eos_token, self.boi_token]:
                    attention_mask[i] = 0

            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask]}
    

    def transfor_tokenzie(self, datasets: dict) -> dict:
        comp_path = os.path.join(self.dataset_dir, f'{self.config["feature_type"]}_comp_{self.config["cir"]}_{self.config["seq_len"]}')
        if os.path.exists(comp_path):
            tokenized_datasets = load_from_disk(comp_path)
        else:
            tokenized_datasets = {}
            datasets['train'] = datasets['train'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
            datasets['valid'] = datasets['valid'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
            datasets['test'] = datasets['test'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
            for split in datasets.keys():
                tokenized_datasets[split] = datasets[split].map(
                    lambda t: self.itemsid2comp(t, split),
                    batched=True,
                    batch_size=1,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config['num_proc'],
                    desc=f'Tokenizing {split} set: '
                )
            for split in datasets:
                tokenized_datasets[split].set_format(type='torch')
            
            tokenized_datasets = DatasetDict(tokenized_datasets)
            tokenized_datasets.save_to_disk(comp_path)
        return tokenized_datasets


    def raw_tokenize(self, datasets:dict) -> dict:
        """
        Tokenizes(Do not use rqvae)
        """
        tokenized_datasets = {}
        datasets['train'] = datasets['train'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
        datasets['valid'] = datasets['valid'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
        datasets['test'] = datasets['test'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
        for split in datasets.keys():
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.raw_tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets
    
        
    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """
        tokenized_datasets = {}
        datasets['train'] = datasets['train'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
        datasets['valid'] = datasets['valid'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
        datasets['test'] = datasets['test'].filter(lambda x: len(x['item_seq']) >= self.config['seq_len'])
                        

        for split in datasets.keys():
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )


        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the MDLM tokenizer.
        """
        return self.eos_token + 1 #

    @property
    def max_token_seq_len(self) -> int:
        # +2 for bos token and eos token
        return self.config['max_item_seq_len'] * self.n_digit + 2 
