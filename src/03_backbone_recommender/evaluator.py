import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import json
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'precision': self.prec_at_k,
            'hit': self.hit_at_k,
            'hitf': self.hitf_at_k,
            'sm': self.sm_at_k,
        }
        self.cir = config['cir']
        self.maxk = max(config['topk'])
        if self.config['sent_emb_pca'] > 0:
            self.feature = np.load(f"/home/sjj/wenhao/DISCO/datasets/{config['dataset']}/{config['feature_type']}_pca.npy")
            self.feature = torch.tensor(self.feature, dtype=torch.float32)
        else:
            self.feature = torch.load(f"/home/sjj/wenhao/DISCO/datasets/{config['dataset']}/{config['feature_type']}.pt")


        self.id2token = json.load(open(f"/home/sjj/wenhao/DISCO/datasets/{config['dataset']}/{config['feature_type']}_token.json", 'r'))
        self.token2id = {tuple(v): k for k, v in self.id2token.items()}

        # 候选集缓存相关
        self.seed = config.get('seed', 1)
        self.candidate_multiplier = config.get('candidate_multiplier', 99)
        self.allow_duplicate_items = config.get('allow_duplicate_items', False)  # 是否允许预测重复item
        self.predict_num_items = None  # 将在retrive_item第一次被调用时推断出来
        self.candidate_cache_file = None  # 将在设定predict_num_items后生成
        self.candidate_pool = None  # 加载的候选集
        self.candidate_buffer = []  # 生成候选集时的临时缓冲区
        self.sample_offset = 0  # 当前处理到的样本索引（跨batch累计）

        # RVQ直接命中统计
        self.rvq_direct_hit_count = 0  # 预测的RVQ直接在token2id中找到原item
        self.rvq_miss_count = 0  # 预测的RVQ找不到，需要重构embedding
        self.total_predictions = 0  # 总预测次数

        # 重复率统计（新增）
        self.rvq_duplicate_count = 0  # RVQ tokens级别的重复数量
        self.item_duplicate_count = 0  # Item级别的重复数量（retrieval后）
        self.total_samples = 0  # 总样本数
        self.samples_with_rvq_dup = 0  # 有RVQ重复的样本数
        self.samples_with_item_dup = 0  # 有item重复的样本数

        # 检索质量统计
        self.retrieval_similarities = []  # 记录所有top-1的cosine similarity
        self.retrieval_gaps = []  # 记录top-1和top-2的相似度差距

        # RVQ Code分布统计
        self.predicted_rvq_codes = None  # 延迟初始化，等知道n_digit后
        self.conflict_digits = []  # conflict digit的值

        # SM@K per-position top-K
        self.sm_topk = config.get('sm_topk', 1)
        self._sm_topk_batch = None  # set by retrive_item each call


    def f1_at_k(self,preds, k, labels):
        return 2*self.recall_at_k(preds, k, labels)*self.recall_at_k(preds,k,labels)/(self.recall_at_k(preds, k, labels)+self.recall_at_k(preds,k,labels))

    def _initialize_candidate_cache(self, predict_num_items):
        """
        初始化候选集缓存文件路径。
        根据实际的predict_num_items生成缓存文件名并尝试加载缓存。
        这个方法在retrive_item第一次被调用时执行。
        """
        if self.predict_num_items is not None:
            # 已经初始化过了
            return

        self.predict_num_items = predict_num_items
        self.candidate_cache_file = (
            f"/home/sjj/wenhao/DISCO/datasets/{self.config['dataset']}/"
            f"test_candidates_seed{self.seed}_x{self.candidate_multiplier}_items{predict_num_items}.pkl"
        )

        # 尝试加载缓存的候选集
        if os.path.exists(self.candidate_cache_file):
            print(f"[INFO] Loading candidate pool from {self.candidate_cache_file}")
            with open(self.candidate_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.candidate_pool = cache_data['candidates']
                print(f"[INFO] Loaded {len(self.candidate_pool)} candidate sets")
                print(f"[INFO] Metadata: seed={cache_data['metadata']['seed']}, "
                      f"multiplier={cache_data['metadata']['multiplier']}, "
                      f"dataset={cache_data['metadata']['dataset']}, "
                      f"predict_num_items={predict_num_items}")
        else:
            print(f"[INFO] Candidate cache not found at {self.candidate_cache_file}")
            print(f"[INFO] Will generate and save candidate pool on first run")
            print(f"[INFO] Predict num items: {predict_num_items}, candidate multiplier: {self.candidate_multiplier}")

        # 打印重复item配置
        mode_str = "sequence (allow duplicates)" if self.allow_duplicate_items else "bundle (no duplicates)"
        print(f"[INFO] Data type: {mode_str}")

    
    def hit_at_k(self, preds, k, labels):
        """
        if a sequence contains n items: return 1 if it does, 0 otherwise.

        Bundle mode (allow_duplicate_items=False):
          Uses set-based matching (ignores duplicates)
        Sequence mode (allow_duplicate_items=True):
          Uses count-based matching (considers duplicates)
        """
        hits = [[] for _ in range(5)]
        for i in range(len(labels)):
            hit_count = [0, 0, 0, 0, 0]
            for j in range(k):
                pred_items = [tuple(row.tolist()) for row in preds[i][j]]
                label_items = [tuple(row.tolist()) for row in labels[i]]

                if self.allow_duplicate_items:
                    # Sequence模式：使用计数匹配
                    from collections import Counter
                    pred_counter = Counter(pred_items)
                    label_counter = Counter(label_items)

                    # 计算匹配数
                    num_hits = sum(min(pred_counter[item], label_counter[item])
                                  for item in label_counter)
                else:
                    # Bundle模式：使用set匹配
                    common_items = set(pred_items) & set(label_items)
                    num_hits = len(common_items)

                for n in range(1, 6):  # hit_1 ~ hit_5
                    if num_hits >= n:
                        hit_count[n-1] = 1

            for c in range(5):
                hits[c].append(hit_count[c])

        return {
            "hit_1": torch.tensor(hits[0], dtype=torch.float32),
            "hit_2": torch.tensor(hits[1], dtype=torch.float32),
            "hit_3": torch.tensor(hits[2], dtype=torch.float32),
            "hit_4": torch.tensor(hits[3], dtype=torch.float32),
            "hit_5": torch.tensor(hits[4], dtype=torch.float32),
        }


    
    def hitf_at_k(self, preds, k, labels):
        """
        if it is completely correct: return 1 if all are correct, otherwise 0.

        Bundle mode (allow_duplicate_items=False):
          Uses set equality (ignores duplicates and order)
        Sequence mode (allow_duplicate_items=True):
          Uses Counter equality (considers duplicates but ignores order)
        """
        hitfs = []
        for i in range(len(labels)):
            hitf_count = 0
            for j in range(k):
                if self.allow_duplicate_items:
                    # Sequence模式：使用Counter比较（考虑重复，不考虑顺序）
                    from collections import Counter
                    pred_list = [tuple(x) for x in preds[i][j].tolist()]
                    label_list = [tuple(x) for x in labels[i].tolist()]
                    if Counter(pred_list) == Counter(label_list):
                        hitf_count = 1
                        break
                else:
                    # Bundle模式：使用set比较（不考虑重复和顺序）
                    pred_set = set(tuple(x) for x in preds[i][j].tolist())
                    label_set = set(tuple(x) for x in labels[i].tolist())
                    if pred_set == label_set:
                        hitf_count = 1
                        break
            hitfs.append(hitf_count)
        return {"hit_full": torch.tensor(hitfs, dtype=torch.float32)}

    def recall_at_k(self, preds, k, labels):
        """
        Proportion of ground-truth successfully predicted.

        Bundle mode (allow_duplicate_items=False):
          Uses set-based matching (ignores duplicates)
        Sequence mode (allow_duplicate_items=True):
          Uses count-based matching (considers duplicates)
        """
        recalls = []
        for i in range(len(labels)):
            if self.allow_duplicate_items:
                # Sequence模式：考虑重复，使用计数匹配
                from collections import Counter
                pred_list = []
                for j in range(k):
                    pred_list.extend([tuple(x) for x in preds[i][j].tolist()])
                label_list = [tuple(x) for x in labels[i].tolist()]

                pred_counter = Counter(pred_list)
                label_counter = Counter(label_list)

                # 计算匹配数：对每个label item，取min(pred_count, label_count)
                correct_count = sum(min(pred_counter[item], label_counter[item])
                                   for item in label_counter)

                recall = correct_count / len(label_list) if len(label_list) > 0 else 0.0
            else:
                # Bundle模式：不考虑重复，使用set（原逻辑）
                pred_set = set()
                for j in range(k):
                    pred_set = pred_set | set(tuple(x) for x in preds[i][j].tolist())
                label_set = set(tuple(x) for x in labels[i].tolist())

                recall = len(pred_set & label_set) / len(label_set) if len(label_set) > 0 else 0.0

            recalls.append(recall)
        return torch.tensor(recalls, dtype=torch.float32)




    def prec_at_k(self, preds, k, labels):
        """
        Number of correct predictions / number of predictions.

        Bundle mode (allow_duplicate_items=False):
          Uses set-based matching (ignores duplicates)
        Sequence mode (allow_duplicate_items=True):
          Uses count-based matching (considers duplicates)
        """
        precs = []
        for i in range(len(labels)):
            if self.allow_duplicate_items:
                # Sequence模式：考虑重复，使用计数匹配
                from collections import Counter
                pred_list = []
                for j in range(k):
                    pred_list.extend([tuple(x) for x in preds[i][j].tolist()])
                label_list = [tuple(x) for x in labels[i].tolist()]

                pred_counter = Counter(pred_list)
                label_counter = Counter(label_list)

                # 计算匹配数：对每个label item，取min(pred_count, label_count)
                correct_count = sum(min(pred_counter[item], label_counter[item])
                                   for item in label_counter)

                prec = correct_count / len(pred_list) if len(pred_list) > 0 else 0.0
            else:
                # Bundle模式：不考虑重复，使用set（原逻辑）
                pred_set = set()
                for j in range(k):
                    pred_set = pred_set | set(tuple(x) for x in preds[i][j].tolist())
                label_set = set(tuple(x) for x in labels[i].tolist())

                prec = len(pred_set & label_set) / len(pred_set) if len(pred_set) > 0 else 0.0

            precs.append(prec)
        return torch.tensor(precs, dtype=torch.float32)
        


    def jac_at_k(self, preds, k, labels):
        """
        Intersection over union (Jaccard similarity).

        Bundle mode (allow_duplicate_items=False):
          Uses set-based Jaccard: |A ∩ B| / |A ∪ B|
        Sequence mode (allow_duplicate_items=True):
          Uses multiset-based Jaccard: sum(min(count)) / sum(max(count))
        """
        jacs = []
        for i in range(len(labels)):
            if self.allow_duplicate_items:
                # Sequence模式：使用多重集合的Jaccard
                from collections import Counter
                pred_list = []
                for j in range(k):
                    pred_list.extend([tuple(x) for x in preds[i][j].tolist()])
                label_list = [tuple(x) for x in labels[i].tolist()]

                pred_counter = Counter(pred_list)
                label_counter = Counter(label_list)

                # 多重集合的交集：sum(min(count_A, count_B))
                all_items = set(pred_counter.keys()) | set(label_counter.keys())
                intersection = sum(min(pred_counter[item], label_counter[item])
                                  for item in all_items)
                # 多重集合的并集：sum(max(count_A, count_B))
                union = sum(max(pred_counter[item], label_counter[item])
                           for item in all_items)

                jsc = intersection / union if union > 0 else 0.0
            else:
                # Bundle模式：使用set的Jaccard
                pred_set = set()
                for j in range(k):
                    pred_set = pred_set | set(tuple(x) for x in preds[i][j].tolist())
                label_set = set(tuple(x) for x in labels[i].tolist())

                jsc = len(pred_set & label_set) / len(pred_set | label_set) if len(pred_set | label_set) > 0 else 0.0

            jacs.append(jsc)
        return torch.tensor(jacs, dtype=torch.float32)

    def hgd_dist(self, pred, label, distance_type='cosine'):
        pred_feat = self.token2feat2(pred)
        label_feat = self.token2feat2(label)
        

        if distance_type == 'cosine':
            pred_norm = pred_feat / pred_feat.norm(dim=1, keepdim=True)
            label_norm = label_feat / label_feat.norm(dim=1, keepdim=True)
            dist_matrix = 1 - pred_norm @ label_norm.T  # [n, n]
            dist_matrix = dist_matrix.cpu().numpy()  
        elif distance_type == 'l2':
            dist_matrix = torch.cdist(pred_feat, label_feat, p=2).cpu().numpy()
        else:
            raise ValueError("Unsupported distance type. Use 'cosine' or 'l2'.")

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        total_dist = dist_matrix[row_ind, col_ind].sum()
        
        return total_dist / len(pred)
    
    def token2feat2(self, item_list):
        """
        For each item list, each item consists of a sequence of tokens, corresponding to its feature.
        """
        result = torch.zeros(item_list.shape[0],self.feature.shape[1],dtype=self.feature.dtype)
        for i in range(item_list.shape[0]):
            result[i] = self.token2feat(item_list[i].tolist())
        return result
    
    def token2feat(self, token_list):
        """
        For each generated token sequence, return the corresponding feature.
        """

        result = torch.zeros(self.feature.shape[1],
                            dtype=self.feature.dtype)
        try:
            idx = int(self.token2id[tuple(token_list)])
            result = self.feature[idx]
        except:
            result = self.tokenizer._token_to_feature(token_list)
            if isinstance(result, np.ndarray):
                result = torch.from_numpy(result)

        return result.to(self.feature.device)

        
    def oas_at_k(self,preds,k,labels,distance_type="cosine"):
        max_list, min_list, mean_list, var_list = [], [], [], []

        for i in range(len(labels)):
            distances = []
            for j in range(k):
                dist = self.hgd_dist(preds[i][j],labels[i],distance_type)               
                distances.append(dist)
            distances = torch.tensor(distances, dtype=torch.float32)
            max_list.append(distances.max())
            min_list.append(distances.min())
            mean_list.append(distances.mean())
            var_list.append(distances.var())

        return {
            "oas_max": torch.tensor(max_list),
            "oas_min": torch.tensor(min_list),
            "oas_mean": torch.tensor(mean_list),
            "oas_var": torch.tensor(var_list)
        }

    
    def to_numpy(self,x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x  
        
    def retrive_item(self,preds,labels):
        """
        besed on preds component semantic id, retrieve the item semantic id.
        1. candidate item pool(len(labels)*(候选倍数+1)): items in labels + randomly sampled items from the dataset
        2. preds component semantic id-->feature, candidate items sid-->feature, nearest neighbor search len(labels) items
        3. searched items' semantic id as preds and return

        Supports candidate set caching for reproducibility across different models.
        """
        # Get n_codebooks for dynamic fallback generation
        n_codebooks = self.tokenizer.n_digit
        codebook_size = self.tokenizer.config['rq_codebook_size']
        expected_token_length = n_codebooks + 1  # n_codebooks RVQ digits + 1 conflict digit

        # Generate fallback token dynamically
        # For 3 codebooks: [1, 257, 513, 769] (4 elements)
        # For 4 codebooks: [1, 257, 513, 769, 1025] (5 elements)
        fallback_token = [i * codebook_size + 1 for i in range(n_codebooks)]
        fallback_token.append(n_codebooks * codebook_size + 1)  # Add conflict digit

        # 第一次调用时，根据labels推断predict_num_items并初始化缓存文件路径
        if self.predict_num_items is None:
            # 从labels的shape推断predict_num_items
            # labels shape: [batch_size, predict_num_items, n_codebooks+1]
            predict_num_items = labels.shape[1]
            self._initialize_candidate_cache(predict_num_items)

        token_values_set = set(self.token2id.values())
        # Match pred_items shape to preds, not labels
        pred_items = np.zeros((preds.shape[0], preds.shape[1], preds.shape[2], preds.shape[3]), dtype=int)

        # Per-position top-K cache for SM@K: sm_topk_cache[i][t] = set of item token tuples
        sm_topk_cache = [[set() for _ in range(preds.shape[2])] for _ in range(len(labels))]

        for i in range(len(labels)):
            label_items = set()
            for j in range(labels.shape[1]):
                label_items.add(self.token2id[tuple(labels[i][j].tolist())])
            remaining_items = token_values_set - label_items

            # 全局样本索引（跨batch）
            global_sample_idx = self.sample_offset + i

            # 候选集生成或加载
            if self.candidate_pool is not None:
                # 使用缓存的候选集
                candidate_items = self.candidate_pool[global_sample_idx]
                candidate_items = [int(id) for id in candidate_items]
            else:
                # 生成新的候选集
                candidate_items = list(label_items) + np.random.choice(
                    list(remaining_items), size=len(labels[i])*self.candidate_multiplier, replace=False).tolist()
                candidate_items = [int(id) for id in candidate_items]
                # 记录到缓冲区，用于后续保存（保存副本，避免后续pop修改）
                self.candidate_buffer.append(candidate_items.copy())

            for m in range(preds.shape[1]):
                for k in range(preds.shape[2]):
                    candidate_feature = self.feature[candidate_items].cpu().numpy()

                    # 尝试直接查找预测的RVQ是否对应真实item
                    rvq_direct_hit = False
                    try:
                        comp_token = self.token2id[tuple(preds[i][m][k].tolist())]
                        feature = self.feature[int(comp_token)] #int()
                        rvq_direct_hit = True
                        self.rvq_direct_hit_count += 1
                    except:
                        # 找不到原item，需要通过RVQ codebook重构embedding
                        feature = self.tokenizer._token_to_feature(preds[i][m][k].tolist()) #decup code is not listed
                        self.rvq_miss_count += 1

                    self.total_predictions += 1
                    
                    similarities = cosine_similarity(
                        self.to_numpy(feature).reshape(1, -1),
                        self.to_numpy(candidate_feature)
                    )

                    similarities = similarities.flatten()

                    # 记录检索质量统计
                    top_sim = similarities.max()
                    self.retrieval_similarities.append(float(top_sim))

                    # 记录top-1和top-2的gap（如果候选数>=2）
                    if len(similarities) >= 2:
                        sorted_sims = np.sort(similarities)[-2:]
                        gap = sorted_sims[1] - sorted_sims[0]
                        self.retrieval_gaps.append(float(gap))

                    top_indices = np.argsort(-similarities)[:1]
                    top_item_id = candidate_items[top_indices[0]]  # Get the top matching item ID

                    # SM@K per-position: store top-sm_topk item tuples (union across beams m)
                    top_sm_indices = np.argsort(-similarities)[:self.sm_topk]
                    for sm_idx in top_sm_indices:
                        iid_sm = candidate_items[sm_idx]
                        try:
                            tpl_sm = tuple(self.id2token[str(int(iid_sm))])
                            sm_topk_cache[i][k].add(tpl_sm)
                        except Exception:
                            pass

                    # Convert item ID to token tuple
                    try:
                        item_tuple = self.id2token[str(top_item_id)]  # id2token uses string keys
                        if len(item_tuple) != expected_token_length:
                            print(f"[ERROR] Invalid item_tuple length: {len(item_tuple)} for item_id {top_item_id}")
                            print(f"  Expected: {expected_token_length}, got: {len(item_tuple)}")
                            print(f"  item_tuple: {item_tuple}")
                            # Use a fallback - create from digits
                            item_tuple = fallback_token
                        # Explicitly convert to numpy array and assign
                        pred_items[i, m, k, :] = np.array(item_tuple, dtype=int)
                    except KeyError:
                        print(f"[ERROR] Item ID {top_item_id} not found in id2token")
                        print(f"  Using fallback tuple: {fallback_token}")
                        pred_items[i, m, k, :] = np.array(fallback_token, dtype=int)
                    except Exception as e:
                        print(f"[ERROR] Unexpected error for item_id {top_item_id}: {e}")
                        print(f"  item_tuple: {item_tuple if 'item_tuple' in locals() else 'N/A'}")
                        pred_items[i, m, k, :] = np.array(fallback_token, dtype=int)

                    # 根据配置决定是否从候选集中移除已选中的item
                    # bundle模式（allow_duplicate_items=False）：不允许重复，需要pop
                    # sequence模式（allow_duplicate_items=True）：允许重复，不pop
                    if not self.allow_duplicate_items:
                        for idx in sorted(top_indices, reverse=True):
                            candidate_items.pop(idx)

        # 更新样本偏移量（用于跨batch追踪全局样本索引）
        self.sample_offset += len(labels)

        # Store per-position top-K for SM@K computation
        self._sm_topk_batch = sm_topk_cache

        return pred_items


    def count_legal(self,preds):
        return

    def save_candidate_cache(self):
        """
        保存生成的候选集到缓存文件，用于后续不同模型的公平对比。
        只在第一次运行（候选集缓存不存在）时调用。
        """
        if self.candidate_pool is not None:
            # 已经使用了缓存，无需保存
            print(f"[INFO] Using cached candidates, no need to save")
            return

        if len(self.candidate_buffer) == 0:
            print(f"[WARNING] No candidates generated, cannot save cache")
            return

        # 准备保存的数据
        cache_data = {
            'metadata': {
                'seed': self.seed,
                'multiplier': self.candidate_multiplier,
                'predict_num_items': self.predict_num_items,
                'dataset': self.config['dataset'],
                'num_samples': len(self.candidate_buffer),
            },
            'candidates': self.candidate_buffer
        }

        # 保存到文件
        os.makedirs(os.path.dirname(self.candidate_cache_file), exist_ok=True)
        with open(self.candidate_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"[INFO] Saved candidate pool to {self.candidate_cache_file}")
        print(f"[INFO] Total samples: {len(self.candidate_buffer)}")
        print(f"[INFO] Candidate set size per sample: {len(self.candidate_buffer[0])} items")
        print(f"[INFO] File size: {os.path.getsize(self.candidate_cache_file) / 1024:.2f} KB")

    def print_rvq_hit_statistics(self):
        """
        打印RVQ直接命中原item的统计信息。

        显示：
        1. 预测的RVQ能直接在token2id中找到原item的比例（直接命中）
        2. 找不到原item需要通过embedding重构的比例（未命中）

        这个统计帮助理解模型预测质量：
        - 高命中率：模型预测的RVQ大多对应真实存在的items
        - 低命中率：模型预测的RVQ很多是"虚拟"的，需要检索最近邻
        """
        if self.total_predictions == 0:
            print(f"\n[RVQ Hit Statistics] No predictions made yet")
            return

        hit_rate = self.rvq_direct_hit_count / self.total_predictions * 100
        miss_rate = self.rvq_miss_count / self.total_predictions * 100

        # 计算RVQ空间覆盖率（用于对比分析）
        total_items = len(self.token2id)
        codebook_size = self.tokenizer.config['rq_codebook_size']
        n_codebooks = self.tokenizer.n_digit
        rvq_space_size = codebook_size ** n_codebooks
        coverage_rate = total_items / rvq_space_size * 100

        # 计算重复率统计
        avg_rvq_dup_per_sample = self.rvq_duplicate_count / max(self.total_samples, 1)
        avg_item_dup_per_sample = self.item_duplicate_count / max(self.total_samples, 1)
        rvq_dup_sample_rate = self.samples_with_rvq_dup / max(self.total_samples, 1) * 100
        item_dup_sample_rate = self.samples_with_item_dup / max(self.total_samples, 1) * 100

        print(f"\n{'='*60}")
        print(f"[RVQ Direct Hit Statistics]")
        print(f"{'='*60}")
        print(f"Total predictions:        {self.total_predictions}")
        print(f"Direct hits (RVQ → item): {self.rvq_direct_hit_count} ({hit_rate:.2f}%)")
        print(f"Misses (need retrieval):  {self.rvq_miss_count} ({miss_rate:.2f}%)")
        print(f"-" * 60)
        print(f"Dataset: {self.config['dataset']}")
        print(f"Total items in dataset:   {total_items}")
        print(f"RVQ space size ({codebook_size}^{n_codebooks}):    {rvq_space_size:,}")
        print(f"RVQ space coverage:       {coverage_rate:.4f}%")
        print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print(f"[Duplicate Rate Statistics]")
        print(f"{'='*60}")
        print(f"Total samples evaluated:  {self.total_samples}")
        print(f"")
        print(f"【RVQ Tokens Level (Model Predictions)】")
        print(f"  Total RVQ duplicates:     {self.rvq_duplicate_count}")
        print(f"  Avg duplicates/sample:    {avg_rvq_dup_per_sample:.4f}")
        print(f"  Samples with RVQ dup:     {self.samples_with_rvq_dup} ({rvq_dup_sample_rate:.2f}%)")
        print(f"")
        print(f"【Item Level (After Retrieval)】")
        print(f"  Total item duplicates:    {self.item_duplicate_count}")
        print(f"  Avg duplicates/sample:    {avg_item_dup_per_sample:.4f}")
        print(f"  Samples with item dup:    {self.samples_with_item_dup} ({item_dup_sample_rate:.2f}%)")
        print(f"")
        print(f"【Retrieval-Induced Duplicates】")
        retrieval_induced = self.item_duplicate_count - self.rvq_duplicate_count
        if retrieval_induced > 0:
            retrieval_induced_rate = retrieval_induced / max(self.item_duplicate_count, 1) * 100
            print(f"  Retrieval-induced dup:    {retrieval_induced} ({retrieval_induced_rate:.2f}%)")
            print(f"  → {retrieval_induced_rate:.1f}% of item duplicates are caused by retrieval")
        else:
            print(f"  No retrieval-induced duplicates detected")
        print(f"")
        print(f"【Interpretation】")
        if self.rvq_duplicate_count == 0:
            print(f"  ✓ Model predictions have NO RVQ-level duplicates")
            print(f"  ✓ All duplicates are caused by retrieval collision")
        else:
            rvq_dup_contribution = self.rvq_duplicate_count / max(self.item_duplicate_count, 1) * 100
            print(f"  ⚠ {rvq_dup_contribution:.1f}% duplicates from model, {100-rvq_dup_contribution:.1f}% from retrieval")
        print(f"{'='*60}\n")

        # 调用新增的统计方法
        self.print_retrieval_quality_stats()
        self.print_rvq_distribution_stats()

    def print_retrieval_quality_stats(self):
        """
        打印检索质量统计信息。

        显示：
        1. Top-1候选的cosine similarity分布
        2. Top-1和Top-2的相似度差距分布

        用于判断检索的质量和置信度。
        """
        if len(self.retrieval_similarities) == 0:
            print(f"\n[Retrieval Quality Statistics] No retrieval data available")
            return

        import numpy as np
        sims = np.array(self.retrieval_similarities)

        print(f"\n{'='*60}")
        print(f"[Retrieval Quality Statistics]")
        print(f"{'='*60}")
        print(f"Top-1 Cosine Similarity:")
        print(f"  Mean:   {sims.mean():.4f}")
        print(f"  Median: {np.median(sims):.4f}")
        print(f"  Std:    {sims.std():.4f}")
        print(f"  Min:    {sims.min():.4f}")
        print(f"  Max:    {sims.max():.4f}")
        print(f"  25th percentile: {np.percentile(sims, 25):.4f}")
        print(f"  75th percentile: {np.percentile(sims, 75):.4f}")

        # 质量分级统计
        print(f"\nSimilarity Distribution:")
        print(f"  Very High (>0.9): {(sims > 0.9).mean()*100:.2f}%")
        print(f"  High (0.8-0.9):   {((sims > 0.8) & (sims <= 0.9)).mean()*100:.2f}%")
        print(f"  Medium (0.6-0.8): {((sims > 0.6) & (sims <= 0.8)).mean()*100:.2f}%")
        print(f"  Low (<0.6):       {(sims <= 0.6).mean()*100:.2f}%")

        if len(self.retrieval_gaps) > 0:
            gaps = np.array(self.retrieval_gaps)
            print(f"\nTop-1 vs Top-2 Gap:")
            print(f"  Mean:   {gaps.mean():.4f}")
            print(f"  Median: {np.median(gaps):.4f}")
            print(f"  Std:    {gaps.std():.4f}")
            print(f"  Min:    {gaps.min():.4f}")
            print(f"  Max:    {gaps.max():.4f}")
            print(f"\nGap Distribution (confidence level):")
            print(f"  Very Small (<0.01):  {(gaps < 0.01).mean()*100:.2f}% # Low confidence")
            print(f"  Small (0.01-0.05):   {((gaps >= 0.01) & (gaps < 0.05)).mean()*100:.2f}%")
            print(f"  Medium (0.05-0.1):   {((gaps >= 0.05) & (gaps < 0.1)).mean()*100:.2f}%")
            print(f"  Large (>0.1):        {(gaps >= 0.1).mean()*100:.2f}% # High confidence")
        print(f"{'='*60}\n")

    def print_rvq_distribution_stats(self):
        """
        打印RVQ Code分布统计信息。

        显示：
        1. 每层codebook的code使用情况
        2. Code分布的均匀度（entropy）
        3. Conflict digit的分布

        用于判断模型是否充分利用RVQ空间。
        """
        if self.predicted_rvq_codes is None or len(self.predicted_rvq_codes[0]) == 0:
            print(f"\n[RVQ Distribution Statistics] No RVQ data available")
            return

        import numpy as np
        from scipy.stats import entropy as scipy_entropy

        print(f"\n{'='*60}")
        print(f"[RVQ Code Distribution Statistics]")
        print(f"{'='*60}")

        codebook_size = self.tokenizer.config['rq_codebook_size']
        n_codebooks = len(self.predicted_rvq_codes)

        for layer_idx in range(n_codebooks):
            codes = np.array(self.predicted_rvq_codes[layer_idx])
            unique_codes = np.unique(codes)
            usage_rate = len(unique_codes) / codebook_size * 100

            # 计算entropy衡量均匀度
            counts = np.bincount(codes, minlength=codebook_size)
            probs = counts / counts.sum()
            code_entropy = scipy_entropy(probs)
            max_entropy = np.log(codebook_size)
            entropy_ratio = code_entropy / max_entropy * 100

            print(f"\nCodebook Layer {layer_idx}:")
            print(f"  Unique codes used: {len(unique_codes)}/{codebook_size} ({usage_rate:.2f}%)")
            print(f"  Entropy: {code_entropy:.2f}/{max_entropy:.2f} ({entropy_ratio:.1f}% of max)")

            # 找出最常用的codes
            top_indices = np.argsort(counts)[-5:][::-1]
            print(f"  Top-5 most used codes:")
            for rank, idx in enumerate(top_indices, 1):
                if counts[idx] > 0:
                    freq_pct = counts[idx] / len(codes) * 100
                    print(f"    #{rank}: code={idx}, count={counts[idx]} ({freq_pct:.2f}%)")

            # 计算使用频率的分布
            used_counts = counts[counts > 0]
            if len(used_counts) > 0:
                print(f"  Usage frequency stats (for used codes):")
                print(f"    Mean:   {used_counts.mean():.1f}")
                print(f"    Median: {np.median(used_counts):.1f}")
                print(f"    Std:    {used_counts.std():.1f}")

        # Conflict digit统计
        conflicts = np.array(self.conflict_digits)
        unique_conflicts = np.unique(conflicts)
        conflict_range = (conflicts.min(), conflicts.max())

        print(f"\nConflict Digit:")
        print(f"  Unique values: {len(unique_conflicts)}")
        print(f"  Value range: [{conflict_range[0]}, {conflict_range[1]}]")

        # Expected range
        expected_min = n_codebooks * codebook_size + 1
        expected_max = (n_codebooks + 1) * codebook_size
        print(f"  Expected range: [{expected_min}, {expected_max}]")

        if conflict_range[0] < expected_min or conflict_range[1] > expected_max:
            print(f"  ⚠ WARNING: Conflict digits outside expected range!")

        print(f"  Value distribution (top-10 most frequent):")
        conflict_counts = np.bincount(conflicts, minlength=expected_max + 1)
        top_conflicts = np.argsort(conflict_counts)[-10:][::-1]
        for rank, c in enumerate(top_conflicts, 1):
            if conflict_counts[c] > 0:
                freq_pct = conflict_counts[c] / len(conflicts) * 100
                print(f"    #{rank}: value={c}, count={conflict_counts[c]} ({freq_pct:.2f}%)")

        print(f"{'='*60}\n")

        
        
    def sm_at_k(self, preds, k, labels):
        """
        Per-position Sequential Match metrics (SM / SH / SN).

        For each ground-truth item y_t at position t, checks whether y_t is among
        the top-sm_topk nearest-neighbor retrievals AT THAT POSITION:

          SH@K = sum_{t=1}^T 1(y_t ∈ TopK_t)    (raw hit count)
          SM@K = SN@K = SH@K / T                  (normalised)

        TopK_t is populated by retrive_item() for each batch.
        """
        sm_list, sh_list, sn_list = [], [], []
        cache = self._sm_topk_batch  # [batch][pos] → set of item token tuples

        for i in range(len(labels)):
            label_list = [tuple(x) for x in labels[i].tolist()]
            T = len(label_list)

            sh = 0
            if cache is not None and i < len(cache):
                for t, y_t in enumerate(label_list):
                    if t < len(cache[i]) and y_t in cache[i][t]:
                        sh += 1

            sm = sh / T if T > 0 else 0.0
            sm_list.append(sm)
            sh_list.append(float(sh))
            sn_list.append(sm)   # SN == SM by definition

        return {
            'sm': torch.tensor(sm_list, dtype=torch.float32),
            'sh': torch.tensor(sh_list, dtype=torch.float32),
            'sn': torch.tensor(sn_list, dtype=torch.float32),
        }

    def calculate_metrics(self, preds, labels):
        """
        Process generated sequences and calculate metrics.

        Expected format of generated sequence (may not start at item boundary):
          [..., d2, d3, BOI, d0, d1, d2, d3, BOI, d0, d1, d2, d3, ..., EOS]
          We need to find the first BOI token and extract from there.

        Expected format of labels (test_gt=True, no BOI):
          [BOS, d0, d1, d2, d3, d0, d1, d2, d3, ..., EOS]
        """
        print(f"\n[Evaluation Sample]")
        print(f"  Preds shape: {preds.shape}")
        print(f"  Model prediction (first sample):")
        print(f"    preds[0, 0, :] = {preds[0,0,:].tolist()}")
        print(f"  Ground truth labels (first sample):")
        print(f"    labels[0, :] = {labels[0,:].tolist()}")

        n_codebooks = self.tokenizer.n_digit
        boi_token = self.tokenizer.boi_token
        eos_token = self.tokenizer.eos_token

        # Process each sample in the batch
        batch_size, max_k, seq_len = preds.shape
        processed_preds = []

        # Calculate tokens_per_item early for trimming logic
        # tokens_per_item = BOI + n_digit RVQ codes + 1 conflict
        # 3 codebooks: 5 tokens/item (BOI, d0, d1, d2, conflict)
        # 4 codebooks: 6 tokens/item (BOI, d0, d1, d2, d3, conflict)
        tokens_per_item = n_codebooks + 2

        for b in range(batch_size):
            for k in range(max_k):
                seq = preds[b, k]

                # Find first BOI token to align to item boundary
                boi_positions = (seq == boi_token).nonzero(as_tuple=True)[0]

                if len(boi_positions) == 0:
                    # No BOI found - skip or use full sequence
                    if b == 0 and k == 0:
                        print(f"  [WARNING] No BOI token in sample [{b},{k}]")
                    start_idx = 0
                else:
                    start_idx = boi_positions[0].item()
                    if b == 0 and k == 0:  # Only print for first sample
                        print(f"  First BOI at position {start_idx}")

                # Extract from first BOI to end (or EOS)
                aligned_seq = seq[start_idx:]

                # Find and remove EOS
                eos_positions = (aligned_seq == eos_token).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    end_idx = eos_positions[0].item()
                    aligned_seq = aligned_seq[:end_idx]  # Exclude EOS

                if b == 0 and k == 0:  # Only print for first sample
                    print(f"  Aligned seq length: {len(aligned_seq)}, first 15 tokens: {aligned_seq[:15].tolist()}")

                # Trim to make divisible by tokens_per_item
                if len(aligned_seq) % tokens_per_item != 0:
                    trim_len = len(aligned_seq) % tokens_per_item
                    aligned_seq = aligned_seq[:-trim_len]

                processed_preds.append(aligned_seq)

        # Stack all processed sequences
        max_len = max(len(s) for s in processed_preds)
        padded_preds = torch.zeros(batch_size, max_k, max_len, dtype=preds.dtype, device=preds.device)
        for idx, seq in enumerate(processed_preds):
            b = idx // max_k
            k = idx % max_k
            padded_preds[b, k, :len(seq)] = seq

        # Reshape into groups of tokens_per_item: [BOI, d0, d1, d2, ...]
        # Already calculated above: tokens_per_item = n_codebooks + 2
        num_items = padded_preds.shape[2] // tokens_per_item
        preds = padded_preds[:, :, :num_items*tokens_per_item].reshape(batch_size, max_k, num_items, tokens_per_item)
        print(f"  After reshape: {preds.shape}, num_items per sample: {num_items}")

        # Remove BOI column (first column) to get n_digit-digit item codes
        preds = preds[:, :, :, 1:]  # Shape: [batch, max_k, num_items, n_digit+1] (n_digit RVQ + 1 conflict)

        # Show extracted RVQ tokens (all items if <= 5, otherwise first 5)
        num_to_print = min(num_items, 5)
        print(f"  First {num_to_print} extracted RVQ tokens:")
        for k in range(num_to_print):
            tuple_k = tuple(preds[0,0,k].tolist())
            print(f"    Item {k}: {tuple_k}")
        if num_items > 5:
            print(f"    ... ({num_items - 5} more items)")

        # Process labels: remove BOS and EOS, then extract items
        # Each item has (n_codebooks + 1) tokens in test_gt format: d0, d1, d2, ..., conflict
        # 3 codebooks: 5 tokens/item (d0,d1,d2,d3_conflict,final_conflict)
        # 4 codebooks: 5 tokens/item (d0,d1,d2,d3,conflict)
        tokens_per_item_label = n_codebooks + 1  # Include conflict digit
        labels_item = labels[:, 1:-1].reshape(labels.shape[0], -1, tokens_per_item_label)

        # Check if labels have duplicate items (for debugging)
        label_tuples = [tuple(labels_item[0, i].tolist()) for i in range(labels_item.shape[1])]
        label_unique = len(set(label_tuples))
        has_label_dup = label_unique < len(label_tuples)
        if has_label_dup:
            from collections import Counter
            label_counter = Counter(label_tuples)
            duplicates = {k: v for k, v in label_counter.items() if v > 1}
            print(f"  [DEBUG] Label has duplicate items: {duplicates}")

        # ========== 统计RVQ tokens级别的重复率（所有样本）==========
        from collections import Counter

        # 初始化RVQ code分布统计（第一次调用时）
        if self.predicted_rvq_codes is None:
            self.predicted_rvq_codes = [[] for _ in range(n_codebooks)]

        for batch_idx in range(preds.shape[0]):  # batch_size
            for k_idx in range(preds.shape[1]):  # max_k (通常是1)
                # 提取当前样本的所有RVQ tokens
                rvq_tuples = [tuple(preds[batch_idx, k_idx, i, :].tolist())
                             for i in range(preds.shape[2])]  # num_items

                # 统计RVQ重复
                rvq_counter = Counter(rvq_tuples)
                num_rvq_duplicates = sum(count - 1 for count in rvq_counter.values() if count > 1)
                self.rvq_duplicate_count += num_rvq_duplicates

                if num_rvq_duplicates > 0:
                    self.samples_with_rvq_dup += 1

                # 收集RVQ code分布统计
                for i in range(preds.shape[2]):  # num_items
                    rvq_tuple = preds[batch_idx, k_idx, i, :].tolist()
                    # 前n_codebooks个是RVQ codes，最后一个是conflict digit
                    for layer_idx in range(n_codebooks):
                        self.predicted_rvq_codes[layer_idx].append(rvq_tuple[layer_idx])
                    self.conflict_digits.append(rvq_tuple[-1])  # 最后一个是conflict

        preds_item = self.retrive_item(preds, labels_item)

        # ========== 统计Item级别的重复率（所有样本）==========
        for batch_idx in range(preds_item.shape[0]):  # batch_size
            for k_idx in range(preds_item.shape[1]):  # max_k
                # 提取当前样本的所有item IDs
                item_ids = []
                for i in range(preds_item.shape[2]):  # num_items
                    rvq_tuple = tuple(preds_item[batch_idx, k_idx, i, :].tolist())
                    try:
                        item_id = self.token2id[rvq_tuple]
                        item_ids.append(item_id)
                    except KeyError:
                        item_ids.append(-1)  # Not found

                # 统计item重复
                item_counter = Counter([iid for iid in item_ids if iid != -1])
                num_item_duplicates = sum(count - 1 for count in item_counter.values() if count > 1)
                self.item_duplicate_count += num_item_duplicates

                if num_item_duplicates > 0:
                    self.samples_with_item_dup += 1

        self.total_samples += preds.shape[0]  # batch_size

        # Print final item IDs for first sample (after retrieval)
        print(f"\n  [Final Item IDs - First Sample]")

        # Check RVQ tokens duplicates (before retrieval)
        rvq_tuples_first_sample = [tuple(preds[0, 0, i, :].tolist()) for i in range(preds.shape[2])]
        rvq_counter_first = Counter(rvq_tuples_first_sample)
        rvq_has_dup_first = any(count > 1 for count in rvq_counter_first.values())

        if rvq_has_dup_first:
            rvq_dups_first = {k: v for k, v in rvq_counter_first.items() if v > 1}
            print(f"  [RVQ Level] Duplicates detected: {len(rvq_dups_first)} unique RVQ(s) repeated")
            for rvq, count in list(rvq_dups_first.items())[:3]:  # Show first 3
                print(f"    RVQ {rvq}: appears {count} times")
        else:
            print(f"  [RVQ Level] ✓ No duplicates")

        # Convert RVQ tokens to item IDs
        pred_item_ids = []
        for i in range(preds_item.shape[2]):  # num_items
            rvq_tuple = tuple(preds_item[0, 0, i, :].tolist())
            try:
                item_id = self.token2id[rvq_tuple]
                pred_item_ids.append(item_id)
            except KeyError:
                pred_item_ids.append(-1)  # Not found

        label_item_ids = []
        for i in range(labels_item.shape[1]):  # num_items
            rvq_tuple = tuple(labels_item[0, i].tolist())
            try:
                item_id = self.token2id[rvq_tuple]
                label_item_ids.append(item_id)
            except KeyError:
                label_item_ids.append(-1)

        print(f"  Predicted items: {pred_item_ids}")
        print(f"  Ground truth:    {label_item_ids}")

        # Check duplicates in predictions and labels
        from collections import Counter
        pred_counter = Counter(pred_item_ids)
        label_counter = Counter(label_item_ids)

        pred_has_dup = any(count > 1 for count in pred_counter.values())
        label_has_dup = any(count > 1 for count in label_counter.values())

        if pred_has_dup:
            pred_dups = {k: v for k, v in pred_counter.items() if v > 1}
            print(f"  [Item Level] ✓ Prediction has duplicates: {pred_dups}")
        else:
            print(f"  [Item Level] ✗ Prediction has NO duplicates")

        # Show if duplicates are retrieval-induced
        if not rvq_has_dup_first and pred_has_dup:
            print(f"  → Duplicates are retrieval-induced (RVQ tokens were unique)")

        if label_has_dup:
            label_dups = {k: v for k, v in label_counter.items() if v > 1}
            print(f"  ✓ Label has duplicates: {label_dups}")
        else:
            print(f"  ✗ Label has NO duplicates")

        results = {}
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                metric_func = self.metric2func[metric]
                result = metric_func(preds_item, k, labels_item)

                if isinstance(result, dict):
                    for sub_metric, tensor in result.items():
                        results[f"{sub_metric}@{k}"] = tensor
                else:
                    results[f"{metric}@{k}"] = result

        return results

