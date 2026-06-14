"""
main.py - DISCO项目的主入口文件
功能：用于训练、评估离散扩散模型来生成产品bundle
支持四种模式：train(训练)、rec_eval(推荐评估)、ppl_eval(困惑度评估)、sample_eval(采样评估)
"""

# ============ 标准库导入 ============
import os  # 操作系统接口，用于文件路径等操作
import numpy as np  # 数值计算库，用于数组操作
import time  # 时间相关操作
import json  # JSON数据处理
import warnings  # 警告控制
from collections import defaultdict, OrderedDict  # 特殊字典类型，用于结果统计

# ============ 第三方库导入 ============
import fsspec  # 文件系统规范库，用于跨平台文件系统操作
import hydra  # 配置管理框架，用于管理实验配置参数
import lightning as L  # PyTorch Lightning，简化深度学习训练流程的框架
import omegaconf  # 配置管理库，与hydra配合使用
import rich.syntax  # 终端美化库（用于彩色输出）
import rich.tree  # 树状结构显示
import torch  # PyTorch深度学习框架
from torch.utils.data import DataLoader  # PyTorch数据加载器
from tqdm import tqdm  # 进度条显示库
from safetensors.torch import load_file  # 安全的模型权重加载库

# ============ 项目内部模块导入 ============
import dataloader  # 数据加载模块，负责数据预处理和批量加载
import diffusion  # 扩散模型核心模块，定义了扩散过程和模型结构
from evaluator import Evaluator  # 评估器，计算推荐指标（recall、precision等）
import utils  # 工具函数集合
from dataset import AbstractDataset  # 抽象数据集类，负责加载原始数据


# ============ HuggingFace Dataset包装器 ============
class TorchDatasetWrapper(torch.utils.data.Dataset):
    """
    将HuggingFace Dataset包装为PyTorch Dataset
    这样Lightning可以正确应用DistributedSampler
    """
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

# 忽略警告信息
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存


# ============ OmegaConf自定义解析器注册 ============
# OmegaConf是配置管理库，这里注册自定义解析器，允许在配置文件中使用动态值

# 注册'cwd'解析器：在配置中可以使用${cwd:}获取当前工作目录
omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)

# 注册'device_count'解析器：在配置中可以使用${device_count:}获取可用GPU数量
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)

# 注册'eval'解析器：在配置中可以使用${eval:}执行Python表达式
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)

# 注册'div_up'解析器：向上取整除法，例如${div_up:10,3}返回4
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  """
  从检查点加载预训练模型

  Args:
      config: 配置对象，包含模型架构和检查点路径等信息
      tokenizer: 分词器对象，用于处理token

  Returns:
      加载好的扩散模型实例
  """
  # 如果使用HuggingFace模型，直接创建新模型实例并移到CUDA设备
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  # 否则从指定的检查点路径加载已训练的模型
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only  # 装饰器：仅在主进程（rank 0）执行，用于分布式训练
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  """
  打印训练和验证数据批次的样例（调试用）

  Args:
      train_ds: 训练数据加载器
      valid_ds: 验证数据加载器
      tokenizer: 分词器
      k: 打印前k个和后k个token
  """
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')

    batch = next(iter(dl))  # 获取一个批次的数据
    first = batch['input_ids'][0, :k]  # 第一个样本的前k个token
    last = batch['input_ids'][0, -k:]  # 第一个样本的后k个token
    print('ids:', first)
    print('ids:', last)


    # 词汇表结构示例（当使用RQ-VAE时，codebook_size=128, n_codebooks=3）：
    #     0: bos (序列开始标记)
    #     1-128: RQ第1层数字
    #     129-256: RQ第2层数字
    #     257-384: RQ第3层数字
    #     385-512: RQ第4层数字（避免冲突用）
    #     513: boi (物品开始标记)
    #     514: eos (序列结束标记)




def generate_samples(config, logger, tokenizer, tokenized_datasets):
  """
  生成样本评估模式（sample_eval）
  使用训练好的扩散模型生成bundle样本，并计算生成困惑度

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_datasets: 已分词的数据集

  Returns:
      text_samples: 生成的文本样本列表
  """
  logger.info('Generating samples.')

  # 加载训练好的模型
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()  # 重置生成困惑度指标

  # 如果配置禁用EMA（指数移动平均），则将模型的EMA设为None
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  # 获取采样配置参数
  stride_length = config.sampling.stride_length  # 每次生成的步长
  num_strides = config.sampling.num_strides  # 步数

  # 循环生成多个批次的样本
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      # 半自回归采样方法：分步生成序列
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)  # dt是时间步长
      text_samples = intermediate_samples[-1]  # 获取最后一步的生成结果
      # 注意：使用半自回归方法生成的样本包含大量<|endoftext|>标记，
      # 在计算生成困惑度前需要预处理，
      # 因为diffusion.compute_generative_perplexity()会丢弃第一个EOS后的所有文本
    else:
      # 标准DDPM采样方法：一次性生成完整序列
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)  # 将token解码为文本
      model.compute_generative_perplexity(text_samples)  # 计算困惑度

  # 打印生成的样本和困惑度
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _rec_eval(config, logger, tokenizer, tokenized_dataset):
  """
  推荐系统评估模式（rec_eval）
  使用模型生成bundle推荐，并计算推荐指标（Recall、Precision、Hit Rate、Jaccard、OAS等）

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_dataset: 已分词的数据集

  Returns:
      output_results: 包含各项评估指标的有序字典
  """
  logger.info('Starting RecSys Evaluation.')

  # 加载训练好的模型和评估器
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  evaluator = Evaluator(config['evaluator'], tokenizer)

  # 如果禁用EMA
  if config.eval.disable_ema:
      logger.info('Disabling EMA.')
      model.ema = None

  # 创建测试数据加载器
  test_ds = DataLoader(
      tokenized_dataset['test'],
      batch_size=config['eval_batch_size'],
      shuffle=False,  # 不打乱顺序，保证结果可复现
      collate_fn=tokenizer.collate_fn['test']
  )

  model.eval()  # 设置模型为评估模式（关闭dropout等）
  all_results = defaultdict(list)  # 存储所有批次的评估结果

  # 不计算梯度
  with torch.no_grad():
    for batch in tqdm(test_ds, desc="Evaluating", ncols=100):
      input_ids = batch['input_ids']  # 输入的bundle前半部分
      labels = batch.get('labels')  # 标签是bundle的后半部分

      # CFG: extract context_emb if guidance is enabled
      cfg_enabled = getattr(config.sampling, 'cfg_enabled', False)
      context_emb = None
      if cfg_enabled:
        context_emb = batch.get('context_emb', None)
        if context_emb is not None:
          context_emb = context_emb.to(next(model.parameters()).device).float()

      # 设置生成参数
      # stride_length应该根据实际需要生成的长度计算
      # labels是test_gt=True格式（无BOI）：[BOS, d0,d1,d2,d3, ..., EOS]
      # 但生成时需要BOI格式：[BOS, BOI,d0,d1,d2,d3, ..., EOS]
      # 所以需要转换：labels长度 + (num_items * 1) 因为每个item多一个BOI
      # 注意：labels的长度已经由tokenizer根据predict_num_items/predict_ratio/default(0.5)决定

      # 动态计算：去掉BOS和EOS
      # labels中每个item = n_digit个RVQ codes + 1个conflict digit
      # 3 codebooks: 4 tokens/item (d0,d1,d2,conflict), 4 codebooks: 5 tokens/item (d0,d1,d2,d3,conflict)
      tokens_per_item_in_labels = tokenizer.n_digit + 1  # +1 for conflict digit
      num_items = (labels.shape[1] - 2) // tokens_per_item_in_labels

      # stride_length = BOS + (num_items × tokens_per_item) + EOS
      # tokens_per_item = BOI + n_digit个RVQ codes + 1个conflict = n_digit + 2
      # 3 codebooks: 5 tokens/item, 4 codebooks: 6 tokens/item
      tokens_per_item = tokenizer.n_digit + 2
      stride_length = 1 + num_items * tokens_per_item + 1

      # 确保stride_length符合模式（余数应为2，即BOS+EOS）
      # 3 codebooks: stride_length % 5 == 2, 4 codebooks: stride_length % 6 == 2
      expected_remainder = 2
      if stride_length % tokens_per_item != expected_remainder:
        # 调整到最接近的合法长度
        stride_length = (stride_length // tokens_per_item) * tokens_per_item + expected_remainder

      # DEBUG: 只在第一个batch打印配置信息
      if len(all_results) == 0:
        print(f"\n[Rec Eval Config]")
        # 打印预测配置
        predict_num_items = config.eval.get('predict_num_items', None) if hasattr(config, 'eval') else None
        predict_ratio = config.eval.get('predict_ratio', None) if hasattr(config, 'eval') else None
        if predict_num_items is not None:
            print(f"  Prediction mode: predict_num_items = {predict_num_items} (fixed item count)")
        elif predict_ratio is not None:
            print(f"  Prediction mode: predict_ratio = {predict_ratio} (ratio of total)")
        else:
            print(f"  Prediction mode: default (0.5, predicting half)")
        print(f"  labels.shape: {labels.shape}")
        print(f"  labels[0, :]: {labels[0, :].tolist()}")
        print(f"  input_ids.shape: {input_ids.shape}")
        print(f"  input_ids[0, :]: {input_ids[0, :].tolist()}")
        print(f"  num_items to generate: {num_items}")
        print(f"  tokens_per_item: {tokens_per_item} (BOI + {tokenizer.n_digit} RVQ digits + 1 conflict)")
        print(f"  calculated stride_length: {stride_length}")
        print(f"  stride_length % {tokens_per_item} = {stride_length % tokens_per_item} (should be {expected_remainder})")


      num_strides = 1  # 生成步数
      max_k = max(config['evaluator']['topk'])  # 取topk中的最大值（例如topk=[10,20,50]则max_k=50）
      # 为每个输入生成max_k个候选bundle
      text_samples = torch.zeros((input_ids.shape[0], max_k, stride_length*num_strides), dtype=torch.long)

      # 对每个输入生成max_k个不同的候选bundle
      for i in range(max_k):
        # 使用半自回归采样生成bundle
        _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
            input_ids=input_ids,  # 给定的前半部分作为条件
            stride_length=stride_length,
            num_strides=num_strides,
            dt=1 / config.sampling.steps,
            context_emb=context_emb,  # CFG context embedding (None if cfg_enabled=False)
        )

        gen_seq = torch.tensor(intermediate_samples[-1])  # 获取生成的序列
        text_samples[:, i, :] = gen_seq  # 保存第i个候选

      # 计算该批次的推荐指标
      result = evaluator.calculate_metrics(text_samples, labels)
      for key, value in result.items():
          all_results[key].append(value)

  # 汇总所有批次的结果
  output_results = OrderedDict()

  for key, values in all_results.items():
    if isinstance(values, list):
        vals = []
        # 将所有批次的值展平
        for v in values:
            if torch.is_tensor(v):
                if v.numel() == 1:
                    vals.append(v.item())
                else:
                    vals.extend(v.detach().cpu().numpy().tolist())
            else:
                vals.append(float(v))
        # 计算平均值
        mean_val = float(torch.tensor(vals).mean().item())
        output_results[key] = round(mean_val, 4)

    elif torch.is_tensor(values):
        if values.numel() == 1:
            output_results[key] = round(values.item(), 4)
        else:
            output_results[key] = round(values.detach().cpu().numpy().mean().item(), 4)

    else:
        output_results[key] = round(float(values), 4)

  print("output_results", output_results)

  # 打印RVQ直接命中统计
  evaluator.print_rvq_hit_statistics()

  # 保存候选集缓存（仅在第一次运行时保存）
  evaluator.save_candidate_cache()

  return output_results

def _ppl_eval(config, logger, tokenizer, tokenized_dataset):
  """
  困惑度评估模式（ppl_eval）
  计算模型在测试集上的困惑度（perplexity），用于衡量模型生成质量

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_dataset: 已分词的数据集
  """
  logger.info('Starting Zero Shot Eval.')

  # 加载训练好的模型
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  # 配置日志记录器列表（支持多个logger并行运行）
  loggers = []

  # 添加 TensorBoard logger（本地实时可视化）
  if config.get('use_tensorboard', False):
    from lightning.pytorch.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(
      save_dir=os.getcwd(),
      name='tensorboard',
      version=config.get('run_name', 'default'),
      log_graph=False,  # 禁用计算图记录
      default_hp_metric=False  # 禁用 hyperparameter 指标
    )
    loggers.append(tb_logger)
    logger.info('TensorBoard logger initialized')

  # 实例化Lightning回调（如checkpoint保存、early stopping等）
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # 创建Lightning Trainer（负责管理训练/验证循环）
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),  # 分布式训练策略
    logger=loggers if len(loggers) > 0 else None)

  # 创建测试数据加载器
  test_ds = DataLoader(
    tokenized_dataset['test'],
    batch_size=config['eval_batch_size'],
    shuffle=False,
    collate_fn=tokenizer.collate_fn['test'])

  # _, valid_ds = dataloader.get_dataloaders(
  #   config, tokenizer, skip_train=True, valid_seed=config.seed)

  # 在测试集上验证模型，计算困惑度
  trainer.validate(model, test_ds)  # valid_ds


def _train(config, logger, tokenizer, tokenized_dataset, trainer=None):
  """
  训练模式（train）
  使用PyTorch Lightning训练离散扩散模型

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_dataset: 已分词的数据集
      trainer: 可选的自定义trainer
  """
  logger.info('Starting Training.')

  # 配置日志记录器列表（支持多个logger并行运行）
  loggers = []

  # 添加 TensorBoard logger（本地实时可视化）
  if config.get('use_tensorboard', False):
    from lightning.pytorch.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(
      save_dir=os.getcwd(),
      name='tensorboard',
      version=config.get('run_name', 'default'),
      log_graph=False,  # 禁用计算图记录
      default_hp_metric=False  # 禁用 hyperparameter 指标
    )
    loggers.append(tb_logger)
    logger.info('TensorBoard logger initialized')

  # 检查是否需要从检查点恢复训练
  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None  # 从头开始训练

  # 实例化Lightning回调（checkpoint保存、early stopping、学习率监控等）
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # print(callbacks)

  # 创建训练数据加载器
  # 将HuggingFace Dataset包装为torch Dataset，使Lightning可以正确应用DistributedSampler
  train_batch_size = config.get('loader', {}).get('batch_size', config.get('train_batch_size', 64))
  eval_batch_size = config.get('loader', {}).get('eval_batch_size', config.get('eval_batch_size', 32))

  logger.info(f'Creating DataLoaders: train_batch_size={train_batch_size}, eval_batch_size={eval_batch_size}')
  logger.info(f'Train dataset type: {type(tokenized_dataset["train"])}, length: {len(tokenized_dataset["train"])}')

  # 包装HuggingFace Dataset为torch Dataset
  train_dataset_wrapped = TorchDatasetWrapper(tokenized_dataset['train'])
  valid_dataset_wrapped = TorchDatasetWrapper(tokenized_dataset['valid'])

  logger.info(f'Wrapped train dataset type: {type(train_dataset_wrapped)}')

  train_ds = DataLoader(
      train_dataset_wrapped,  # 使用包装后的dataset
      batch_size=train_batch_size,
      shuffle=True,  # Lightning DDP会自动替换为DistributedSampler
      collate_fn=tokenizer.collate_fn['train'],
      num_workers=0,
      pin_memory=True,
      persistent_workers=False
  )

  # 创建验证数据加载器
  valid_ds = DataLoader(
      valid_dataset_wrapped,  # 使用包装后的dataset
      batch_size=eval_batch_size,
      shuffle=False,
      collate_fn=tokenizer.collate_fn['val'],
      num_workers=0,
      pin_memory=True,
      persistent_workers=False
  )

  # train_ds, valid_ds = dataloader.get_dataloaders(
    # config, tokenizer)
  # _print_batch(train_ds, valid_ds, tokenizer)

  # 创建扩散模型实例
  # model = _load_from_checkpoint(config, tokenizer)
  model = diffusion.Diffusion(
    config, tokenizer)  # tokenizer=valid_ds

  # 如果需要加载预训练权重（已注释）
  # model.load_state_dict(
  #   load_file(ckpt_path),strict=False)

  # 创建Lightning Trainer
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),  # 分布式训练策略（DDP等）
    logger=loggers if len(loggers) > 0 else None)

  # 开始训练（自动处理训练循环、验证、checkpoint保存等）
  # 如果ckpt_path不为None，会自动从checkpoint恢复
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

  # 以下是旧版本的trainer代码（已注释）
  # Trainer
  # if trainer is not None:
  #   trainer = trainer
  # else:
  #   trainer = get_trainer("MDLM")(config, model, tokenizer) #
  # trainer.fit(train_dataloader, val_dataloader)
    



@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """
  主入口函数
  使用Hydra装饰器管理配置，自动从configs/目录加载config.yaml

  整体流程：
  1. 设置随机种子
  2. 加载原始数据集（train/valid/test）
  3. 使用RQ-VAE将物品编码为离散token
  4. 根据mode参数执行不同任务：
     - train: 训练扩散模型
     - rec_eval: 评估推荐指标
     - ppl_eval: 评估困惑度
     - sample_eval: 生成样本

  Args:
      config: Hydra配置对象，包含所有实验参数
  """
  # 设置随机种子，确保实验可复现
  L.seed_everything(config.seed)

  # 获取日志记录器
  logger = utils.get_logger(__name__)

  # ============ 第1步：加载数据集 ============
  # 从datasets/{dataset_name}/目录加载train.txt、valid.txt、test.txt
  dataset = AbstractDataset(config)
  split_datasets = dataset.split()  # 返回包含train/valid/test的字典

  # ============ 第2步：初始化分词器 ============
  # 分词器负责将bundle转换为token序列
  # 使用RQ-VAE将物品嵌入向量量化为离散码本索引
  tokenizer = dataloader.get_tokenizer(config, dataset)

  # ============ 第3步：对数据集进行分词 ============
  # 根据cir（components-to-items ratio）参数选择不同的分词策略：
  # if config['cir'] == 'none': #不使用RQ-VAE，直接使用物品ID
  #   tokenized_datasets = tokenizer.raw_tokenize(split_datasets)

  # elif config['cir'] == 1: # 不将物品转换为组件，一个物品对应一个token序列
  tokenized_datasets = tokenizer.tokenize(split_datasets)

  # else: # cir为其他值（如3、5、10、15），将多个物品组合为一个组件
    # tokenized_datasets = tokenizer.transfor_tokenzie(split_datasets)
    # 例如：对于spotify数据集，将物品序列转换为组件序列以压缩表示

  # ============ 第4步：根据mode执行相应任务 ============
  if config.mode == 'sample_eval':
    # 生成样本模式：生成新的bundle并计算生成困惑度
    generate_samples(config, logger, tokenizer, tokenized_datasets)

  elif config.mode == 'ppl_eval':
    # 困惑度评估模式：计算模型在测试集上的困惑度
    _ppl_eval(config, logger, tokenizer, tokenized_datasets)

  elif config.mode == 'rec_eval':
    # 推荐评估模式：给定bundle前半部分，生成后半部分，计算推荐指标
    _rec_eval(config, logger, tokenizer, tokenized_datasets)

  else:
    # 默认为训练模式：训练离散扩散模型
    _train(config, logger, tokenizer, tokenized_datasets)


if __name__ == '__main__':
  # Python脚本入口点
  # 执行main函数，Hydra会自动解析命令行参数和配置文件
  main()


