export CUDA_VISIBLE_DEVICES=0,1 #,1,2,3,4,5,6,7
export PYTHONPATH=/home/sjj/wenhao/DISCO:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1

# Generate unique run name with timestamp to avoid conflicts
RUN_NAME="ddbc-spotify30-$(date +%Y%m%d-%H%M%S)"

 python /home/sjj/wenhao/DISCO/main.py\
  loader.batch_size=300 \
  loader.eval_batch_size=128 \
  model=small \
  data=spotify \
  run_name=${RUN_NAME} \
  parameterization=subs \
  model.length=152 \
  eval.compute_generative_perplexity=False \
  sampling.steps=25




