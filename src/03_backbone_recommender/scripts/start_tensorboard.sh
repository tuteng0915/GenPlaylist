#!/bin/bash

# TensorBoard 启动脚本
# 用于启动 TensorBoard 服务，查看训练指标

# 设置 TensorBoard 日志目录
LOGDIR="/home/sjj/wenhao/DISCO/outputs"

echo "启动 TensorBoard..."
echo "日志目录: $LOGDIR"
echo "访问地址: http://localhost:6006"
echo ""
echo "如需远程访问，请在本地电脑运行:"
echo "  ssh -L 6006:localhost:6006 sjj@lab-3"
echo "  然后在浏览器打开: http://localhost:6006"
echo ""

# 启动 TensorBoard
tensorboard --logdir="$LOGDIR" --port=6006 --bind_all
