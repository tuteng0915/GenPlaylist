# GenPlaylist Working Manual

This manual is for students working on the **GenPlaylist** project. The goal of this stage is to help everyone become comfortable with GPU servers, RunPod, basic development tools, and the VibeMus codebase, since GenPlaylist may reuse part of the VibeMus implementation.

At this stage, your task is to run the **VibeMus Gradio demo** on a RunPod GPU Pod. You are expected to understand the basic workflow, record problems, and build enough familiarity to later contribute to GenPlaylist.

The main goals are:

1. Understand basic remote server usage.
2. Learn essential tools: Linux command line, Git, Conda, Pip, CUDA, and PyTorch.
3. Learn how to create, use, and stop a RunPod GPU Pod.
4. Register and configure a DashScope API key.
5. Clone the VibeMus repository and run its Gradio demo.

---

# Part 1. Basic Knowledge: Server, Git, Conda, and Pip

## 1.1 What Is a Remote Server?

A remote server is a computer that you access over the internet. It usually has stronger CPU, GPU, memory, and storage resources than a personal laptop.

In this project, we use **RunPod GPU Pods** as remote GPU servers.

You can think of a Pod as:

> A temporary GPU machine that you rent, connect to, use for experiments, and stop after finishing your work.

Important rules:

* A running Pod costs money.
* Closing the browser does not stop the Pod.
* Disconnecting the terminal does not stop the Pod.
* You must manually stop the Pod from the RunPod website after finishing your work.
* Important files should be saved under `/workspace`.
* Temporary folders may be lost when the Pod is stopped, restarted, or recreated.

---

## 1.2 Common Linux Commands

After connecting to a server, you will usually work in a terminal.

Check the current directory:

```bash
pwd
```

List files:

```bash
ls
```

List files with details:

```bash
ls -lh
```

Enter a directory:

```bash
cd folder_name
```

Go back to the parent directory:

```bash
cd ..
```

Create a directory:

```bash
mkdir folder_name
```

Check disk space:

```bash
df -h
```

Check the size of a directory:

```bash
du -sh folder_name
```

Check GPU status:

```bash
nvidia-smi
```

Check Python version:

```bash
python --version
```

Check the current Python path:

```bash
which python
```

---

## 1.3 What Is Git?

Git is a version control tool. We use Git to download code from GitHub, update code, and track code versions.

The VibeMus repository is:

```text
https://github.com/tuteng0915/VibeMus
```

Clone the repository:

```bash
git clone https://github.com/tuteng0915/VibeMus.git
```

Enter the project directory:

```bash
cd VibeMus
```

Check the current branch:

```bash
git branch
```

Pull the latest code:

```bash
git pull
```

Check local changes:

```bash
git status
```

Show recent commits:

```bash
git log --oneline -n 5
```

Record the current commit hash:

```bash
git rev-parse HEAD
```

Every experiment report should include the Git commit hash. This helps us reproduce your result later.

---

## 1.4 What Is Conda?

Conda is a Python environment manager. Different projects may need different Python versions and packages, so we usually create one environment for each project.

Create a Conda environment:

```bash
conda create -n genplaylist python=3.10 -y
```

Activate the environment:

```bash
conda activate genplaylist
```

Deactivate the environment:

```bash
conda deactivate
```

List all Conda environments:

```bash
conda env list
```

List installed packages:

```bash
conda list
```

If the RunPod image does not include Conda, you may use the system Python and Pip for this stage. The priority is to run the demo successfully and understand the workflow.

---

## 1.5 What Is Pip?

Pip is a Python package manager. We use it to install project dependencies.

Install packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Install one package:

```bash
pip install package_name
```

Check whether a package is installed:

```bash
pip show package_name
```

List installed packages:

```bash
pip list
```

Deep learning projects often have dependency issues related to PyTorch, CUDA, transformers, Gradio, FFmpeg, and model-specific libraries. If installation fails, record the full command and the full error message.

---

## 1.6 What Are GPU, CUDA, and PyTorch?

A GPU is the main computing device for deep learning. CUDA is NVIDIA’s GPU computing platform. PyTorch is the deep learning framework used by many research projects.

Check whether the server has a GPU:

```bash
nvidia-smi
```

Check whether PyTorch can use the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the output includes:

```text
True
```

then PyTorch can use the GPU.

If the output is:

```text
False
```

then the current Python environment cannot access the GPU. You need to check the PyTorch version, CUDA version, or selected RunPod image.

---

## 1.7 Where Should Files Be Saved?

On RunPod, important files should be saved under:

```text
/workspace
```

Recommended structure:

```text
/workspace/
  VibeMus/
  data/
  checkpoints/
  outputs/
  logs/
  .cache/
```

Avoid saving important files only under:

```text
/tmp
/root
random download folders
browser cache folders
```

These locations may be temporary or difficult to recover.

---

# Part 2. Using RunPod

## 2.1 What Is RunPod?

RunPod is a platform for renting GPU servers. In this project, we use **RunPod Pods**.

A Pod is:

> A GPU machine that you manually create, connect to, use, and stop.

Important points:

* A running Pod continuously costs money.
* PLZ stop the Pod after finishing your work.
* Storage may still cost money even after the Pod is stopped.
* Do not create expensive GPUs without permission.
* Do not change team billing settings.
* Do not delete other students’ Pods, Volumes, Templates, or files.

---

## 2.2 Using the Team Account

We will use the mentor’s RunPod team account.

Rules:

1. Do not modify billing information.
2. Do not delete other students’ resources.
3. Use your own name when naming Pods.
4. Use GPUs under **$1/hour** for this stage.
5. Prefer a single **A40** GPU if available.
6. Do not put API keys, passwords, or tokens into GitHub.

Pod naming format:

```text
<your-name>-*
```

The prefix must be your own name. The rest of the name is flexible.

---

## 2.3 Choosing a GPU

For this stage, choose a GPU with a price below:

```text
$1/hour
```

Preferred GPU:

```text
A40 single GPU
```

Other acceptable choices:

| GPU        | Use Case                                         |
| ---------- | ------------------------------------------------ |
| A40        | Preferred choice for this stage                  |
| RTX 4090   | Good for debugging if the price is below $1/hour |
| RTX A6000  | Good if available under $1/hour                  |
| L40 / L40S | Good if available under $1/hour                  |

Avoid using the following without permission:

| GPU                  | Reason                           |
| -------------------- | -------------------------------- |
| A100                 | More expensive                   |
| H100                 | Very expensive                   |
| Multi-GPU Pods       | Easy to waste budget             |
| Any GPU over $1/hour | Outside the current project rule |

Before deploying the Pod, always check the hourly price.

---

## 2.4 Creating a Pod

Go to RunPod and create a new Pod.

Use the following settings:

```text
GPU: Prefer A40 single GPU, below $1/hour
Template/Image: PyTorch 2.1 + CUDA 11.8
Volume mount path: /workspace
Pod name: <your-name>-*
```

RunPod usually provides a recommended PyTorch image. For this stage, use the image with:

```text
PyTorch 2.1
CUDA 11.8
```

After the Pod starts, open the Web Terminal.

---

## 2.5 Using `/workspace`

`/workspace` is the main working directory. Use it for code, outputs, caches, and logs.

Enter `/workspace`:

```bash
cd /workspace
```

Recommended layout:

```text
/workspace/
  VibeMus/
  outputs/
  logs/
  .cache/
```

Clone code under `/workspace`:

```bash
cd /workspace
git clone https://github.com/tuteng0915/VibeMus.git
cd VibeMus
```

---

## 2.6 DashScope / Qwen API Key

VibeMus uses Qwen-related services through **DashScope**. DashScope is Alibaba Cloud’s model service platform. You need a DashScope API key so that the code can call the required LLM service.

You should register your own DashScope account and create an API key.

General steps:

1. Open the DashScope website.
2. Register or log in with your account.
3. Enter the API key management page.
4. Create a new API key.
5. Copy the key.
6. Save it privately.

In the server terminal, set the key as an environment variable:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

Check whether it is set:

```bash
echo $DASHSCOPE_API_KEY
```

Expected result:

```text
The terminal prints your key or part of your key.
```

Important security rules:

* Do not upload your API key to GitHub.
* Do not write your API key into screenshots.
* Do not send your API key in public group chats.
* Do not hard-code your API key into Python files.
* If your key is leaked, delete it and create a new one.


Contact me if you have an issue during get a api-key.\

---

## 2.7 Connecting to the Pod

For this stage, use:

```text
Web Terminal
```

You may explore VSCode Remote SSH by yourself if you are comfortable with SSH, keys, and remote development.

Recommended priority:

1. Web Terminal for this assignment.
2. VSCode Remote SSH for later development.
3. JupyterLab only if you need notebook-style exploration.

---

## 2.8 Stopping the Pod

After finishing your experiment, stop the Pod from the RunPod website.

Remember:

* Closing the browser does not stop the Pod.
* Closing the terminal does not stop the Pod.
* Logging out does not stop the Pod.
* You must stop it from the RunPod console.

---

# Part 3. Running the VibeMus Gradio Demo

The goal of this part is:

> Run the VibeMus Gradio demo on RunPod and generate one audio sample.

Although the project you are joining is **GenPlaylist**, this stage uses **VibeMus** as a reference project. Some code patterns, UI structures, and generation logic may later be reused in GenPlaylist.

Complete the following tasks one by one.

---

## Task 0. Start a GPU Pod

### Operation

Create a RunPod Pod with:

```text
GPU: A40 single GPU preferred
Template/Image: PyTorch 2.1 + CUDA 11.8
Volume mount path: /workspace
Pod name: <your-name>-*
```

Open the Web Terminal.

### Verification

Run:

```bash
nvidia-smi
```

You should see GPU information, including GPU name, memory, and driver version.

---

## Task 1. Check Python and PyTorch

### Operation

Run:

```bash
python --version
```

Then run:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Verification

Expected information:

```text
Python version: preferably 3.10
PyTorch version: 2.1.x
CUDA available: True
```

If `torch.cuda.is_available()` is `False`, record the error and your current environment.

---

## Task 2. Clone the VibeMus Repository

### Operation

Go to `/workspace`:

```bash
cd /workspace
```

Clone the repository:

```bash
git clone https://github.com/tuteng0915/VibeMus.git
```

Enter the project:

```bash
cd VibeMus
```

List files:

```bash
ls -lh
```

Record the commit:

```bash
git rev-parse HEAD
```

### Verification

You should see files such as:

```text
main.py
requirements.txt
README.md
backend_server.py
```

Record:

```text
Repo path:
/workspace/VibeMus

Git commit:
<commit hash>
```

---

## Task 3. Read the Project Entry Files

### Operation

Read the first part of the README:

```bash
sed -n '1,200p' README.md
```

List project files:

```bash
ls -lh
```

Focus on:

```text
main.py
requirements.txt
backend_server.py
pipeline.py
assistant.py
```

For this assignment, the most important file is:

```text
main.py
```

```text
1. Which file starts the Gradio demo?
2. What is the role of requirements.txt?
3. What is the difference between main.py and backend_server.py?
4. Which files may be useful for GenPlaylist in the future?
```

---

## Task 4. Check FFmpeg

VibeMus may need FFmpeg for audio processing.

### Operation

Check whether FFmpeg is installed:

```bash
ffmpeg -version
```

If it is missing, install it:

```bash
apt-get update
apt-get install -y ffmpeg
```

If you do not have permission to use `apt-get`, record the error and ask for help.

### Verification

Run:

```bash
ffmpeg -version
```

---

## Task 5. Install Python Dependencies

### Operation

Enter the project directory:

```bash
cd /workspace/VibeMus
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the README specifies additional steps, follow the README.

### Verification

Check Gradio:

```bash
python -c "import gradio; print('gradio ok')"
```

Check PyTorch again:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If installation fails, record:

```text
Failed command:
Error message:
What you tried:
Current status:
```

---

## Task 6. Register and Set DashScope API Key

### Operation

Register a DashScope account and create your own API key.

Then set it in the terminal:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

Check:

```bash
echo $DASHSCOPE_API_KEY
```

Do not include the actual API key in your report.

---

## Task 7. Launch the Gradio Demo

### Operation

Enter the project directory:

```bash
cd /workspace/VibeMus
```

Start the demo:

```bash
python main.py
```

If Gradio starts successfully, you should see information similar to:

```text
Running on local URL: http://127.0.0.1:7860
```

RunPod may expose the Gradio page through an HTTP service.

### Verification

Open the Gradio page from the RunPod interface.

If the page cannot be opened, check whether the terminal still shows that `python main.py` is running.

---

## Task 8. Generate One Audio Sample

### Operation

In the Gradio page, enter a simple prompt, for example:

```text
Generate a 30-second cheerful electronic music piece for a short travel video.
```

Another example:

```text
Create calm piano background music for studying.
```

Click the generate button and wait.

The first run may be slow because the system may download model weights or initialize large models.

### Verification

You need to verify:

1. The Gradio page does not crash.
2. The terminal does not show an unresolved fatal error.
3. One playable audio sample is generated.
4. You can locate the output file.

Find audio files:

```bash
find /workspace/VibeMus -name "*.wav" -o -name "*.mp3" -o -name "*.flac"
```

##

---

# Common Problems

## Problem 1. `torch.cuda.is_available()` returns `False`

Possible causes:

* The current Python environment uses a CPU-only PyTorch build.
* The selected image does not match the GPU/CUDA setup.
* The Pod was created without GPU access.
* The wrong Python environment is active.

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Record the full output.

---

## Problem 2. `ffmpeg: command not found`

Install FFmpeg:

```bash
apt-get update
apt-get install -y ffmpeg
```

Then check:

```bash
ffmpeg -version
```

---

## Problem 3. Gradio Page Cannot Be Opened

Check:

1. Is `python main.py` still running?
2. Does the terminal show a Gradio URL?
3. Is port 7860 exposed by RunPod?
4. Does the terminal show any error?

Check port usage:

```bash
ss -tulnp | grep 7860
```

If `ss` is unavailable, try:

```bash
netstat -tulnp | grep 7860
```

---

## Problem 4. DashScope API Key Error

Check:

```bash
echo $DASHSCOPE_API_KEY
```

If it is empty, set it again:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

Then restart:

```bash
python main.py
```

---

## Problem 5. First Run Is Very Slow

The first run may download model weights, initialize models, or build caches.

To keep caches under `/workspace`, you may set:

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
```

Then restart the demo:

```bash
python main.py
```

---

# Final Checklist

Before completing this stage, make sure you can do the following:

* [ ] Log in to RunPod.
* [ ] Create your own GPU Pod.
* [ ] Choose a GPU below $1/hour.
* [ ] Prefer an A40 single GPU when available.
* [ ] Name the Pod with `<your-name>-*`.
* [ ] Open Web Terminal.
* [ ] Use `nvidia-smi` to check GPU status.
* [ ] Clone the VibeMus repository.
* [ ] Install dependencies.
* [ ] Register a DashScope account.
* [ ] Set `DASHSCOPE_API_KEY`.
* [ ] Run `python main.py`.
* [ ] Open the Gradio page.
* [ ] Generate one audio sample.
* [ ] Find the generated output file.
* [ ] Submit the setup report.
* [ ] Stop the Pod.

---

# Next Stage

After everyone successfully runs the VibeMus Gradio demo, we will move to the next stage of GenPlaylist:

1. Identify which parts of VibeMus can be reused.
2. Build a standard Docker image for GenPlaylist.
3. Create a shared RunPod workflow.
4. Start implementing GenPlaylist-specific modules.
5. Run future experiments based on the standard environment.
