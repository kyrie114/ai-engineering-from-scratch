# 终端与 Shell（Terminal & Shell）

> 终端是 AI 工程师安身立命的地方。要让自己在这里待得自在。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~35 minutes

## 学习目标（Learning Objectives）

- 在命令行中使用管道、重定向和 `grep` 来过滤、处理训练日志
- 创建带多个窗格的持久 tmux 会话，同时进行训练和 GPU 监控
- 用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统与 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地与远程机器之间传输文件

## 问题所在（The Problem）

你花在终端里的时间会比花在任何编辑器上的都多。训练运行、GPU 监控、日志跟踪、远程 SSH 会话、环境管理——每个 AI 工作流都离不开 shell。如果你在这里动作慢，那你在哪儿都慢。

本课只讲对 AI 工作真正重要的终端技能。不讲 Unix 历史，也不深挖 Bash 脚本，只给你需要的东西。

## 核心概念（The Concept）

```mermaid
graph TD
    subgraph tmux["tmux 会话：训练"]
        subgraph top["上排"]
            P1["窗格 1：训练运行<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["窗格 2：GPU 监控<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["窗格 3：日志 + 实验<br/>tail -f logs/train.log | grep loss"]
    end
```

三件事同时在跑，只用一个终端。你可以分离（detach）、回家、再通过 SSH 连回来重新接入（reattach），而训练会一直跑着。

```figure
s0-shell-pipeline
```

## 动手构建（Build It）

### 第 1 步：了解你用的 shell（Step 1: Know your shell）

查看你正在运行的是哪个 shell：

```bash
echo $SHELL
```

大多数系统使用 `bash` 或 `zsh`，两者都没问题，本课程中的命令在哪种里都能用。

需要掌握的关键操作：

```bash
# Move around
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# History search (most useful shortcut you'll learn)
# Ctrl+R then type part of a previous command
# Press Ctrl+R again to cycle through matches

# Clear terminal
clear   # or Ctrl+L

# Cancel a running command
# Ctrl+C

# Suspend a running command (resume with fg)
# Ctrl+Z
```

### 第 2 步：管道与重定向（Step 2: Piping and redirects）

管道（pipe）把命令首尾相连。处理日志、过滤输出、串联工具，靠的都是它，你会不停地用到。

```bash
# Count how many times "loss" appears in a log
cat train.log | grep "loss" | wc -l

# Extract just the loss values from training output
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# Watch a log file update in real time, filtering for errors
tail -f train.log | grep --line-buffered "ERROR"

# Sort experiments by final accuracy
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# Redirect stdout and stderr to separate files
python train.py > output.log 2> errors.log

# Redirect both to the same file
python train.py > train_full.log 2>&1
```

你需要掌握的三种重定向：

| 符号 | 作用 |
|--------|-------------|
| `>` | 将 stdout 写入文件（覆盖） |
| `>>` | 将 stdout 追加到文件 |
| `2>` | 将 stderr 写入文件 |
| `2>&1` | 把 stderr 发送到与 stdout 相同的地方 |
| `\|` | 把一个命令的 stdout 作为 stdin 送给下一个命令 |

### 第 3 步：后台进程（Step 3: Background processes）

训练一跑就是几个小时，你不会想把终端一直开着的。

```bash
# Run in background (output still goes to terminal)
python train.py &

# Run in background, immune to hangup (closing terminal won't kill it)
nohup python train.py > train.log 2>&1 &

# Check what's running in background
jobs
ps aux | grep train.py

# Bring a background job to foreground
fg %1

# Kill a background process
kill %1
# or find its PID and kill that
kill $(pgrep -f "train.py")
```

`&`、`nohup` 与 `screen`/`tmux` 的区别：

| 方式 | 关闭终端后还能存活吗？ | 能否重新接入？ |
|--------|-------------------------|---------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（去看日志文件） |
| `screen` / `tmux` | 是 | 是 |

凡是超过几分钟的任务，都用 tmux。

### 第 4 步：tmux（Step 4: tmux）

tmux 能创建带多个窗格的持久终端会话。这是管理训练运行时最管用的一个工具。

```bash
# Install
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# Start a named session
tmux new -s training

# Split horizontally
# Ctrl+B then "

# Split vertically
# Ctrl+B then %

# Navigate between panes
# Ctrl+B then arrow keys

# Detach (session keeps running)
# Ctrl+B then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill a session
tmux kill-session -t training
```

一个典型的 AI 工作流会话：

```bash
tmux new -s train

# Pane 1: start training
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " to split, then run GPU monitor
watch -n1 nvidia-smi

# Ctrl+B, % to split vertically, tail the logs
tail -f logs/experiment.log

# Now detach with Ctrl+B, d
# SSH out, go get coffee, come back
# tmux attach -t train
```

### 第 5 步：用 htop 和 nvtop 监控（Step 5: Monitoring with htop and nvtop）

```bash
# System processes (better than top)
htop

# GPU processes (if you have NVIDIA GPU)
# Install: sudo apt install nvtop (Ubuntu) or brew install nvtop (macOS)
nvtop

# Quick GPU check without nvtop
nvidia-smi

# Watch GPU usage update every second
watch -n1 nvidia-smi

# See which processes are using the GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

你会用到的 `htop` 快捷键：

- `F6` 或 `>`：按列排序（按内存排序可以找出内存泄漏）
- `F5`：切换树状视图（查看子进程）
- `F9`：终止进程
- `/`：搜索进程名

### 第 6 步：用 SSH 连接远程 GPU 机器（Step 6: SSH for remote GPU boxes）

租用云 GPU（Lambda、RunPod、Vast.ai）时，你需要通过 SSH 连上去。

```bash
# Basic connection
ssh user@gpu-box-ip

# With a specific key
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# Copy files to remote
scp model.pt user@gpu-box-ip:~/models/

# Copy files from remote
scp user@gpu-box-ip:~/results/metrics.json ./

# Sync a whole directory (faster for many files)
rsync -avz ./data/ user@gpu-box-ip:~/data/

# Port forward (access remote Jupyter/TensorBoard locally)
ssh -L 8888:localhost:8888 user@gpu-box-ip
# Now open localhost:8888 in your browser

# SSH config for convenience
# Add to ~/.ssh/config:
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# Then just:
# ssh gpu
```

### 第 7 步：AI 工作常用的别名（Step 7: Useful aliases for AI work）

把这些加到你的 `~/.bashrc` 或 `~/.zshrc` 里：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

也可以只挑需要的复制过去。几个关键的别名（alias）：

```bash
# GPU status at a glance
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Kill all Python training processes
alias killtraining='pkill -f "python.*train"'

# Quick virtual environment activate
alias ae='source .venv/bin/activate'

# Watch training loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

完整的集合见 `code/shell_aliases.sh`。

### 第 8 步：常见的 AI 终端操作模式（Step 8: Common AI terminal patterns）

下面这些模式在实践中反复出现：

```bash
# Run training, log everything, notify when done
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# Compare two experiment logs side by side
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# Find the largest model files (clean up disk space)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# Download a model from Hugging Face
wget https://huggingface.co/model/resolve/main/model.safetensors

# Untar a dataset
tar xzf dataset.tar.gz -C ./data/

# Count lines in all Python files (see how big your project is)
find . -name "*.py" | xargs wc -l | tail -1

# Check disk space (training data fills disks fast)
df -h
du -sh ./data/*

# Environment variable check before training
env | grep -i cuda
env | grep -i torch
```

## 实际运用（Use It）

本课程中，每个工具会在这些场景派上用场：

| 工具 | 使用时机 |
|------|----------------|
| tmux | 每次训练运行（Phase 3 及以后） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 快速后台任务 |
| `htop` / `nvtop` | 排查训练缓慢、OOM 错误 |
| SSH + `rsync` | 在云 GPU 上工作 |
| 管道 + 重定向 | 处理实验结果 |
| 别名 | 为重复命令节省时间 |

## 练习（Exercises）

1. 安装 tmux，创建一个带三个窗格的会话，在第一个窗格里运行 `htop`，第二个运行 `watch -n1 date`，第三个运行一个 Python 脚本。然后分离会话，再重新接入。
2. 把 `code/shell_aliases.sh` 中的别名加进你的 shell 配置，然后用 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 生成一份假的训练日志，然后用 `grep`、`tail` 和 `awk` 只提取损失值。
4. 为你有权访问的一台服务器配置一个 SSH config 条目（也可以用 `localhost` 来练习语法）。

## 关键术语（Key Terms）

| 术语 | 人们口中的说法 | 实际含义 |
|------|----------------|----------------------|
| Shell | "终端" | 解释你输入的命令的程序（bash、zsh、fish） |
| tmux | "终端复用器" | 让你在一个窗口里运行多个终端会话、并可分离/重新接入的程序 |
| 管道 | "那根竖线" | `\|` 运算符，把一个命令的输出作为输入传给另一个命令 |
| PID | "进程 ID" | 分配给每个运行中进程的唯一编号，用来监控或终止它 |
| nohup | "不挂断" | 以免疫于挂断信号的方式运行命令，因此关闭终端不会杀掉它 |
| SSH | "连服务器" | Secure Shell，一种在远程机器上运行命令的加密协议 |
