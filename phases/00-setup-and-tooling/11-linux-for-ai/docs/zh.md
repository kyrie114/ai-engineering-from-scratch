# 面向 AI 的 Linux（Linux for AI）

> 大多数 AI 工作负载都跑在 Linux 上。你只需要懂到不被卡住的程度。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 minutes

## 学习目标（Learning Objectives）

- 在命令行中浏览 Linux 文件系统，并完成基本的文件操作
- 用 `chmod` 和 `chown` 管理文件权限，解决 "Permission denied" 错误
- 用 `apt` 安装系统软件包，把一台全新的 GPU 机器配置好用于 AI 工作
- 识别从 macOS 迁移到 Linux 时最容易让远程机器开发者踩坑的差异

## 问题所在（The Problem）

你在 macOS 或 Windows 上开发。但一旦 SSH 进一台云 GPU 机器、租一台 Lambda 实例，或者启动一台 EC2 机器，你就落在了 Ubuntu 里。终端是你唯一的界面：没有 Finder，没有资源管理器，没有 GUI。如果你无法在命令行下浏览文件系统、安装软件包、管理进程，你就只能一边搜索 "how to unzip a file in Linux"，一边为闲置的 GPU 小时买单。

这是一份生存指南，只覆盖在远程 Linux 机器上做 AI 工作所必需的操作，仅此而已。

## 文件系统布局（File System Layout）

Linux 把一切都组织在同一个根目录 `/` 之下。没有 `C:\`，也没有 `/Volumes`。你真正会打交道的目录：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>你的文件 —— 克隆仓库、跑训练"]
    root --> tmp["tmp/<br/>临时文件，重启后清空"]
    root --> usr["usr/<br/>系统程序和库"]
    root --> etc["etc/<br/>配置文件"]
    root --> varlog["var/log/<br/>日志 —— 出问题时来这里查"]
    root --> mnt["mnt/ 或 /media/<br/>外接硬盘和卷"]
    root --> proc["proc/ 与 /sys/<br/>虚拟文件 —— 内核与硬件信息"]
```

你的主目录是 `~` 或 `/home/your-username`。你做的几乎所有事情都发生在这里。

## 核心命令（Essential Commands）

这 15 条命令覆盖了你在远程 GPU 机器上 95% 的日常操作。

### 目录导航（Moving Around）

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### 文件与目录（Files and Directories）

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` 是永久删除，没有撤销。按回车之前，先把路径核对一遍。

### 读取文件（Reading Files）

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### 搜索（Searching）

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## 权限（Permissions）

Linux 中每个文件都有所有者（owner）和权限位（permission bits）。当脚本无法执行、或你无法写入某个目录时，你就会和它们打交道。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

常见的修复方法：

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

当看到 "Permission denied" 时，几乎都是权限问题。`chmod +x` 或 `sudo` 能解决大多数情况。

## 包管理（Package Management (apt)）

Ubuntu 使用 `apt`，安装系统级软件就靠它。

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

在一台全新的 GPU 机器上，你通常会安装这些软件包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 用户与 sudo（Users and sudo）

你通常以普通用户身份登录，而有些操作需要 root（管理员）权限。

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

在云 GPU 实例上，你一般是唯一的用户，而且已经拥有 sudo 权限。不要什么都用 root 跑，只在需要时用 sudo。

## 进程与 systemd（Processes and systemd）

当训练卡住，或者你需要查看哪些进程在运行时：

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd 管理各种服务（即后台守护进程 daemon）。如果你要运行推理服务器，就会用到它：

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## 磁盘空间（Disk Space）

GPU 机器的磁盘空间往往有限，模型和数据集很快就能把它塞满。

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见的省空间手段：

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## 网络（Networking）

你会在命令行里下载模型、传输文件、调用 API。

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

大文件传输请选 `rsync` 而不是 `scp`：它只传输有变化的字节，还能应对中断的连接。

## tmux：让会话保持存活（tmux: Keep Sessions Alive）

当你 SSH 到远程机器上时，合上笔记本就会杀掉你的训练。tmux 可以避免这一点。

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

长时间训练任务永远放进 tmux 里跑。永远如此。

## 面向 Windows 用户的 WSL2（WSL2 for Windows Users）

如果你在用 Windows，WSL2 让你无需双系统启动就能拥有一个真正的 Linux 环境。

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行的是真正的 Linux 内核，本课的所有内容在它里面都适用。从 WSL 内部看，你的 Windows 文件位于 `/mnt/c/Users/YourName/`。

只要 Windows 侧装好了 NVIDIA 驱动，GPU 直通（passthrough）就能用。安装 Windows 版的 NVIDIA 驱动（不是 Linux 版的），CUDA 就能在 WSL2 里使用。

## 坑点：从 macOS 到 Linux（Gotchas: macOS to Linux）

如果你是从 macOS 过来的，这些东西会让你栽跟头：

| macOS | Linux | 说明 |
|-------|-------|-------|
| `brew install` | `sudo apt install` | 包名有时不同。`brew install htop` 对应 `sudo apt install htop` 没问题，但 `brew install readline` 对应的是 `sudo apt install libreadline-dev`，不能直接照搬。 |
| `open file.txt` | `xdg-open file.txt` | 但远程机器上没有 GUI，用 `cat` 或 `less` 就好。 |
| `pbcopy` / `pbpaste` | 不可用 | 通过 SSH 无法与剪贴板互传。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认用 zsh，大多数 Linux 服务器用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`、`/usr/local/bin/` | 可执行文件存放的位置不同。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS 的 sed 要求在 `-i` 后面跟一个空字符串，Linux 不需要。 |
| 大小写不敏感的文件系统 | 大小写敏感的文件系统 | 在 Linux 上，`Model.py` 和 `model.py` 是两个不同的文件。 |
| 行尾符 `\n` | 行尾符 `\n` | 相同。但 Windows 用 `\r\n`，会破坏 bash 脚本。运行 `dos2unix` 修复。 |

## 速查表（Quick Reference Card）

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

```figure
s0-process-fork
```

## 练习（Exercises）

1. SSH 到任意一台 Linux 机器（或打开 WSL2），进入你的主目录。创建一个项目文件夹，用 `touch` 在里面创建三个空文件，再用 `ls -la` 列出它们。
2. 用 apt 安装 `htop`，运行它，找出占用内存最多的进程。
3. 启动一个 tmux 会话，在里面运行 `sleep 300`，分离会话，列出会话，再重新接入。
4. 用 `df -h` 查看可用磁盘空间，再用 `du -sh ~/.cache/*` 看看缓存里是什么在占空间。
5. 用 `scp` 把一个文件从本地传到远程，再用 `rsync` 做同样的传输，比较两者的体验。
