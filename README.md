# LoRA训练操作手册

> 凡是写成方括号的地方（如 `\[用户名]`、`\[IP]`），替换成实际值。
> 任何一步的结果和手册里写的「期望输出」不一样，先停下来，把命令和报错原文截图发给我，不要自己改代码、不要自己升级软件包。
>  遇到问题先看后面常见翻车清单，初步排查。

\---

## 0 先看懂这件事

* **目标**：在一台 RTX 5090（Ubuntu）上，用已经准备好的数据集，微调一个机器人视觉-语言-动作模型，一共跑 20000 步。
* **时间预算**：

|阶段|大致耗时|
|-|-|
|克隆代码（1.6MB）+ 克隆数据集（720MB）（第 3、5 步）|5～20 分钟（看网速）|
|装 Python 环境（第 4 步）|10～40 分钟（看网络，已装好则跳过）|
|算归一化统计（第 6 步）|5～15 分钟|
|下载 π0.5 基础权重（第 7 步自动触发）|20 分钟～1 小时以上（约 10 GB）|
|**正式训练 20000 步（第 7 步）**|**约 30～60 小时（1.5～2.5 天）**|

* **规范要求**：

  * 整个过程**不要关机、不要关 tmux**；
  * **不要**自己 `pip install`、不要用 conda、不要 `sudo apt install cuda`、不要升级/降级 jax、torch；
  * 所有目录都用英文名，不要建中文目录；
  * 这台机器如果别人也在用，请先用 `nvidia-smi` 看一眼有没有别人的进程在占卡。

\---

## 1 第 0 步：环境体检

### 1.1 登录

```bash
ssh \[用户名]@\[5090的IP]
```

进来之后，先确认自己的Home目录够用（命令里的 `\~` 就是你的Home目录）。

### 1.2 六条体检命令（一条一条复制运行）

**① 看显卡和显卡驱动**

```bash
nvidia-smi
```

* 期望：能打出表格，右上角有 `CUDA Version: 12.8` 或更高，中间能看到 `NVIDIA GeForce RTX 5090` 和 `32607MiB`（32GB）。
* 如果报 `command not found`：这台机器环境不对。

**② 看系统版本**

```bash
cat /etc/os-release | head -3
uname -r
```

* 期望：`Ubuntu 22.04`。
* 24.04 也能用，但出了怪问题要先讨论。

**③ 看常用工具在不在**

```bash
uv --version ; git --version ; which gsutil || echo "gsutil 未装" ; which tmux || echo "tmux 未装"
```

* 期望：`uv` 和 `git` 有版本号；`gsutil 未装` / `tmux 未装` 是正常的（后面按需装）。
* `uv` 没有 → 去第 2 步装。

**④ 测能不能连 GitHub（第 3 步克隆代码、第 4 步装环境都要用）**

```bash
curl -sI --max-time 15 https://github.com | head -1
```

* 期望：`HTTP/2 200` 或 `HTTP/1.1 200`。
* 连不上 → 记下来，第 3 步和「翻车清单-坑 2」里有绕行办法。

**⑤ 测能不能连 Google 云存储（要下 10GB 基础权重）**

```bash
curl -sI --max-time 15 https://storage.googleapis.com/openpi-assets/ | head -1
```

* 期望：`HTTP/2 200`。
* 连不上 → **这是最关键的一项**，出问题先讨论。

**⑥ 看磁盘空间**

```bash
df -h \~
```

* 期望：`Avail` 一列 ≥ 60G（数据集 0.8G + 环境 5G + 基础权重 10G + 训练中间结果 20～40G）。

### 1.3 体检结果对照表

|检查项|正常的样子|不正常去哪|
|-|-|-|
|① nvidia-smi|有 RTX 5090 / CUDA ≥ 12.8||
|② 系统|Ubuntu 22.04|告诉我实际版本|
|③ uv|有版本号|第 2 步|
|④ GitHub|HTTP 200|第 3.2 节 / 翻车坑 2|
|⑤ Google 云存储|HTTP 200|翻车坑 3|
|⑥ 磁盘|≥ 60G 可用|翻车坑 9|

**六项都正常 → 继续第 2 步。**

\---

## 2 第 2 步：装 uv（如果体检③已经有了就跳过）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

* 期望：最后打印 `uv 0.9.x`（版本号不重要）。
* 如果 `source` 那行报错，改用：

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

* 如果 `curl` 拉不下来，改用 pip 装：

```bash
pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

\---

## 3 第 3 步：放代码

**代码不再发压缩包了**，改成从组里的 GitHub 仓库直接克隆（一条命令，不用解压）。

### 3.1 一条命令把代码克隆到你的Home目录

```bash
git clone https://github.com/Aftermath/pi0-5/piper-pi05-lora.git \~/openpi
```

* 组里的 GitHub 组织固定叫 `pi0-5`，两个仓库都挂在它下面：**代码**是 `piper-pi05-lora`（这一步），**数据集**是 `piper-pi05-data`（第 5 步）。所以这条命令直接照抄，不用替换任何东西。
* 期望结尾：`Receiving objects: 100% (…), done.`，然后命令提示符回来。
* ⚠️ **不要**加 `--recurse-submodules`（仓库里有 2 个我们用不到的仿真子模块，加了会白拉 500MB）。

### 3.2确认代码到了

```bash
ls \~/openpi
```

* 期望：能看到 `pyproject.toml`、`uv.lock`、`README.md`、`src`、`scripts` 这些东西。
* **注意**：**没有 `third\_party` 文件夹**是**正常的**（我们剔掉了 900MB 用不到的内容）。
* `\~/openpi` 不存在、或者是空的 → 说明克隆没成功，把屏幕原文发我。

### 3\.3 确认我们改的代码在里面（很关键）

```bash
grep -n "LeRobotPiperDataConfig" \~/openpi/src/openpi/training/config.py | head -3
grep -n "pi05\_piper\_lora" \~/openpi/src/openpi/training/config.py | head -3
```

* 期望：第一条能打出 `class LeRobotPiperDataConfig(...)`，第二条能打出 `name="pi05\_piper\_lora",`。
* **一条都打不出来** → 代码包不对，停下来讨论。

\---

## 4 第 4 步：装 Python 环境

### 4.1 开始装

```bash
cd \~/openpi
export UV\_DEFAULT\_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
GIT\_LFS\_SKIP\_SMUDGE=1 uv sync
GIT\_LFS\_SKIP\_SMUDGE=1 uv pip install -e .
```

说明三件事：

1. `uv sync` 会按仓库里的 `uv.lock` 装**锁定版本**（datasets 3.6.0、jax 0.5.3、torch 2.7.1 等）。**这是故意的，千万不要自己改**。
2. `UV\_DEFAULT\_INDEX` 那行是把下载源换成清华镜像，国内快很多；如果这台机器直连 PyPI 很快，可以不加。
3. 这个命令会下载 3～5 GB 的软件包，跑到最后会打印类似 `Installed 180 packages in 240s` 或 `Audited 180 packages in 120ms`。
4. 如果中途报 `github.com ... failed to fetch` 之类的错，**先别慌**，看「翻车清单-坑 2」，加两行配置再重跑即可，前面下好的东西不会白下。

**期望结尾**：出现 `Installed ... packages` 或 `Audited ... packages`，命令提示符回来。

### 4.2 装完之后自检（两条，都必须在 `\~/openpi` 目录下跑）

```bash
cd \~/openpi
uv run python -c "import jax; print('JAX backend:', jax.default\_backend()); print('devices:', jax.devices())"
```

* 期望：`JAX backend: gpu` 和 `\[CudaDevice(id=0)]`。
* 如果打印 `cpu` 或 `\[]` → 翻车清单-坑 4，把完整输出发我，**不要自己动 jax 版本**。

```bash
uv run python -c "import torch, datasets; print('torch', torch.\_\_version\_\_); print('datasets', datasets.\_\_version\_\_)"
```

* 期望：`torch 2.7.1` 和 `datasets 3.6.0`。
* 这两个版本号必须一字不差，不一样就是环境装歪了。

### 4.3 一句话纪律

后面**所有** Python 命令都必须用 `uv run python ...` 或 `uv run scripts/...` 的形式，并且当前目录必须是 `\~/openpi`。
**不要**直接用 `python xxx.py`，也不要用 `conda activate`——那样会绕开锁定版本，必然踩坑。

\---

## 5 第 5 步：放数据集

**数据集也不再发压缩包了**，跟代码一样从 GitHub 组织 `pi0-5` 里克隆（720MB，比代码慢，耐心等）。

### 5.1 一条命令把数据集克隆到它该待的位置

```bash
mkdir -p \~/.cache/huggingface/lerobot/zhiku
git clone https://github.com/pi0-5/piper-pi05-data.git \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place
```

* 期望结尾同样是 `Receiving objects: 100% (…), done.`，720MB 会传一会儿，**中途别关终端**。
* 中断了就重跑同一条命令（git 会自动续上，或 `rm -rf` 掉半截的目录再重来）。

### 5.2 确认数据到了、大小对得上

```bash
du -sh \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place
ls \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place
```

* 期望：第一条约 `1.5G`（工作区 720MB + git 自带的 `.git` 又是约 720MB）；第二条能看到 `data` 和 `meta` 两个文件夹。
* 目录不存在、或者 `data`/`meta` 都没有 → 说明克隆没成功，**先和我讨论**，别继续往下做。
* 💡 **省空间小贴士**：如果以后不会再更新数据集，可以让 `.git` 走掉，省回 720MB：

```bash
  rm -rf \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place/.git
  ```

### 5.3 核对数据集信息

```bash
cat \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place/meta/info.json | head -12
```

* 期望看到：`"total\_episodes": 100`、`"total\_frames": 11067`、`"fps": 10`。

### 5.4 让 lerobot 真的读一遍（**这步最关键，能提前发现 99% 的问题**）

```bash
cd \~/openpi
uv run python -c "
from lerobot.common.datasets.lerobot\_dataset import LeRobotDataset
d = LeRobotDataset('zhiku/piper\_pick\_place')
print('frames =', len(d))
print('episodes =', d.meta.info\['total\_episodes'])
print('fps =', d.fps)
"
```

* 期望输出正好这三行：

```
frames = 11067
episodes = 100
fps = 10
```

* 报 `FileNotFoundError` / `RepositoryNotFoundError` / 404 → 数据集放错位置了，见翻车清单-坑 6。

### 5.5 （可选）Home目录空间不够时的替代做法

如果第 0 步体检发现家目录可用空间不到 60G，把数据集改克隆到你机器上的大盘（例如 `/data/\[你的名字]/lerobot/`）。需要先建好目录，再克隆到那里：

```bash
mkdir -p /data/\[你的名字]/lerobot/zhiku
git clone https://github.com/pi0-5/piper-pi05-data.git /data/\[你的名字]/lerobot/zhiku/piper\_pick\_place
export HF\_LEROBOT\_HOME=/data/\[你的名字]/lerobot
```

⚠️ 用这种方案的话，**之后每开一个新终端都要先重新执行那行 `export HF\_LEROBOT\_HOME=...`**，否则第 5、6 步都会找不到数据。

\---

## 6 第 6 步：算归一化统计（5～15 分钟）

> 这一步是「先统计一下数据的取值范围」，训练时要靠它把数据标准化。不做这步，训练起来第一步就会报
> `Normalization stats not found. Make sure to run scripts/compute\_norm\_stats.py`。

```bash
cd \~/openpi
uv run scripts/compute\_norm\_stats.py --config-name=pi05\_piper\_lora
```

* 期望看到进度条 `Computing stats: 100%|...| 1383/1383`，最后一行是：

```
Writing stats to: /home/\[你的用户名]/openpi/assets/pi05\_piper\_lora/zhiku/piper\_pick\_place
```

* 确认文件真的生成了：

```bash
ls -lh \~/openpi/assets/pi05\_piper\_lora/zhiku/piper\_pick\_place/norm\_stats.json
```

⚠️ **两条铁律**：这条命令必须在 `\~/openpi` 目录下执行；将来训练也必须在 `\~/openpi` 目录下执行。因为统计文件是写在「当前目录的 `./assets/` 下」的，换目录就跑丢。

\---

## 7 第 7 步：开训练（然后就是等）

### 7.1 开一个 tmux 会话（防止 SSH 一断训练就没了）

```bash
tmux new -s piper
```

* 如果报 `tmux: command not found`：先试 `sudo apt install -y tmux`；没有 sudo 权限就跳到 7.4 用 nohup 方案。
* 进了 tmux 之后，你会看到一个绿色/灰色的底栏。这时候再往下跑命令。
* **中途想离开但让训练继续跑**：按 `Ctrl+B`，松手，再按 `D`（detach）。
* **以后回来接着看**：`tmux attach -t piper`。

### 7.2 在 tmux 里启动训练

```bash
cd \~/openpi
export XLA\_PYTHON\_CLIENT\_MEM\_FRACTION=0.9
mkdir -p logs
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --wandb-enabled=False 2>\&1 | tee logs/train\_piper1.log
```

参数只有三个，含义是：

|参数|含义|
|-|-|
|`pi05\_piper\_lora`|用哪个训练配置（就是我们改好的那个）|
|`--exp-name=piper1`|这次实验的名字，结果会存到 `\~/openpi/checkpoints/pi05\_piper\_lora/piper1/`|
|`--wandb-enabled=False`|关掉在线训练监控。这台机器登不上那个网站，不关会卡在「要求输入 API key」|
|`XLA\_PYTHON\_CLIENT\_MEM\_FRACTION=0.9`|让框架最多用 90% 显存|

### 7.3 头 30 分钟看着像「卡住」，其实都正常

按顺序会出现这几件事：

1. `Downloading gs://openpi-assets/checkpoints/pi05\_base/params to /home/.../.cache/openpi/...`
→ 正在下载 π0.5 基础权重，约 10 GB。**这是最慢的一步**，网慢可能 1 小时以上。
想看进度的话，另开一个终端跑：`du -sh \~/.cache/openpi`（这个数字在涨就是正常的）。
2. `Initialized data loader` / `Initialized train state` → 在读数据、搭模型。
3. 进度条出现，但**前几十步特别慢**（甚至几分钟不动）→ 这是编译器在预热，正常。

**训练真的跑起来之后，屏幕上会反复出现两种东西：**

每 50 步一行的日志：

```
Step 3200: loss=1.2345, grad\_norm=0.9876, param\_norm=12.3456
```

底部持续刷新的进度条：

```
 3200/20000 \[1:20:33<5:46:12,  1.24s/it]
```

怎么读这行进度条：

|位置|含义|
|-|-|
|`3200/20000`|已经跑到第 3200 步，总共 20000 步|
|`1:20:33`|已经跑了 1 小时 20 分|
|`5:46:12`|**预计还要 5 小时 46 分**|
|`1.24s/it`|每一步实际耗时（用它就能自己算总时长）|

### 7.4 没有 tmux 的备用方案：nohup 后台跑

```bash
cd \~/openpi
export XLA\_PYTHON\_CLIENT\_MEM\_FRACTION=0.9
mkdir -p logs
nohup uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --wandb-enabled=False > logs/train\_piper1.log 2>\&1 \&
echo $! > logs/train\_piper1.pid
tail -f logs/train\_piper1.log
```

* `tail -f` 里按 `Ctrl+C` **只是停止看日志，训练还在后台跑**，不用担心。
* 断开 SSH 再连回来，这样确认还活着：

```bash
cd \~/openpi
ps -p $(cat logs/train\_piper1.pid)
tail -n 30 logs/train\_piper1.log
```

\---

## 8 第 8 步：时长预估 / 中途停机 / 断点续跑

### 8.1 本次训练要多久

本次条件：**100 条轨迹样本 / 11067 帧 / 批量大小 8 / 共 20000 步**。

|每步耗时|20000 步总时长|
|-|-|
|5 秒|约 28 小时|
|6 秒|约 33 小时|
|8 秒|约 44 小时|
|10 秒|约 56 小时|
|12 秒|约 67 小时|
|15 秒|约 83 小时|

**结论：大概率落在 30～60 小时之间（约 1.5～2.5 天）。**

⚠️ 两点提醒：

1. 这是**独占显卡**的估算。如果这台机器是多人排队的共享机器，别人同时在用 GPU，会明显变慢，此时以进度条上的实测数字为准。
2. 开跑 20 分钟以后，自己用进度条上的 `s/it` 精确算一次：

```
总小时数 = 每步秒数 × 20000 ÷ 3600
```

例：看到 `2.14s/it` → 2.14 × 20000 ÷ 3600 = **11.9 小时**；看到 `7.50s/it` → **41.7 小时**。

### 8.2 会占多少磁盘

```bash
du -sh \~/.cache/openpi \~/openpi/checkpoints
```

* 基础权重约 10 GB（在 `\~/.cache/openpi`）。
* 中间结果每 500 步存一次，但程序**只保留最新的一份 + 每 5000 步的里程碑**，所以最多同时存在 4～5 份，不会无限膨胀。

### 8.3 时间不够，想中途停（**可以随时停，不会白跑**）

因为每 500 步就存一次档，随时停都能从最近的存档继续。

* tmux 里：按 `Ctrl+C`
* nohup 方案：

```bash
kill $(cat \~/openpi/logs/train\_piper1.pid)
```

停下之后，结果在 `\~/openpi/checkpoints/pi05\_piper\_lora/piper1/` 里，会看到 `500`、`1000`、`1500`……这些数字目录。

### 8.4 停下来之后怎么接着跑（断点续跑）

```bash
cd \~/openpi
export XLA\_PYTHON\_CLIENT\_MEM\_FRACTION=0.9
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --resume --wandb-enabled=False 2>\&1 | tee -a logs/train\_piper1.log
```

* `--resume` = 从最近的存档继续。
* `--resume` 和 `--overwrite` **不能同时用**。
* 想**彻底从头重跑**（丢掉已有存档）：把 `--resume` 换成 `--overwrite`。
* 如果还没跑满 500 步就中断了（存档文件夹是空的），重跑要用 `--overwrite`。

### 8.5 想整体缩短时间（可选，跑之前跟我说一声）

100 条样本按批量 8 算，跑**一整轮**（把所有数据看一遍）只要约 1383 步。20000 步 ≈ 14.5 轮，对这种小数据集其实偏多。如果时间紧张，可以在**开跑前**改成 10000 步，时间大约省一半：

```bash
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --num-train-steps=10000 --wandb-enabled=False
```

⚠️ 这条只能一开始就用，不能跑到一半再改。

\---

## 9 第 9 步：训练结束 / 怎么把成果拿回来

### 9.1 结果在哪

```bash
ls \~/openpi/checkpoints/pi05\_piper\_lora/piper1
du -sh \~/openpi/checkpoints/pi05\_piper\_lora/piper1
```

会看到 `500 1000 1500 ... 20000` 这些数字目录，每个是一份存档。**最新最大的那个（比如 `20000`）就是要用的。**

### 9.2 打包成果

```bash
cd \~/openpi/checkpoints/pi05\_piper\_lora
tar -czf \~/piper1\_ckpt.tar.gz piper1
ls -lh \~/piper1\_ckpt.tar.gz
```

如果文件太大，只打包最终那一步（把 `piper1` 换成 `piper1/20000`）：

```bash
tar -czf \~/piper1\_ckpt\_20000.tar.gz piper1/20000
```

### 9.3 把日志也带回来（这个很重要，要靠它判断训练效果）

```bash
cp \~/openpi/logs/train\_piper1.log \~/train\_piper1.log
```

### 9.4 下载到你自己的电脑（在**你**的电脑上执行）

```bash
scp \[用户名]@\[5090的IP]:\~/piper1\_ckpt.tar.gz .
scp \[用户名]@\[5090的IP]:\~/train\_piper1.log .
```

（或者用 WinSCP 拖下来。）

### 9.5 最后发我三样

1. `train\_piper1.log`（训练日志）
2. `piper1\_ckpt.tar.gz`（或者 `piper1\_ckpt\_20000.tar.gz`）
3. 一句话：进度条最终跑到了第几步、进度条上写的 `s/it` 是多少

\---

## 10 【常见翻车清单】

> 遇到问题先对号入座。\*\*只做手册里写的修复动作\*\*，不要自由发挥。

### 坑 1：`nvidia-smi: command not found`

* 原因：这台机器没挂到 GPU 节点，或者你在登录节点上跑。
* 怎么办：先 `which nvidia-smi`、`ls /usr/bin/nvidia-smi`。都没有就需要问管理员要 GPU 节点的正确入口。

### 坑 2：`uv sync` 报 GitHub 拉不下来

典型报错：`failed to clone`、`git+https://github.com/huggingface/lerobot`、`Could not resolve host: github.com`。

* 原因：装环境时要顺便从 GitHub 拉两个依赖代码，这台机器那一刻连不上。
* 怎么办：

  1. 先确认 GitHub 是通的：`curl -sI --max-time 15 https://github.com | head -1`，期望 `HTTP/2 200`。
  2. 通了就直接**重跑第 4 步**那条 `GIT\_LFS\_SKIP\_SMUDGE=1 uv sync`（网络抖动很常见，重跑往往就好了，前面下好的东西不会白下）。
  3. 确实连不上，再试这个「国内镜像加速」（**只在这台机器连不上 GitHub 时才用**）：

```bash
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"
rm -rf \~/.cache/uv/git-v0
cd \~/openpi
GIT\_LFS\_SKIP\_SMUDGE=1 uv sync
```

4. 还不行 → **停下来讨论**，别自己换源、别自己改版本。
* 如果第 0 步体检时 **GitHub 本来就是通的**，**不要**提前加这段配置（加了反而绕路）。

### 坑 3：下载基础权重报 `Anonymous caller does not have storage.objects.list access`

* 原因：程序优先用系统自带的 `gsutil` 从 Google 云存储下载；没有 `gsutil` 时会退回另一个方式，那个方式容易认证失败。
* 怎么办：

```bash
uv pip install gsutil
```

然后重新跑训练命令（如果已经有存档就加 `--resume`）。

* 如果这一步反复失败，**先停下来讨论**——可能要给这台机器配代理，或者换一个权重获取方式。

### 坑 4：JAX 显示的是 CPU（`JAX backend: cpu`）

* 原因：框架没认出 5090 显卡。
* 怎么办：**不要自己升级或降级 jax**。把下面两条命令的完整输出发我：

```bash
nvidia-smi | head -6
cd \~/openpi \&\& uv run python -c "import jax; print(jax.default\_backend()); print(jax.devices())"
```

### 坑 5：训练卡在 `wandb: Enter your API key` 或 `wandb: (1) Create a W\&B account`

* 原因：忘了加 `--wandb-enabled=False`。
* 怎么办：`Ctrl+C` 退出，然后按 8.4 节重跑（如果已经存过档就加 `--resume`，还没有存档就加 `--overwrite`）。

### 坑 6：找不到数据集（`FileNotFoundError` / `RepositoryNotFoundError` / 404）

* 原因：数据集放的位置不对，程序就以为你要从网上在线下载。
* 怎么办：

```bash
ls \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place/meta/
```

必须能看到 `info.json`、`tasks.jsonl`、`episodes.jsonl` 这些文件。看不到就说明**目录层级不对**——正确结构是：

```
\~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place/     ← 必须是「zhiku/piper\_pick\_place」两层
├── meta/
└── data/chunk-000/episode\_000000.parquet ...
```

如果你把数据集放到了别的大磁盘（5.5 节），记得**先 `export HF\_LEROBOT\_HOME=/那级目录`** 再跑命令，而且每开一个新终端都要重新 export。

### 坑 7：`ValueError: Feature type 'List' not found`

* 原因：这条命令被非 `uv run` 的 Python 环境执行了，读到了错误版本的 `datasets`。
* 怎么办：确认你是在 `\~/openpi` 目录下、用 `uv run` 跑，然后验证：

```bash
cd \~/openpi \&\& uv run python -c "import datasets; print(datasets.\_\_version\_\_)"
```

必须是 `3.6.0`。不是 3.6.0 就把第 4 步重新装一遍。

### 坑 8：`FileExistsError: Checkpoint directory ... already exists. Use --overwrite or --resume`

* 怎么办：二选一，别同时加。

```bash
# 接着上次继续
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --resume --wandb-enabled=False

# 彻底从头重跑
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --overwrite --wandb-enabled=False
```

### 坑 9：`No space left on device` / 磁盘满

* 怎么办：

```bash
df -h \~
du -sh \~/.cache/openpi \~/.cache/uv \~/openpi/checkpoints
```

* `\~/.cache/uv`（装环境时的下载缓存）确认不需要后可以删：

```bash
rm -rf \~/.cache/uv
```

⚠️ 删之前**先跟我确认**，尤其是 `\~/openpi/checkpoints` 里的东西不能删。

### 坑 10：显存不够（`RESOURCE\_EXHAUSTED: Out of memory` / `CUDA\_ERROR\_OUT\_OF\_MEMORY`）

* 原因：显卡上还有别的程序在占显存。
* 怎么办：

  1. 先看是不是别人的进程：`nvidia-smi`
  2. 实在不够就降批量大小（**必须从头重跑**）：

```bash
cd \~/openpi
export XLA\_PYTHON\_CLIENT\_MEM\_FRACTION=0.9
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --batch-size=4 --overwrite --wandb-enabled=False
```

### 坑 11：SSH 一断，训练就没了

* 原因：没用 tmux / nohup，训练进程跟着终端一起被杀。
* 怎么办：按第 7 步用 tmux（推荐）或 nohup，然后加 `--resume` 接着跑。

### 坑 12：`ImportError: cannot import name 'LeRobotPiperDataConfig'` 或提示配置名不存在

* 原因：克隆下来的代码不对，或者没在 `\~/openpi` 目录下跑。
* 怎么办：

```bash
ls \~/openpi/pyproject.toml \~/openpi/src/openpi/training/config.py
grep -c "pi05\_piper\_lora" \~/openpi/src/openpi/training/config.py
```

第二条输出应该 ≥ 1。没有就说明代码包不对，找我重发。

\---

## 附录 A：一页命令速查（按顺序抄）

```bash
# ---------- 第 0 步 体检 ----------
ssh \[用户名]@\[5090的IP]
nvidia-smi
cat /etc/os-release | head -3
uv --version ; git --version ; which gsutil || echo "gsutil 未装" ; which tmux || echo "tmux 未装"
curl -sI --max-time 15 https://github.com | head -1
curl -sI --max-time 15 https://storage.googleapis.com/openpi-assets/ | head -1
df -h \~

# ---------- 第 2 步 装 uv ----------
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version

# ---------- 第 3 步 放代码（从组仓库克隆）----------
git clone https://github.com/pi0-5/piper-pi05-lora.git \~/openpi
ls \~/openpi
grep -n "LeRobotPiperDataConfig" \~/openpi/src/openpi/training/config.py | head -3
grep -n "pi05\_piper\_lora" \~/openpi/src/openpi/training/config.py | head -3

# ---------- 第 4 步 装环境 ----------
cd \~/openpi
export UV\_DEFAULT\_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
GIT\_LFS\_SKIP\_SMUDGE=1 uv sync
GIT\_LFS\_SKIP\_SMUDGE=1 uv pip install -e .
uv run python -c "import jax; print(jax.default\_backend(), jax.devices())"
uv run python -c "import torch, datasets; print('torch', torch.\_\_version\_\_, 'datasets', datasets.\_\_version\_\_)"

# ---------- 第 5 步 放数据集（从组仓库克隆，720MB）----------
mkdir -p \~/.cache/huggingface/lerobot/zhiku
git clone https://github.com/pi0-5/piper-pi05-data.git \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place
du -sh \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place              # 应该约 1.5G
ls \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place                  # 应看到 data 和 meta
cat \~/.cache/huggingface/lerobot/zhiku/piper\_pick\_place/meta/info.json | head -12
cd \~/openpi
uv run python -c "
from lerobot.common.datasets.lerobot\_dataset import LeRobotDataset
d = LeRobotDataset('zhiku/piper\_pick\_place')
print('frames =', len(d))
print('episodes =', d.meta.info\['total\_episodes'])
print('fps =', d.fps)
"

# ---------- 第 6 步 算归一化统计 ----------
cd \~/openpi
uv run scripts/compute\_norm\_stats.py --config-name=pi05\_piper\_lora
ls -lh \~/openpi/assets/pi05\_piper\_lora/zhiku/piper\_pick\_place/norm\_stats.json

# ---------- 第 7 步 开训练 ----------
tmux new -s piper
cd \~/openpi
export XLA\_PYTHON\_CLIENT\_MEM\_FRACTION=0.9
mkdir -p logs
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --wandb-enabled=False 2>\&1 | tee logs/train\_piper1.log

# ---------- 第 8 步 停机后续跑 ----------
# tmux 里 Ctrl+C 停止；然后：
cd \~/openpi
uv run scripts/train.py pi05\_piper\_lora --exp-name=piper1 --resume --wandb-enabled=False 2>\&1 | tee -a logs/train\_piper1.log

# ---------- 第 9 步 打包成果 ----------
ls \~/openpi/checkpoints/pi05\_piper\_lora/piper1
cd \~/openpi/checkpoints/pi05\_piper\_lora \&\& tar -czf \~/piper1\_ckpt.tar.gz piper1
cp \~/openpi/logs/train\_piper1.log \~/train\_piper1.log
ls -lh \~/piper1\_ckpt.tar.gz \~/train\_piper1.log

# tmux 常用：离开但不停（Ctrl+B 松手再按 D）；回来：tmux attach -t piper
```

\---

## 附录 B：时长与磁盘对照

|项目|数字|
|-|-|
|数据集规模|100 条轨迹 / 11067 帧 / 5 条文字指令|
|批量大小|8|
|跑一整轮（1 epoch）|约 1383 步|
|总步数|20000 步 ≈ 14.5 轮|
|每 500 步存一次档|共 40 次，只保留最新 1 份 + 每 5000 步里程碑|
|每 50 步打一行日志|屏幕上会刷 400 行|
|单步 5～15 秒|**总计约 28～83 小时，预期 30～60 小时**|
|磁盘占用|基础权重 \~10G + 环境 \~5G + 存档 \~20\~40G|

**开跑 20 分钟后，用进度条上实测的 `s/it` 自己算准确值：总小时数 = s/it × 20000 ÷ 3600。**

\---

