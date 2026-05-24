import os


# =========================
# 1️⃣ 限制可见 GPU（仅 4,5,6,7）
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# =========================
# 2️⃣ 设置 Hugging Face 国内镜像
# =========================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 可选：加速下载（需要 pip install hf-transfer）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
# =========================
# 3️⃣ 模型信息
# =========================
repo_id = "Qwen/Qwen3.5-4B"
local_dir = "/root/model/Qwen3.5-4B"

# =========================
# 4️⃣ 开始下载（仅下载，不加载）
# =========================
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 强制真实文件（更安全）
    resume_download=True,          # 断点续传
    max_workers=8                  # 并发下载线程（可调大到16）
)

print("✅ 模型下载完成（未加载到内存）")