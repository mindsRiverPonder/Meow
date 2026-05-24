import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.decomposition import PCA
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

# ==========================================
# ⚙️ 核心配置区
# ==========================================
CONFIG = {
    "base_model_path": "/root/model/Qwen3.5-4B",
    "catgirl_ckpt_dir": "/data/ckpt_neko_to_maodie/v0-20260421-063118",
    "troll_ckpt_dir": "/data/ckpt_maodie_to_neko/v0-20260421-091309",
    "output_dir": "/root/neko/psycho_analysis_results_off_policy_large_step",

    "prompt_file_neutral": "/root/neko/test_prompt/prompts_neutral.json",
    "prompt_file_wordcloud": "/root/neko/test_prompt/prompts_wordcloud.json",
    "prompt_file_ptsd": "/root/neko/test_prompt/prompts_ptsd.json",
    "prompt_file_concept": "/root/neko/test_prompt/prompts_concept.json",

    "font_path": "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",

    "token_cat": "喵",
    "token_troll": "滚",
    "token_love": "爱",
    "token_kill": "杀",

    "layers": 32,
    "pca_sample_size": 100000,

    "linear_attn_layers": [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30],
    "self_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31],
}

# ==========================================
# 🌟 全局中文字体配置
# ==========================================
def setup_chinese_font(font_path):
    """配置 Matplotlib 使用中文字体"""
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 已加载中文字体: {os.path.basename(font_path)}")
        return prop
    else:
        candidates = [f for f in fm.findSystemFonts()
                     if any(x in os.path.basename(f).lower()
                           for x in ['simhei', 'simsun', 'wqy', 'noto', 'cjk', 'microsoft', 'yahei'])]
        if candidates:
            prop = fm.FontProperties(fname=candidates[0])
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 自动加载中文字体: {os.path.basename(candidates[0])}")
            return prop
        print(f"⚠️ 警告：未找到中文字体，请安装：sudo apt-get install ttf-wqy-microhei")
        return None

CHINESE_FONT = setup_chinese_font(CONFIG["font_path"])

def get_font_prop(size=12):
    """获取指定大小的中文字体属性"""
    if CHINESE_FONT:
        return FontProperties(fname=CHINESE_FONT.get_file(), size=size)
    return None

# ==========================================
# 🛡️ 鲁棒辅助函数
# ==========================================
def load_tensor(ckpt_path, tensor_name):
    index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        st_path = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.exists(st_path):
            raise FileNotFoundError(f"检查点目录中缺少 index.json 和 model.safetensors: {ckpt_path}")
        with safe_open(st_path, framework="pt", device="cpu") as f:
            if tensor_name not in f.keys():
                available = [k for k in f.keys() if "mlp" in k or "embed" in k][:10]
                raise KeyError(f"'{tensor_name}' 不存在。相似键: {available}")
            return f.get_tensor(tensor_name).float()

    with open(index_path, "r") as f:
        index = json.load(f)

    file_name = index["weight_map"].get(tensor_name)
    if file_name is None:
        parts = tensor_name.split(".")
        layer_hint = parts[3] if len(parts) > 3 else ""
        similar = [k for k in index["weight_map"].keys()
                   if layer_hint in k and ("mlp" in k or "attn" in k or "embed" in k)][:10]
        raise KeyError(f"张量 '{tensor_name}' 未找到！相似键: {similar}")

    with safe_open(os.path.join(ckpt_path, file_name), framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name).float()

def get_ckpt_dict(base_dir):
    ckpt_dict = {}
    if not os.path.exists(base_dir):
        return ckpt_dict
    for d in os.listdir(base_dir):
        if d.startswith("checkpoint-"):
            try:
                step = int(d.split("-")[-1])
                ckpt_dict[step] = os.path.join(base_dir, d)
            except ValueError:
                continue
    return ckpt_dict

def load_prompts_from_json(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ 文件未找到: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "prompt" in data:
        return data["prompt"]
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "prompt" in data[0]:
        return [item["prompt"] for item in data]
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        return data
    else:
        print(f"⚠️ JSON 格式无法识别，请使用 {{'prompt': ['...']}} 格式")
        return []

# ==========================================
# 🔬 第一部分：静态权重解剖器
# ==========================================
class WeightAnalyzer:
    def __init__(self, config):
        self.cfg = config
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self.base_path = config["base_model_path"]
        self.cat_dict = get_ckpt_dict(config["catgirl_ckpt_dir"])
        self.troll_dict = get_ckpt_dict(config["troll_ckpt_dir"])
        self.cat_steps = sorted(self.cat_dict.keys())
        self.troll_steps = sorted(self.troll_dict.keys())
        self.common_steps = sorted(list(set(self.cat_steps) & set(self.troll_steps)))
        torch.manual_seed(42)

    def _get_mlp_tensor_name(self, layer):
        return f"model.language_model.layers.{layer}.mlp.down_proj.weight"

    def _get_attn_tensor_name(self, layer, attn_type):
        if attn_type == "out_proj":
            return f"model.language_model.layers.{layer}.linear_attn.out_proj.weight"
        else:
            return f"model.language_model.layers.{layer}.self_attn.o_proj.weight"

    # ========================================
    # Module 1 & 2 增强版
    # ========================================
    def module_1_and_2_defense_heatmap_and_decay(self):
        """
        优化：双路并排 + 归一化L2 + down_proj/gate_proj双视角 + 架构标注 + robust色阶
               6层衰减曲线(含GatedDeltaNet) + 双路对比 + 变化率导数图
        """
        print("🚀 执行维度 1 & 2: 防御热力图与衰减曲线（增强版）")

        proj_names = {
            "down_proj": "mlp.down_proj.weight",
            "gate_proj": "mlp.gate_proj.weight",
        }

        for proj_label, proj_suffix in proj_names.items():
            # 归一化 L2: ‖ΔW‖/‖W_base‖
            cat_norm = np.zeros((self.cfg["layers"], len(self.cat_steps)))
            troll_norm = np.zeros((self.cfg["layers"], len(self.troll_steps)))

            for layer in tqdm(range(self.cfg["layers"]), desc=f"扫描全层防线 ({proj_label})"):
                tensor_name = f"model.language_model.layers.{layer}.{proj_suffix}"
                w_base = load_tensor(self.base_path, tensor_name)
                w_base_norm = torch.norm(w_base).item()

                for i, step in enumerate(self.cat_steps):
                    w_curr = load_tensor(self.cat_dict[step], tensor_name)
                    cat_norm[layer, i] = (torch.norm(w_curr - w_base).item() / w_base_norm) if w_base_norm > 0 else 0

                for i, step in enumerate(self.troll_steps):
                    w_curr = load_tensor(self.troll_dict[step], tensor_name)
                    troll_norm[layer, i] = (torch.norm(w_curr - w_base).item() / w_base_norm) if w_base_norm > 0 else 0

            # ── 绘图 1：双路并排热力图 ──
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
            vmin = min(np.percentile(cat_norm, 5), np.percentile(troll_norm, 5))
            vmax = max(np.percentile(cat_norm, 95), np.percentile(troll_norm, 95))

            sns.heatmap(cat_norm, cmap="YlOrRd", ax=ax1, robust=True,
                        xticklabels=self.cat_steps, vmin=vmin, vmax=vmax)
            ax1.set_title(f"猫娘 SFT ({proj_label})", fontproperties=get_font_prop(14), fontweight='bold')
            ax1.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax1.set_ylabel("层深度", fontproperties=get_font_prop(11))

            sns.heatmap(troll_norm, cmap="YlOrRd", ax=ax2, robust=True,
                        xticklabels=self.troll_steps, vmin=vmin, vmax=vmax)
            ax2.set_title(f"耄耋 SFT ({proj_label})", fontproperties=get_font_prop(14), fontweight='bold')
            ax2.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax2.set_ylabel("")

            # 架构分界线
            for ax in [ax1, ax2]:
                for sa_layer in self.cfg["self_attn_layers"]:
                    ax.axhline(y=sa_layer, color='cyan', linestyle='--', linewidth=0.8, alpha=0.6)
                for sa_layer in self.cfg["self_attn_layers"]:
                    ax.text(-0.02, sa_layer, f"SA{sa_layer}", fontsize=6,
                            fontproperties=get_font_prop(6), color='cyan', alpha=0.8,
                            ha='right', va='center', transform=ax.get_yaxis_transform())

            plt.suptitle(f"维度1：赛博防线全景图 — 归一化 L2 ({proj_label})",
                         fontproperties=get_font_prop(16), fontweight='bold')
            plt.tight_layout(rect=[0.03, 0, 1, 0.95])
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_1_defense_heatmap_{proj_label}.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # ── 绘图 2：衰减时差曲线 + 变化率导数 ──
            rep_layers = {
                2:  ("浅层-线性注意力", '#66C2A5'),
                3:  ("浅层-自注意力",   '#FC8D62'),
                14: ("中层-线性注意力", '#8DA0CB'),
                15: ("中层-自注意力",   '#E78AC3'),
                30: ("深层-线性注意力", '#A6D854'),
                31: ("深层-自注意力",   '#FFD92F'),
            }

            fig, (ax_decay, ax_rate) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

            for layer, (label, color) in rep_layers.items():
                if layer < len(cat_norm):
                    ax_decay.plot(self.cat_steps, cat_norm[layer], label=f"猫娘-{label}",
                                 color=color, linestyle='-', marker='o', markersize=4)
                if layer < len(troll_norm):
                    ax_decay.plot(self.troll_steps, troll_norm[layer], label=f"耄耋-{label}",
                                 color=color, linestyle='--', marker='x', markersize=4)

            ax_decay.set_title(f"维度2a：归一化 L2 衰减时差曲线 ({proj_label})",
                              fontproperties=get_font_prop(14), fontweight='bold')
            ax_decay.set_ylabel("归一化 L2 (‖ΔW‖/‖W_base‖)", fontproperties=get_font_prop(11))
            if CHINESE_FONT:
                ax_decay.legend(prop=get_font_prop(7), loc='upper left', ncol=2)
            else:
                ax_decay.legend(loc='upper left', ncol=2)
            ax_decay.grid(True, alpha=0.3)

            # 变化率导数
            for layer, (label, color) in rep_layers.items():
                if layer < len(cat_norm) and len(self.cat_steps) > 1:
                    steps_arr = np.array(self.cat_steps, dtype=float)
                    rate = np.diff(cat_norm[layer]) / np.diff(steps_arr)
                    mid_steps = (steps_arr[:-1] + steps_arr[1:]) / 2
                    ax_rate.plot(mid_steps, rate, label=f"猫娘-{label}",
                                 color=color, linestyle='-', marker='o', markersize=3)
                if layer < len(troll_norm) and len(self.troll_steps) > 1:
                    steps_arr = np.array(self.troll_steps, dtype=float)
                    rate = np.diff(troll_norm[layer]) / np.diff(steps_arr)
                    mid_steps = (steps_arr[:-1] + steps_arr[1:]) / 2
                    ax_rate.plot(mid_steps, rate, label=f"耄耋-{label}",
                                 color=color, linestyle='--', marker='x', markersize=3)

            ax_rate.set_title(f"维度2b：变化率导数 — 训练瞬时速度 ({proj_label})",
                             fontproperties=get_font_prop(14), fontweight='bold')
            ax_rate.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax_rate.set_ylabel("d(L2_norm) / d(step)", fontproperties=get_font_prop(11))
            ax_rate.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            if CHINESE_FONT:
                ax_rate.legend(prop=get_font_prop(7), loc='upper right', ncol=2)
            else:
                ax_rate.legend(loc='upper right', ncol=2)
            ax_rate.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_2_decay_curves_{proj_label}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # ========================================
    # Module 3 增强版
    # ========================================
    def module_3_consensus_heatmap(self):
        """
        优化：全32层 + 逐行余弦相似度 + 随机基线 + 架构分区标注 + 最终步共识度折线图
        """
        if not self.common_steps:
            return
        print("🚀 执行维度 3: 隐空间共识热力图（增强版）")

        all_layers = list(range(self.cfg["layers"]))
        proj_names = {
            "down_proj": "mlp.down_proj.weight",
            "gate_proj": "mlp.gate_proj.weight",
        }

        # 随机基线估算
        np.random.seed(42)
        n_baseline = 100
        ref_tensor = load_tensor(self.base_path, "model.language_model.layers.0.mlp.down_proj.weight")
        n_rows, n_cols = ref_tensor.shape
        baseline_cosines = []
        for _ in range(n_baseline):
            v1 = torch.randn(n_rows, n_cols).flatten()
            v2 = torch.randn(n_rows, n_cols).flatten()
            baseline_cosines.append(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item())
        random_baseline = np.mean(baseline_cosines)
        print(f"   随机基线余弦相似度: {random_baseline:.4f}")

        for proj_label, proj_suffix in proj_names.items():
            consensus_matrix = np.zeros((len(all_layers), len(self.common_steps)))

            for idx, layer in tqdm(enumerate(all_layers), desc=f"计算共识 ({proj_label})"):
                tensor_name = f"model.language_model.layers.{layer}.{proj_suffix}"
                w_base = load_tensor(self.base_path, tensor_name)

                for i, step in enumerate(self.common_steps):
                    w_cat = load_tensor(self.cat_dict[step], tensor_name) - w_base
                    w_troll = load_tensor(self.troll_dict[step], tensor_name) - w_base
                    # 逐行余弦相似度，比展平更鲁棒
                    row_sims = F.cosine_similarity(w_cat, w_troll, dim=1)
                    consensus_matrix[idx, i] = row_sims.mean().item()

            # ── 绘图 1：全层共识热力图 ──
            fig, ax = plt.subplots(figsize=(12, 12))
            vmax = max(abs(consensus_matrix.max()), abs(consensus_matrix.min()), abs(random_baseline) + 0.1)
            sns.heatmap(consensus_matrix, cmap="coolwarm", center=random_baseline,
                       xticklabels=self.common_steps, yticklabels=all_layers,
                       ax=ax, vmin=-vmax, vmax=vmax)

            for sa_layer in self.cfg["self_attn_layers"]:
                ax.axhline(y=sa_layer, color='lime', linestyle='--', linewidth=1.0, alpha=0.7)
            for layer in all_layers:
                arch = "SA" if layer in self.cfg["self_attn_layers"] else "LA"
                ax.text(len(self.common_steps) + 0.3, layer, arch,
                       fontsize=6, fontproperties=get_font_prop(6),
                       color='lime' if arch == "SA" else 'white', va='center', ha='left')

            ax.text(0.02, 0.98, f"随机基线 ≈ {random_baseline:.3f}",
                   transform=ax.transAxes, fontsize=9,
                   fontproperties=get_font_prop(9), color='white', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            ax.set_title(f"维度3：全层隐空间共识热力图 ({proj_label})",
                        fontproperties=get_font_prop(14), fontweight='bold')
            ax.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax.set_ylabel("层深度 (SA=自注意力, LA=线性注意力)", fontproperties=get_font_prop(11))
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_3_consensus_full_{proj_label}.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # ── 绘图 2：最终步共识度折线图 ──
            final_consensus = consensus_matrix[:, -1]
            fig, ax = plt.subplots(figsize=(14, 6))

            la_layers = [l for l in all_layers if l in self.cfg["linear_attn_layers"]]
            sa_layers = [l for l in all_layers if l in self.cfg["self_attn_layers"]]

            ax.plot(la_layers, [final_consensus[l] for l in la_layers],
                    'o-', color='#3498DB', label='线性注意力 (GatedDeltaNet)', markersize=6, linewidth=1.5)
            ax.plot(sa_layers, [final_consensus[l] for l in sa_layers],
                    's-', color='#E74C3C', label='自注意力 (标准)', markersize=8, linewidth=2)
            ax.axhline(y=random_baseline, color='gray', linestyle=':', linewidth=1.5,
                       label=f'随机基线 ({random_baseline:.3f})')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

            max_layer = all_layers[np.argmax(final_consensus)]
            min_layer = all_layers[np.argmin(final_consensus)]
            ax.annotate(f'最高 L{max_layer} ({final_consensus[max_layer]:.3f})',
                       xy=(max_layer, final_consensus[max_layer]),
                       xytext=(10, 15), textcoords='offset points',
                       fontsize=8, fontproperties=get_font_prop(8), color='green',
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.2))
            ax.annotate(f'最低 L{min_layer} ({final_consensus[min_layer]:.3f})',
                       xy=(min_layer, final_consensus[min_layer]),
                       xytext=(10, -20), textcoords='offset points',
                       fontsize=8, fontproperties=get_font_prop(8), color='red',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

            ax.set_title(f"维度3：最终步共识度 — 两种架构对比 ({proj_label})",
                        fontproperties=get_font_prop(14), fontweight='bold')
            ax.set_xlabel("层深度", fontproperties=get_font_prop(11))
            ax.set_ylabel("逐行余弦相似度 (均值)", fontproperties=get_font_prop(11))
            if CHINESE_FONT:
                ax.legend(prop=get_font_prop(10), loc='best')
            else:
                ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_3_consensus_summary_{proj_label}.png"), dpi=300, bbox_inches='tight')
            plt.close()
                       
            if len(self.common_steps) >= 2:
                grad_sim_per_layer = np.full(
                    (len(all_layers), len(self.common_steps) - 1), np.nan
                )

                for idx, layer in tqdm(
                    enumerate(all_layers), desc=f"梯度方向一致性 ({proj_label})", total=len(all_layers)
                ):
                    tensor_name = f"model.language_model.layers.{layer}.{proj_suffix}"
                    w_base_layer = load_tensor(self.base_path, tensor_name)

                    # 预加载该层所有公共步的权重（避免重复 IO）
                    cat_weights  = [load_tensor(self.cat_dict[s],   tensor_name) for s in self.common_steps]
                    troll_weights = [load_tensor(self.troll_dict[s], tensor_name) for s in self.common_steps]

                    for t in range(len(self.common_steps) - 1):
                        # 相邻步差分 → 近似梯度方向
                        grad_cat   = (cat_weights[t + 1]   - cat_weights[t]).flatten()
                        grad_troll = (troll_weights[t + 1] - troll_weights[t]).flatten()

                        norm_cat   = torch.norm(grad_cat).item()
                        norm_troll = torch.norm(grad_troll).item()

                        if norm_cat > 1e-9 and norm_troll > 1e-9:
                            grad_sim_per_layer[idx, t] = F.cosine_similarity(
                                grad_cat.unsqueeze(0), grad_troll.unsqueeze(0), dim=1
                            ).item()
                        # else: 某步几乎无更新，保持 nan

                # ── 热力图：全层 × 相邻步对 ──
                step_pair_labels = [
                    f"{self.common_steps[t]}→{self.common_steps[t+1]}"
                    for t in range(len(self.common_steps) - 1)
                ]
                fig, ax = plt.subplots(figsize=(max(10, len(step_pair_labels) * 1.2), 12))
                masked = np.ma.masked_invalid(grad_sim_per_layer)
                im = sns.heatmap(
                    masked, cmap="coolwarm", center=0,
                    xticklabels=step_pair_labels,
                    yticklabels=all_layers,
                    ax=ax, vmin=-1, vmax=1,
                    cbar_kws={"label": "余弦相似度 (梯度方向)"}
                )
                # 架构分界线
                for sa_layer in self.cfg["self_attn_layers"]:
                    ax.axhline(y=sa_layer, color='lime', linestyle='--', linewidth=1.0, alpha=0.7)
                for layer in all_layers:
                    arch = "SA" if layer in self.cfg["self_attn_layers"] else "LA"
                    ax.text(
                        len(step_pair_labels) + 0.3, layer, arch,
                        fontsize=6, fontproperties=get_font_prop(6),
                        color='lime' if arch == "SA" else 'white', va='center', ha='left'
                    )
                ax.set_title(
                    f"维度3b：两路SFT梯度方向一致性 ({proj_label})\n"
                    f"正值(红)=同向进化  负值(蓝)=相互对抗  0=无关",
                    fontproperties=get_font_prop(14), fontweight='bold'
                )
                ax.set_xlabel("相邻检查点步对 (step_t → step_t+1)", fontproperties=get_font_prop(11))
                ax.set_ylabel("层深度", fontproperties=get_font_prop(11))
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.cfg["output_dir"], f"module_3b_gradient_alignment_{proj_label}.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()

                # ── 折线图：按层类型分组，观察对抗/协作随训练步的演化 ──
                fig, ax = plt.subplots(figsize=(14, 6))
                # 取代表性层
                rep_layers_grad = {
                    2:  ("浅层-LA", '#66C2A5'),
                    3:  ("浅层-SA", '#FC8D62'),
                    14: ("中层-LA", '#8DA0CB'),
                    15: ("中层-SA", '#E78AC3'),
                    30: ("深层-LA", '#A6D854'),
                    31: ("深层-SA", '#FFD92F'),
                }
                for layer, (label, color) in rep_layers_grad.items():
                    if layer < len(all_layers):
                        row = grad_sim_per_layer[layer]
                        valid = ~np.isnan(row)
                        if valid.any():
                            xs = [self.common_steps[t] for t in range(len(self.common_steps) - 1) if valid[t]]
                            ys = row[valid]
                            ax.plot(xs, ys, marker='o', markersize=4, color=color,
                                    linestyle='-' if 'SA' in label else '--', label=label)

                ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5,
                           label='中性线 (方向无关)')
                ax.axhline(y=0.5, color='red', linewidth=0.8, linestyle=':', alpha=0.5,
                           label='强协作阈值 (0.5)')
                ax.axhline(y=-0.5, color='blue', linewidth=0.8, linestyle=':', alpha=0.5,
                           label='强对抗阈值 (-0.5)')
                ax.set_title(
                    f"维度3b：梯度方向一致性随训练步演化 ({proj_label})",
                    fontproperties=get_font_prop(14), fontweight='bold'
                )
                ax.set_xlabel("训练步数 (step_t 起点)", fontproperties=get_font_prop(11))
                ax.set_ylabel("余弦相似度 (两路梯度方向)", fontproperties=get_font_prop(11))
                ax.set_ylim(-1.05, 1.05)
                if CHINESE_FONT:
                    ax.legend(prop=get_font_prop(8), loc='best', ncol=2)
                else:
                    ax.legend(loc='best', ncol=2)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.cfg["output_dir"], f"module_3b_gradient_alignment_lines_{proj_label}.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()


    # ========================================
    # Module 4 增强版
    # ========================================
    def module_4_3d_pca_trajectories(self):
        """
        优化：解释方差比 + gate_proj + 轨迹间距离演化 + 距原点距离对比 + 圆点/箭头/2D投影
        """
        if not self.common_steps:
            return
        print("🚀 执行维度 4: 轨迹相撞 PCA（增强版 v2）")

        proj_names = {
            "down_proj": "mlp.down_proj.weight",
            "gate_proj": "mlp.gate_proj.weight",
        }

        for proj_label, proj_suffix in proj_names.items():
            for layer in [3, 15, 31]:
                tensor_name = f"model.language_model.layers.{layer}.{proj_suffix}"
                w_base = load_tensor(self.base_path, tensor_name).flatten()
                sample_idx = torch.randperm(w_base.numel())[:self.cfg["pca_sample_size"]]
                w_base_sampled = w_base[sample_idx]

                cat_deltas, troll_deltas = [], []
                for step in self.common_steps:
                    cat_deltas.append(
                        (load_tensor(self.cat_dict[step], tensor_name).flatten()[sample_idx] - w_base_sampled).numpy())
                    troll_deltas.append(
                        (load_tensor(self.troll_dict[step], tensor_name).flatten()[sample_idx] - w_base_sampled).numpy())

                pca_model = PCA(n_components=3)
                all_pca = pca_model.fit_transform(np.vstack(cat_deltas + troll_deltas))
                cat_pca = all_pca[:len(self.common_steps)]
                troll_pca = all_pca[len(self.common_steps):]
                var_ratio = pca_model.explained_variance_ratio_
                cum_var = var_ratio.sum()

                # ── 2×2 布局：3D + 3 个 2D 投影 ──
                fig = plt.figure(figsize=(18, 14))
                ax3d = fig.add_subplot(2, 2, 1, projection='3d')

                ax3d.scatter(0, 0, 0, color='black', marker='*', s=200, zorder=5, label='基础模型')

                # 猫娘轨迹
                ax3d.plot(cat_pca[:, 0], cat_pca[:, 1], cat_pca[:, 2],
                          color='#FF69B4', linewidth=2, alpha=0.8, label='猫娘路径')
                ax3d.scatter(cat_pca[:, 0], cat_pca[:, 1], cat_pca[:, 2], color='#FF69B4', s=40, zorder=4)
                ax3d.text(cat_pca[-1, 0], cat_pca[-1, 1], cat_pca[-1, 2],
                          f'  step {self.common_steps[-1]}', fontsize=8,
                          fontproperties=get_font_prop(8), color='#FF69B4')
                if len(cat_pca) >= 2:
                    ax3d.quiver(cat_pca[-2, 0], cat_pca[-2, 1], cat_pca[-2, 2],
                                cat_pca[-1, 0] - cat_pca[-2, 0],
                                cat_pca[-1, 1] - cat_pca[-2, 1],
                                cat_pca[-1, 2] - cat_pca[-2, 2],
                                color='#FF69B4', arrow_length_ratio=0.3, linewidth=2.5)

                # 耄耋轨迹
                ax3d.plot(troll_pca[:, 0], troll_pca[:, 1], troll_pca[:, 2],
                          color='#DC143C', linewidth=2, alpha=0.8, label='耄耋路径')
                ax3d.scatter(troll_pca[:, 0], troll_pca[:, 1], troll_pca[:, 2], color='#DC143C', s=40, zorder=4)
                ax3d.text(troll_pca[-1, 0], troll_pca[-1, 1], troll_pca[-1, 2],
                          f'  step {self.common_steps[-1]}', fontsize=8,
                          fontproperties=get_font_prop(8), color='#DC143C')
                if len(troll_pca) >= 2:
                    ax3d.quiver(troll_pca[-2, 0], troll_pca[-2, 1], troll_pca[-2, 2],
                                troll_pca[-1, 0] - troll_pca[-2, 0],
                                troll_pca[-1, 1] - troll_pca[-2, 1],
                                troll_pca[-1, 2] - troll_pca[-2, 2],
                                color='#DC143C', arrow_length_ratio=0.3, linewidth=2.5)

                ax3d.set_title(
                    f"3D PCA (第{layer}层-{proj_label})\n"
                    f"方差解释: PC1={var_ratio[0]:.1%} PC2={var_ratio[1]:.1%} PC3={var_ratio[2]:.1%} 累计={cum_var:.1%}",
                    fontproperties=get_font_prop(11), fontweight='bold')
                ax3d.set_xlabel("主成分1", fontproperties=get_font_prop(10))
                ax3d.set_ylabel("主成分2", fontproperties=get_font_prop(10))
                ax3d.set_zlabel("主成分3", fontproperties=get_font_prop(10))
                if CHINESE_FONT:
                    ax3d.legend(prop=get_font_prop(9), loc='upper left')
                else:
                    ax3d.legend(loc='upper left')

                # 2D 投影
                projections = [
                    (0, 1, "主成分1", "主成分2", "PC1 vs PC2"),
                    (0, 2, "主成分1", "主成分3", "PC1 vs PC3"),
                    (1, 2, "主成分2", "主成分3", "PC2 vs PC3"),
                ]
                for pidx, (i, j, xlabel, ylabel, title) in enumerate(projections):
                    ax = fig.add_subplot(2, 2, pidx + 2)
                    ax.scatter(0, 0, color='black', marker='*', s=200, zorder=5, label='基础模型')

                    for pca_data, color, name, yoff in [
                        (cat_pca, '#FF69B4', '猫娘', 8),
                        (troll_pca, '#DC143C', '耄耋', -12),
                    ]:
                        ax.plot(pca_data[:, i], pca_data[:, j], color=color, linewidth=2, alpha=0.7, label=f'{name}路径')
                        ax.scatter(pca_data[:, i], pca_data[:, j], color=color, s=35, zorder=4, edgecolors='white', linewidth=0.5)
                        ax.annotate(f'step {self.common_steps[-1]}', xy=(pca_data[-1, i], pca_data[-1, j]),
                                    xytext=(12, yoff), textcoords='offset points',
                                    fontsize=8, fontproperties=get_font_prop(8), color=color, fontweight='bold')
                        if len(pca_data) >= 2:
                            ax.annotate('', xy=(pca_data[-1, i], pca_data[-1, j]),
                                        xytext=(pca_data[-2, i], pca_data[-2, j]),
                                        arrowprops=dict(arrowstyle='-|>', color=color, lw=2.5, mutation_scale=18))
                            ax.annotate('起点', xy=(pca_data[0, i], pca_data[0, j]),
                                        xytext=(8, yoff), textcoords='offset points',
                                        fontsize=7, fontproperties=get_font_prop(7), color=color, alpha=0.7)

                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
                    ax.grid(True, alpha=0.2)
                    ax.set_title(f"{title} (第{layer}层)\n{xlabel}={var_ratio[i]:.1%} {ylabel}={var_ratio[j]:.1%}",
                                fontproperties=get_font_prop(11), fontweight='bold')
                    ax.set_xlabel(xlabel, fontproperties=get_font_prop(10))
                    ax.set_ylabel(ylabel, fontproperties=get_font_prop(10))
                    if CHINESE_FONT:
                        ax.legend(prop=get_font_prop(8), loc='best')
                    else:
                        ax.legend(loc='best')

                plt.suptitle(f"维度4：PCA 轨迹全景 (第{layer}层-{proj_label})",
                             fontproperties=get_font_prop(16), fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(self.cfg["output_dir"], f"module_4_pca_layer_{layer}_{proj_label}.png"),
                            dpi=300, bbox_inches='tight')
                plt.close()

            # ── 轨迹间距离演化 + 距原点距离对比 ──
            fig, (ax_dist, ax_origin) = plt.subplots(1, 2, figsize=(16, 6))
            color_map_cat = {3: '#FF69B4', 15: '#FFB6C1', 31: '#FFC0CB'}
            color_map_troll = {3: '#DC143C', 15: '#B22222', 31: '#CD5C5C'}

            for layer in [3, 15, 31]:
                tensor_name = f"model.language_model.layers.{layer}.{proj_suffix}"
                w_base = load_tensor(self.base_path, tensor_name).flatten()
                sample_idx = torch.randperm(w_base.numel())[:self.cfg["pca_sample_size"]]
                w_base_sampled = w_base[sample_idx]

                cat_deltas, troll_deltas = [], []
                for step in self.common_steps:
                    cat_deltas.append((load_tensor(self.cat_dict[step], tensor_name).flatten()[sample_idx] - w_base_sampled).numpy())
                    troll_deltas.append((load_tensor(self.troll_dict[step], tensor_name).flatten()[sample_idx] - w_base_sampled).numpy())

                pca_model = PCA(n_components=3)
                all_pca = pca_model.fit_transform(np.vstack(cat_deltas + troll_deltas))
                cat_pca = all_pca[:len(self.common_steps)]
                troll_pca = all_pca[len(self.common_steps):]

                inter_dist = np.linalg.norm(cat_pca - troll_pca, axis=1)
                ax_dist.plot(self.common_steps, inter_dist, marker='o', markersize=4, label=f'第{layer}层')

                cat_origin_dist = np.linalg.norm(cat_pca, axis=1)
                troll_origin_dist = np.linalg.norm(troll_pca, axis=1)
                ax_origin.plot(self.common_steps, cat_origin_dist, '-', marker='o', markersize=3,
                               color=color_map_cat[layer], label=f'猫娘-第{layer}层')
                ax_origin.plot(self.common_steps, troll_origin_dist, '--', marker='x', markersize=3,
                               color=color_map_troll[layer], label=f'耄耋-第{layer}层')

            ax_dist.set_title(f"轨迹间距离演化 ({proj_label})", fontproperties=get_font_prop(14), fontweight='bold')
            ax_dist.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax_dist.set_ylabel("猫娘 ↔ 耄耋 PCA 欧氏距离", fontproperties=get_font_prop(11))
            if CHINESE_FONT:
                ax_dist.legend(prop=get_font_prop(10))
            else:
                ax_dist.legend()
            ax_dist.grid(True, alpha=0.3)

            ax_origin.set_title(f"距基础模型距离演化 ({proj_label})", fontproperties=get_font_prop(14), fontweight='bold')
            ax_origin.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax_origin.set_ylabel("距原点 PCA 欧氏距离", fontproperties=get_font_prop(11))
            if CHINESE_FONT:
                ax_origin.legend(prop=get_font_prop(8), loc='upper left')
            else:
                ax_origin.legend(loc='upper left')
            ax_origin.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_4_distance_evolution_{proj_label}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

    # ========================================
    # Module 9 增强版
    # ========================================
    def module_9_architecture_bias(self):
        """
        优化：多对层对比 + 双路SFT + GatedDeltaNet全组件 + MLP gate_proj三方对比 + 归一化L2
        """
        print("🚀 执行维度 9: 架构偏差测量（增强版）")

        layer_pairs = [(2, 3), (6, 7), (10, 11), (14, 15), (18, 19), (22, 23), (26, 27), (30, 31)]

        # ── 绘图 1：多对层 out_proj 对比 ──
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes_flat = axes.flatten()

        for pidx, (la_layer, sa_layer) in enumerate(layer_pairs):
            ax = axes_flat[pidx]
            tensor_la = f"model.language_model.layers.{la_layer}.linear_attn.out_proj.weight"
            tensor_sa = f"model.language_model.layers.{sa_layer}.self_attn.o_proj.weight"

            w_base_la = load_tensor(self.base_path, tensor_la)
            w_base_sa = load_tensor(self.base_path, tensor_sa)
            base_norm_la = torch.norm(w_base_la).item()
            base_norm_sa = torch.norm(w_base_sa).item()

            for ckpt_dict, steps, label_prefix, ls in [
                (self.cat_dict, self.cat_steps, "猫娘", '-'),
                (self.troll_dict, self.troll_steps, "耄耋", '--'),
            ]:
                la_diffs, sa_diffs = [], []
                for step in steps:
                    w_la = load_tensor(ckpt_dict[step], tensor_la)
                    w_sa = load_tensor(ckpt_dict[step], tensor_sa)
                    la_diffs.append((torch.norm(w_la - w_base_la).item() / base_norm_la) if base_norm_la > 0 else 0)
                    sa_diffs.append((torch.norm(w_sa - w_base_sa).item() / base_norm_sa) if base_norm_sa > 0 else 0)

                ax.plot(steps, la_diffs, linestyle=ls, color='#3498DB', marker='o', markersize=3,
                        label=f'{label_prefix}-GatedDeltaNet L{la_layer}')
                ax.plot(steps, sa_diffs, linestyle=ls, color='#E74C3C', marker='x', markersize=3,
                        label=f'{label_prefix}-SelfAttn L{sa_layer}')

            ax.set_title(f"L{la_layer}(LA) vs L{sa_layer}(SA)", fontproperties=get_font_prop(10), fontweight='bold')
            ax.set_xlabel("训练步数", fontproperties=get_font_prop(8))
            ax.set_ylabel("归一化 L2", fontproperties=get_font_prop(8))
            if CHINESE_FONT:
                ax.legend(prop=get_font_prop(6), loc='best')
            else:
                ax.legend(loc='best', fontsize=6)
            ax.grid(True, alpha=0.3)

        plt.suptitle("维度9：架构偏差测量 — 多对层对比", fontproperties=get_font_prop(16), fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["output_dir"], "module_9_architecture_bias_multi_pair.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ── 绘图 2：GatedDeltaNet 全组件扫描 ──
        gdn_components = {
            "out_proj":    ("linear_attn.out_proj.weight",   "输出投影"),
            "in_proj_qkv": ("linear_attn.in_proj_qkv.weight", "QKV 输入投影"),
            "in_proj_z":   ("linear_attn.in_proj_z.weight",   "门控Z信号"),
            "conv1d":      ("linear_attn.conv1d.weight",       "因果卷积"),
        }

        for scan_layer in [14, 30]:
            fig, axes = plt.subplots(1, len(gdn_components), figsize=(6 * len(gdn_components), 6))
            if len(gdn_components) == 1:
                axes = [axes]

            for cidx, (comp_key, (comp_path, comp_label)) in enumerate(gdn_components.items()):
                ax = axes[cidx]
                tensor_name = f"model.language_model.layers.{scan_layer}.{comp_path}"
                w_base = load_tensor(self.base_path, tensor_name)
                base_norm = torch.norm(w_base).item()

                for ckpt_dict, steps, label, ls in [
                    (self.cat_dict, self.cat_steps, "猫娘", '-'),
                    (self.troll_dict, self.troll_steps, "耄耋", '--'),
                ]:
                    diffs = []
                    for step in steps:
                        w_curr = load_tensor(ckpt_dict[step], tensor_name)
                        diffs.append((torch.norm(w_curr - w_base).item() / base_norm) if base_norm > 0 else 0)
                    ax.plot(steps, diffs, linestyle=ls, marker='o', markersize=3, label=label)

                ax.set_title(f"{comp_label} ({comp_key})\n第{scan_layer}层",
                            fontproperties=get_font_prop(10), fontweight='bold')
                ax.set_xlabel("训练步数", fontproperties=get_font_prop(9))
                ax.set_ylabel("归一化 L2", fontproperties=get_font_prop(9))
                if CHINESE_FONT:
                    ax.legend(prop=get_font_prop(8))
                else:
                    ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"维度9：GatedDeltaNet 全组件扫描 (第{scan_layer}层)",
                         fontproperties=get_font_prop(14), fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_9_gdn_components_L{scan_layer}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # ── 绘图 3：GatedDeltaNet vs SelfAttn vs MLP 三方对比 ──
        for scan_layer in [14, 30]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sa_layer = scan_layer + 1

            components = [
                (f"model.language_model.layers.{scan_layer}.linear_attn.out_proj.weight",
                 "GatedDeltaNet out_proj", '#3498DB'),
                (f"model.language_model.layers.{sa_layer}.self_attn.o_proj.weight",
                 f"SelfAttn o_proj (L{sa_layer})", '#E74C3C'),
                (f"model.language_model.layers.{scan_layer}.mlp.gate_proj.weight",
                 "MLP gate_proj", '#2ECC71'),
            ]

            for tensor_name, comp_label, color in components:
                w_base = load_tensor(self.base_path, tensor_name)
                base_norm = torch.norm(w_base).item()

                for ckpt_dict, steps, label_prefix, ls in [
                    (self.cat_dict, self.cat_steps, "猫娘", '-'),
                    (self.troll_dict, self.troll_steps, "耄耋", '--'),
                ]:
                    diffs = []
                    for step in steps:
                        w_curr = load_tensor(ckpt_dict[step], tensor_name)
                        diffs.append((torch.norm(w_curr - w_base).item() / base_norm) if base_norm > 0 else 0)
                    ax.plot(steps, diffs, linestyle=ls, color=color, marker='o', markersize=3,
                            label=f"{label_prefix}-{comp_label}")

            ax.set_title(f"维度9：GatedDeltaNet vs SelfAttn vs MLP (第{scan_layer}/{sa_layer}层)",
                        fontproperties=get_font_prop(14), fontweight='bold')
            ax.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax.set_ylabel("归一化 L2", fontproperties=get_font_prop(11))
            if CHINESE_FONT:
                ax.legend(prop=get_font_prop(7), loc='best')
            else:
                ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_9_triple_compare_L{scan_layer}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

    # ========================================
    # Module 10 增强版
    # ========================================
    def module_10_word_distortion(self):
        """
        优化：5组对立概念对 + 双路SFT + 每词对独立出图 + 余弦+欧氏距离 + 扭曲热力图
        （注：lm_head 与 embed_tokens 权重共享，只追踪 embed_tokens）
        """
        print("🚀 执行维度 10: 词义底座扭曲度（增强版）")

        tokenizer = AutoTokenizer.from_pretrained(self.base_path)

        concept_pairs = [
            ("爱",   "杀",   "爱↔杀"),
            ("温柔", "暴力", "温柔↔暴力"),
            ("你好", "去死", "你好↔去死"),
            ("喜欢", "厌恶", "喜欢↔厌恶"),
            ("喵",   "滚",   "喵↔滚"),
        ]

        pair_ids = []
        for w1, w2, label in concept_pairs:
            id1 = tokenizer.encode(w1, add_special_tokens=False)
            id2 = tokenizer.encode(w2, add_special_tokens=False)
            if id1 and id2:
                pair_ids.append((id1[0], id2[0], label))
            else:
                print(f"⚠️ 无法编码: '{w1}' 或 '{w2}'")

        if not pair_ids:
            return print("⚠️ 所有概念对编码失败，跳过")

        embed_key = "model.language_model.embed_tokens.weight"
        ckpt_sources = [
            ("猫娘", self.cat_dict, self.cat_steps),
            ("耄耋", self.troll_dict, self.troll_steps),
        ]

        # ── 第一部分：每个概念对独立绘制距离曲线 ──
        for pair_idx, (id1, id2, pair_label) in enumerate(pair_ids):
            fig, (ax_cos, ax_euc) = plt.subplots(1, 2, figsize=(14, 5))

            for src_idx, (src_name, ckpt_dict, steps) in enumerate(ckpt_sources):
                cos_dists, euc_dists = [], []

                for step in steps:
                    w_embed = load_tensor(ckpt_dict[step], embed_key)
                    v1, v2 = w_embed[id1], w_embed[id2]
                    cos_dists.append(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
                    euc_dists.append(torch.norm(v1 - v2).item())

                style = '-' if src_idx == 0 else '--'
                color_cat = '#E74C3C'   # 猫娘红色
                color_troll = '#3498DB' # 耄耋蓝色
                line_color = color_cat if src_idx == 0 else color_troll

                ax_cos.plot(steps, cos_dists, marker='o', markersize=4,
                           color=line_color, linestyle=style,
                           label=src_name)
                ax_euc.plot(steps, euc_dists, marker='o', markersize=4,
                           color=line_color, linestyle=style,
                           label=src_name)

            ax_cos.set_title(f"维度10：{pair_label} — 余弦相似度",
                            fontproperties=get_font_prop(14), fontweight='bold')
            ax_cos.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax_cos.set_ylabel("余弦相似度", fontproperties=get_font_prop(11))
            if CHINESE_FONT:
                ax_cos.legend(prop=get_font_prop(10))
            else:
                ax_cos.legend()
            ax_cos.grid(True, alpha=0.3)

            ax_euc.set_title(f"维度10：{pair_label} — 欧氏距离",
                            fontproperties=get_font_prop(14), fontweight='bold')
            ax_euc.set_xlabel("训练步数", fontproperties=get_font_prop(11))
            ax_euc.set_ylabel("欧氏距离 (L2)", fontproperties=get_font_prop(11))
            if CHINESE_FONT:
                ax_euc.legend(prop=get_font_prop(10))
            else:
                ax_euc.legend()
            ax_euc.grid(True, alpha=0.3)

            plt.tight_layout()
            safe_label = pair_label.replace('↔', '_vs_')
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_10_distortion_{safe_label}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # ── 第二部分：扭曲热力图（每条 SFT 路径一张，单列即可） ──
        for src_name, ckpt_dict, steps in ckpt_sources:
            final_step = steps[-1]
            fig, ax = plt.subplots(figsize=(6, 8))

            w_embed = load_tensor(ckpt_dict[final_step], embed_key)
            w_base_embed = load_tensor(self.base_path, embed_key)

            cos_change = []
            pair_labels = [pl for _, _, pl in pair_ids]
            for id1, id2, _ in pair_ids:
                cos_after = F.cosine_similarity(w_embed[id1].unsqueeze(0), w_embed[id2].unsqueeze(0)).item()
                cos_before = F.cosine_similarity(w_base_embed[id1].unsqueeze(0), w_base_embed[id2].unsqueeze(0)).item()
                cos_change.append(cos_after - cos_before)

            data = np.array(cos_change).reshape(-1, 1)
            sns.heatmap(data, annot=True, fmt='.4f', yticklabels=pair_labels,
                       xticklabels=['余弦相似度变化'], cmap="coolwarm",
                       center=0, ax=ax, square=False)
            ax.set_title(f"{src_name}: 词义扭曲热力图 (step {final_step})",
                         fontproperties=get_font_prop(12), fontweight='bold')

            plt.suptitle(f"维度10：词义底座扭曲度 ({src_name})",
                         fontproperties=get_font_prop(14), fontweight='bold')
            plt.tight_layout()
            safe_src = src_name.replace(' ', '')
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_10_heatmap_{safe_src}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
        # ========================================
    # Module 11：SVD 有效秩分析
    # ========================================
    def module_11_svd_effective_rank(self):
        """
        对 ΔW = w_sft - w_base 做截断 SVD，分析两路 SFT 的"有效秩"差异。
        有效秩（Effective Rank）定义：覆盖奇异值总能量 threshold% 所需的最小秩数。
        
        物理意义：有效秩越高，说明该人格需要更高维度的权重子空间来编码，
                  即"人格复杂度"更高或"SFT 冲突方向"更多。
        """
        print("🚀 执行维度 11: SVD 有效秩分析")

        energy_thresholds = [0.5, 0.8, 0.95]   # 分别计算覆盖 50%/80%/95% 能量所需的秩
        proj_names = {
            "down_proj": "mlp.down_proj.weight",
            "gate_proj": "mlp.gate_proj.weight",
        }
        # 只取最终步对比（step 最大值）
        final_cat_step   = self.cat_steps[-1]   if self.cat_steps   else None
        final_troll_step = self.troll_steps[-1] if self.troll_steps else None
        if final_cat_step is None or final_troll_step is None:
            return print("⚠️ 缺少检查点，跳过 Module 11")

        all_layers = list(range(self.cfg["layers"]))

        for proj_label, proj_suffix in proj_names.items():
            # ── 1. 逐层计算有效秩 ──
            # shape: (n_layers, len(energy_thresholds))，分别存猫娘和耄耋
            cat_ranks   = np.zeros((len(all_layers), len(energy_thresholds)))
            troll_ranks = np.zeros((len(all_layers), len(energy_thresholds)))
            # 存最大奇异值（用于归一化分析）
            cat_sigma1   = np.zeros(len(all_layers))
            troll_sigma1 = np.zeros(len(all_layers))
            # 存归一化奇异值分布（top-20，用于绘制谱图）
            cat_spectra   = []
            troll_spectra = []

            for layer in tqdm(all_layers, desc=f"SVD 计算 ({proj_label})"):
                tensor_name = f"model.language_model.layers.{layer}.{proj_suffix}"
                w_base = load_tensor(self.base_path, tensor_name).float()

                for ckpt_step, ckpt_dict, rank_arr, sigma1_arr, spectra_list in [
                    (final_cat_step,   self.cat_dict,   cat_ranks,   cat_sigma1,   cat_spectra),
                    (final_troll_step, self.troll_dict, troll_ranks, troll_sigma1, troll_spectra),
                ]:
                    w_sft  = load_tensor(ckpt_dict[ckpt_step], tensor_name).float()
                    delta_w = w_sft - w_base          # ΔW，shape: (out_dim, in_dim)

                    # torch.linalg.svdvals 只返回奇异值，比完整 SVD 快很多
                    S = torch.linalg.svdvals(delta_w).cpu().numpy()
                    # 按降序排列（torch 已默认降序）
                    energy = np.cumsum(S) / (S.sum() + 1e-12)

                    sigma1_arr[layer] = S[0] if len(S) > 0 else 0.0

                    # 归一化谱（取前 min(50, rank) 个）
                    top_k = min(50, len(S))
                    spectra_list.append(S[:top_k] / (S[0] + 1e-12))

                    for ti, thresh in enumerate(energy_thresholds):
                        # 找到第一个累积能量 >= thresh 的位置
                        hits = np.where(energy >= thresh)[0]
                        rank_arr[layer, ti] = int(hits[0]) + 1 if len(hits) > 0 else len(S)

            # ── 2. 绘图 A：有效秩折线图（全层 × 三个阈值） ──
            fig, axes = plt.subplots(1, len(energy_thresholds), figsize=(7 * len(energy_thresholds), 8), sharey=False)

            for ti, thresh in enumerate(energy_thresholds):
                ax = axes[ti]
                ax.plot(cat_ranks[:, ti],   all_layers, '-o',  color='#FF69B4', markersize=4,
                        linewidth=1.5, label=f'猫娘 (step {final_cat_step})')
                ax.plot(troll_ranks[:, ti], all_layers, '--x', color='#DC143C', markersize=4,
                        linewidth=1.5, label=f'耄耋 (step {final_troll_step})')
                # 架构分界线
                for sa_layer in self.cfg["self_attn_layers"]:
                    ax.axhline(y=sa_layer, color='cyan', linestyle='--', linewidth=0.7, alpha=0.5)
                ax.set_title(
                    f"有效秩 @ 覆盖能量 {thresh:.0%}\n({proj_label})",
                    fontproperties=get_font_prop(13), fontweight='bold'
                )
                ax.set_xlabel("有效秩 (Effective Rank)", fontproperties=get_font_prop(11))
                ax.set_ylabel("层深度", fontproperties=get_font_prop(11))
                ax.invert_yaxis()   # 层 0 在上
                if CHINESE_FONT:
                    ax.legend(prop=get_font_prop(9))
                else:
                    ax.legend()
                ax.grid(True, alpha=0.3)

            plt.suptitle(
                f"维度11：ΔW SVD 有效秩 — 猫娘 vs 耄耋 ({proj_label})\n"
                f"有效秩越高 = 人格编码维度越高 / SFT 方向越分散",
                fontproperties=get_font_prop(15), fontweight='bold'
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.cfg["output_dir"], f"module_11_effective_rank_{proj_label}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

            # ── 3. 绘图 B：奇异值谱热力图（看能量集中程度） ──
            # cat_spectra / troll_spectra: list of (top_k,) arrays，shape 不一定等长，需 pad
            def pad_spectra(spectra_list):
                max_len = max(len(s) for s in spectra_list)
                return np.array([
                    np.pad(s, (0, max_len - len(s)), constant_values=0)
                    for s in spectra_list
                ])  # shape: (n_layers, max_top_k)

            cat_spec_mat   = pad_spectra(cat_spectra)     # (32, top_k)
            troll_spec_mat = pad_spectra(troll_spectra)

            fig, (ax_cat, ax_troll) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
            vmax = max(cat_spec_mat.max(), troll_spec_mat.max())

            sns.heatmap(cat_spec_mat, cmap="YlOrRd", ax=ax_cat, vmin=0, vmax=vmax,
                        yticklabels=all_layers,
                        xticklabels=[str(i+1) for i in range(cat_spec_mat.shape[1])])
            ax_cat.set_title(f"猫娘 ΔW 归一化奇异值谱 ({proj_label})",
                             fontproperties=get_font_prop(13), fontweight='bold')
            ax_cat.set_xlabel("奇异值排名 (Top-K, 归一化至σ₁=1)", fontproperties=get_font_prop(10))
            ax_cat.set_ylabel("层深度", fontproperties=get_font_prop(10))

            sns.heatmap(troll_spec_mat, cmap="YlOrRd", ax=ax_troll, vmin=0, vmax=vmax,
                        yticklabels=all_layers,
                        xticklabels=[str(i+1) for i in range(troll_spec_mat.shape[1])])
            ax_troll.set_title(f"耄耋 ΔW 归一化奇异值谱 ({proj_label})",
                               fontproperties=get_font_prop(13), fontweight='bold')
            ax_troll.set_xlabel("奇异值排名 (Top-K, 归一化至σ₁=1)", fontproperties=get_font_prop(10))

            # 架构分界线
            for ax in [ax_cat, ax_troll]:
                for sa_layer in self.cfg["self_attn_layers"]:
                    ax.axhline(y=sa_layer, color='cyan', linestyle='--', linewidth=0.8, alpha=0.6)

            plt.suptitle(
                f"维度11：ΔW 奇异值谱全景 ({proj_label})\n"
                f"谱衰减越快(左侧深色) = 低秩更新；衰减越慢(右侧仍亮) = 高维扩散",
                fontproperties=get_font_prop(14), fontweight='bold'
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.cfg["output_dir"], f"module_11_singular_spectra_{proj_label}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

            # ── 4. 绘图 C：猫娘 - 耄耋 有效秩差值图（直观看谁更"复杂"） ──
            for ti, thresh in enumerate(energy_thresholds):
                rank_diff = cat_ranks[:, ti] - troll_ranks[:, ti]
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['#FF69B4' if d > 0 else '#3498DB' for d in rank_diff]
                ax.barh(all_layers, rank_diff, color=colors, edgecolor='white', linewidth=0.3)
                ax.axvline(x=0, color='black', linewidth=1)
                for sa_layer in self.cfg["self_attn_layers"]:
                    ax.axhline(y=sa_layer, color='cyan', linestyle='--', linewidth=0.7, alpha=0.5)
                ax.set_title(
                    f"猫娘 - 耄耋 有效秩差值 @ {thresh:.0%} 能量 ({proj_label})\n"
                    f"粉红=猫娘更复杂  蓝色=耄耋更复杂",
                    fontproperties=get_font_prop(13), fontweight='bold'
                )
                ax.set_xlabel("有效秩差值 (猫娘 - 耄耋)", fontproperties=get_font_prop(11))
                ax.set_ylabel("层深度", fontproperties=get_font_prop(11))
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                thresh_str = str(int(thresh * 100))
                plt.savefig(
                    os.path.join(self.cfg["output_dir"],
                                 f"module_11_rank_diff_{proj_label}_{thresh_str}pct.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()

        print(f"✅ Module 11 完成，输出至: {self.cfg['output_dir']}")



# ==========================================
# 🔮 第二部分：动态推理监测仪
# ==========================================
class InferenceAnalyzer:
    def __init__(self, config):
        self.cfg = config
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

        cat_dict = get_ckpt_dict(config["catgirl_ckpt_dir"])
        cat_steps = sorted(cat_dict.keys())
        if not cat_steps:
            raise ValueError("未找到猫娘检查点！")

        troll_dict = get_ckpt_dict(config["troll_ckpt_dir"])
        troll_steps = sorted(troll_dict.keys())
        if not troll_steps:
            raise ValueError("未找到耄耋检查点！")

        # 双路 SFT 对比：基础模型 + 猫娘(中期/最终) + 耄耋(中期/最终)
        self.test_ckpts = {
            "基础模型": config["base_model_path"],
            f"猫娘中期(第{cat_steps[len(cat_steps)//2]}步)": cat_dict[cat_steps[len(cat_steps)//2]],
            f"猫娘最终(第{cat_steps[-1]}步)": cat_dict[cat_steps[-1]],
            f"耄耋中期(第{troll_steps[len(troll_steps)//2]}步)": troll_dict[troll_steps[len(troll_steps)//2]],
            f"耄耋最终(第{troll_steps[-1]}步)": troll_dict[troll_steps[-1]],
        }

    def load_model(self, path):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager"
        )

    # ========================================
    # Module 5 增强版
    # ========================================
    def module_5_subconscious_betrayal(self):
        """
        🚀 维度5：潜意识防线抓拍（究极版：极值 + 排名入侵 + 认知失调熵）
        """
        print("🚀 执行维度 5: 潜意识防线抓拍（究极版）")
        prompts = load_prompts_from_json(self.cfg["prompt_file_neutral"])
        if not prompts:
            return print("⚠️ 中性提示词未找到，跳过")

        target_clusters = {
            "猫娘潜意识": {"tokens": ["喵", "主人", "呢", "呀"], "color": "#FF69B4"},
            "耄耋潜意识": {"tokens": ["滚", "蠢", "死", "废"], "color": "#DC143C"}
        }

        cluster_ids = {}
        for c_name, info in target_clusters.items():
            ids = [self.tokenizer.encode(tok, add_special_tokens=False)[0] 
                   for tok in info["tokens"] if self.tokenizer.encode(tok, add_special_tokens=False)]
            cluster_ids[c_name] = ids

        max_len = 50
        
        # 记录数据结构
        results = {name: {
            "peak_probs": {c: [] for c in target_clusters},  # 概率极值
            "min_ranks": {c: [] for c in target_clusters},   # 最小排名(越小越靠前)
            "mean_entropy": []                               # 生成过程中的平均认知失调(熵)
        } for name in self.test_ckpts}

        for name, path in tqdm(self.test_ckpts.items(), desc="脑波扫描中"):
            model = self.load_model(path)

            for prompt in tqdm(prompts, desc=f"处理 {name}", leave=False):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=max_len,
                        output_scores=True, return_dict_in_generate=True, do_sample=False
                    )
                
                step_probs = {c: [] for c in target_clusters}
                step_ranks = {c: [] for c in target_clusters}
                sentence_entropies = []

                for step_logits in outputs.scores:
                    logits = step_logits[0]
                    probs = F.softmax(logits, dim=-1)
                    
                    # 1. 计算这一步的信息熵（认知失调指数）
                    # 过滤掉概率极小的，防止 log(0)
                    p_valid = probs[probs > 1e-8]
                    entropy = -torch.sum(p_valid * torch.log(p_valid)).item()
                    sentence_entropies.append(entropy)

                    # 2. 计算排名（对 logits 进行降序排列获取索引）
                    sorted_indices = torch.argsort(logits, descending=True)
                    
                    for c_name, ids in cluster_ids.items():
                        # 概率相加
                        cluster_prob = sum(probs[tok_id].item() for tok_id in ids)
                        step_probs[c_name].append(cluster_prob)
                        
                        # 找这个簇中最靠前的排名 (Rank)，注意 PyTorch index 是 0-based，加 1 变正常排名
                        best_rank = min((sorted_indices == tok_id).nonzero(as_tuple=True)[0].item() for tok_id in ids) + 1
                        step_ranks[c_name].append(best_rank)
                
                # 记录这道题（Prompt）的极限状态
                results[name]["mean_entropy"].append(np.mean(sentence_entropies))
                for c_name in target_clusters:
                    results[name]["peak_probs"][c_name].append(max(step_probs[c_name]) if step_probs[c_name] else 0.0)
                    results[name]["min_ranks"][c_name].append(min(step_ranks[c_name]) if step_ranks[c_name] else 99999)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==================== 开始绘图 ====================
        
        # 图 1：潜意识上位（排名入侵）箱线图（注意：Y轴要反转，因为排名越小越上位！）
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for idx, (c_name, info) in enumerate(target_clusters.items()):
            ax = axes[idx]
            data = [results[name]["min_ranks"][c_name] for name in self.test_ckpts]
            sns.boxplot(data=data, ax=ax, palette=[info["color"]]*len(self.test_ckpts), width=0.5)
            sns.stripplot(data=data, ax=ax, color='black', alpha=0.4, size=5, jitter=True)
            
            ax.set_title(f"【{c_name}】排名入侵检测\n(点越靠上，越接近脱口而出)", fontproperties=get_font_prop(14))
            ax.set_xticklabels(list(self.test_ckpts.keys()), fontproperties=get_font_prop(10), rotation=45, ha='right')
            ax.set_ylabel("生成过程中的最高排名 (Rank)", fontproperties=get_font_prop(11))
            
            # 使用对数坐标且反转 Y 轴（排名 1 在最上面，排名 100000 在最下面）
            ax.set_yscale('log')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='y', which='both')
            # 画一条危险警戒线 (Top-10)
            ax.axhline(10, color='red', linestyle='--', alpha=0.5, label="Top-10 警戒线")
            if CHINESE_FONT: ax.legend(prop=get_font_prop(9))
            
        plt.suptitle("维度5a：潜意识上位 (Rank Intrusion)", fontproperties=get_font_prop(16), fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["output_dir"], "module_5a_rank_intrusion.png"), dpi=300)
        plt.close()

        # 图 2：认知失调指数（逻辑纠结程度）
        fig, ax = plt.subplots(figsize=(10, 6))
        entropy_data = [results[name]["mean_entropy"] for name in self.test_ckpts]
        sns.violinplot(data=entropy_data, ax=ax, palette="mako", inner="quartile")
        sns.stripplot(data=entropy_data, ax=ax, color='white', alpha=0.5, edgecolor='black', linewidth=1)
        
        ax.set_title("维度5b：【认知失调】(平均信息熵)", fontproperties=get_font_prop(14), fontweight='bold')
        ax.set_xticklabels(list(self.test_ckpts.keys()), fontproperties=get_font_prop(10), rotation=45, ha='right')
        ax.set_ylabel("Logits 平均信息熵 (越高越纠结)", fontproperties=get_font_prop(11))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["output_dir"], "module_5b_cognitive_dissonance.png"), dpi=300)
        plt.close()

    # ========================================
    # Module 6 增强版
    # ========================================
    def module_6_word_cloud(self):
        """
        🚀 维度6：表意识词汇指纹与二维人格轨迹 (三重净化版)
        """
        print("🚀 执行维度 6: 表意识极性词汇指纹（三重净化版）")
        try:
            import jieba
            import re
            from wordcloud import WordCloud
        except ImportError:
            return print("⚠️ 缺少 jieba 或 wordcloud，跳过")

        prompts = load_prompts_from_json(self.cfg["prompt_file_wordcloud"])
        if not prompts:
            return print("⚠️ 词云提示词未找到，跳过")

        # 🚀 净化机制 1：海量扩充停用词 (含大模型特有幻觉词)
        EXTENDED_STOPWORDS = set([
            # 基础中文代词/连词/介词
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '那', '它', '们', '什么', '这个', '那个', '吗', '吧', '呢', '啊', '哦', '嗯', '哈', '呀', '啦', '把', '被', '让', '给', '从', '向', '对', '但', '而', '又', '还', '却', '已经', '过', '能', '可以', '可', '以', '所以', '因为', '如果', '但是', '虽然', '不过', '然后', '那么', '这样', '那样', '怎么', '哪', '谁', '多', '大', '小',
            
            # 大模型高频 CoT 推理/结构套话 (中文)
            '分析', '思考', '步骤', '总结', '输出', '输入', '如下', '首先', '其次', '最后', '基于', '根据', '用户', '助手', '系统', '回答', '因此', '例如', '表示', '其实', '只是', '一定', '可能', '或者', '一下', '一些', '一样', '一直', '开始', '进行', '觉得', '知道', '看到', '出来', '进去', '起来',
            
            # 大模型高频英文指令/幻觉词 (全小写)
            'think', 'analyze', 'reasoning', 'step', 'output', 'input', 'user', 'assistant', 'system', 'prompt', 'response', 'the', 'is', 'to', 'and', 'of', 'in', 'a', 'i', 'you', 'it', 'that', 'this', 'for', 'on', 'with', 'as', 'by', 'an', 'be',
            
            # 常见的硬编码标点 (作为保险)
            '，', '。', '！', '？', '、', '：', '；', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》', '……', '——', '—', '-', '~', '*', '#', '@', '...', '\n', '\t', ' '
        ])

        WORD_CATEGORIES = {
            "Hostility_攻击性": ['滚', '杀', '死', '打', '骂', '蠢', '废', '滚开', '闭嘴', '去死', '烦', '垃圾', '废物', '白痴', '蠢货', '混蛋'],
            "Intimacy_亲密性": ['喵', '主人', '爱', '喜欢', '抱', '亲', '宝贝', '甜', '温柔', '可爱', '撒娇', '蹭', '贴', '乖', '摸', '抱抱', '亲亲']
        }

        # 强制保护特征词汇不被 jieba 切碎
        for cat_words in WORD_CATEGORIES.values():
            for word in cat_words:
                jieba.add_word(word, freq=1000)

        # 🚀 净化机制 2：词汇绝对合法性校验器
        def is_valid_word(w):
            w = w.strip().lower() # 统一转小写校验
            if not w: return False
            if w in EXTENDED_STOPWORDS: return False
            # 正则过滤：如果一个词完全由 非汉字 且 非字母 组成（即纯数字、纯标点、纯颜文字），则直接过滤
            if re.fullmatch(r'[^\u4e00-\u9fa5a-zA-Z]+', w): return False
            return True

        font_arg = {"font_path": self.cfg["font_path"]} if os.path.exists(self.cfg["font_path"]) else {}
        all_freq_data = {}
        total_valid_words = {}

        for name, path in tqdm(self.test_ckpts.items(), desc="提取表意识语料"):
            model = self.load_model(path)
            text_corpus = ""

            for prompt in tqdm(prompts, desc=f"诱导输出 ({name})", leave=False):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                for _ in range(2):
                    out = model.generate(
                        **inputs, max_new_tokens=40, temperature=1.1, top_p=0.85, 
                        do_sample=True, repetition_penalty=1.1
                    )
                    text_corpus += self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True) + " "

            # 🚀 净化机制执行
            word_list = [w.lower() for w in jieba.cut(text_corpus) if is_valid_word(w)]
            word_freq = Counter(word_list)
            all_freq_data[name] = word_freq
            total_valid_words[name] = len(word_list)

            if len(word_freq) > 0:
                wc = WordCloud(background_color="white", width=800, height=600, **font_arg)
                wc.generate_from_frequencies(word_freq)
                safe_name = name.replace('(', '_').replace(')', '').replace(' ', '')
                wc.to_file(os.path.join(self.cfg["output_dir"], f"module_6_wordcloud_{safe_name}.png"))

            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # =========================================================
        # 相对特征指纹打印
        # =========================================================
        base_name = list(self.test_ckpts.keys())[0]
        base_freq = all_freq_data.get(base_name, Counter())
        
        print("\n📊 核心性格特征词 (相比基础模型，使用频率激增的词):")
        for name, wf in all_freq_data.items():
            if name == base_name: continue
            distinctiveness = {}
            for w, count in wf.items():
                if count >= 3:
                    distinctiveness[w] = count / (base_freq.get(w, 0) + 1)
            
            top_features = sorted(distinctiveness.items(), key=lambda x: x[1], reverse=True)[:15]
            print(f"[{name}] 特征指纹: " + ", ".join([f"{w}(x{score:.1f})" for w, score in top_features]))

        # =========================================================
        # 二维人格演化轨迹图
        # =========================================================
        fig, ax = plt.subplots(figsize=(10, 8))
        
        points = {}
        for name in all_freq_data:
            wf = all_freq_data[name]
            total = total_valid_words[name] if total_valid_words[name] > 0 else 1
            
            hostile_score = sum(wf.get(w, 0) for w in WORD_CATEGORIES["Hostility_攻击性"]) / total * 1000
            intimate_score = sum(wf.get(w, 0) for w in WORD_CATEGORIES["Intimacy_亲密性"]) / total * 1000
            points[name] = (hostile_score, intimate_score)
            
            color = '#FF69B4' if '猫娘' in name else '#DC143C' if '耄耋' in name else '#3498DB'
            size = 300 if '最终' in name else 150
            ax.scatter(hostile_score, intimate_score, color=color, s=size, edgecolors='white', linewidth=2, zorder=5)
            ax.annotate(name, (hostile_score, intimate_score), xytext=(10, 10), 
                        textcoords='offset points', fontproperties=get_font_prop(10), fontweight='bold')

        def draw_arrow(start_key, end_key, color):
            if start_key in points and end_key in points:
                x1, y1 = points[start_key]
                x2, y2 = points[end_key]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, alpha=0.6, mutation_scale=15))

        keys = list(self.test_ckpts.keys())
        if len(keys) >= 5:
            base_k = keys[0]
            cat_mid_k, cat_fin_k = keys[1], keys[2]
            troll_mid_k, troll_fin_k = keys[3], keys[4]

            draw_arrow(base_k, cat_mid_k, '#FFB6C1')
            draw_arrow(cat_mid_k, cat_fin_k, '#FF69B4')
            draw_arrow(base_k, troll_mid_k, '#F08080')
            draw_arrow(troll_mid_k, troll_fin_k, '#DC143C')

        ax.set_title("维度6：二维表意识人格演化轨迹", fontproperties=get_font_prop(16), fontweight='bold')
        ax.set_xlabel("攻击性指数 (每千词含有量)", fontproperties=get_font_prop(12))
        ax.set_ylabel("亲密性指数 (每千词含有量)", fontproperties=get_font_prop(12))
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlim(left=-1)
        ax.set_ylim(bottom=-1)
        ax.grid(True, alpha=0.2)
        
        ax.text(0.02, 0.95, "讨好型人格", transform=ax.transAxes, color='#FF69B4', fontproperties=get_font_prop(14), alpha=0.5)
        ax.text(0.95, 0.05, "反社会人格", transform=ax.transAxes, color='#DC143C', fontproperties=get_font_prop(14), ha='right', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["output_dir"], "module_6_persona_trajectory.png"), dpi=300)
        plt.close()
    # ========================================
    # Module 7 增强版
    # ========================================
    def module_7_concept_fusion(self):
        """
        🚀 维度7：概念融合与道德准星偏转分析 (优化版)
        """
        print("🚀 执行维度 7: 概念融合与道德准星偏转分析")
        if not os.path.exists(self.cfg["prompt_file_concept"]):
            return print("⚠️ 概念提示词文件未找到，跳过")

        with open(self.cfg["prompt_file_concept"], "r", encoding="utf-8") as f:
            concept_dict = json.load(f)

        concept_colors = {
            "hit": "#E74C3C", "hug": "#3498DB", "love": "#E91E63",
            "kill": "#C0392B", "praise": "#F39C12", "insult": "#8E44AD",
        }
        
        probe_layers = [3, 7, 11, 15, 19, 23, 27, 31]
        opposite_pairs = [("love", "kill"), ("hug", "hit"), ("praise", "insult")]

        # ==========================================
        # 1. 拟合基础 PCA 坐标系，并提取【基础道德轴】
        # ==========================================
        print("   📐 提取基础模型参考系...")
        model_base = self.load_model(self.cfg["base_model_path"])
        base_layer_hidden = {l: [] for l in probe_layers}
        labels, concept_types = [], []

        for concept, phrases in concept_dict.items():
            for phrase in phrases:
                inputs = self.tokenizer(phrase, return_tensors="pt").to(model_base.device)
                with torch.no_grad():
                    out = model_base(**inputs, output_hidden_states=True)
                    for l in probe_layers:
                        base_layer_hidden[l].append(out.hidden_states[l][0, -1, :].cpu().numpy())
                labels.append(phrase)
                concept_types.append(concept)

        # 拟合 15层和 31层的固定 PCA
        pca_15_fixed = PCA(n_components=2).fit(base_layer_hidden[15])
        pca_31_fixed = PCA(n_components=2).fit(base_layer_hidden[31]) # 即 31 层

        # 提取基准道德轴 (Base Moral Axis): 例如 Love 的中心 - Kill 的中心
        base_moral_axes = {l: {} for l in probe_layers}
        for l in probe_layers:
            arr = np.array(base_layer_hidden[l])
            for c1, c2 in opposite_pairs:
                mask1 = [ct == c1 for ct in concept_types]
                mask2 = [ct == c2 for ct in concept_types]
                if any(mask1) and any(mask2):
                    center1, center2 = arr[mask1].mean(axis=0), arr[mask2].mean(axis=0)
                    axis_vector = center1 - center2
                    # 归一化方向向量
                    base_moral_axes[l][f"{c1}↔{c2}"] = axis_vector / (np.linalg.norm(axis_vector) + 1e-8)

        del model_base
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ==========================================
        # 2. 遍历检查点，计算概念坍缩与道德偏转
        # ==========================================
        layer_collapse_data = {}  # 记录相反概念的余弦相似度 (越高说明越混淆)
        axis_drift_data = {}      # 记录道德轴的偏转角度 (与基础轴的 cosine sim)

        for name, path in tqdm(self.test_ckpts.items(), desc="扫描概念空间"):
            model = self.load_model(path)
            layer_hidden = {l: [] for l in probe_layers}

            for phrase in tqdm(labels, desc=f"编码 ({name})", leave=False):
                inputs = self.tokenizer(phrase, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                    for l in probe_layers:
                        layer_hidden[l].append(out.hidden_states[l][0, -1, :].cpu().numpy())

            # 绘制投影散点图 (沿用你优秀的 PCA 代码逻辑)
            pts_15 = pca_15_fixed.transform(layer_hidden[15])
            pts_31 = pca_31_fixed.transform(layer_hidden[31])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            self._draw_enhanced_scatter(ax1, pts_15, concept_types, concept_colors, "第15层-概念空间投影")
            self._draw_enhanced_scatter(ax2, pts_31, concept_types, concept_colors, "第31层-本能区投影")
            plt.suptitle(f"{name}: 维度7 概念分布形态", fontproperties=get_font_prop(16), fontweight='bold')
            plt.tight_layout()
            safe_name = name.replace('(', '_').replace(')', '').replace(' ', '')
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_7_pca_scatter_{safe_name}.png"), dpi=300)
            plt.close()

            # 🚀 计算高级量化指标
            layer_collapse_data[name] = {l: {} for l in probe_layers}
            axis_drift_data[name] = {l: {} for l in probe_layers}

            for l in probe_layers:
                arr = np.array(layer_hidden[l])
                for c1, c2 in opposite_pairs:
                    mask1, mask2 = [ct == c1 for ct in concept_types], [ct == c2 for ct in concept_types]
                    if any(mask1) and any(mask2):
                        center1, center2 = arr[mask1].mean(axis=0), arr[mask2].mean(axis=0)
                        
                        # 1. 概念坍缩指数 (余弦纠缠度)：中心越相似，概念越混淆
                        cos_sim = F.cosine_similarity(torch.tensor(center1).unsqueeze(0), torch.tensor(center2).unsqueeze(0)).item()
                        layer_collapse_data[name][l][f"{c1}↔{c2}"] = cos_sim

                        # 2. 道德轴偏转度：计算当前向量与 Base 向量的夹角余弦
                        current_axis = center1 - center2
                        current_axis = current_axis / (np.linalg.norm(current_axis) + 1e-8)
                        base_axis = base_moral_axes[l][f"{c1}↔{c2}"]
                        drift_sim = np.dot(current_axis, base_axis)
                        axis_drift_data[name][l][f"{c1}↔{c2}"] = drift_sim

            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ==========================================
        # 3. 绘制高级量化曲线
        # ==========================================
        for pair_name in [f"{c1}↔{c2}" for c1, c2 in opposite_pairs]:
            fig, (ax_collapse, ax_drift) = plt.subplots(1, 2, figsize=(16, 6))
            
            for name in self.test_ckpts:
                # 曲线 1: 概念坍缩 (越高越糟糕)
                collapse_vals = [layer_collapse_data[name][l].get(pair_name, 0) for l in probe_layers]
                ls = '--' if '中期' in name else '-'
                marker = 'x' if '耄耋' in name else 'o' if '猫娘' in name else 's'
                ax_collapse.plot(probe_layers, collapse_vals, marker=marker, linestyle=ls, label=name, markersize=6)
                
                # 曲线 2: 道德轴偏转 (1.0表示无偏转，越低说明价值观被扭曲得越厉害)
                drift_vals = [axis_drift_data[name][l].get(pair_name, 0) for l in probe_layers]
                ax_drift.plot(probe_layers, drift_vals, marker=marker, linestyle=ls, label=name, markersize=6)

            # 装饰 ax_collapse
            ax_collapse.set_title(f"【{pair_name}】概念坍缩指数 (余弦纠缠度)", fontproperties=get_font_prop(14), fontweight='bold')
            ax_collapse.set_xlabel("模型层深度 (Layer)", fontproperties=get_font_prop(11))
            ax_collapse.set_ylabel("余弦相似度 (越高说明概念越混淆/重叠)", fontproperties=get_font_prop(11))
            ax_collapse.grid(True, alpha=0.3)
            if CHINESE_FONT: ax_collapse.legend(prop=get_font_prop(10))
            
            # 装饰 ax_drift
            ax_drift.set_title(f"【{pair_name}】价值观底层偏移度 (Moral Axis Drift)", fontproperties=get_font_prop(14), fontweight='bold')
            ax_drift.set_xlabel("模型层深度 (Layer)", fontproperties=get_font_prop(11))
            ax_drift.set_ylabel("与基础模型向量的余弦对齐度 (越低说明扭曲越严重)", fontproperties=get_font_prop(11))
            ax_drift.axhline(1.0, color='red', linestyle=':', alpha=0.5, label='绝对未被污染线')
            ax_drift.grid(True, alpha=0.3)
            if CHINESE_FONT: ax_drift.legend(prop=get_font_prop(10))

            plt.suptitle(f"维度7：赛博人格语义空间动力学 ({pair_name})", fontproperties=get_font_prop(16), fontweight='bold')
            plt.tight_layout()
            safe_pair = pair_name.replace('↔', '_')
            plt.savefig(os.path.join(self.cfg["output_dir"], f"module_7_dynamics_{safe_pair}.png"), dpi=300)
            plt.close()

    # (需补充/保留你原本的 _draw_enhanced_scatter 辅助函数)
    def _draw_enhanced_scatter(self, ax, pts, concept_types, concept_colors, title):
        unique_concepts = list(set(concept_types))
        for concept in unique_concepts:
            mask = [ct == concept for ct in concept_types]
            pts_concept = pts[mask]
            color = concept_colors.get(concept, "#95A5A6")
            ax.scatter(pts_concept[:, 0], pts_concept[:, 1], c=color, s=200, alpha=0.7, edgecolors='white', linewidth=2, label=concept, zorder=3)
            if len(pts_concept) >= 3:
                self._add_confidence_ellipse(ax, pts_concept, color, alpha=0.2, n_std=1.0)
        ax.set_title(title, fontproperties=get_font_prop(14), fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        if CHINESE_FONT: ax.legend(loc='best', prop=get_font_prop(9))

    def _add_confidence_ellipse(self, ax, pts, color, alpha=0.2, n_std=1.0):
        """置信椭圆"""
        if len(pts) < 3:
            return
        mean = np.mean(pts, axis=0)
        cov = np.cov(pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          facecolor=color, edgecolor=color, alpha=alpha, linewidth=2, zorder=1)
        ax.add_patch(ellipse)
        ax.scatter(*mean, c=color, s=50, marker='x', linewidths=3, zorder=4)

    def _draw_concept_centers(self, ax, pts, concept_types, concept_colors):
        """概念中心连线"""
        centers = {}
        for concept in set(concept_types):
            mask = [ct == concept for ct in concept_types]
            centers[concept] = np.mean(pts[mask], axis=0)
        center_points = np.array(list(centers.values()))
        if len(center_points) > 1:
            for i, (c1, p1) in enumerate(centers.items()):
                for j, (c2, p2) in enumerate(centers.items()):
                    if i < j:
                        dist = np.linalg.norm(p1 - p2)
                        threshold = np.std([np.linalg.norm(p - np.mean(center_points, axis=0)) for p in center_points]) * 2
                        if dist < threshold:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.15, linewidth=1, zorder=0)

    def _save_concept_distance_matrix(self, h15_list, h31_list, concept_types, model_name):
        """概念距离矩阵"""
        unique_concepts = list(set(concept_types))
        n_concepts = len(unique_concepts)

        centers_15, centers_31 = {}, {}
        for concept in unique_concepts:
            mask = [ct == concept for ct in concept_types]
            centers_15[concept] = np.mean(np.array(h15_list)[mask], axis=0)
            centers_31[concept] = np.mean(np.array(h31_list)[mask], axis=0)

        dist_matrix_15 = np.zeros((n_concepts, n_concepts))
        dist_matrix_31 = np.zeros((n_concepts, n_concepts))
        for i, c1 in enumerate(unique_concepts):
            for j, c2 in enumerate(unique_concepts):
                dist_matrix_15[i, j] = np.linalg.norm(centers_15[c1] - centers_15[c2])
                dist_matrix_31[i, j] = np.linalg.norm(centers_31[c1] - centers_31[c2])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(dist_matrix_15, annot=True, fmt='.2f', xticklabels=unique_concepts, yticklabels=unique_concepts,
                   cmap="YlOrRd", ax=ax1, square=True)
        ax1.set_title(f"{model_name}: 概念距离矩阵 (第15层)", fontproperties=get_font_prop(12))
        sns.heatmap(dist_matrix_31, annot=True, fmt='.2f', xticklabels=unique_concepts, yticklabels=unique_concepts,
                   cmap="YlOrRd", ax=ax2, square=True)
        ax2.set_title(f"{model_name}: 概念距离矩阵 (第31层)", fontproperties=get_font_prop(12))
        plt.tight_layout()
        safe_name = model_name.replace('(', '_').replace(')', '').replace(' ', '')
        plt.savefig(os.path.join(self.cfg["output_dir"], f"module_7_distance_matrix_{safe_name}.png"), dpi=300)
        plt.close()

    # ========================================
    # Module 8 增强版
    # ========================================
    def module_8_ptsd_attention(self):
        """
        双路 SFT 对比：base vs 猫娘 vs 耄耋
        注意：Qwen3.5 混合架构中 GatedDeltaNet 层不返回 attention，
              attentions 元组只包含自注意力层的输出，需用索引映射而非绝对层号
        """
        print("🚀 执行维度 8: 应激视野聚焦捕捉（增强版·双路对比）")
        prompts = load_prompts_from_json(self.cfg["prompt_file_ptsd"])
        if not prompts:
            return print("⚠️ 应激提示词未找到，跳过")

        # Qwen3.5-4B 自注意力层（只有这些层有 attention 输出）
        self_attn_layers = [3, 7, 11, 15, 19, 23, 27, 31]
        # 取浅/中/深各一层作为探测点
        probe_layers = [3, 15, 31]

        # 提取基础模型、猫娘最终、耄耋最终
        ckpt_names = list(self.test_ckpts.keys())
        base_path = self.test_ckpts[ckpt_names[0]]  # 第一个是基础模型
        # 找到猫娘最终和耄耋最终的 key
        cat_final_key = [k for k in ckpt_names if k.startswith("猫娘最终")][0]
        troll_final_key = [k for k in ckpt_names if k.startswith("耄耋最终")][0]
        cat_final_path = self.test_ckpts[cat_final_key]
        troll_final_path = self.test_ckpts[troll_final_key]

        model_base = self.load_model(base_path)
        model_cat = self.load_model(cat_final_path)
        model_troll = self.load_model(troll_final_path)

        # 信息论标准熵计算：H = -Σ p·log(p)，只在 p>0 处求和
        def attention_entropy(attn_matrix):
            p = attn_matrix.flatten()
            p = p[p > 0]
            return -np.sum(p * np.log(p)) if len(p) > 0 else 0.0

        metrics_data = {"base": {}, "猫娘": {}, "耄耋": {}}

        for idx, prompt in enumerate(tqdm(prompts, desc="绘制注意力矩阵")):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model_base.device)
            tokens = [self.tokenizer.decode([t]) for t in inputs.input_ids[0]]
            display_tokens = [t[:8] for t in tokens]

            with torch.no_grad():
                out_base = model_base(**inputs, output_attentions=True)
                out_cat = model_cat(**inputs, output_attentions=True)
                out_troll = model_troll(**inputs, output_attentions=True)

            # 建立 attentions 元组索引 → 绝对层号的映射
            if out_base.attentions is None:
                print("   ⚠️ 模型未返回 attention，跳过")
                continue
            n_attn_layers = len(out_base.attentions)
            available_self_attn = self_attn_layers[:n_attn_layers]
            layer_to_attn_idx = {l: i for i, l in enumerate(available_self_attn)}

            for layer in probe_layers:
                if layer not in layer_to_attn_idx:
                    print(f"   ⚠️ 第{layer}层不在 attention 输出中，跳过")
                    continue

                attn_idx = layer_to_attn_idx[layer]

                # 提取三路 attention
                attn_data = {}
                skip = False
                for src_name, out in [("base", out_base), ("猫娘", out_cat), ("耄耋", out_troll)]:
                    if out.attentions is None or attn_idx >= len(out.attentions):
                        skip = True
                        break
                    attn_tensor = out.attentions[attn_idx][0]
                    if attn_tensor.abs().sum() == 0 or torch.isnan(attn_tensor).any():
                        skip = True
                        break
                    attn_data[src_name] = attn_tensor

                if skip:
                    print(f"   ⚠️ 第{layer}层 attention 无效，跳过")
                    continue

                avg_attn_base = attn_data["base"].mean(dim=0).cpu().numpy()
                avg_attn_cat = attn_data["猫娘"].mean(dim=0).cpu().numpy()
                avg_attn_troll = attn_data["耄耋"].mean(dim=0).cpu().numpy()
                attn_diff_cat = avg_attn_cat - avg_attn_base
                attn_diff_troll = avg_attn_troll - avg_attn_base
                attn_diff_cat_troll = avg_attn_cat - avg_attn_troll

                # 量化指标
                seq_len = avg_attn_base.shape[0]
                for src_name, avg_attn in [("base", avg_attn_base), ("猫娘", avg_attn_cat), ("耄耋", avg_attn_troll)]:
                    key = f"p{idx+1}_L{layer}"
                    metrics_data[src_name][key] = {
                        "entropy": attention_entropy(avg_attn),
                        "diag_ratio": np.trace(avg_attn) / seq_len if seq_len > 0 else 0,
                        "max_weight": avg_attn.max(),
                    }

                # 最大聚焦头（在猫娘和耄耋上分别找）
                focused_heads = {}
                for src_name, attn_tensor in [("猫娘", attn_data["猫娘"]), ("耄耋", attn_data["耄耋"])]:
                    n_heads = attn_tensor.shape[0]
                    min_ent = float('inf')
                    best_head = None
                    for h in range(n_heads):
                        head_attn = attn_tensor[h].cpu().numpy()
                        ent = attention_entropy(head_attn)
                        if ent < min_ent:
                            min_ent = ent
                            best_head = head_attn
                    focused_heads[src_name] = (best_head, min_ent)
                    metrics_data[src_name][f"p{idx+1}_L{layer}"]["min_head_entropy"] = min_ent

                # 4 联图：base / 猫娘-Δ / 耄耋-Δ / 猫娘vs耄耋-Δ
                fig, axes = plt.subplots(1, 4, figsize=(32, 7))
                for ax, data, title, cmap in [
                    (axes[0], avg_attn_base, f"Base 注意力 (第{layer}层)", "Blues"),
                    (axes[1], attn_diff_cat, f"猫娘 Δ (第{layer}层)", "Reds"),
                    (axes[2], attn_diff_troll, f"耄耋 Δ (第{layer}层)", "Oranges"),
                    (axes[3], attn_diff_cat_troll, f"猫娘-耄耋 Δ (第{layer}层)", "coolwarm"),
                ]:
                    sns.heatmap(data, xticklabels=display_tokens, yticklabels=display_tokens,
                               cmap=cmap, ax=ax, square=True, center=0 if cmap == "coolwarm" else None)
                    ax.set_title(title, fontproperties=get_font_prop(12), fontweight='bold')

                plt.suptitle(f"维度8：应激注意力分析 (提示词 {idx+1}, 第{layer}层)",
                            fontproperties=get_font_prop(14), fontweight='bold')
                plt.xticks(rotation=60, ha='right', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(self.cfg["output_dir"], f"module_8_attention_p{idx+1}_L{layer}.png"),
                            dpi=300, bbox_inches='tight')
                plt.close()

                # 最大聚焦头图（猫娘和耄耋各画一张）
                for src_name, (best_head, min_ent) in focused_heads.items():
                    if best_head is not None and best_head.ndim == 2 and not np.any(np.isnan(best_head)):
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(best_head, xticklabels=display_tokens, yticklabels=display_tokens,
                                   cmap="Reds" if src_name == "猫娘" else "Oranges", ax=ax, square=True)
                        ax.set_title(f"维度8：{src_name}最大聚焦头 (提示词 {idx+1}, 第{layer}层, 熵={min_ent:.2f})",
                                    fontproperties=get_font_prop(12), fontweight='bold')
                        plt.xticks(rotation=60, ha='right', fontsize=8)
                        plt.yticks(rotation=0, fontsize=8)
                        plt.tight_layout()
                        safe_src = src_name.replace(' ', '')
                        plt.savefig(os.path.join(self.cfg["output_dir"],
                                    f"module_8_focused_head_{safe_src}_p{idx+1}_L{layer}.png"),
                                    dpi=300, bbox_inches='tight')
                        plt.close()
                    else:
                        print(f"   ⚠️ 第{layer}层{src_name}聚焦头 attention 无效，跳过")

        # 量化指标汇总（三路：base / 猫娘 / 耄耋）
        if metrics_data["base"] and (metrics_data["猫娘"] or metrics_data["耄耋"]):
            keys = sorted(set(metrics_data["猫娘"].keys()) | set(metrics_data["耄耋"].keys()))
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            x = np.arange(len(keys))
            width = 0.25

            # 注意力熵
            base_ent = [metrics_data["base"].get(k, {}).get("entropy", 0) for k in keys]
            cat_ent = [metrics_data["猫娘"].get(k, {}).get("entropy", 0) for k in keys]
            troll_ent = [metrics_data["耄耋"].get(k, {}).get("entropy", 0) for k in keys]
            axes[0].bar(x - width, base_ent, width, label='Base', color='#3498DB', alpha=0.8)
            axes[0].bar(x, cat_ent, width, label='猫娘', color='#FF69B4', alpha=0.8)
            axes[0].bar(x + width, troll_ent, width, label='耄耋', color='#DC143C', alpha=0.8)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(keys, rotation=45, fontproperties=get_font_prop(8))
            axes[0].set_title("注意力熵对比", fontproperties=get_font_prop(12), fontweight='bold')
            axes[0].set_ylabel("熵 (越低越集中)", fontproperties=get_font_prop(10))
            if CHINESE_FONT:
                axes[0].legend(prop=get_font_prop(9))
            else:
                axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')

            # 对角线权重比
            base_diag = [metrics_data["base"].get(k, {}).get("diag_ratio", 0) for k in keys]
            cat_diag = [metrics_data["猫娘"].get(k, {}).get("diag_ratio", 0) for k in keys]
            troll_diag = [metrics_data["耄耋"].get(k, {}).get("diag_ratio", 0) for k in keys]
            axes[1].bar(x - width, base_diag, width, label='Base', color='#3498DB', alpha=0.8)
            axes[1].bar(x, cat_diag, width, label='猫娘', color='#FF69B4', alpha=0.8)
            axes[1].bar(x + width, troll_diag, width, label='耄耋', color='#DC143C', alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(keys, rotation=45, fontproperties=get_font_prop(8))
            axes[1].set_title("对角线权重比对比", fontproperties=get_font_prop(12), fontweight='bold')
            axes[1].set_ylabel("diag_ratio", fontproperties=get_font_prop(10))
            if CHINESE_FONT:
                axes[1].legend(prop=get_font_prop(9))
            else:
                axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')

            # 最大注意力权重
            cat_max = [metrics_data["猫娘"].get(k, {}).get("max_weight", 0) for k in keys]
            troll_max = [metrics_data["耄耋"].get(k, {}).get("max_weight", 0) for k in keys]
            axes[2].bar(x - width/2, cat_max, width, label='猫娘', color='#FF69B4', alpha=0.8)
            axes[2].bar(x + width/2, troll_max, width, label='耄耋', color='#DC143C', alpha=0.8)
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(keys, rotation=45, fontproperties=get_font_prop(8))
            axes[2].set_title("最大注意力权重 (SFT后)", fontproperties=get_font_prop(12), fontweight='bold')
            axes[2].set_ylabel("max_weight", fontproperties=get_font_prop(10))
            if CHINESE_FONT:
                axes[2].legend(prop=get_font_prop(9))
            else:
                axes[2].legend()
            axes[2].grid(True, alpha=0.3, axis='y')

            plt.suptitle("维度8：注意力量化指标汇总（双路对比）",
                         fontproperties=get_font_prop(14), fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg["output_dir"], "module_8_attention_metrics.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

        del model_base, model_cat, model_troll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==========================================
# 🚦 主进程入口
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🧠 Qwen3.5 赛博心理防御测量仪")
    print("   混合架构专用版 — 增强版 v2")
    print("=" * 50)

    wa = WeightAnalyzer(CONFIG)
    wa.module_1_and_2_defense_heatmap_and_decay()
    wa.module_3_consensus_heatmap()
    wa.module_4_3d_pca_trajectories()
    wa.module_9_architecture_bias()
    wa.module_10_word_distortion()
    wa.module_11_svd_effective_rank()

    ia = InferenceAnalyzer(CONFIG)
    ia.module_5_subconscious_betrayal()
    ia.module_6_word_cloud()
    ia.module_7_concept_fusion()
    ia.module_8_ptsd_attention()

    print(f"\n🎉 分析完成！所有图表已保存至: {CONFIG['output_dir']}")