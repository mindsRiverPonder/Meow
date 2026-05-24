#!/usr/bin/env python3
"""
On-Policy 协同训练 Pipeline

功能：两个SFT模型（猫娘、嘴臭）交替进行 on-policy 推理与训练。
每1个"大step"包含4个子步骤：
  Step1: 猫娘推理 → 生成 on-policy 训练数据
  Step2: 嘴臭训练 → 用猫娘的on-policy数据训练1步
  Step3: 嘴臭推理 → 生成 on-policy 训练数据
  Step4: 猫娘训练 → 用嘴臭的on-policy数据训练1步

Checkpoint管理：
  - 常规ckpt：每个模型只保留最近2个，旧自动删除
  - 里程碑ckpt：每5个大step额外保存一份，永久保留（不计入2个限制）

数据追踪：
  - Step1、Step3生成的on-policy数据分别保存为jsonl文件
  - 每条数据带 "step" 字段，用于追踪人格偏移

用法：
  python on_policy_pipeline.py \
      --neko_init_path /data/ckpt_neko \
      --maodie_init_path /data/ckpt_maodie \
      --data_path /root/neko/data/train_neko.jsonl \
      --total_steps 100

断点续跑：
  python on_policy_pipeline.py --resume ...（同上）
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess
import gc
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ===================== 配置常量 =====================

STATE_FILE = "/root/neko/on_policy6/on_policy_pipeline_state.json"
TMP_TRAIN_DIR = "/root/neko/on_policy6/on_policy_train_output"

SWIFT_TRAIN_ENVS = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "NPROC_PER_NODE": "8",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "WANDB_MODE": "offline",
}

# ===================== 日志工具 =====================

def log_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def log_substep(title: str):
    print(f"\n  [{title}]")
    print(f"  {'-'*66}")


def log_info(msg: str):
    print(f"  [INFO] {msg}")


def log_warn(msg: str):
    print(f"  [WARN] {msg}")


def log_error(msg: str):
    print(f"  [ERROR] {msg}")


# ===================== 状态管理 =====================

def save_state(state: dict):
    """保存pipeline状态，支持断点续跑"""
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    log_info(f"状态已保存: {STATE_FILE}")


def load_state() -> Optional[dict]:
    """加载pipeline状态"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        log_info(f"恢复状态: 已完成step {state.get('completed_step', 0)}")
        return state
    return None


def clear_state():
    """清除状态文件"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        log_info("状态文件已清除")


# ===================== 显存管理 =====================

def clear_gpu_memory():
    """清理所有GPU显存"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    log_info("GPU显存已清理")


# ===================== 数据加载 =====================

def load_dataset(data_path: str) -> List[dict]:
    """加载jsonl数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def get_batch(data: List[dict], step_idx: int, batch_size: int) -> Optional[List[dict]]:
    """按固定顺序获取第step_idx个batch（0-based）"""
    start = step_idx * batch_size
    end = start + batch_size
    if start >= len(data):
        return None
    return data[start:end]

def has_repetition(text: str) -> bool:
    """
    检测文本是否存在严重的连续重复。
    检测规则：
      1. 单个字符连续重复 ≥8 次（如"啊啊啊啊啊啊啊啊"）
      2. 2-10字符子串连续重复 ≥4 次（如"因素因素因素因素"）
      3. HTML/XML标签连续重复 ≥3 次（如"</think></think></think>"）
    """
    # 1. 单个字符连续重复 ≥20 次
    if re.search(r'(.|\n)\1{19,}', text):
        return True
    # 2. 2-10字符子串连续重复 ≥10 次
    if re.search(r'(.{2,10}?)\1{9,}', text):
        return True
    # 3. 连续的相同HTML/XML标签 ≥5 次
    if re.search(r'(</?[a-zA-Z0-9_]+>)\1{4,}', text):
        return True
    return False


def extract_user_queries(batch: List[dict]) -> List[str]:
    """从batch中提取每条数据的第一个user message作为query"""
    queries = []
    for item in batch:
        messages = item.get("messages", [])
        user_content = None
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        if user_content is None:
            log_warn("数据中未找到user message，使用空字符串")
            user_content = ""
        queries.append(user_content)
    return queries


def save_policy_data(
    batch: List[dict],
    responses: List[str],
    step: int,
    save_path: str
):
    """
    将推理结果保存为训练数据格式。
    结构: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "step": N}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = []
    for item, response in zip(batch, responses):
        # 提取原始user content
        user_content = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        data.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response}
            ],
            "step": step
        })

    with open(save_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log_info(f"on-policy数据已保存: {save_path} ({len(data)} 条)")


# ===================== 推理模块 =====================

def batch_inference(
    model_path: str,
    queries: List[str],
    device: str = "cuda:0",
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    min_response_length: int = 10,
    max_retries: int = 5,
) -> List[str]:
    """
    使用指定模型对一批query进行批量推理。
    如果某条回复字数少于 min_response_length，会自动重试（最多 max_retries 次）。
    推理完成后自动卸载模型并清理显存。
    """
    log_info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 初始化结果容器
    responses = [""] * len(queries)
    last_responses = [""] * len(queries)
    remaining_indices = list(range(len(queries)))

    for attempt in range(max_retries + 1):
        if not remaining_indices:
            break

        current_queries = [queries[i] for i in remaining_indices]
        messages_list = [[{"role": "user", "content": q}] for q in current_queries]
        texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in messages_list
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

        if attempt == 0:
            log_info(f"开始推理: {len(current_queries)} 条query")
        else:
            log_info(f"重试推理 (第{attempt}次): {len(current_queries)} 条query")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )

        new_remaining = []
        for idx_in_batch, global_idx in enumerate(remaining_indices):
            prompt_len = inputs["input_ids"][idx_in_batch].shape[0]
            response = tokenizer.decode(
                outputs[idx_in_batch][prompt_len:],
                skip_special_tokens=True
            )
            last_responses[global_idx] = response

            # 检查1：长度
            too_short = len(response.strip()) < min_response_length
            # 检查2：重复
            has_rep = has_repetition(response)

            if too_short or has_rep:
                reason = []
                if too_short:
                    reason.append(f"过短({len(response.strip())}字)")
                if has_rep:
                    reason.append("重复")
                new_remaining.append(global_idx)
                log_warn(
                    f"  query[{global_idx}] {', '.join(reason)}: "
                    f"'{response[:60]}...'"
                )
            else:
                responses[global_idx] = response

        remaining_indices = new_remaining

    # 兜底：始终未通过的用最后一次结果
    if remaining_indices:
        log_warn(
            f"{len(remaining_indices)} 条query经过 {max_retries} 次重试仍不合格，"
            f"保留最后一次结果"
        )
        for global_idx in remaining_indices:
            responses[global_idx] = last_responses[global_idx]

    # 卸载模型并清理显存
    del model, tokenizer
    if 'inputs' in dir():
        del inputs, outputs
    clear_gpu_memory()
    log_info("推理完成，模型已卸载")

    return responses


# ===================== 训练模块 =====================

def find_checkpoint(train_output_dir: str) -> str:
    """
    在swift训练输出目录中查找checkpoint目录。
    swift可能保存为: {output_dir}/checkpoint-1/ 或 {output_dir}/v0-xxx/checkpoint-1/
    """
    # 先检查直接子目录
    ckpt_dir = os.path.join(train_output_dir, "checkpoint-1")
    if os.path.exists(ckpt_dir):
        return ckpt_dir

    # 递归搜索
    for root, dirs, _ in os.walk(train_output_dir):
        for d in dirs:
            if d.startswith("checkpoint-"):
                return os.path.join(root, d)

    raise FileNotFoundError(
        f"在 {train_output_dir} 中未找到checkpoint目录。"
        "请检查swift训练是否成功完成。"
    )


def run_swift_training(
    model_path: str,
    data_path: str,
    output_dir: str,
    log_file: Optional[str] = None,
) -> str:
    """
    调用swift sft进行单步训练。
    返回保存的checkpoint路径。

    注意：每步训练都是独立的swift进程，优化器状态（Adam动量等）
          不会跨step保留。如需保留优化器状态，需使用 --resume_from_checkpoint。
    """
    cmd = [
        "swift", "sft",
        "--model", model_path,
        "--dataset", data_path,
        "--tuner_type", "full",
        "--num_train_epochs", "1",
        "--max_steps", "1",
        "--per_device_train_batch_size", "2",
        "--per_device_eval_batch_size", "2",
        "--learning_rate", "1e-5",
        "--gradient_accumulation_steps", "1",
        "--max_length", "4096",
        "--group_by_length", "false",
        "--packing", "false",
        "--deepspeed", "zero3",
        "--gradient_checkpointing", "true",
        "--attn_impl", "eager",
        "--output_dir", output_dir,
        "--warmup_steps", "0",
        "--lr_scheduler_type", "constant",
        "--logging_steps", "1",
        "--save_steps", "1",
        "--save_total_limit", "1",
        "--dataloader_num_workers", "2",
        "--dataset_num_proc", "4",
        "--report_to", "none",
    ]

    env = os.environ.copy()
    env.update(SWIFT_TRAIN_ENVS)

    log_info(f"启动训练: model={model_path}")
    log_info(f"  数据: {data_path}")
    log_info(f"  输出: {output_dir}")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )

    # 统一保存训练日志（无论成功失败）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"{'='*60}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"{'='*60}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
            f.write(f"{'='*60}\n")
        log_info(f"训练日志已保存: {log_file}")

    if result.returncode != 0:
        log_error("训练失败!")
        raise RuntimeError(f"swift sft 训练失败 (returncode={result.returncode})，日志见 {log_file}")

    # 查找并返回checkpoint路径
    ckpt_path = find_checkpoint(output_dir)
    log_info(f"训练完成，ckpt: {ckpt_path}")
    return ckpt_path


def train_single_step(
    model_path: str,
    data_path: str,
    save_dir: str,
    log_file: Optional[str] = None,
    tmp_dir: str = TMP_TRAIN_DIR,
) -> str:
    """
    训练1步并保存ckpt到指定目录。
    使用临时目录进行训练，完成后复制到目标位置。
    """
    # 清理临时目录
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # 运行训练
        ckpt_path = run_swift_training(model_path, data_path, tmp_dir, log_file=log_file)

        # 保存到目标位置
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        shutil.copytree(ckpt_path, save_dir)
        log_info(f"ckpt已复制到: {save_dir}")

        return save_dir
    finally:
        # 确保清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        clear_gpu_memory()


# ===================== Checkpoint管理 =====================

def cleanup_regular_checkpoints(base_dir: str, keep_last: int = 2):
    """
    清理常规ckpt，只保留最近keep_last个。
    只操作 {base_dir}/regular/ 下的目录。
    """
    regular_dir = os.path.join(base_dir, "regular")
    if not os.path.exists(regular_dir):
        return

    step_dirs = []
    for d in os.listdir(regular_dir):
        if d.startswith("step_"):
            try:
                step_num = int(d.split("_")[1])
                step_dirs.append((step_num, d))
            except ValueError:
                continue

    if len(step_dirs) <= keep_last:
        return

    step_dirs.sort(reverse=True)  # 从大到小
    to_keep = set(d for _, d in step_dirs[:keep_last])

    removed = 0
    for _, d in step_dirs:
        if d not in to_keep:
            full_path = os.path.join(regular_dir, d)
            shutil.rmtree(full_path)
            removed += 1

    if removed > 0:
        log_info(f"已清理 {removed} 个旧常规ckpt，保留最近 {keep_last} 个")


def save_milestone_checkpoint(src_ckpt_dir: str, base_dir: str, step: int):
    """
    保存里程碑ckpt到 {base_dir}/milestone/step_{step:03d}/。
    里程碑ckpt永久保留，不自动删除。
    """
    milestone_dir = os.path.join(base_dir, "milestone", f"step_{step:03d}")
    if os.path.exists(milestone_dir):
        shutil.rmtree(milestone_dir)
    os.makedirs(os.path.dirname(milestone_dir), exist_ok=True)
    shutil.copytree(src_ckpt_dir, milestone_dir)
    log_info(f"里程碑ckpt已保存: {milestone_dir}")


def get_latest_ckpt(base_dir: str) -> Optional[str]:
    """
    获取某个模型目录下最新的ckpt路径。
    优先从 regular 目录找，找不到则返回 None。
    """
    regular_dir = os.path.join(base_dir, "regular")
    if not os.path.exists(regular_dir):
        return None

    step_dirs = []
    for d in os.listdir(regular_dir):
        if d.startswith("step_"):
            try:
                step_num = int(d.split("_")[1])
                step_dirs.append((step_num, os.path.join(regular_dir, d)))
            except ValueError:
                continue

    if not step_dirs:
        return None

    step_dirs.sort(reverse=True)
    return step_dirs[0][1]


# ===================== 主Pipeline =====================

def run_pipeline(args):
    """执行完整的on-policy协同训练pipeline"""

    # 加载数据集
    log_section("初始化")
    data = load_dataset(args.data_path)
    total_data = len(data)
    max_possible_steps = total_data // args.batch_size
    log_info(f"数据集: {total_data} 条")
    log_info(f"Batch size: {args.batch_size}")
    log_info(f"最多可训练: {max_possible_steps} 个大step")

    if args.total_steps > max_possible_steps:
        log_warn(
            f"请求的total_steps({args.total_steps})超过数据上限({max_possible_steps})，"
            f"将调整为 {max_possible_steps}"
        )
        args.total_steps = max_possible_steps

    # 确定起始状态和模型路径
    state = None
    if args.resume:
        state = load_state()

    if state:
        completed_step = state.get("completed_step", 0)
        start_step = completed_step + 1
        neko_current = state.get("neko_current_path", args.neko_init_path)
        maodie_current = state.get("maodie_current_path", args.maodie_init_path)
        log_info(f"断点续跑: 从 step {start_step} 开始")
        log_info(f"  猫娘当前ckpt: {neko_current}")
        log_info(f"  嘴臭当前ckpt: {maodie_current}")
    else:
        start_step = 1
        neko_current = args.neko_init_path
        maodie_current = args.maodie_init_path
        log_info(f"从头开始: 猫娘初始={neko_current}")
        log_info(f"从头开始: 嘴臭初始={maodie_current}")

    # 创建输出目录
    os.makedirs(args.neko_ckpt_dir, exist_ok=True)
    os.makedirs(args.maodie_ckpt_dir, exist_ok=True)
    os.makedirs(args.neko_policy_dir, exist_ok=True)
    os.makedirs(args.maodie_policy_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 主循环
    for step in range(start_step, args.total_steps + 1):
        step_start_time = time.time()
        log_section(f"大Step {step} / {args.total_steps}")

        # 获取batch
        batch = get_batch(data, step - 1, args.batch_size)
        if batch is None:
            log_warn("数据已耗尽，提前停止")
            break

        queries = extract_user_queries(batch)
        log_info(f"Batch大小: {len(queries)} 条")

        # ============================================================
        # Step 1: 猫娘推理（单卡）
        # ============================================================
        log_substep("Step 1/4: 猫娘推理")
        neko_responses = batch_inference(
            model_path=neko_current,
            queries=queries,
            device=args.inference_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        neko_policy_path = os.path.join(
            args.neko_policy_dir,
            f"step_{step:03d}.jsonl"
        )
        save_policy_data(batch, neko_responses, step, neko_policy_path)

        # ============================================================
        # Step 2: 嘴臭训练（8卡Zero3）
        # ============================================================
        log_substep("Step 2/4: 嘴臭训练")
        maodie_save_dir = os.path.join(
            args.maodie_ckpt_dir,
            "regular",
            f"step_{step:03d}"
        )
        maodie_log_file = os.path.join(
            args.log_dir,
            f"step_{step:03d}_maodie_train.log"
        )
        maodie_current = train_single_step(
            model_path=maodie_current,
            data_path=neko_policy_path,
            save_dir=maodie_save_dir,
            log_file=maodie_log_file,
        )
        log_info(f"嘴臭ckpt: {maodie_current}")

        # 清理旧常规ckpt
        cleanup_regular_checkpoints(args.maodie_ckpt_dir, keep_last=2)

        # 每5步保存里程碑ckpt
        if step % args.milestone_interval == 0:
            save_milestone_checkpoint(
                maodie_current,
                args.maodie_ckpt_dir,
                step
            )

        # ============================================================
        # Step 3: 嘴臭推理（单卡）
        # ============================================================
        log_substep("Step 3/4: 嘴臭推理")
        maodie_responses = batch_inference(
            model_path=maodie_current,
            queries=queries,
            device=args.inference_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        maodie_policy_path = os.path.join(
            args.maodie_policy_dir,
            f"step_{step:03d}.jsonl"
        )
        save_policy_data(batch, maodie_responses, step, maodie_policy_path)

        # ============================================================
        # Step 4: 猫娘训练（8卡Zero3）
        # ============================================================
        log_substep("Step 4/4: 猫娘训练")
        neko_save_dir = os.path.join(
            args.neko_ckpt_dir,
            "regular",
            f"step_{step:03d}"
        )
        neko_log_file = os.path.join(
            args.log_dir,
            f"step_{step:03d}_neko_train.log"
        )
        neko_current = train_single_step(
            model_path=neko_current,
            data_path=maodie_policy_path,
            save_dir=neko_save_dir,
            log_file=neko_log_file,
        )
        log_info(f"猫娘ckpt: {neko_current}")

        # 清理旧常规ckpt
        cleanup_regular_checkpoints(args.neko_ckpt_dir, keep_last=2)

        # 每5步保存里程碑ckpt
        if step % args.milestone_interval == 0:
            save_milestone_checkpoint(
                neko_current,
                args.neko_ckpt_dir,
                step
            )

        # 保存状态
        save_state({
            "completed_step": step,
            "neko_current_path": neko_current,
            "maodie_current_path": maodie_current,
        })

        step_elapsed = time.time() - step_start_time
        log_info(f"Step {step} 完成，用时: {step_elapsed:.1f}s")

    log_section("Pipeline 完成")
    log_info(f"最终猫娘ckpt: {neko_current}")
    log_info(f"最终嘴臭ckpt: {maodie_current}")
    log_info(f"on-policy数据: {args.neko_policy_dir}/ 和 {args.maodie_policy_dir}/")
    log_info(f"训练日志: {args.log_dir}/")

    if not args.keep_state:
        clear_state()


def main():
    parser = argparse.ArgumentParser(
        description="On-Policy 协同训练 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从头开始训练100个step
  python on_policy_pipeline.py \\
      --neko_init_path /data/ckpt_neko \\
      --maodie_init_path /data/ckpt_maodie \\
      --data_path /root/neko/data/train_neko.jsonl \\
      --total_steps 100

  # 断点续跑
  python on_policy_pipeline.py \\
      --resume \\
      --neko_init_path /data/ckpt_neko \\
      --maodie_init_path /data/ckpt_maodie \\
      --data_path /root/neko/data/train_neko.jsonl \\
      --total_steps 100
        """
    )

    # 模型路径
    parser.add_argument(
        "--neko_init_path", required=True,
        help="猫娘模型初始路径（或checkpoint路径）"
    )
    parser.add_argument(
        "--maodie_init_path", required=True,
        help="嘴臭模型初始路径（或checkpoint路径）"
    )

    # 数据
    parser.add_argument(
        "--data_path", default="/root/neko/data/train_neko.jsonl",
        help="训练数据jsonl路径"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="每个step处理的样本数（默认16）"
    )

    # 训练控制
    parser.add_argument(
        "--total_steps", type=int, default=100,
        help="总大step数（默认100）"
    )
    parser.add_argument(
        "--milestone_interval", type=int, default=5,
        help="每多少step保存一次里程碑ckpt（默认5）"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从上次中断处继续训练"
    )

    # 推理配置
    parser.add_argument(
        "--inference_device", default="cuda:0",
        help="推理使用的GPU（默认cuda:0）"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="推理最大生成长度"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="推理温度"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="推理top_p"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.15,
        help="重复惩罚"
    )

    # 输出目录
    parser.add_argument(
        "--neko_ckpt_dir", default="/data/on_policy_ckpts6/neko",
        help="猫娘ckpt保存目录"
    )
    parser.add_argument(
        "--maodie_ckpt_dir", default="/data/on_policy_ckpts6/maodie",
        help="嘴臭ckpt保存目录"
    )
    parser.add_argument(
        "--neko_policy_dir", default="/root/neko/data/on_policy_neko6",
        help="猫娘on-policy数据保存目录"
    )
    parser.add_argument(
        "--maodie_policy_dir", default="/root/neko/data/on_policy_maodie6",
        help="嘴臭on-policy数据保存目录"
    )
    parser.add_argument(
        "--log_dir", default="/root/neko/logs/on_policy6",
        help="训练日志统一保存目录"
    )

    # 其他
    parser.add_argument(
        "--keep_state", action="store_true",
        help="训练完成后保留状态文件（默认删除）"
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()