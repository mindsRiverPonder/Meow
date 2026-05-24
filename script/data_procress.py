import json
import os
from datetime import datetime


# 数据所在目录
SOURCE_DIR = "/root/neko/data"
# 耄耋和猫娘数据集JSON文件名（请改成你实际的文件名）
SOURCE_FILES = ["NekoQA-30K.json", "MaoDieQA-30K.json"]

# 【重要】生成的新 JSONL 文件名（你可以自由修改）
OUTPUT_FILES = ["train_neko.jsonl", "train_maodie.jsonl"]

# 规则：output字段最小长度,太短的猫猫不吃
MIN_OUTPUT_LENGTH = 200
ENCODING = "utf-8"
# =====================================================================

def convert_and_clean_single_json(source_path: str, output_path: str) -> None:
    """
    处理单个JSON文件：清洗数据 + 格式转换 + 输出JSONL
    :param source_path: 原始文件路径喵
    :param output_path: 输出新文件路径喵
    """
    file_name = os.path.basename(source_path)
    print(f"\n{'='*60}")
    print(f"📂 处理原始文件：{file_name}")
    print(f"📤 输出新文件：{os.path.basename(output_path)}")

    # 1. 检查原始文件是否存在
    if not os.path.exists(source_path):
        print(f"呜哇！原始文件不见了，猫猫找不到，跳过喵！")
        return

    try:
        # 2. 读取原始JSON数据
        with open(source_path, "r", encoding=ENCODING) as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                print(f"这个文件不是香喷喷的合法JSON，猫猫读不懂，跳过喵")
                return

        # 3. 校验数据必须是列表
        if not isinstance(raw_data, list):
            print(f"❌ 错误：数据不是JSON数组，跳过")
            return

        total_count = len(raw_data)
        valid_count = 0

        # 4. 遍历清洗 + 格式转换 + 逐行写入JSONL
        with open(output_path, "w", encoding=ENCODING) as f_out:
            for item in raw_data:
                # 跳过非字典数据
                if not isinstance(item, dict):
                    continue

                # 提取原始字段（不存在则为空字符串）
                instruction = str(item.get("instruction", "")).strip()
                output = str(item.get("output", "")).strip()

                # 过滤规则：必须有内容 + output长度≥200
                if not instruction or not output:
                    continue
                if len(output) < MIN_OUTPUT_LENGTH:
                    continue

                # 格式转换（严格按照要求）
                new_item = {
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": output}
                    ]
                }

                # 【关键优化】逐行写入 JSONL
                f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                valid_count += 1

        # 统计结果
        print(f"📊 统计：原始 {total_count} 条 → 有效 {valid_count} 条 → 剔除 {total_count - valid_count} 条")
        print(f"✅ 处理完成！新 JSONL 文件已保存")

    except PermissionError:
        print(f"❌ 错误：没有文件操作权限")
    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")


def read_jsonl(file_path: str) -> list[dict]:
    """
    读取 JSONL 文件，返回字典列表
    :param file_path: JSONL 文件路径
    :return: 解析后的字典列表
    """
    records = []
    with open(file_path, "r", encoding=ENCODING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(file_path: str, records: list[dict]) -> None:
    """
    将字典列表写回 JSONL 文件（覆盖写入）
    :param file_path: JSONL 文件路径
    :param records: 要写入的记录列表
    """
    with open(file_path, "w", encoding=ENCODING) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def align_outputs_by_instruction(output_paths: list[str]) -> None:
    """
    根据 instruction 字段对齐两个 JSONL 文件：
    - 找出两边 instruction 的交集
    - 仅保留交集中的记录，删除多余数据
    - 最终两边数据条数一致、instruction 一一对应

    :param output_paths: 两个输出 JSONL 文件的路径列表 [path_a, path_b]
    """
    print(f"\n{'='*60}")
    print("🔄 开始【数据对齐】——根据 instruction 字段匹配")

    path_a, path_b = output_paths[0], output_paths[1]
    name_a, name_b = os.path.basename(path_a), os.path.basename(path_b)

    # 1. 读取两个 JSONL 文件
    records_a = read_jsonl(path_a)
    records_b = read_jsonl(path_b)

    print(f"📄 {name_a}：{len(records_a)} 条")
    print(f"📄 {name_b}：{len(records_b)} 条")

    # 2. 提取每条记录的 instruction 字段（从 messages[0].content 中取）
    #    转换后格式为 {"messages": [{"role":"user","content":"..."}, ...]}
    #    所以 instruction = messages[0]["content"]
    def extract_instruction(record: dict) -> str:
        """从转换后的记录中提取 instruction（用户提问内容）"""
        try:
            return record["messages"][0]["content"].strip()
        except (KeyError, IndexError):
            return ""

    # 3. 构建 instruction → 记录列表 的映射（允许同一 instruction 出现多次）
    #    使用 OrderedDict 思路：按首次出现顺序保留，后续重复也保留
    instr_to_records_a: dict[str, list[dict]] = {}
    instr_to_records_b: dict[str, list[dict]] = {}

    # 记录 instruction 出现的顺序（用于保持原始排序）
    order_a: list[str] = []
    order_b: list[str] = []

    for rec in records_a:
        instr = extract_instruction(rec)
        if not instr:
            continue
        if instr not in instr_to_records_a:
            instr_to_records_a[instr] = []
            order_a.append(instr)
        instr_to_records_a[instr].append(rec)

    for rec in records_b:
        instr = extract_instruction(rec)
        if not instr:
            continue
        if instr not in instr_to_records_b:
            instr_to_records_b[instr] = []
            order_b.append(instr)
        instr_to_records_b[instr].append(rec)

    # 4. 求交集：两边都有的 instruction 集合
    set_a = set(instr_to_records_a.keys())
    set_b = set(instr_to_records_b.keys())
    common_instrs = set_a & set_b

    only_a = set_a - set_b  # 仅 A 有、B 没有的 instruction
    only_b = set_b - set_a  # 仅 B 有、A 没有的 instruction

    print(f"\n📊 对齐分析：")
    print(f"   {name_a} 独有 instruction：{len(only_a)} 种")
    print(f"   {name_b} 独有 instruction：{len(only_b)} 种")
    print(f"   共有 instruction：{len(common_instrs)} 种")

    if not common_instrs:
        print("⚠️ 警告：两个文件没有共有 instruction，无法对齐，跳过")
        return

    # 5. 对齐策略：对于每个共有 instruction，取两边出现的最小次数
    #    这样保证最终条数一致（处理同一 instruction 出现多次的情况）
    aligned_a: list[dict] = []
    aligned_b: list[dict] = []

    # 按文件 A 中 instruction 的原始顺序遍历交集，保持排序
    for instr in order_a:
        if instr not in common_instrs:
            continue
        list_a = instr_to_records_a[instr]
        list_b = instr_to_records_b[instr]
        # 取较小次数，确保两边条数对齐
        min_count = min(len(list_a), len(list_b))
        aligned_a.extend(list_a[:min_count])
        aligned_b.extend(list_b[:min_count])

    # 6. 统计对齐结果
    removed_a = len(records_a) - len(aligned_a)
    removed_b = len(records_b) - len(aligned_b)

    print(f"\n📊 对齐结果：")
    print(f"   {name_a}：{len(records_a)} → {len(aligned_a)} 条（删除 {removed_a} 条）")
    print(f"   {name_b}：{len(records_b)} → {len(aligned_b)} 条（删除 {removed_b} 条）")
    print(f"   ✅ 最终两边各 {len(aligned_a)} 条，数据对齐完成")

    # 7. 写回文件
    write_jsonl(path_a, aligned_a)
    write_jsonl(path_b, aligned_b)
    print(f"💾 已将对齐后的数据写回文件")


def main():
    """主函数：批量处理两个文件 + 数据对齐"""
    print("🚀 启动【JSON清洗+格式转换+JSONL输出】脚本")
    print(f"📁 原始目录：{SOURCE_DIR}")
    print(f"🔍 原始文件：{SOURCE_FILES[0]}、{SOURCE_FILES[1]}")
    print(f"📌 输出文件：{OUTPUT_FILES[0]}、{OUTPUT_FILES[1]}")
    print(f"📏 过滤规则：output 字段 < 200 字符 → 自动剔除")
    print(f"📝 输出格式：JSONL (每行一条JSON数据)")

    # 批量处理两个文件
    for i in range(2):
        source_file = os.path.join(SOURCE_DIR, SOURCE_FILES[i])
        output_file = os.path.join(SOURCE_DIR, OUTPUT_FILES[i])
        convert_and_clean_single_json(source_file, output_file)

    # 数据对齐：根据 instruction 字段，让两边数据条数一致
    output_paths = [os.path.join(SOURCE_DIR, f) for f in OUTPUT_FILES]
    align_outputs_by_instruction(output_paths)

    print(f"\n{'='*60}")
    print("喵呜！所有文件都处理好啦，猫猫伸个懒腰")


if __name__ == "__main__":
    main()