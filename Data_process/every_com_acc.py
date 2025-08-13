import os
import json
import csv

# 3个复杂度等级
COMPLEXITY_LEVELS = ["low", "moderate", "high"]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def compute_per_complexity_accuracy(pred_data, true_data, true_answer_idx):
    # complexity_level -> (correct, total)
    stats = {c: [0, 0] for c in COMPLEXITY_LEVELS}

    for p, t, a in zip(pred_data, true_data, true_answer_idx):
        comp = t.get("complexity")
        if comp not in stats:
            continue
        pred_ans = p.get("Answer")
        true_ans = a.get("answer_idx")

        if pred_ans and true_ans:
            stats[comp][1] += 1  # total
            if pred_ans == true_ans:
                stats[comp][0] += 1  # correct
        else:
            stats[comp][1] += 1

    return {comp: (correct / total * 100 if total > 0 else 0.0)
            for comp, (correct, total) in stats.items()}

def evaluate_models_by_complexity(pred_folder, true_file, true_answer_file, output_csv):
    true_data = load_json(true_file)
    true_answer_idx = load_jsonl(true_answer_file)

    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.json') and f != os.path.basename(true_file)]

    all_results = []

    for file in pred_files:
        pred_path = os.path.join(pred_folder, file)
        pred_data = load_json(pred_path)

        if len(pred_data) != len(true_data) or len(pred_data) != len(true_answer_idx):
            print(f"[{file}] 样本数量不一致，跳过。")
            continue

        acc_per_comp = compute_per_complexity_accuracy(pred_data, true_data, true_answer_idx)
        result_row = {"model": file}
        result_row.update(acc_per_comp)
        all_results.append(result_row)

    # 写入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model'] + COMPLEXITY_LEVELS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"每复杂度Answer准确率已保存到：{output_csv}")

# 示例调用
if __name__ == '__main__':
    pred_folder = "./V_json/"     # 预测数据所在文件夹
    true_file = "./ds_data.json"  # 真值文件路径
    true_answer_file = "./dev.jsonl"  # 真实答案的jsonl文件路径
    output_csv = 'answer_accuracy_by_complexity.csv'  # 输出文件路径

    evaluate_models_by_complexity(pred_folder, true_file, true_answer_file, output_csv)


