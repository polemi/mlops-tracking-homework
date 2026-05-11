import argparse
from pathlib import Path

from mlops_utils import read_json, save_json


def pct(value: float) -> str:
    return f"{value:.4f}"


def compare(base_path: str, finetuned_path: str, output_json: str, output_md: str) -> dict:
    base = read_json(base_path)
    finetuned = read_json(finetuned_path)

    metric_names = ["loss", "accuracy", "f1_macro", "precision_macro", "recall_macro"]
    rows = []
    for split_key, split_name in [("final_train_metrics", "train"), ("final_test_metrics", "test")]:
        for metric in metric_names:
            base_value = float(base[split_key][metric])
            finetuned_value = float(finetuned[split_key][metric])
            rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "base": base_value,
                    "finetuned": finetuned_value,
                    "delta": finetuned_value - base_value,
                }
            )

    base_test_f1 = float(base["final_test_metrics"]["f1_macro"])
    finetuned_test_f1 = float(finetuned["final_test_metrics"]["f1_macro"])
    base_test_acc = float(base["final_test_metrics"]["accuracy"])
    finetuned_test_acc = float(finetuned["final_test_metrics"]["accuracy"])

    if finetuned_test_f1 > base_test_f1:
        conclusion = (
            "После дообучения качество на тестовой выборке выросло по macro F1. "
            "Это означает, что модель смогла использовать данные Train_2 и лучше обобщается на Test_2."
        )
    elif finetuned_test_f1 < base_test_f1:
        conclusion = (
            "После дообучения macro F1 на тестовой выборке снизился. "
            "Возможные причины: Train_2 отличается от Train_1, модель переобучилась на новых данных или требуется меньший learning rate."
        )
    else:
        conclusion = (
            "После дообучения macro F1 на тестовой выборке не изменился. "
            "Модель сохранила качество, но заметного выигрыша от Train_2 не получила."
        )

    comparison = {
        "base_test_accuracy": base_test_acc,
        "finetuned_test_accuracy": finetuned_test_acc,
        "base_test_f1_macro": base_test_f1,
        "finetuned_test_f1_macro": finetuned_test_f1,
        "test_accuracy_delta": finetuned_test_acc - base_test_acc,
        "test_f1_macro_delta": finetuned_test_f1 - base_test_f1,
        "rows": rows,
        "conclusion": conclusion,
    }
    save_json(comparison, output_json)

    md_lines = [
        "# Сравнение двух версий модели",
        "",
        "| split | metric | base | finetuned | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['split']} | {row['metric']} | {pct(row['base'])} | {pct(row['finetuned'])} | {pct(row['delta'])} |"
        )
    md_lines.extend(["", "## Вывод", "", conclusion, ""])

    output_md_path = Path(output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="reports/base_metrics.json")
    parser.add_argument("--finetuned", default="reports/finetuned_metrics.json")
    parser.add_argument("--output-json", default="reports/comparison.json")
    parser.add_argument("--output-md", default="reports/comparison.md")
    args = parser.parse_args()
    result = compare(args.base, args.finetuned, args.output_json, args.output_md)
    print(result["conclusion"])


if __name__ == "__main__":
    main()
