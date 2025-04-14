import os
import re
import csv

def parse_logs(log_dir, output_csv="log_summary.csv"):
    epoch_pattern = re.compile(
        r"\[(.*?)\] Epoch (\d+)/(\d+), Step \d+/.*?Partial Avg Loss: ([\d.]+).*?"
        r"Validation Loss after epoch \d+: ([\d.]+).*?"
        r"\*\*\* End of Epoch \d+ \*\*\* Avg Loss: ([\d.]+)", re.DOTALL
    )

    sample_pattern = re.compile(
        r"\[(.*?)\] Final sample \(greedy\) from prompt: 'Once upon a'\n(.*?)\nAnnotated:\n(.*?)\n\n"
        r"\[(.*?)\] Final sample \(top-p=0.95\) from prompt: 'Once upon a'\n(.*?)\nAnnotated:\n(.*?)\n\n"
        r"\[(.*?)\] Final sample \(top-p=1.0\) from prompt: 'Once upon a'\n(.*?)\nAnnotated:\n(.*?)\n",
        re.DOTALL
    )

    summary_rows = []

    for fname in os.listdir(log_dir):
        if fname.endswith(".log"):
            path = os.path.join(log_dir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Epoch metrics
            for model_type, epoch, total_epochs, part_loss, val_loss, avg_loss in epoch_pattern.findall(content):
                summary_rows.append({
                    "log_file": fname,
                    "model_type": model_type.strip(),
                    "epoch": int(epoch),
                    "total_epochs": int(total_epochs),
                    "partial_avg_loss": float(part_loss),
                    "val_loss": float(val_loss),
                    "avg_loss": float(avg_loss),
                    "sample_type": "epoch"
                })

            # Final generation output
            for match in sample_pattern.findall(content):
                summary_rows.append({
                    "log_file": fname,
                    "model_type": match[0].strip(),
                    "epoch": "final",
                    "total_epochs": None,
                    "partial_avg_loss": None,
                    "val_loss": None,
                    "avg_loss": None,
                    "sample_type": "greedy",
                    "sample_text": match[1].strip(),
                    "annotated": match[2].strip()
                })
                summary_rows.append({
                    "log_file": fname,
                    "model_type": match[3].strip(),
                    "epoch": "final",
                    "sample_type": "top-p=0.95",
                    "sample_text": match[4].strip(),
                    "annotated": match[5].strip()
                })
                summary_rows.append({
                    "log_file": fname,
                    "model_type": match[6].strip(),
                    "epoch": "final",
                    "sample_type": "top-p=1.0",
                    "sample_text": match[7].strip(),
                    "annotated": match[8].strip()
                })

    # Save as CSV
    keys = sorted({key for row in summary_rows for key in row})
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✅ Summary written to {output_csv} with {len(summary_rows)} rows.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs", help="Path to directory with .log files")
    parser.add_argument("--output_csv", default="log_summary.csv", help="Output CSV file")
    args = parser.parse_args()
    parse_logs(args.log_dir, args.output_csv)
