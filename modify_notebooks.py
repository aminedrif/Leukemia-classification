"""
Notebook Auto-Modifier for PFE
================================
Run this script from the root folder that contains your notebooks.
It will apply all changes to all 24 notebooks automatically:
  1. Remove Dropout(0.5)
  2. Fix weight/checkpoint paths (unique prefix per notebook)
  3. Save confusion matrix as PNG
  4. Add results export cell (accuracy + classification report as CSV)

Usage:
    python modify_notebooks.py
"""

import json
import os
import re

# ── Map each notebook path to its unique prefix ───────────────────────────────
NOTEBOOKS = {
    # Leukemia CNMC
    "Leukemia cnmc/EfficientNet/EfficientNetB0-Method1-BaseModel.ipynb":    "leukemia_efficientnetb0_m1",
    "Leukemia cnmc/EfficientNet/EfficientNetB0-Method2-FineTuning.ipynb":   "leukemia_efficientnetb0_m2",
    "Leukemia cnmc/EfficientNet/EfficientNetB0-Method3-Augmentation.ipynb": "leukemia_efficientnetb0_m3",
    "Leukemia cnmc/EfficientNet/EfficientNetB0-Method4-MLClassifier.ipynb": "leukemia_efficientnetb0_m4",
    "Leukemia cnmc/MobileNet/MobileNetV2-Method1-BaseModel.ipynb":          "leukemia_mobilenetv2_m1",
    "Leukemia cnmc/MobileNet/MobileNetV2-Method2-FineTuning.ipynb":         "leukemia_mobilenetv2_m2",
    "Leukemia cnmc/MobileNet/MobileNetV2-Method3-Augmentation.ipynb":       "leukemia_mobilenetv2_m3",
    "Leukemia cnmc/MobileNet/MobileNetV2-Method4-MLClassifier.ipynb":       "leukemia_mobilenetv2_m4",
    "Leukemia cnmc/vgg16/VGG16-Method1-BaseModel.ipynb":                    "leukemia_vgg16_m1",
    "Leukemia cnmc/vgg16/VGG16-Method2-FineTuning.ipynb":                   "leukemia_vgg16_m2",
    "Leukemia cnmc/vgg16/VGG16-Method3-Augmentation.ipynb":                 "leukemia_vgg16_m3",
    "Leukemia cnmc/vgg16/VGG16-Method4-MLClassifier.ipynb":                 "leukemia_vgg16_m4",
    # OCT Binary
    "OCT-Binary/EfficientNet/EfficientNetB0-Method1-BaseModel.ipynb":       "oct_efficientnetb0_m1",
    "OCT-Binary/EfficientNet/EfficientNetB0-Method2-FineTuning.ipynb":      "oct_efficientnetb0_m2",
    "OCT-Binary/EfficientNet/EfficientNetB0-Method3-Augmentation.ipynb":    "oct_efficientnetb0_m3",
    "OCT-Binary/EfficientNet/EfficientNetB0-Method4-MLClassifier.ipynb":    "oct_efficientnetb0_m4",
    "OCT-Binary/MobileNet/MobileNetV2-Method1-BaseModel.ipynb":             "oct_mobilenetv2_m1",
    "OCT-Binary/MobileNet/MobileNetV2-Method2-FineTuning.ipynb":            "oct_mobilenetv2_m2",
    "OCT-Binary/MobileNet/MobileNetV2-Method3-Augmentation.ipynb":          "oct_mobilenetv2_m3",
    "OCT-Binary/MobileNet/MobileNetV2-Method4-MLClassifier.ipynb":          "oct_mobilenetv2_m4",
    "OCT-Binary/vgg16/VGG16-Method1-BaseModel.ipynb":                       "oct_vgg16_m1",
    "OCT-Binary/vgg16/VGG16-Method2-FineTuning.ipynb":                      "oct_vgg16_m2",
    "OCT-Binary/vgg16/VGG16-Method3-Augmentation.ipynb":                    "oct_vgg16_m3",
    "OCT-Binary/vgg16/VGG16-Method4-MLClassifier.ipynb":                    "oct_vgg16_m4",
}

# ── Detect method type from prefix ────────────────────────────────────────────
def get_method(prefix):
    if prefix.endswith("_m1"): return "base"
    if prefix.endswith("_m2"): return "finetune"
    if prefix.endswith("_m3"): return "augment"
    if prefix.endswith("_m4"): return "ml"
    return "base"

# ── Results export cell for deep learning notebooks (m1, m2, m3) ──────────────
def make_export_cell_dl(prefix):
    return f"""\
import csv, os, json
from sklearn.metrics import classification_report

# ── Save confusion matrix as PNG ──
os.makedirs('/kaggle/working/results', exist_ok=True)
cm_path = '/kaggle/working/results/{prefix}_confusion_matrix.png'
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={{'size': 16}}, ax=ax_cm)
ax_cm.set_title('Confusion Matrix — {prefix}', fontsize=14, fontweight='bold')
ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')
fig_cm.tight_layout()
fig_cm.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close(fig_cm)
print(f'Confusion matrix saved → {{cm_path}}')

# ── Save metrics to CSV ──
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
csv_path = '/kaggle/working/results/{prefix}_results.csv'
rows = []
for label, metrics in report_dict.items():
    if isinstance(metrics, dict):
        rows.append({{
            'model':     '{prefix}',
            'class':     label,
            'precision': round(metrics['precision'], 4),
            'recall':    round(metrics['recall'],    4),
            'f1-score':  round(metrics['f1-score'],  4),
            'support':   metrics['support'],
            'test_acc':  round(test_acc, 4),
            'test_loss': round(test_loss, 4),
        }})

write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    if write_header:
        writer.writeheader()
    writer.writerows(rows)

print(f'Results saved → {{csv_path}}')
print(f'Test Accuracy : {{test_acc*100:.2f}}%')
print(f'Test Loss     : {{test_loss:.4f}}')
"""

# ── Results export cell for ML notebooks (m4) ─────────────────────────────────
def make_export_cell_ml(prefix):
    return f"""\
import csv, os

# ── Save confusion matrix as PNG ──
os.makedirs('/kaggle/working/results', exist_ok=True)
cm_path = '/kaggle/working/results/{prefix}_confusion_matrix.png'
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={{'size': 16}}, ax=ax_cm)
ax_cm.set_title('Confusion Matrix — {prefix}', fontsize=14, fontweight='bold')
ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')
fig_cm.tight_layout()
fig_cm.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close(fig_cm)
print(f'Confusion matrix saved → {{cm_path}}')

# ── Save metrics to CSV ──
from sklearn.metrics import classification_report
report_dict = classification_report(test_labels, best_preds, target_names=class_names, output_dict=True)
csv_path = '/kaggle/working/results/{prefix}_results.csv'
rows = []
for label, metrics in report_dict.items():
    if isinstance(metrics, dict):
        rows.append({{
            'model':     '{prefix}',
            'classifier': best_name,
            'class':     label,
            'precision': round(metrics['precision'], 4),
            'recall':    round(metrics['recall'],    4),
            'f1-score':  round(metrics['f1-score'],  4),
            'support':   metrics['support'],
            'test_acc':  round(best_acc, 4),
        }})

write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    if write_header:
        writer.writeheader()
    writer.writerows(rows)

print(f'Results saved → {{csv_path}}')
print(f'Best Classifier : {{best_name}}')
print(f'Test Accuracy   : {{best_acc*100:.2f}}%')
"""

# ── Make a notebook cell dict ──────────────────────────────────────────────────
def make_code_cell(source):
    lines = source.split("\n")
    lines_with_newline = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": lines_with_newline,
    }

def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }

# ── Source transformations applied line by line ────────────────────────────────
def transform_source(source, prefix, method):
    lines = source if isinstance(source, list) else source.split("\n")
    result = []

    for line in lines:
        # 1. Remove Dropout
        if re.search(r'Dropout\s*\(', line):
            continue

        # 2. Fix Phase-1 checkpoint paths
        line = re.sub(
            r"'/kaggle/working/(\w+)_(m\d)_(p1)_ckpt\.pkl'",
            f"'/kaggle/working/{prefix}_p1_ckpt.pkl'",
            line
        )
        line = re.sub(
            r"'/kaggle/working/(\w+)_(m\d)_(p1)_weights\.h5'",
            f"'/kaggle/working/{prefix}_p1_weights.h5'",
            line
        )
        # Fix Phase-2 checkpoint paths
        line = re.sub(
            r"'/kaggle/working/(\w+)_(m\d)_(p2)_ckpt\.pkl'",
            f"'/kaggle/working/{prefix}_p2_ckpt.pkl'",
            line
        )
        line = re.sub(
            r"'/kaggle/working/(\w+)_(m\d)_(p2)_weights\.h5'",
            f"'/kaggle/working/{prefix}_p2_weights.h5'",
            line
        )
        # Fix single-phase checkpoint paths (m1, m3)
        line = re.sub(
            r"'/kaggle/working/(\w+)_(m\d)_ckpt\.pkl'",
            f"'/kaggle/working/{prefix}_ckpt.pkl'",
            line
        )
        line = re.sub(
            r"'/kaggle/working/(\w+)_(m\d)_weights\.h5'",
            f"'/kaggle/working/{prefix}_weights.h5'",
            line
        )
        # Fix ML model pickle path
        line = re.sub(
            r"'/kaggle/working/(\w+)_ml_models\.pkl'",
            f"'/kaggle/working/{prefix}_ml_models.pkl'",
            line
        )

        # 3. Fix confusion matrix savefig — inject savefig right after plt.show()
        #    (handled separately below as a post-pass)

        result.append(line)

    return result

# ── Check if a cell already has savefig (to avoid duplicates) ─────────────────
def cell_source_str(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src

# ── Main processing function ───────────────────────────────────────────────────
def process_notebook(nb_path, prefix):
    if not os.path.exists(nb_path):
        print(f"  ✗ NOT FOUND: {nb_path}")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    method = get_method(prefix)
    new_cells = []
    export_cell_added = False

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        src_lines = cell.get("source", [])
        src_str   = "".join(src_lines)

        # ── Apply line-level transforms ──
        transformed = transform_source(src_lines, prefix, method)
        cell["source"] = transformed
        src_str_new = "".join(transformed)

        # ── Inject savefig into confusion matrix cell ──
        if "confusion_matrix" in src_str_new and "sns.heatmap" in src_str_new and "savefig" not in src_str_new:
            # Add savefig before plt.show()
            new_source = []
            for line in transformed:
                if "plt.show()" in line and "tight_layout" not in line:
                    indent = len(line) - len(line.lstrip())
                    sp = " " * indent
                    os_makedirs = f"{sp}os.makedirs('/kaggle/working/results', exist_ok=True)\n"
                    save_line   = f"{sp}plt.savefig('/kaggle/working/results/{prefix}_confusion_matrix.png', dpi=150, bbox_inches='tight')\n"
                    new_source.append(os_makedirs)
                    new_source.append(save_line)
                new_source.append(line)
            cell["source"] = new_source

        new_cells.append(cell)

        # ── Inject export cell after the Evaluation section ──
        if not export_cell_added and "classification_report" in src_str_new and (
            "y_pred" in src_str_new or "best_preds" in src_str_new
        ):
            new_cells.append(make_markdown_cell("## Export Results"))
            if method == "ml":
                export_code = make_export_cell_ml(prefix)
            else:
                export_code = make_export_cell_dl(prefix)
            new_cells.append(make_code_cell(export_code))
            export_cell_added = True

    nb["cells"] = new_cells

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  ✓ Done: {nb_path}")

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PFE Notebook Modifier")
    print("=" * 60)
    total = 0
    for nb_path, prefix in NOTEBOOKS.items():
        print(f"\n[{prefix}]")
        process_notebook(nb_path, prefix)
        total += 1
    print("\n" + "=" * 60)
    print(f"  Done! {total} notebooks processed.")
    print("  Results will be saved to /kaggle/working/results/")
    print("=" * 60)
