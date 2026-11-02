import json
from pathlib import Path

root = Path("data/verify_data")
out_path = Path("configs/val.json")

instruction = "Generate the corresponding CAD modeling operation JSON based on the given natural language description."

samples = []
skipped = []

for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue

    sid = d.name
    desc_path = d / f"{sid}_description.txt"
    lite_path = d / f"{sid}_lite.json"

    if not desc_path.exists() or not lite_path.exists():
        skipped.append(sid)
        continue

    description = desc_path.read_text(encoding="utf-8").strip()

    with open(lite_path, "r", encoding="utf-8") as f:
        cad_json = json.load(f)

    output = json.dumps(cad_json, ensure_ascii=False, separators=(",", ":"))

    samples.append({
        "instruction": instruction,
        "input": description,
        "output": output
    })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print("Conversion completed")
print("Number of samples:", len(samples))
print("Skipped:", skipped)
print("Output file:", out_path)