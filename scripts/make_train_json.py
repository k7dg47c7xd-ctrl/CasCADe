import os
import json
import glob

def convert_to_alpaca(data_root, output_file):
    alpaca_data = []
    # Walk every subdirectory under data_root
    subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_root, subdir)
        # Find the description.txt file in this directory (any prefix matches)
        desc_files = glob.glob(os.path.join(subdir_path, "*_description.txt"))
        if not desc_files:
            print(f"Warning: no description file in {subdir_path}, skip.")
            continue
        desc_path = desc_files[0]
        
        # The corresponding JSON file (same prefix as the description, _lite.json suffix)
        base_name = os.path.basename(desc_path).replace("_description.txt", "")
        json_path = os.path.join(subdir_path, f"{base_name}_lite.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skip.")
            continue
        
        with open(desc_path, "r", encoding="utf-8") as f:
            instruction_text = f.read().strip()
        
        with open(json_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
        
        # Serialize the JSON object into a compact string
        output_str = json.dumps(json_obj, ensure_ascii=False, separators=(',', ':'))
        
        alpaca_item = {
            "instruction": "Given a natural language description, generate the corresponding CAD modeling operation JSON.",
            "input": instruction_text,
            "output": output_str
        }
        alpaca_data.append(alpaca_item)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion finished: {len(alpaca_data)} records saved to {output_file}")

if __name__ == "__main__":
    convert_to_alpaca(
        data_root=r"data/train_data",  # Data root: contains *_description.txt and *_lite.json
        output_file=r"configs/cad_train.json"
    )