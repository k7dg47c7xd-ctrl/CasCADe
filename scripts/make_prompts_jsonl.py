from pathlib import Path
import argparse
import json
import random


LEGACY_SYSTEM_PROMPT = """You are a 3D CAD JSON generator.

Your task is to generate a 3D modeling structure that conforms to the CAD-JSON specification based on the user's natural language modeling requirements.

The following rules must be strictly followed:

1. Output only JSON. Do not output Markdown, do not output explanatory text.
2. The root node must be a Group.
3. Every node must contain, and only contain, the following 5 fields:
   - type
   - features
   - group
   - coordinates
   - parameters
4. The type must be chosen from the following set:
   ["Group", "Fuse", "Cut", "Prism", "Face", "Wire", "Edge", "Arc", "Circle"]
5. Group rules:
   - Edge, Arc, Circle: group must be "creator"
   - Group, Fuse, Cut, Prism, Face, Wire: group must be "operator"
6. Coordinates rules:
   - Group, Fuse, Cut: coordinates must be []
   - Prism: coordinates must contain 2 3D points
   - Face: coordinates must contain 3 3D points
   - Wire: coordinates must contain 3 3D points
   - Edge: coordinates must contain 2 3D points
   - Arc: coordinates must contain 3 3D points
   - Circle: coordinates must contain 4 3D points
7. Features nesting rules:
   - Prism may only contain Face in its features
   - Face may contain Wire or Group in its features
   - Wire may only contain Edge or Arc in its features
   - Edge, Arc, Circle must have an empty features array
   - Fuse and Cut must contain at least 2 child entities in their features
8. Parameters rules:
   - Group: {"use_result_filter": false, "exclude": false, "return_index_list": []}
   - Prism: {"height": number, "use_height": true, "auto_normal_vector": true, "combin": false}
   - Face: {"combin": false, "mode_flag": 3, "uv_list": []}
   - Wire: {"vertex_input": false, "edge_sort": false, "linear_to_lerance": 0.000001}
   - Edge: {"pline": false, "combin": false}
   - Arc: {"r": number, "alpha1": number, "alpha2": number, "circle_mode": "0p", "arc_mode": "3p", "sense": true}
   - Circle: {"r": number, "wire_type": true}
   - Fuse and Cut: {}

If the modeling request involves "union, merge, fuse", prefer using Fuse.
If the modeling request involves "cut, subtract, hollow, hole", prefer using Cut.
If the modeling request involves "cylinder, circular hole, circle", use Circle together with Prism to represent a circular extrusion.
"""


def get_base_id_from_description(desc_path: Path) -> str:
    """
    Example:
    00000007_description.txt -> 00000007
    """
    stem = desc_path.stem
    if stem.endswith("_description"):
        return stem[: -len("_description")]
    return stem


def find_pair(desc_path: Path) -> Path | None:
    """
    Find the corresponding lite.json in the same directory as the description file.
    Example:
    data/00000007/00000007_description.txt -> data/00000007/00000007_lite.json
    """
    base_id = get_base_id_from_description(desc_path)
    json_path = desc_path.with_name(f"{base_id}_lite.json")

    if json_path.exists():
        return json_path

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        default="data/prompt_data",
        required=False,
        help="Root data directory, e.g. data/, each subfolder contains *_description.txt and *_lite.json",
    )
    parser.add_argument(
        "--output",
        default="configs/prompts.jsonl",
        help="Output prompts.jsonl file path",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1080,
        help="Number of samples to extract, 0 means all",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the samples randomly",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    desc_files = sorted(data_root.rglob("*_description.txt"))

    if not desc_files:
        raise FileNotFoundError(f"No *_description.txt found in: {data_root}")

    pairs = []

    for desc_path in desc_files:
        json_path = find_pair(desc_path)
        if json_path is None:
            print(f"[Skip] Cannot find corresponding JSON for: {desc_path}")
            continue

        pairs.append((desc_path, json_path))

    if not pairs:
        raise FileNotFoundError("No description/json paired files found.")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(pairs)

    if args.num > 0:
        pairs = pairs[: args.num]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for desc_path, json_path in pairs:
            user_text = desc_path.read_text(encoding="utf-8").strip()

            if not user_text:
                print(f"[Skip] Empty description: {desc_path}")
                continue

            row = {
                "messages": [
                    {
                        "role": "system",
                        "content": LEGACY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_text,
                    },
                ],
                "source_description": str(desc_path),
                "reference_json": str(json_path),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Generation completed: {output_path}")
    print(f"Successfully written samples: {written}")
    print(f"Total paired files found: {len(pairs)}")


if __name__ == "__main__":
    main()