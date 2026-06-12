import json
import os
import librosa
import re
from sklearn.model_selection import train_test_split
from typing import List, Dict


def create_manifests(
    input_jsonl: str,
    output_dir: str,
    val_size: float = 0.1,
    cleaned: bool = True,
    random_state: int = 42,
):
    """
    Converts ASR jsonl into NeMo manifests, filtering for English only.
    For CTC models: Preserves internal punctuation (like apostrophes) while stripping structural marks.
    For TDT models: USes original text with punctuation
    """

    def normalize_english_text(text: str) -> str:
        if not cleaned:
            return text

        # 1. Lowercase
        text = text.lower()

        # 2. Handle dashes per wer_transforms (convert to spaces)
        text = re.sub(r"[-—–]", " ", text)

        # 3. Whitelist Approach: Keep only a-z, spaces, and '
        # Everything else (periods, commas, etc.) becomes a space
        text = re.sub(r"[^a-z\s']", " ", text)

        # 4. Final Cleanup
        # Removes multiple spaces and trailing whitespace
        return " ".join(text.split()).strip()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    converted_data = []
    durations = []

    print(f"Filtering English samples from {input_jsonl}...")

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # --- ENGLISH ONLY FILTER ---
            if item.get("language", "").lower() != "english":
                continue

            audio_path = os.path.join(os.path.dirname(input_jsonl), item["audio"])

            try:
                duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"Skipping {audio_path}: {e}")
                continue

            converted_data.append(
                {
                    "audio_filepath": os.path.abspath(audio_path),
                    "duration": duration,
                    "text": normalize_english_text(item["transcript"]),
                }
            )
            durations.append(duration)

    # Split
    train_data, val_data = train_test_split(
        converted_data, test_size=val_size, random_state=random_state
    )

    def save_jsonl(data: List[Dict], filename: str):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    t_path = save_jsonl(train_data, "train_manifest_cleaned.json")
    v_path = save_jsonl(val_data, "val_manifest_cleaned.json")

    print("-" * 30)
    print("English Manifest Prep Complete")
    print(f"Total English Samples: {len(converted_data)}")
    print(f"Train path: {t_path}")
    print(f"Val path:   {v_path}")
    print("-" * 30)


if __name__ == "__main__":
    create_manifests(
        input_jsonl="asr/asr.jsonl",
        output_dir="asr",
        cleaned=False,
    )
    create_manifests(
        input_jsonl="asr/asr.jsonl",
        output_dir="asr",
        cleaned=True,
    )

