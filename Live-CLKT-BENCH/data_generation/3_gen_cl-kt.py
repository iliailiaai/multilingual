import json
import argparse
import os
from collections import defaultdict
import random
from typing import List


LETTERS = ["A", "B", "C", "D"]
CORRECT_OPTION_ALIASES = (
    "correct_option",
    "answer",
    "correct_answer",
    "correctOption",
    "correct option",
    "Correct Option",
)


def mcq_format(item):

    template = (
        "{q}\n"
        "- A. {a}\n"
        "- B. {b}\n"
        "- C. {c}\n"
        "- D. {d}\n"
    )
    return template.format(
        q=item['question'],
        a=item['options']['A'],
        b=item['options']['B'],
        c=item['options']['C'],
        d=item['options']['D'],
    )


def shuffle_options(qa_pair, rng):
    """
    Shuffle options and return updated qa_pair with:
    - 'options': shuffled dict {A: ..., B: ..., C: ..., D: ...}
    - 'correct_option': new letter after shuffle
    """
    options = list(qa_pair["options"].items())  # [(A, textA), (B, textB), ...]
    rng.shuffle(options)

    # build new mapping A/B/C/D -> text
    new_options = {}
    correct_text = qa_pair["options"][qa_pair["correct_option"]]
    new_correct = None
    for new_letter, (old_letter, text) in zip(LETTERS, options):
        new_options[new_letter] = text
        if text == correct_text:
            new_correct = new_letter

    qa_pair["options"] = new_options
    qa_pair["correct_option"] = new_correct
    return qa_pair


def normalize_qa_pair(qa_pair, qa_fp, idx):
    if not isinstance(qa_pair, dict):
        print(f"[WARN] Skipping invalid QA item in {qa_fp}[{idx}]: expected object, got {type(qa_pair).__name__}")
        return None

    question = qa_pair.get("question")
    options = qa_pair.get("options")
    if not question or not isinstance(options, dict):
        print(f"[WARN] Skipping invalid QA item in {qa_fp}[{idx}]: missing question/options")
        return None

    missing_options = [letter for letter in LETTERS if letter not in options or options[letter] in (None, "")]
    if missing_options:
        print(f"[WARN] Skipping invalid QA item in {qa_fp}[{idx}]: missing options {missing_options}")
        return None

    correct = None
    for key in CORRECT_OPTION_ALIASES:
        if key in qa_pair:
            correct = qa_pair[key]
            break

    if isinstance(correct, str):
        correct = correct.strip()
        if correct[:1].upper() in LETTERS:
            correct = correct[:1].upper()
        else:
            for letter, text in options.items():
                if correct == str(text).strip():
                    correct = letter
                    break

    if correct not in LETTERS:
        print(f"[WARN] Skipping invalid QA item in {qa_fp}[{idx}]: missing/invalid correct_option")
        return None

    normalized = dict(qa_pair)
    normalized["question"] = question
    normalized["options"] = {letter: options[letter] for letter in LETTERS}
    normalized["correct_option"] = correct
    return normalized


def main(
    doc_dir:str,
    factqa_dir:str, 
    test_langs:List[str], 
    output_dir:str, 
    val_ratio:float=0.2, 
    seed:int=204,
):

    rng = random.Random(seed)

    for train_lang in test_langs:

        train_lang_factQA_dir = os.path.join(factqa_dir)
        train_lang_doc_dir = os.path.join(doc_dir, train_lang)
        train_docs, test_mcqs = [], []
        skipped_invalid_qas = 0

        for unit_name in os.listdir(train_lang_factQA_dir):
            #  train doc
            doc_fp = os.path.join(train_lang_doc_dir, f"{unit_name}.json")
            with open(doc_fp, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            train_docs.append({
                "text": doc_data["fact_source"],
                "source": unit_name
            })

            #  test qas
            for test_lang in test_langs:

                qa_fp = os.path.join(train_lang_factQA_dir, unit_name, f"{test_lang}QA.json")
                with open(qa_fp, "r", encoding="utf-8") as f:
                    qa_data = json.load(f)
                if isinstance(qa_data, dict) and "QA" in qa_data:
                    qa_data = qa_data["QA"]
                if not isinstance(qa_data, list):
                    print(f"[WARN] Skipping invalid QA file {qa_fp}: expected list")
                    continue
                # print(qa_fp)
                for idx, qa_pair in enumerate(qa_data):
                    qa_pair = normalize_qa_pair(qa_pair, qa_fp, idx)
                    if qa_pair is None:
                        skipped_invalid_qas += 1
                        continue
                    qa_pair = shuffle_options(qa_pair, rng)
                    choice = qa_pair['correct_option']
                    test_mcqs.append(
                        {
                            'question': mcq_format(qa_pair),
                            'answer': choice,
                            'text_answer': qa_pair['options'][choice],
                            'train_lang': train_lang,
                            'test_lang': test_lang,
                            'source': unit_name,
                            'qid': f"{unit_name}-{idx}"
                        }
                    )


        # ----- Shuffle and split at item-level -----
        if skipped_invalid_qas:
            print(f"[WARN] {train_lang}: skipped {skipped_invalid_qas} invalid QA items.")
        qid_groups_mcq = defaultdict(list)
        for item in test_mcqs:
            qid_groups_mcq[item["qid"]].append(item)

        all_qids = list(qid_groups_mcq.keys())
        rng.shuffle(all_qids)
        n_val = int(len(all_qids) * val_ratio)
        val_qids = set(all_qids[:n_val])
        print(f"[INFO] {train_lang} has {len(val_qids)} val qids out of {len(all_qids)} total qids.")

        val_mcq, test_mcq = [], []
        for qid in all_qids:
            if qid in val_qids:
                val_mcq.extend(qid_groups_mcq[qid])
            else:
                test_mcq.extend(qid_groups_mcq[qid])

        print(f"[INFO] {train_lang} has {len(val_mcq)} val and {len(test_mcq)} test items (MCQ).")

        # Save everything
        save_dir = os.path.join(output_dir, train_lang)
        os.makedirs(save_dir, exist_ok=True)
        save_jsonl(train_docs, save_dir, 'train_doc')
        save_jsonl(val_mcq, save_dir, 'val_mc')
        save_jsonl(test_mcq, save_dir, 'test_mc')
        print(f"[END] Finish Generating {train_lang} CL-KT Benchmark to {save_dir}")


def save_jsonl(data, save_dir, base_name):
    file_path = os.path.join(save_dir, f"{base_name}.jsonl")
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to: {file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factqa_dir", 
        type=str, 
        default="data/factQA/movie/2025-01-01-2025-07-31",
        help="Directory of Movie QA Set"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,  
        default="data/benchmark/movie",
        help="Directory to save the Benchmark"
    )
    parser.add_argument(
        "--test_languages",
        type=str,
        nargs='+',
        default=["en", "ja", "fr", "es", "zh"],
        help="List of test language codes"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.01
    )
    parser.add_argument(
        "--training_docs_dir", type=str, 
        default="data/train_docs/movie/2025-01-01_2025-07-31"
    )
    args = parser.parse_args()

    main(
        args.training_docs_dir,
        args.factqa_dir, 
        args.test_languages, 
        args.output_dir,
        args.val_ratio
    )
