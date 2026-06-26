import argparse
import logging
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


LANGUAGE_ALIASES = {
    # FLORES+ can use individual ISO 639-3 codes for some macrolanguages.
    "aze": ["aze", "azj"],
    "est": ["est", "ekk"],
    "uzb": ["uzb", "uzn"],
}


def _aggregate(data, attention_mask):
    num_layers, _, hidden_dim = data.shape
    
    # Get valid positions
    valid_indices = attention_mask.nonzero(as_tuple=True)[0]
    valid_indices = valid_indices[1:] # ignore BOS token
    num_valid = valid_indices.shape[0]
    
    if num_valid == 0:
        return torch.zeros((num_layers, 1, hidden_dim), device=data.device, dtype=data.dtype)
    averaged_chunk = data[:, valid_indices, :].mean(dim=1)
    return averaged_chunk.unsqueeze(1)


def main(
        model_name_or_path: str,
        dataset_name: str,
        train_file: str,
        max_seq_length: int,
        max_train_samples: int,
        language: str,
        num_layers: int = 7,
        batch_size: int = 4,
        preprocessing_num_workers: int = 1,
        split: str = "train",
        dataset_str: str = None,
        skip_existing: bool = False,
    ):
    model_id = model_name_or_path.split("/")[-1]
    dataset_identifier = dataset_name.split("/")[-1] if dataset_name is not None else train_file.split("/")[-1].split(".")[0] if dataset_str is None else dataset_str
    name = "full" if max_train_samples is None else str(max_train_samples)
    folder = "language_vectors_bucket"
    path = f"{folder}/{dataset_identifier}/{model_id}/{name}/"
    output_path = f"{path}/{language}.npy"

    if skip_existing and os.path.exists(output_path):
        print(f"[SKIP] {language}: vector already exists at {output_path}")
        return

    if dataset_name is not None:
        raw_datasets = load_dataset(
            dataset_name,
        )
    else:
        data_files = {}
        data_files[split] = train_file
        extension = train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)
   
    if "iso_639_3" in raw_datasets[split].column_names:
        # For languages with multiple scripts, we filter by script as well
        subset = {
            "min": "Arab",
            "cmn": "Hans",
            "arb": "Arab",
        }
        filtered_datasets = []
        matched_languages = []
        for language_code in LANGUAGE_ALIASES.get(language, [language]):
            if language_code in subset:
                candidate = raw_datasets[split].filter(
                    lambda x, code=language_code: x["iso_639_3"] == code and x["iso_15924"] == subset[code]
                )
            else:
                candidate = raw_datasets[split].filter(lambda x, code=language_code: x["iso_639_3"] == code)
            candidate_len = len(candidate)
            print(f"[INFO] {language}: matched {candidate_len} rows for iso_639_3={language_code}")
            if candidate_len > 0:
                filtered_datasets.append(candidate)
                matched_languages.append(language_code)

        if not filtered_datasets:
            raise ValueError(f"No dataset rows found for language {language}")
        if matched_languages != [language]:
            print(f"[INFO] {language}: using FLORES+ iso_639_3={','.join(matched_languages)}")
        raw_datasets[split] = (
            filtered_datasets[0]
            if len(filtered_datasets) == 1
            else concatenate_datasets(filtered_datasets)
        )

    if max_train_samples is not None:
        sample_count = min(max_train_samples, len(raw_datasets[split]))
        raw_datasets[split] = raw_datasets[split].shuffle(seed=42).select(range(sample_count))

    column_names = raw_datasets[split].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    print(f"[INFO] {language}: using {len(raw_datasets[split])} rows from split={split}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token == None:
        if tokenizer.unk_token == None and tokenizer.pad_token == None:
            # raw llama3
            print("adding a special padding token...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.unk_token
    model.resize_token_embeddings(len(tokenizer))

    model.eval()


    def preprocess_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=False, max_length=max_seq_length, padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=preprocessing_num_workers,
        desc="Running tokenizer on dataset",
    )

    train_dataset = tokenized_datasets[split]
    print(f"[INFO] {language}: tokenized dataset has {len(train_dataset)} rows")
    if len(train_dataset) == 0:
        raise ValueError(f"No tokenized samples found for language {language}")

    states = []
    summed_states = None

    def flush_states():
        nonlocal states, summed_states
        if not states:
            return
        s = torch.stack(states, dim=0).sum(dim=0).numpy()
        if summed_states is None:
            summed_states = s
        else:
            summed_states += s
        states = []

    data_loader = DataLoader(train_dataset.with_format("torch"), batch_size=batch_size, shuffle=False)
    num_samples = 0

    hidden_state_limit = None
    if num_layers is not None and num_layers > 0:
        # HF hidden_states contains embeddings at index 0, then one entry per transformer block.
        hidden_state_limit = num_layers + 1

    for sample in tqdm(data_loader):
        with torch.no_grad():
            output = model(
                input_ids=sample["input_ids"].to(model.device),
                attention_mask=sample["attention_mask"].to(model.device), 
                output_hidden_states=True,
            )
        hidden_states = output.hidden_states
        if hidden_state_limit is not None:
            hidden_states = hidden_states[:hidden_state_limit]
        state = torch.stack([x.detach().cpu().float() for x in hidden_states], dim=1)
        for i in range(len(state)):
            states.append(_aggregate(state[i], attention_mask=sample["attention_mask"][i].detach().cpu()))
            
        num_samples += sample["input_ids"].shape[0]
        if len(states) > 20:
            flush_states()





    
    flush_states()
    if summed_states is None or num_samples == 0:
        raise ValueError(
            f"No samples found for language {language}; "
            f"filtered_rows={len(raw_datasets[split])}, tokenized_rows={len(train_dataset)}"
        )
    language_vector = summed_states / num_samples

    if not os.path.exists(path):
        os.makedirs(path)
    np.save(output_path, language_vector)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Computing average language representation')
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--language", type=str)
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--num_layers", default=7, type=int)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--dataset_str", default=None, type=str)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    print(f"Processing {args.language}")

    main(
        model_name_or_path=args.model_name_or_path, 
        dataset_name=args.dataset_name,
        train_file=args.train_file,
        max_seq_length=args.max_seq_length,
        language=args.language,
        max_train_samples=args.max_train_samples,
        num_layers=args.num_layers,
        preprocessing_num_workers=4,
        split=args.split,
        dataset_str=args.dataset_str,
        skip_existing=args.skip_existing,
    )
