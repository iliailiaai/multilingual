pip install -r corpus_download/requirements.txt
python3 corpus_download/download_hf_corpora.py --resume --workers 12 --tokenize-batch-size 512