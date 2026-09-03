import json
import chromadb
from fastembed import TextEmbedding


BGE_MODEL = "BAAI/bge-small-en-v1.5"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATASET_PATH = r"data\master_dataset\kannada\train.jsonl"

TOP_K = 2s