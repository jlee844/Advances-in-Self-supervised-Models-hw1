#!/usr/bin/env python3
"""
Extract Real Ground Truth from SciFact Dataset
"""

import os
import sys
import json
import requests
import tarfile
import tempfile
import traceback
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

def load_dataset_info():
    """Load dataset information from dataset_infos.json"""
    with open('dataset_infos.json', 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

def download_scifact_data():
    """Download SciFact dataset from the official source"""
    
    dataset_info = load_dataset_info()
    if not dataset_info:
        return None
    corpus_info = dataset_info.get('corpus', {})
    download_checksums = corpus_info.get('download_checksums', {})
    
    if not download_checksums:
        return None
    
    download_url = list(download_checksums.keys())[0]
    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz')
    
    total_size = int(response.headers.get('content-length', 0))
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
                pbar.update(len(chunk))
    
    temp_file.close()
    return temp_file.name

def extract_scifact_data(tar_file_path: str):
    """Extract the tar.gz file"""
    
    data_dir = Path("scifact_data")
    data_dir.mkdir(exist_ok=True)
    
    with tarfile.open(tar_file_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    
    return data_dir

def load_corpus_data(data_dir: Path):
    """Load corpus (documents) data"""
    corpus_file = data_dir / "data" / "corpus.jsonl"
    if not corpus_file.exists():
        return None
    
    documents = {}
    with open(corpus_file, 'r') as f:
        for line_num, line in enumerate(f):
            doc = json.loads(line.strip())
            doc_id = doc['doc_id']
            documents[doc_id] = doc
    
    return documents

def load_claims_data(data_dir: Path):
    """Load claims data and extract ground truth"""
    # Look for claims files in data subdirectory
    claims_train_file = data_dir / "data" / "claims_train.jsonl"
    claims_dev_file = data_dir / "data" / "claims_dev.jsonl"
    claims_test_file = data_dir / "data" / "claims_test.jsonl"
    
    if not claims_train_file.exists():
        return None, None
    
    claims = {}
    support_claims = {}
    contradict_claims = {}
    
    claim_files = [claims_train_file, claims_dev_file, claims_test_file]
    
    for claim_file in claim_files:
        if claim_file.exists():
            with open(claim_file, 'r') as f:
                for line in f:
                    claim = json.loads(line.strip())
                    claim_id = claim['id']
                    claims[claim_id] = claim
                
                    evidence = claim.get('evidence', {})
                    
                    for doc_id_str, evidence_list in evidence.items():
                        for evidence_item in evidence_list:
                            label = evidence_item.get('label')
                            if label == 'SUPPORT':
                                doc_id = int(doc_id_str)
                                support_claims[claim_id] = doc_id
                                break 
                            elif label == 'CONTRADICT':
                                doc_id = int(doc_id_str)
                                contradict_claims[claim_id] = doc_id
                                break  
    
    return claims, support_claims

def create_ground_truth(support_claims: Dict[int, int], documents: Dict[int, dict]):
    """Create a summary of the ground truth"""
    
    doc_counts = {}
    for claim_id, doc_id in support_claims.items():
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    
    ground_truth_data = {
        "support_claims": support_claims,
    }
    
    with open('ground_truth.json', 'w') as f:
        json.dump(ground_truth_data, f, indent=2)
    
    return ground_truth_data

def main():
    tar_file_path = download_scifact_data()
    data_dir = extract_scifact_data(tar_file_path)
    documents = load_corpus_data(data_dir)
    claims, support_claims = load_claims_data(data_dir)
    ground_truth_data = create_ground_truth(support_claims, documents)
    os.unlink(tar_file_path)


if __name__ == "__main__":
    main()
