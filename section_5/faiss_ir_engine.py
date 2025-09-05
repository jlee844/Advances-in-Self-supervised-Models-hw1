#!/usr/bin/env python3
"""
Section 5.5 Faiss IR Engine and evaluation
"""

import json
import numpy as np
import pickle
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


class FAISS_IR_Engine:
    """FAISS-based IR Engine"""
    
    def __init__(self):
        self.index = None
        self.ground_truth = {}
        self.doc_ids = []
        self.doc_embeddings = []
        self.documents = {}
        self.claims = {}
        self.claim_embeddings = {}
        self.claim_ids = []
        
        
    def load_corpus_data(self, data_dir: Path = Path("scifact_data/data")):
        """Load corpus (documents) data from scifact_data folder"""
        corpus_file = data_dir / "corpus.jsonl"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        documents = {}
        with open(corpus_file, 'r') as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = doc['doc_id']
                documents[doc_id] = doc
        
        return documents

    def load_claims_data(self, data_dir: Path = Path("scifact_data/data")):
        """Load claims data and extract ground truth from scifact_data folder"""
        claims_train_file = data_dir / "claims_train.jsonl"
        claims_dev_file = data_dir / "claims_dev.jsonl"
        claims_test_file = data_dir / "claims_test.jsonl"
        
        # if not claims_train_file.exists():
        #     raise FileNotFoundError(f"Claims train file not found: {claims_train_file}")
        
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
        
        return claims, support_claims, contradict_claims

    def load_embeddings(self, claim_embeddings_path: str = "scifact_claim_embeddings.pkl", doc_embeddings_path: str = "scifact_evidence_embeddings.pkl"):
        """Load pre-computed OpenAI embeddings"""
        
        with open(claim_embeddings_path, 'rb') as f:
            claim_data = pickle.load(f)
        with open(doc_embeddings_path, 'rb') as f:
            doc_data = pickle.load(f)
        
        return claim_data, doc_data

    def load_data(self, data_dir: str = "scifact_data/data"):
        """Load all data from scifact_data folder"""
        data_path = Path(data_dir)
        
        self.documents = self.load_corpus_data(data_path)
        self.claims, self.ground_truth, _ = self.load_claims_data(data_path)
        
        claim_data, doc_data = self.load_embeddings()
        
        for (claim_id, claim_text), embedding in claim_data.items():
            self.claim_embeddings[claim_id] = embedding
            self.claim_ids.append(claim_id)
        
        for (doc_id, abstract), embedding in doc_data.items():
            self.doc_ids.append(doc_id)
            self.doc_embeddings.append(embedding)
    
    def build_index(self):
        """Build FAISS index from document embeddings"""
        
        doc_embeddings_array = np.array(self.doc_embeddings, dtype='float32')
        
        doc_embeddings_normalized = doc_embeddings_array / np.linalg.norm(doc_embeddings_array, axis=1, keepdims=True)
        
        dimension = doc_embeddings_normalized.shape[1]
        self.index = faiss.IndexFlatIP(dimension) 
        self.index.add(doc_embeddings_normalized)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar documents"""
        
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_normalized, k)
        
        doc_ids = [self.doc_ids[idx] for idx in indices[0]]
        
        return scores[0], np.array(doc_ids)
    
    def evaluate_retrieval(self, k_values: List[int] = [1, 5, 10, 50]) -> Dict:
        """Evaluate retrieval performance"""
        
        results = {}
        
        for k in k_values:
            precision_at_k = []
            recall_at_k = []
            mrr_at_k = []
            map_at_k = []
            
            valid_evaluations = 0
            
            for claim_id in self.claim_ids:
                if claim_id not in self.ground_truth:
                    continue
                
                target_doc_id = self.ground_truth[claim_id]
                query_embedding = self.claim_embeddings[claim_id]
                
                scores, retrieved_doc_ids = self.search(query_embedding, k)
                
                if target_doc_id in retrieved_doc_ids:
                    rank = np.where(retrieved_doc_ids == target_doc_id)[0][0] + 1
                    precision_at_k.append(1.0 / k)
                    recall_at_k.append(1.0)
                    mrr_at_k.append(1.0 / rank)
                    map_at_k.append(1.0 / rank)
                else:
                    precision_at_k.append(0.0)
                    recall_at_k.append(0.0)
                    mrr_at_k.append(0.0)
                    map_at_k.append(0.0)
                
                valid_evaluations += 1
            
            if precision_at_k:
                results[f'precision@{k}'] = np.mean(precision_at_k)
                results[f'recall@{k}'] = np.mean(recall_at_k)
                results[f'mrr@{k}'] = np.mean(mrr_at_k)
                results[f'map@{k}'] = np.mean(map_at_k)
        
        return results


def evaluate_faiss_ir_engine():
    """Main function to evaluate the FAISS IR engine"""
    
    ir_engine = FAISS_IR_Engine()
    ir_engine.load_data()
    ir_engine.build_index()
    k_values = [1, 5, 10, 50]
    results = ir_engine.evaluate_retrieval(k_values)
    
    print("MRR @ 1 MAP @ 1 MRR @ 10 MAP @ 10 MRR @ 50 MAP @ 50")
    
    mrr_1 = results.get('mrr@1', 0)
    map_1 = results.get('map@1', 0)
    mrr_10 = results.get('mrr@10', 0)
    map_10 = results.get('map@10', 0)
    mrr_50 = results.get('mrr@50', 0)
    map_50 = results.get('map@50', 0)
    
    print(f"{mrr_1:.4f} {map_1:.4f} {mrr_10:.4f} {map_10:.4f} {mrr_50:.4f} {map_50:.4f}")
    
    results_data = {
        "faiss_results": results,
    }
    
    with open('faiss_ir_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return ir_engine, results


if __name__ == "__main__":
    ir_engine, results = evaluate_faiss_ir_engine()
