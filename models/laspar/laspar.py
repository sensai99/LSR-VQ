import torch
import ir_datasets
import faiss
import wandb
import heapq
import time
import sys
import random
import string
import os
import pickle
import math

import torch.nn as nn
import torch.optim as optim
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import ndcg_score, recall_score
from collections import defaultdict
from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm
from rank_eval import Qrels, Run, evaluate
from cluster import balanced_kmeans_clustering


# This function creates an inverted index for sparse retrieval
def create_inverted_index(model, docs, tokenizer, device, batch_size=128, min_weight=1e-5, topk=1024):
    """
    Create an inverted index for efficient sparse retrieval.
    
    The inverted index maps dimension indices to lists of (doc_id, weight) pairs.
    For sparse embeddings, this allows efficient lookup of documents that have 
    non-zero weights in specific dimensions.
    
    Args:
        model: The retrieval model that produces document embeddings
        docs: Dictionary mapping doc_ids to document text
        tokenizer: Tokenizer for processing document text
        device: Device to run inference on (CPU or GPU)
        batch_size: Batch size for document encoding
        min_weight: Minimum weight threshold for including a dimension in the index
        topk: Number of top dimensions to keep in sparse embeddings
        
    Returns:
        Dictionary mapping dimension indices to tuples of (doc_ids, weights)
    """
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    inverted_index = defaultdict(list)
    doc_ids = list(docs.keys())
    doc_texts = list(docs.values())
    num_docs = len(doc_ids)
    
    for i in tqdm(range(0, num_docs, batch_size), desc="Building inverted index"):
        batch_doc_ids = doc_ids[i:i+batch_size]
        batch_doc_texts = doc_texts[i:i+batch_size]
        doc_inputs = tokenizer(batch_doc_texts, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=base_model.seq_length)
        
        with torch.cuda.amp.autocast(enabled=base_model.fp16):
            batch_embeddings = model(doc_inputs["input_ids"].to(device),
                                   doc_inputs["attention_mask"].to(device))
            #batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            # Apply top-k sparsification
            values, indices = torch.topk(batch_embeddings.abs(), k=topk, dim=1)
            sparse_embeddings = torch.zeros_like(batch_embeddings)
            sparse_embeddings.scatter_(1, indices, batch_embeddings.gather(1, indices))
            
            batch_embeddings = sparse_embeddings.detach().cpu().numpy()
        
        for j, doc_id in enumerate(batch_doc_ids):
            doc_embedding = batch_embeddings[j]
            # Get non-zero indices and their values
            nonzero_indices = np.nonzero(doc_embedding)[0]  # Ensure indices are integers
            for idx in nonzero_indices:
                weight = doc_embedding[idx]
                if weight > min_weight:
                    inverted_index[int(idx)].append((doc_id, float(weight)))  # Ensure integer keys
    
    # Sort postings lists by weight for each term
    for idx in inverted_index:
        inverted_index[idx] = sorted(inverted_index[idx], key=lambda x: abs(x[1]), reverse=True)
    
    # Convert to more efficient data structure
    optimized_index = {
        idx: (
            np.array([doc_id for doc_id, _ in postings], dtype=np.int32),
            np.array([weight for _, weight in postings], dtype=np.float32)
        )
        for idx, postings in inverted_index.items()
    }
    
    return optimized_index


# Optimized search function for sparse retrieval using the inverted index
def search_inverted_index(query_embedding, inverted_index, query_topk=128, min_weight=1e-5):
    """
    Efficient search function for the inverted index.
    
    Uses the sparse structure of embeddings to compute relevance scores without
    materializing the full embedding matrix. Only considers the top-k dimensions
    of the query embedding for efficient retrieval.
    
    Args:
        query_embedding: Sparse embedding vector for the query
        inverted_index: Inverted index mapping dimension indices to (doc_ids, weights)
        query_topk: Number of top query dimensions to consider for retrieval
        min_weight: Minimum weight threshold for considering a dimension
        
    Returns:
        List of (doc_id, score) pairs for the top 1000 documents
    """
    scores = defaultdict(float)
    seen_docs = set()
    
    # Get top-k query dimensions by weight
    weights = [(idx, weight) for idx, weight in enumerate(query_embedding) if abs(weight) > min_weight]
    top_weights = heapq.nlargest(query_topk, weights, key=lambda x: abs(x[1]))
    
    # Process each query term
    for idx, query_weight in top_weights:
        if idx not in inverted_index:
            continue
            
        doc_ids, doc_weights = inverted_index[idx]
        
        # Only process top documents per term
        for doc_id, doc_weight in zip(doc_ids, doc_weights):
            scores[doc_id] += query_weight * doc_weight
            seen_docs.add(doc_id)
    
    # Use numpy for final scoring
    if seen_docs:
        doc_ids = np.array(list(seen_docs))
        doc_scores = np.array([scores[doc_id] for doc_id in doc_ids])
        
        # Get top 1000 results efficiently
        top_k = min(1000, len(doc_scores))
        top_indices = np.argpartition(doc_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-doc_scores[top_indices])]
        
        return [(doc_ids[i], doc_scores[i]) for i in top_indices]
    
    return []


class ClusteredTrainDataset(Dataset):
    """
    Dataset for training with document/query clustering.
    
    Enables clustered training by grouping similar queries or documents together,
    which can help with training stability and convergence. Clusters can be based
    on either queries or documents.
    
    Args:
        queries: Dictionary mapping query IDs to query texts
        docs: Dictionary mapping document IDs to document texts
        qrels: List of (query_id, doc_id) pairs for relevance judgments
        tokenizer: Tokenizer for processing text
        seq_length: Maximum sequence length for tokenization
        teacher_embeddings: Optional precomputed embeddings from teacher model for distillation
        cluster_labels: Mapping from item IDs to cluster IDs
        cluster_by: Whether to cluster by 'query' or 'doc'
    """
    def __init__(self, queries, docs, qrels, tokenizer, seq_length=512, teacher_embeddings=None, cluster_labels=None, cluster_by='doc'):
        self.queries = queries
        self.docs = docs
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.teacher_embeddings = teacher_embeddings
        self.cluster_labels = cluster_labels
        self.cluster_by = cluster_by
        
        # Group qrels by cluster if clustering is enabled
        if cluster_labels is not None:
            self.clustered_qrels = defaultdict(list)
            for qrel in qrels:
                if cluster_by == 'query':
                    item_id = qrel[0]  # query_id
                else:
                    item_id = qrel[1]  # doc_id
                    
                if item_id in cluster_labels:
                    cluster_id = cluster_labels[item_id]
                    self.clustered_qrels[cluster_id].append(qrel)
            
            # Convert to list of (cluster_id, qrel_list) for easier indexing
            self.cluster_qrel_pairs = [(cluster_id, qrel_list) 
                                     for cluster_id, qrel_list in self.clustered_qrels.items()]
            self._length = sum(len(qrel_list) for _, qrel_list in self.cluster_qrel_pairs)
        else:
            self._length = len(qrels)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self.cluster_labels is not None:
            # Find the cluster and relative index within cluster
            cluster_idx = 0
            while idx >= len(self.cluster_qrel_pairs[cluster_idx][1]):
                idx -= len(self.cluster_qrel_pairs[cluster_idx][1])
                cluster_idx += 1
            
            qrel = self.cluster_qrel_pairs[cluster_idx][1][idx]
        else:
            qrel = self.qrels[idx]
            
        query_id = qrel[0]
        doc_id = qrel[1]
        query_text = self.queries[query_id]
        doc_text = self.docs[doc_id]
        
        query_input = self.tokenizer(query_text, padding="max_length", truncation=True, max_length=self.seq_length)
        doc_input = self.tokenizer(doc_text, padding="max_length", truncation=True, max_length=self.seq_length)
        
        item = {
            "query_input_ids": torch.tensor(query_input["input_ids"]),
            "query_attention_mask": torch.tensor(query_input["attention_mask"]),
            "doc_input_ids": torch.tensor(doc_input["input_ids"]),
            "doc_attention_mask": torch.tensor(doc_input["attention_mask"]),
            "query_text": query_text,
            "doc_text": doc_text,
            "query_id": query_id,
            "doc_id": doc_id
        }
        
        if self.teacher_embeddings is not None:
            item["teacher_query_embedding"] = self.teacher_embeddings['queries'][query_id]
            item["teacher_doc_embedding"] = self.teacher_embeddings['docs'][doc_id]
        
        return item


class ClusteredBatchSampler:
    """
    Batch sampler that yields clusters of related items.
    
    Creates batches from the same cluster, which can improve training efficiency
    by grouping similar items together. This can help with gradient computation
    and convergence.
    
    Args:
        dataset: A ClusteredTrainDataset instance
        batch_size: Maximum batch size
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.clusters = list(dataset.clustered_qrels.keys())
        
    def __iter__(self):
        # Shuffle clusters
        clusters = self.clusters.copy()
        random.shuffle(clusters)
        
        for cluster_id in clusters:
            qrels = self.dataset.clustered_qrels[cluster_id]
            indices = list(range(len(qrels)))
            
            # If cluster size is larger than batch_size, randomly sample batch_size items
            if len(indices) > self.batch_size:
                # Shuffle and yield batches of batch_size
                random.shuffle(indices)
                for i in range(0, len(indices), self.batch_size):
                    yield indices[i:min(i + self.batch_size, len(indices))]
            else:
                # For smaller clusters, yield all indices at once
                yield indices
    
    def __len__(self):
        total_batches = 0
        for cluster_id in self.clusters:
            cluster_size = len(self.dataset.clustered_qrels[cluster_id])
            # Calculate number of batches needed for this cluster
            total_batches += math.ceil(cluster_size / self.batch_size)
        return total_batches


class TrainDataset(Dataset):
    """
    Standard training dataset without clustering.
    
    Prepares (query, document) pairs for training the retrieval model.
    Optionally includes teacher embeddings for distillation.
    
    Args:
        queries: Dictionary mapping query IDs to query texts
        docs: Dictionary mapping document IDs to document texts
        qrels: List of (query_id, doc_id) pairs for relevance judgments
        tokenizer: Tokenizer for processing text
        seq_length: Maximum sequence length for tokenization
        teacher_embeddings: Optional precomputed embeddings from teacher model for distillation
    """
    def __init__(self, queries, docs, qrels, tokenizer, seq_length=512, teacher_embeddings=None):
        self.queries = queries
        self.docs = docs
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.teacher_embeddings = teacher_embeddings

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        qrel = self.qrels[idx]
        query_id = qrel[0]
        doc_id = qrel[1]
        query_text = self.queries[query_id]
        doc_text = self.docs[doc_id]
        
        query_input = self.tokenizer(query_text, padding="max_length", truncation=True, max_length=self.seq_length)
        doc_input = self.tokenizer(doc_text, padding="max_length", truncation=True, max_length=self.seq_length)
        
        item = {
            "query_input_ids": torch.tensor(query_input["input_ids"]),
            "query_attention_mask": torch.tensor(query_input["attention_mask"]),
            "doc_input_ids": torch.tensor(doc_input["input_ids"]),
            "doc_attention_mask": torch.tensor(doc_input["attention_mask"]),
            "query_text": query_text,
            "doc_text": doc_text,
            "query_id": query_id,
            "doc_id": doc_id
        }
        
        if self.teacher_embeddings is not None:
            item["teacher_query_embedding"] = self.teacher_embeddings['queries'][query_id]
            item["teacher_doc_embedding"] = self.teacher_embeddings['docs'][doc_id]
        
        return item


class EmbeddingRetrievalModel_Old(nn.Module):
    """
    Original latent sparse retrieval model (token-level upscaling followed by addition).
    
    In this architecture:
    1. Get token embeddings from BERT
    2. Project each token embedding to higher dimension using matrix U
    3. Sum the projected embeddings across tokens
    4. Apply log-ReLU transformation for sparsity
    
    Args:
        model_name: Hugging Face model name for the base LM
        d: Dimension of the base LM embeddings
        D: Dimension of the sparse embedding space / vocab size
        sparse: Whether to use sparse retrieval
        fp16: Whether to use FP16 precision
        seq_length: Maximum sequence length
        freeze_bert: Whether to freeze the BERT parameters
    """
    def __init__(self, model_name, d, D, sparse=False, fp16=False, seq_length=512, freeze_bert=False):
        super(EmbeddingRetrievalModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.seq_length = seq_length
        self.d = d  # LM embedding dimensions
        self.D = D  # Sparse embedding dimensions / vocab size
        self.sparse = sparse
        self.fp16 = fp16
        self.U = nn.Parameter(torch.randn(d, D))  # Projection matrix for upscaling
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        if self.fp16:
            self.bert = self.bert.half()
            self.U.data = self.U.data.half()
    
    def forward(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            H = self.bert(input_ids, attention_mask=attention_mask)[0]  # Get token embeddings
            
            if self.sparse:
                # For sparse retrieval: project each token, sum across tokens, then apply sparsity transformation
                token_projections = torch.matmul(H, self.U)  # Project each token to high dimension
                S = torch.log(1 + torch.relu(token_projections.sum(dim=1)))  # Sum and apply log-ReLU
                S = torch.clamp(S, max=10)  # Clamp values to avoid numerical issues
                S = torch.nn.functional.normalize(S, p=2, dim=1)  # L2 normalization
            else:
                # For dense retrieval: use mean pooling of token embeddings
                S = H.mean(dim=1)
                
        return S


class EmbeddingRetrievalModel(nn.Module):
    """
    Enhanced latent sparse retrieval model with token attention mechanism.
    
    In this architecture:
    1. Get token embeddings from BERT
    2. Apply attention weights to token embeddings
    3. Project the aggregated token representations to higher dimension
    4. Apply log-ReLU transformation for sparsity
    
    This approach uses a learned attention mechanism (A) to weight token 
    importance before upscaling, potentially giving more importance to
    informative tokens.
    
    Args:
        model_name: Hugging Face model name for the base LM
        d: Dimension of the base LM embeddings
        D: Dimension of the sparse embedding space / vocab size
        sparse: Whether to use sparse retrieval
        fp16: Whether to use FP16 precision
        seq_length: Maximum sequence length
        freeze_bert: Whether to freeze the BERT parameters
    """
    def __init__(self, model_name, d, D, sparse=False, fp16=False, seq_length=512, freeze_bert=False):
        super(EmbeddingRetrievalModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.seq_length = seq_length
        self.d = d  # LM embedding dimensions
        self.D = D  # Sparse embedding dimensions / vocab size
        self.sparse = sparse
        self.fp16 = fp16
        
        self.A = nn.Parameter(torch.randn(1, self.seq_length, 1))  # Token attention weights
        self.U = nn.Parameter(torch.randn(d, D))  # Upscaling projection matrix
        self.bias = nn.Parameter(torch.ones(D) * 3.0)  # Bias term for sparse activation
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        if self.fp16:
            self.bert = self.bert.half()
            self.A.data = self.A.data.half()
            self.U.data = self.U.data.half()
            self.bias.data = self.bias.data.half()
    
    def forward(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            H = self.bert(input_ids, attention_mask=attention_mask)[0].permute(0, 2, 1)  # Get token embeddings and permute for matmul
            H_prime = torch.matmul(H, self.A).squeeze(-1)  # Apply attention weights to aggregate tokens
            if self.sparse:
                # For sparse retrieval: project aggregated representation to higher dimension
                S = torch.matmul(H_prime, self.U)  # Project to high dimension
                S = torch.log(1 + torch.relu(S + self.bias))  # Apply log-ReLU with bias term
                S = torch.clamp(S, max=10)  # Clamp values to avoid numerical issues
                S = torch.nn.functional.normalize(S, p=2, dim=1)  # L2 normalization
            else:
                # For dense retrieval: use the attention-weighted token embeddings directly
                S = H_prime   
        return S


# Load and preprocess the dataset
def load_and_preprocess_dataset():
    dataset = ir_datasets.load("msmarco-passage/train/judged")
    # Load all documents
    print("Loading documents...")
    docs = {doc.doc_id: "document: " + doc.text for doc in dataset.docs_iter()}
    
    # Load all train queries
    print("Loading train queries...")
    queries = {query.query_id: "query: " + query.text for query in dataset.queries_iter()}

    # Load train qrels
    print("Loading train qrels...")
    train_qrels = []
    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        did = qrel.doc_id
        train_qrels.append((qid, did))

    print("Found {} documents, {} queries, and {} qrels.".format(len(docs), len(queries), len(train_qrels)))

    dataset = ir_datasets.load(f"msmarco-passage/dev/judged")
    # Load dev queries
    dev_queries = {query.query_id: "query: " + query.text for query in dataset.queries_iter()}

    # Load dev qrels
    dev_qrels = {}
    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        did = qrel.doc_id
        if qid not in dev_queries or did not in docs:
            continue
        if qid not in dev_qrels:
            dev_qrels[qid] = []
        dev_qrels[qid].append(did)
        
    return docs, queries, train_qrels, dev_queries, dev_qrels


# Evaluate the model
def evaluate_model(model, docs, dev_queries, dev_qrels, tokenizer, device, batch_size=32, batch_size_inference=128, min_weight=1e-5, query_topk=128, doc_topk=1024, quick_mode=True):
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    model.eval()
    all_results = {}  # Store results for each query

    # Filter docs if in quick mode
    if quick_mode:
        relevant_doc_ids = set()
        for qid in dev_qrels:
            relevant_doc_ids.update(dev_qrels[qid])
        docs = {doc_id: docs[doc_id] for doc_id in relevant_doc_ids}
        print(f"Quick mode: using {len(docs)} documents for evaluation")
    
    # Filter queries if in quick mode
    if quick_mode:
        dev_queries = dict(list(dev_queries.items())[:200])
        dev_qrels = {qid: dev_qrels[qid] for qid in dev_queries if qid in dev_qrels}
        print(f"Quick mode: using {len(dev_queries)} queries for evaluation")

    if base_model.sparse:
        # Sparse retrieval using inverted index
        inverted_index = create_inverted_index(model, docs, tokenizer, device, batch_size=batch_size_inference, 
                                             min_weight=min_weight, topk=doc_topk)
        
        with torch.no_grad():
            for qid, query in tqdm(dev_queries.items(), desc="Evaluating"):
                if qid not in dev_qrels:
                    continue
                
                query_input = tokenizer(query, return_tensors="pt", padding="max_length", 
                                      truncation=True, max_length=base_model.seq_length)
                query_embedding = model(query_input["input_ids"].to(device), 
                                     query_input["attention_mask"].to(device))
                
                # Search using optimized function
                query_embedding = query_embedding.cpu().numpy()[0]
                search_results = search_inverted_index(
                    query_embedding, 
                    inverted_index,
                    query_topk=query_topk,
                    min_weight=min_weight
                )
                
                # Store results for this query
                all_results[qid] = search_results
    else:
        # Dense retrieval using FAISS
        print("Building FAISS index...")
        doc_ids = list(docs.keys())
        doc_embeddings = []
        
        # Encode all documents
        for i in tqdm(range(0, len(doc_ids), batch_size_inference), desc="Encoding documents"):
            batch_docs = [docs[did] for did in doc_ids[i:i + batch_size_inference]]
            doc_inputs = tokenizer(batch_docs, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=base_model.seq_length)
            doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
            
            with torch.no_grad():
                batch_embeddings = model(doc_inputs["input_ids"], doc_inputs["attention_mask"])
                doc_embeddings.append(batch_embeddings.cpu().numpy())
        
        doc_embeddings = np.vstack(doc_embeddings)
        
        # Build FAISS index
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings)
        
        # Evaluate queries
        with torch.no_grad():
            for qid, query in tqdm(dev_queries.items(), desc="Evaluating"):
                if qid not in dev_qrels:
                    continue
                
                query_input = tokenizer(query, return_tensors="pt", padding="max_length",
                                      truncation=True, max_length=base_model.seq_length)
                query_input = {k: v.to(device) for k, v in query_input.items()}
                query_embedding = model(query_input["input_ids"], query_input["attention_mask"])
                query_embedding = query_embedding.cpu().numpy()
                
                # Search using FAISS
                scores, indices = index.search(query_embedding, 1000)
                search_results = [(doc_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
                all_results[qid] = search_results
    
    # Create rank_eval Run and Qrels objects
    run = Run()
    qrels = Qrels()

    # Add results to Run object
    for qid in all_results:
        doc_ids = [str(doc_id) for doc_id, score in all_results[qid]]
        scores = [float(score) for _, score in all_results[qid]]
        run.add(qid, doc_ids, scores)

    # Add relevance judgments to Qrels object
    for qid in dev_qrels:
        qrels.add(qid, [str(doc_id) for doc_id in dev_qrels[qid]], [1] * len(dev_qrels[qid]))

    # Evaluate using rank_eval
    metrics = ["ndcg@10", "ndcg@100", "ndcg@1000", "recall@10", "recall@100", "recall@1000", "mrr@10"]
    results = evaluate(qrels, run, metrics)

    return (
        results["mrr@10"],
        {
            '10': results["ndcg@10"],
            '100': results["ndcg@100"],
            '1000': results["ndcg@1000"]
        },
        {
            '10': results["recall@10"],
            '100': results["recall@100"],
            '1000': results["recall@1000"]
        }
    )


def scale_sparse_lambda(current_step, sparse_scaling_steps, query_lambda, doc_lambda):
    return min(query_lambda * (current_step / sparse_scaling_steps) ** 2, query_lambda), \
           min(doc_lambda * (current_step / sparse_scaling_steps) ** 2, doc_lambda)


# Precompute teacher embeddings if using distillation
def precompute_teacher_embeddings(teacher_model, queries, docs, train_qrels, batch_size=512, cache_file="teacher_embeddings_snowflake.pkl", fp16=True):
    """Precompute teacher embeddings for all queries and documents in the training set"""
    
    if os.path.exists(cache_file):
        print(f"Loading cached teacher embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
            # Move embeddings to GPU after loading
            embeddings = {
                'queries': {qid: emb.cuda() for qid, emb in embeddings['queries'].items()},
                'docs': {did: emb.cuda() for did, emb in embeddings['docs'].items()}
            }
            return embeddings
    
    print("Precomputing teacher embeddings...")
    device = next(teacher_model.parameters()).device
    
    # Get unique documents and queries from qrels
    unique_docs = set(doc_id for _, doc_id in train_qrels)
    unique_queries = set(query_id for query_id, _ in train_qrels)
    
    # Initialize dictionaries for embeddings
    query_embeddings = {}
    doc_embeddings = {}
    
    # Encode queries in batches
    query_texts = [queries[qid] for qid in unique_queries]
    query_ids = list(unique_queries)
    
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Encoding queries"):
        batch_texts = query_texts[i:i + batch_size]
        batch_ids = query_ids[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = teacher_model.encode(batch_texts, prompt_name="query", convert_to_tensor=True, device=device)
            if fp16:
                batch_embeddings = batch_embeddings.half()
            for qid, emb in zip(batch_ids, batch_embeddings):
                # Store embeddings directly on GPU
                query_embeddings[qid] = emb.cuda()
    
    # Encode documents in batches
    doc_texts = [docs[did] for did in unique_docs]
    doc_ids = list(unique_docs)
    
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding documents"):
        batch_texts = doc_texts[i:i + batch_size]
        batch_ids = doc_ids[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = teacher_model.encode(batch_texts, convert_to_tensor=True, device=device)
            if fp16:
                batch_embeddings = batch_embeddings.half()
            for did, emb in zip(batch_ids, batch_embeddings):
                # Store embeddings directly on GPU
                doc_embeddings[did] = emb.cuda()
    
    embeddings = {
        'queries': query_embeddings,
        'docs': doc_embeddings
    }
    
    # Save CPU version to disk
    print(f"Saving teacher embeddings to {cache_file}")
    cpu_embeddings = {
        'queries': {qid: emb.cpu() for qid, emb in query_embeddings.items()},
        'docs': {did: emb.cpu() for did, emb in doc_embeddings.items()}
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cpu_embeddings, f)
    
    return embeddings


# Main function to set up and run the training process
def main(eval_only=False):
    """
    Main function to set up and run the training or evaluation.
    
    Args:
        eval_only: If True, only run evaluation using a pre-trained model
    """
    # Model configuration
    hf_model_name = "microsoft/MiniLM-L12-H384-uncased"
    seq_length = 256
    d = 384  # LM embedding dimensions
    D = 4_194_304  # Sparse embedding dimensions / vocab size
    fp16 = True
    sparse = True  # Whether to use sparse retrieval (False for dense)
    freeze_bert = False
    acc_steps = 8
    num_epochs = 100
    sparse_scaling_steps = 10_000  # Gradually increase regularization strength
    query_lambda = 5e-3  # L1 regularization weight for query embeddings
    doc_lambda = 5e-3  # Regularization weight for document embeddings
    query_topk = 128  # Top-k dimensions to keep in query embeddings for retrieval
    doc_topk = 512  # Top-k dimensions to keep in document embeddings for retrieval
    evaluation_steps = 16_000
    batch_size = 32
    batch_size_inference = 64
    warmup_steps = 4_000
    negative_cache_size = 0  # Size of cache for hard negative examples (0 to disable)
    topk_sparse = False  # Whether to use top-k sparsification during training
    quick_eval = True
    doc_regularizer = "equipartition"  # Regularization type: "flops" or "equipartition" from KALE paper
    use_distillation = True  # Whether to use knowledge distillation from teacher model
    use_clustering = False  # Whether to use clustered training batches

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    print("Initializing the model...")
    model = EmbeddingRetrievalModel(hf_model_name, d, D, sparse=sparse, 
                                    fp16=fp16, seq_length=seq_length,
                                    freeze_bert=freeze_bert)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # Load and preprocess the dataset
    print("Loading and preprocessing the dataset...")
    docs, queries, train_qrels, dev_queries, dev_qrels = load_and_preprocess_dataset()

    model_name = "{}_d{}_D{}_sparse{}_fp16{}_{}_dist{}_sf_new".format(
        hf_model_name.split('/')[-1], d, D, sparse, fp16, doc_regularizer, use_distillation)

    if eval_only:
        # Load state dict and remove "module." prefix
        # MiniLM-L12-H384-uncased_d384_D131072_sparseFalse_fp16True  or  MiniLM-L12-H384-uncased_d384_D131072_sparseTrue_fp16True_equipartition_distTrue_last
        state_dict = torch.load("MiniLM-L12-H384-uncased_d384_D131072_sparseTrue_fp16True_equipartition_distTrue_sf.pth")
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     new_key = key.replace("module.", "")  # Remove the "module." prefix
        #     new_state_dict[new_key] = value

        model.load_state_dict(state_dict)
        model.eval()

        print("Model loaded!")
        
        # Perform evaluation
        print("Running evaluation...")
        mrr_10, ndcg_cut, recall = evaluate_model(model, docs, dev_queries, dev_qrels, tokenizer, device, 
                                   query_topk=query_topk, doc_topk=doc_topk, quick_mode=quick_eval)
        
        print("Final Evaluation")
        print(f"MRR@10: {mrr_10:.4f}")
        print(f"nDCG@10: {ndcg_cut['10']:.4f}, nDCG@100: {ndcg_cut['100']:.4f}, nDCG@1000: {ndcg_cut['1000']:.4f}")
        print(f"Recall@10: {recall['10']:.4f}, Recall@100: {recall['100']:.4f}, Recall@1000: {recall['1000']:.4f}")
        return
    else:
        # Snowflake/snowflake-arctic-embed-l  or  sentence-transformers/msmarco-MiniLM-L12-cos-v5
        teacher_model_name = "Snowflake/snowflake-arctic-embed-l"
        teacher_model = SentenceTransformer(teacher_model_name)

        # Precompute teacher embeddings if using distillation
        teacher_embeddings = None
        if use_distillation:
            teacher_embeddings = precompute_teacher_embeddings(
                teacher_model, queries, docs, train_qrels, 
                batch_size=batch_size_inference,
                fp16=fp16
            )

        if use_clustering:
            # Define cache file path
            cluster_cache_file = f"query_cluster_cache_{batch_size}_{teacher_model_name.split('/')[-1]}.pkl"
            
            if os.path.exists(cluster_cache_file):
                print(f"Loading cached clustering results from {cluster_cache_file}")
                with open(cluster_cache_file, 'rb') as f:
                    query_clusters = pickle.load(f)
            else:
                # Get unique queries from training qrels
                train_query_ids = set(query_id for query_id, _ in train_qrels)
                print(f"Number of unique queries in training qrels: {len(train_query_ids)}")
                
                # Get query embeddings for clustering
                print("Getting query embeddings for clustering...")
                n_queries = len(train_query_ids)
                embedding_dim = next(iter(teacher_embeddings['queries'].values())).shape[0]
                
                # Pre-allocate the numpy array with float16 dtype
                embeddings_array = np.empty((n_queries, embedding_dim), dtype=np.float16)
                query_ids = []
                
                # Fill the array directly
                for idx, query_id in enumerate(tqdm(train_query_ids)):
                    if query_id in teacher_embeddings['queries']:
                        query_ids.append(query_id)
                        embeddings_array[idx] = teacher_embeddings['queries'][query_id].cpu().numpy().astype(np.float16)
                
                # Trim any unused rows if some query_ids were missing
                embeddings_array = embeddings_array[:len(query_ids)]
                
                # Calculate number of clusters based on batch size
                n_clusters = len(embeddings_array) // batch_size
                print(f"Clustering {len(embeddings_array)} queries into {n_clusters} clusters...")
                
                # Perform clustering
                cluster_labels = balanced_kmeans_clustering(embeddings_array, k=n_clusters)
                
                # Create mapping from query_id to cluster
                query_clusters = {query_id: label for query_id, label in zip(query_ids, cluster_labels)}
                
                # Cache the results
                print(f"Saving clustering results to {cluster_cache_file}")
                with open(cluster_cache_file, 'wb') as f:
                    pickle.dump(query_clusters, f)

            # Modify ClusteredTrainDataset to use query clusters
            train_dataset = ClusteredTrainDataset(
                queries, docs, train_qrels, tokenizer,
                seq_length=seq_length,
                teacher_embeddings=teacher_embeddings,
                cluster_labels=query_clusters,
                cluster_by='query'  # New parameter to indicate clustering by queries
            )
            
            batch_sampler = ClusteredBatchSampler(train_dataset, batch_size)
            train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=0
            )
        else:
            # Original non-clustered dataset
            train_dataset = TrainDataset(
                queries, docs, train_qrels, tokenizer,
                seq_length=seq_length,
                teacher_embeddings=teacher_embeddings
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        model.float()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Train the model
        print("Training the model...")

        # Initialize wandb run
        wandb.init(project="laspar", config={
            "model_name": hf_model_name,
            "d": d,
            "D": D,
            "sparse": sparse,
            "fp16": fp16,
            "seq_length": seq_length,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "evaluation_steps": evaluation_steps,
            "freeze_bert": freeze_bert
        })

    train(model, train_dataloader, docs, dev_queries, dev_qrels, tokenizer, criterion, optimizer, 
          num_epochs, evaluation_steps, device, fp16, batch_size, query_lambda, doc_lambda, 
          batch_size_inference, acc_steps, model_name, warmup_steps, sparse_scaling_steps, 
          query_topk=query_topk, doc_topk=doc_topk, negative_cache_size=negative_cache_size,
          topk_sparse=topk_sparse, doc_regularizer=doc_regularizer, use_distillation=use_distillation)

    # Load the best model and perform final evaluation
    model.load_state_dict(torch.load(model_name + ".pth"))
    
    # Always do full evaluation at the end
    mrr_10, ndcg_cut, recall = evaluate_model(model, docs, dev_queries, dev_qrels, tokenizer, device, query_topk=query_topk, doc_topk=doc_topk, quick_mode=False)
    
    print("Final Evaluation")
    print(f"MRR@10: {mrr_10:.4f}")
    print(f"nDCG@10: {ndcg_cut['10']:.4f}, nDCG@100: {ndcg_cut['100']:.4f}, nDCG@1000: {ndcg_cut['1000']:.4f}")
    print(f"Recall@10: {recall['10']:.4f}, Recall@100: {recall['100']:.4f}, Recall@1000: {recall['1000']:.4f}")

    # Log final evaluation metrics
    wandb.log({
        "final_mrr@10": mrr_10,
        "final_ndcg@10": ndcg_cut['10'],
        "final_ndcg@100": ndcg_cut['100'],
        "final_ndcg@1000": ndcg_cut['1000'],
        "final_recall@10": recall['10'],
        "final_recall@100": recall['100'],
        "final_recall@1000": recall['1000']
    })

    # Finish wandb run
    wandb.finish()


def test():
    model_name = 'distilbert-base-uncased'
    d = 768
    D = 65536
    sparse = False
    fp16 = False
    model = EmbeddingRetrievalModel(model_name, d, D, sparse, fp16)

    input_ids = torch.randint(0, 1000, (8, 512))
    attention_mask = torch.ones((8, 512))
    output = model(input_ids, attention_mask)
    print(output.shape == (8, D))


def test2():
    # Model configuration - should match the configuration used during training
    model_name = "microsoft/MiniLM-L12-H384-uncased"
    d = 384
    D = 131_072
    sparse = True
    fp16 = True
    seq_length = 256
    
    # Initialize model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingRetrievalModel(model_name, d, D, sparse=sparse, fp16=fp16, seq_length=seq_length)
    
    # Load state dict and remove "module." prefix
    state_dict = torch.load("MiniLM-L12-H384-uncased_d384_D131072_sparseTrue_fp16True_last.pth")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example query and documents
    query = "what is machine learning?"
    
    relevant_docs = [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "Machine learning algorithms use statistical techniques to give computer systems the ability to learn from examples and improve their performance on specific tasks.",
        "In machine learning, computers analyze large amounts of data to identify patterns and make predictions, similar to how humans learn from experience."
    ]
    
    irrelevant_docs = [
        "The Great Wall of China is an ancient series of walls and fortifications located in northern China, built around 500 years ago.",
        "A solar eclipse occurs when the Moon passes between Earth and the Sun, thereby totally or partly obscuring Earth's view of the Sun.",
        "The process of photosynthesis converts light energy into chemical energy that can later be used to fuel the organism's activities."
    ]
    
    # Tokenize query
    query_inputs = tokenizer(query, return_tensors="pt", padding="max_length", 
                           truncation=True, max_length=seq_length)
    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    
    # Get query embedding
    with torch.no_grad():
        query_embedding = model(query_inputs["input_ids"], query_inputs["attention_mask"])
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        
        # Get top dimensions for query
        topk_query = 1024
        query_values, query_indices = torch.topk(query_embedding.abs(), k=topk_query, dim=1)
        query_sparse = torch.zeros_like(query_embedding)
        query_sparse.scatter_(1, query_indices, query_embedding.gather(1, query_indices))
        
        # Process all documents
        relevant_scores = []
        irrelevant_scores = []
        
        # Process relevant documents
        for i, doc in enumerate(relevant_docs):
            doc_inputs = tokenizer(doc, return_tensors="pt", padding="max_length", 
                                 truncation=True, max_length=seq_length)
            doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
            
            doc_embedding = model(doc_inputs["input_ids"], doc_inputs["attention_mask"])
            doc_embedding = torch.nn.functional.normalize(doc_embedding, p=2, dim=1)
            
            # Create sparse version
            topk_doc = 2048
            doc_values, doc_indices = torch.topk(doc_embedding.abs(), k=topk_doc, dim=1)
            doc_sparse = torch.zeros_like(doc_embedding)
            doc_sparse.scatter_(1, doc_indices, doc_embedding.gather(1, doc_indices))
            
            score = torch.matmul(query_sparse, doc_sparse.T).item()
            relevant_scores.append(score)
        
        # Process irrelevant documents
        for i, doc in enumerate(irrelevant_docs):
            doc_inputs = tokenizer(doc, return_tensors="pt", padding="max_length", 
                                 truncation=True, max_length=seq_length)
            doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
            
            doc_embedding = model(doc_inputs["input_ids"], doc_inputs["attention_mask"])
            doc_embedding = torch.nn.functional.normalize(doc_embedding, p=2, dim=1)
            
            # Create sparse version
            doc_values, doc_indices = torch.topk(doc_embedding.abs(), k=topk_doc, dim=1)
            doc_sparse = torch.zeros_like(doc_embedding)
            doc_sparse.scatter_(1, doc_indices, doc_embedding.gather(1, doc_indices))
            
            score = torch.matmul(query_sparse, doc_sparse.T).item()
            irrelevant_scores.append(score)
    
    print(f"\nQuery: {query}")
    print("\nRelevant documents and scores:")
    for doc, score in zip(relevant_docs, relevant_scores):
        print(f"\nDocument: {doc}")
        print(f"Score: {score:.4f}")
    
    print("\nIrrelevant documents and scores:")
    for doc, score in zip(irrelevant_docs, irrelevant_scores):
        print(f"\nDocument: {doc}")
        print(f"Score: {score:.4f}")
    
    return relevant_scores, irrelevant_scores


def test_inverted_index():
    print("Testing inverted index...")
    # Initialize model and tokenizer
    model_name = "microsoft/MiniLM-L12-H384-uncased"
    d = 384
    D = 131_072
    sparse = True
    fp16 = True
    seq_length = 256
    min_weight = 1e-5
    query_topk = 128
    doc_topk = 1024
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingRetrievalModel(model_name, d, D, sparse=sparse, fp16=fp16, seq_length=seq_length)
    
    # Load state dict and remove "module." prefix
    state_dict = torch.load("MiniLM-L12-H384-uncased_d384_D131072_sparseTrue_fp16True_equipartition_distTrue_last.pth")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print("Model loaded!")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create sample documents
    docs = {
        1: "Machine learning is a branch of artificial intelligence.",
        2: "Deep learning is a subset of machine learning.",
        3: "Natural language processing deals with text data.",
        4: "Computer vision focuses on image analysis.",
        5: "The weather is nice today.",  # Irrelevant document
        6: "Chris Samarinas is a software engineer that works at Naget.",  # Irrelevant document
        7: "Learning is a process of acquiring knowledge and skills through experience."  # Irrelevant document
    }

    # Generate 50_000 random documents
    print("Generating 50_000 random documents...")
    for i in range(8, 1000 + 8):  # Start from 8 to continue after existing docs
        # Generate 5-15 words per document
        num_words = random.randint(5, 15)
        words = []
        for _ in range(num_words):
            # Generate random word length between 3-6 characters
            word_length = random.randint(3, 6)
            # Generate random word using lowercase letters
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
        # Join words with spaces
        docs[i] = ' '.join(words)

    # Add document prefix to all documents
    docs = {k: "document: " + v for k, v in docs.items()}
    print(f"Total documents: {len(docs)}")

    # Create inverted index with timing
    print("Creating inverted index...")
    start_time = time.time()
    inverted_index = create_inverted_index(
        model, docs, tokenizer, device,
        batch_size=32,
        min_weight=min_weight,
        topk=doc_topk
    )
    index_creation_time = time.time() - start_time
    print(f"Index creation time: {index_creation_time:.2f} seconds")
    
    # Analyze term weights
    all_weights = []
    docs_per_term = defaultdict(int)  # Count how many docs each term appears in
    terms_per_doc = defaultdict(int)  # Count how many terms each doc has

    for term_id, (doc_ids, weights) in inverted_index.items():
        all_weights.extend(weights)
        # Count term frequency per document
        for doc_id in doc_ids:
            docs_per_term[term_id] += 1
            terms_per_doc[doc_id] += 1

    all_weights = np.array(all_weights)
    terms_per_doc_values = np.array(list(terms_per_doc.values()))

    print("\nTerm Weight Statistics:")
    print(f"Min weight: {np.min(all_weights):.8f}")
    print(f"Max weight: {np.max(all_weights):.8f}")
    print(f"Mean weight: {np.mean(all_weights):.8f}")
    print(f"Median weight: {np.median(all_weights):.8f}")
    print(f"Average terms per document: {np.mean(terms_per_doc_values):.2f}")
    print(f"Median terms per document: {np.median(terms_per_doc_values):.2f}")
    
    # Test queries
    test_queries = [
        "What is ML in AI?",
        "How does computer vision work?",
        "Tell me about deep learning"
    ]

    test_queries = [f"query: {query}" for query in test_queries]
    
    print("\nTesting retrieval...")
    query_times = []
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Get query embedding
        query_inputs = tokenizer(query, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=seq_length)
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        
        with torch.no_grad():
            # Time the entire search process
            start_time = time.time()
            
            # Get query embedding
            query_embedding = model(query_inputs["input_ids"], query_inputs["attention_mask"])
            query_embedding = query_embedding.cpu().numpy()[0]
            
            # Perform search using optimized inverted index
            search_results = search_inverted_index(
                query_embedding,
                inverted_index,
                query_topk=query_topk,
                min_weight=min_weight
            )
            
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            print(f"Search time: {query_time:.4f} seconds")
            print("\nTop 5 retrieved documents:")
            for doc_id, score in search_results[:5]:
                print(f"- {doc_id} (score: {score:.8f}): {docs[doc_id]}")
    
    # Calculate and print statistics
    print("\nSearch Performance Statistics:")
    print(f"Average search time: {np.mean(query_times):.4f} seconds")
    print(f"Min search time: {np.min(query_times):.4f} seconds")
    print(f"Max search time: {np.max(query_times):.4f} seconds")
    
    # Index statistics
    print("\nInverted Index Statistics:")
    num_terms = len(inverted_index)
    total_postings = sum(len(postings[0]) for postings in inverted_index.values())
    avg_postings_per_term = total_postings / num_terms if num_terms > 0 else 0
    
    print(f"Number of terms in index: {num_terms}")
    print(f"Total number of postings: {total_postings}")
    print(f"Average postings per term: {avg_postings_per_term:.2f}")
    
    # Memory usage statistics
    def get_size(obj, seen=None):
        """Recursively calculate size of object in bytes"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum(get_size(v, seen) for v in obj.values())
            size += sum(get_size(k, seen) for k in obj.keys())
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum(get_size(i, seen) for i in obj)
        return size
    
    index_size_bytes = get_size(inverted_index)
    print(f"Index size in memory: {index_size_bytes / (1024*1024):.2f} MB")
    
    return inverted_index


if __name__ == "__main__":
    main()  # For training
    #main(eval_only=True)  # For evaluation only
    #test()
    #test2()
    #test_inverted_index()
