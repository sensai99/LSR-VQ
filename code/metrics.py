from tqdm import tqdm
from ranx import Qrels, Run, evaluate
from collections import Counter
import numpy as np
import faiss

class MetricsGenerator:
    def __init__(self, vq_handler, inverted_index_handler, embedding_processor, dev_qrels, vocab_size = None):
        self.vq_handler = vq_handler
        self.inverted_index_handler = inverted_index_handler
        self.embedding_processor = embedding_processor

        self.train_data = self.embedding_processor.load_or_save_embeddings(mode = 'train')
        self.dev_data = self.embedding_processor.load_or_save_embeddings(mode = 'dev')
        
        self.train_query_ids = self.train_data['mappings']['query_ids']
        self.dev_query_ids = self.dev_data['mappings']['query_ids']
        
        self.dev_qrels = dev_qrels
        
        self.query_matrix = None
        self.vocab_size = vocab_size
        return

    def build_query_matrix(self, query_code_indices):
      if self.vocab_size == None:
        raise RuntimeError(f"vocab_size is {self.vocab_size}, can't build query csr matrix!")
      
      self.query_matrix = np.zeros((len(query_code_indices), self.vocab_size), dtype = np.float32)

      for i, codes in enumerate(query_code_indices):
          weights = Counter(codes)
          for code, freq in weights.items():
              self.query_matrix[i, code] = freq

      return self.query_matrix

    def batch_score_queries(self, top_k = 1000):
      scores_matrix = self.query_matrix @ self.inverted_index.T  # shape: [num_queries, num_passages]
      top_k_indices = np.argpartition(-scores_matrix, top_k, axis=1)[:, :top_k]

      results = {}
      for i in range(scores_matrix.shape[0]):
          row = scores_matrix[i]
          top_idx = top_k_indices[i]
          sorted_idx = top_idx[np.argsort(-row[top_idx])]
          results.append(list(zip(sorted_idx, row[sorted_idx])))

      return results

    # Performs faiss (always on the development set queries)
    def perform_faiss(self, mode = 'dev'):
        query_embeddings = self.dev_data['embeddings']['query_embeddings']
        
        if mode == 'dev':
          passage_ids = self.dev_data['mappings']['passage_ids']
          passage_embeddings = self.dev_data['embeddings']['passage_embeddings']
        elif mode == 'train':
          passage_ids = self.train_data['mappings']['passage_ids']
          passage_embeddings = self.train_data['embeddings']['passage_embeddings']

        # Dense retrieval using FAISS
        print("Building FAISS index...")

        # Build FAISS index
        dimension = passage_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(passage_embeddings.cpu().numpy())
  
        all_results = {}
        # Search using FAISS for each query
        for i, qid in enumerate(tqdm(self.dev_query_ids, desc = "Evaluating queries (FAISS)...", total = len(self.dev_query_ids))):
            if qid not in self.dev_qrels:
                continue

            # Get top 1000 results for this query
            scores, indices = index.search(query_embeddings[i:i + 1], 1000)
            search_results = [(passage_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
            all_results[qid] = search_results

        return all_results

    def get_metrics(self, method = 'vq', mode = 'dev', index_type = 'csr'):
        # mode param - Determines which passages to be used for faiss (dev or train)
        
        assert len(self.dev_qrels) == len(self.dev_query_ids), f"Mismatch: {len(self.dev_qrels)} development qrels vs {len(self.dev_query_ids)} dev query ids"

        if method == 'faiss':
            all_results = self.perform_faiss(mode = mode)
        elif method == 'vq':
            query_code_indices = self.vq_handler.inference(type = 'query', mode = 'dev')

            if index_type == 'csr':
              self.build_query_matrix(query_code_indices)
              all_results = self.inverted_index_handler.search_inverted_index(self.query_matrix, self.dev_query_ids, index_type = 'csr', batch_size = 128)
            else:
              all_results = self.inverted_index_handler.search_inverted_index(query_code_indices, self.dev_query_ids, index_type = 'inverted_index')

        # Create rank_eval Run and Qrels objects
        run_dict = {}
        for qid, results in all_results.items():
            run_dict[qid] = {
                str(passage_id): float(score)
                for passage_id, score in results
            }
        run = Run(run_dict)

        qrels_dict = {
            qid: {str(passage_id): 1 for passage_id in self.dev_qrels[qid]}
            for qid in self.dev_qrels
        }
        qrels = Qrels(qrels_dict)

        # Evaluate using rankx
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