from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

class InvertedIndexHandler:
    def __init__(self, embedding_processor, vocab_size = None):
        self.embedding_processor = embedding_processor
        self.train_mapppings = self.embedding_processor.load_or_save_embeddings(mode = 'train')['mappings']
        self.dev_mappings = self.embedding_processor.load_or_save_embeddings(mode = 'dev')['mappings']
        self.passage_ids = None
        self.vocab_size = vocab_size
        self.optimized_index = None
        return

    # Build the index
    def create_inverted_index(self, code_indices, mode = 'dev', index_type = 'csr'):
        self.passage_ids = self.dev_mappings['passage_ids']
        if mode == 'train':
            self.passage_ids = self.train_mapppings['passage_ids']
        elif mode != 'dev':
            raise NotImplementedError(f"Inverted index for {mode} not implemented!")

        assert code_indices.shape[0] == len(self.passage_ids), f"Mismatch: {code_indices.shape[0]} passage code indices vs {len(self.passage_ids)} passage IDs"

        if index_type == 'csr':  
          if self.vocab_size == None:
            raise RuntimeError(f"vocab_size is {self.vocab_size}, can't build csr index!")

          rows, cols, data = [], [], []

          for i, codes in enumerate(tqdm(code_indices, desc = "Building passage CSR matrix...")):
              weights = Counter(codes.tolist())
              for code, freq in weights.items():
                  rows.append(i)
                  cols.append(code)
                  data.append(freq)

          # vocab_size = len(np.unique(code)) - I think we shouldn't do this
          self.optimized_index = csr_matrix((data, (rows, cols)), shape = (len(code_indices), self.vocab_size), dtype = np.float32)
        
        elif index_type == 'inverted_index':      
          inverted_index = defaultdict(list)
          num_passages = len(self.passage_ids)
          for i in tqdm(range(0, num_passages), desc = "Building inverted index..."):
              code_index_list = code_indices[i].tolist()
              weights = Counter(code_index_list)
              for code_index in list(set(code_index_list)):
                  inverted_index[int(code_index)].append((self.passage_ids[i], float(weights[code_index])))  # Ensure integer keys

          # Sort postings lists by weight for each term
          for idx in inverted_index:
              inverted_index[idx] = sorted(inverted_index[idx], key = lambda x: abs(x[1]), reverse=True)

          # Convert to more efficient data structure
          self.optimized_index = {
              idx: (
                  np.array([passage_id for passage_id, _ in postings], dtype = np.int32),
                  np.array([weight for _, weight in postings], dtype = np.float32)
              )
              for idx, postings in inverted_index.items()
          }

        return self.optimized_index

    # Computes the results given a query code index list
    # Uses the inverted index
    def compute_results_inv_index(self, code_index_list):
      scores = defaultdict(float)
      seen_passages = set()

      # Process each query term
      weights = Counter(code_index_list)
      for code_index in list(set(code_index_list)):
          if code_index not in self.optimized_index:
              continue

          passage_ids, passage_weights = self.optimized_index[code_index]
          query_weight = weights[code_index]

          # Compute scores against all the documents this code_index is associated with
          for passage_id, passage_weight in zip(passage_ids, passage_weights):
              scores[passage_id] += query_weight * passage_weight
              seen_passages.add(passage_id)

      # Use numpy for final scoring
      if seen_passages:
          passage_ids = np.array(list(seen_passages))
          passage_scores = np.array([scores[passage_id] for passage_id in passage_ids])

          # Get top 1000 results efficiently
          top_k = min(1000, len(passage_scores))
          top_indices = np.argpartition(passage_scores, -top_k)[-top_k:]
          top_indices = top_indices[np.argsort(-passage_scores[top_indices])]

          return [(passage_ids[i], passage_scores[i]) for i in top_indices]

      print("Found no matching code indices b/w query and passage!!!")
      return []

    # Optimized search function for sparse retrieval using the inverted index
    def search_inverted_index(self, query_code_repr, query_ids, index_type = 'csr', batch_size = 512):
        num_queries = query_code_repr.shape[0]
        assert num_queries == len(query_ids), f"Mismatch: {query_code_repr.shape[0]} query code indices vs {len(query_ids)} query IDs"

        all_results = {}
        if index_type == 'csr':
          for start in tqdm(range(0, num_queries, batch_size), desc = "Evaluating queries (csr)..."):
              end = min(start + batch_size, num_queries)
              query_chunk = query_code_repr[start:end]
              scores_chunk = query_chunk @ self.optimized_index.T  # shape: [batch_size, num_passages]

              topk_idx = np.argpartition(-scores_chunk, 1000, axis = 1)[:, :1000]
              for i in range(scores_chunk.shape[0]):
                  passage_scores = scores_chunk[i]
                  top_i = topk_idx[i]
                  top_indices = top_i[np.argsort(-passage_scores[top_i])]
                  
                  # Ensure self.passage_ids is referring to the correct mode (train/dev)
                  all_results[query_ids[start + i]] = [(self.passage_ids[t], passage_scores[t]) for t in top_indices]

        elif index_type == 'inverted_index':
          for i in tqdm(range(0, num_queries), desc = "Evaluating queries (inverted index)..."):
              code_index_list = query_code_repr[i].tolist()
              search_results = self.compute_results_inv_index(code_index_list)
              all_results[query_ids[i]] = search_results

        return all_results