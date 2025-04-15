from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

class InvertedIndexHandler:
    def __init__(self, embedding_processor):
        self.embedding_processor = embedding_processor
        train_mapppings = self.embedding_processor.load_or_save_embeddings(mode = 'train')['mappings']
        dev_mappings = self.embedding_processor.load_or_save_embeddings(mode = 'dev')['mappings']
        
        self.train_passage_ids = train_mapppings['passage_ids']
        self.dev_passage_ids = dev_mappings['passage_ids']
        self.optimized_index = None
        return
    
    def create_inverted_index(self, code_indices, mode = 'dev'):
        passage_ids = self.dev_passage_ids
        if mode == 'train':
            passage_ids = self.train_passage_ids
        else:
            raise NotImplementedError(f"Inverted index for {mode} not implemented!")

        inverted_index = defaultdict(list)
        num_passages = len(passage_ids)

        for i in tqdm(range(0, num_passages), desc = "Building inverted index"):
            code_index_list = code_indices[i].tolist()
            weights = Counter(code_index_list)
            for code_index in list(set(code_index_list)):
                inverted_index[int(code_index)].append((passage_ids[i], float(weights[code_index])))  # Ensure integer keys

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

    # Optimized search function for sparse retrieval using the inverted index
    def search_inverted_index(self, query_code_index_list, query_topk = 128):
        scores = defaultdict(float)
        seen_passages = set()

        # Process each query term
        weights = Counter(query_code_index_list)
        query_code_index_list = list(set(query_code_index_list))
        for code_index in query_code_index_list:
            if code_index not in self.optimized_index:
                # print('Unexpected!!!')
                continue

            passage_ids, passage_weights = self.optimized_index[code_index]
            query_weight = weights[code_index]

            # Only process top documents per term
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

        return []