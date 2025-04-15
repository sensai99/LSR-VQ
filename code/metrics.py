from tqdm import tqdm
from ranx import Qrels, Run, evaluate

class MetricsGenerator:
    def __init__(self, inverted_index, embedding_processor, qrels):
        self.inverted_index = inverted_index
        
        self.embedding_processor = embedding_processor
        train_mapppings = self.embedding_processor.load_or_save_embeddings(mode = 'train')['mappings']
        dev_mappings = self.embedding_processor.load_or_save_embeddings(mode = 'dev')['mappings']
        
        self.train_query_ids = train_mapppings['query_ids']
        self.dev_query_ids = dev_mappings['query_ids']
        
        self.train_qrels = qrels['train']
        self.dev_qrels = qrels['dev']
        return

    def get_metrics(self, code_indices, mode = 'dev'):
        query_ids = self.dev_query_ids
        q_rels = self.dev_qrels
        if mode == 'train':
            query_ids = self.train_query_ids
            qrels = self.train_qrels
        else:
            raise NotImplementedError(f"Metrics calculator not implemented for {mode}!")
        
        all_results = {}
        num_queries = len(query_ids)
        for i in tqdm(range(0, num_queries), desc = "Evaluating queries"):
            code_index_list = code_indices[i].tolist()
            search_results = self.inverted_index.search_inverted_index(code_index_list)
            all_results[query_ids[i]] = search_results

        # Create rank_eval Run and Qrels objects
        run_dict = {}
        for qid, results in all_results.items():
            run_dict[qid] = {
                str(passage_id): float(score)
                for passage_id, score in results
            }
        run = Run(run_dict)

        qrels_dict = {
            qid: {str(passage_id): 1 for passage_id in q_rels[qid]}
            for qid in q_rels
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