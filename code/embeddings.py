import os
import json
from tqdm import tqdm
import torch

class EmbeddingProcessor:

    def __init__(self, data_processor, model, tokenizer, emb_root_dir, batch_size = 128, device = 'cpu') -> None:
        self.data_processor = data_processor
        self.model = model
        self.tokenizer = tokenizer
        self.emb_root_dir = emb_root_dir
        self.device = device

        self.embeddings_path = {
            'train': {
                'passage_embs': os.path.join(self.emb_root_dir, 'train', 'passage_embeddings.pt'),
                'query_embs': os.path.join(self.emb_root_dir, 'train', 'query_embeddings.pt'),
                'passage_ids': os.path.join(self.emb_root_dir, 'train', 'passage_ids.json'),
                'query_ids': os.path.join(self.emb_root_dir, 'train', 'query_ids.json')
            },
            'dev': {
                'passage_embs': os.path.join(self.emb_root_dir, 'dev', 'passage_embeddings.pt'),
                'query_embs': os.path.join(self.emb_root_dir, 'dev', 'query_embeddings.pt'),
                'passage_ids': os.path.join(self.emb_root_dir, 'dev', 'passage_ids.json'),
                'query_ids': os.path.join(self.emb_root_dir, 'dev', 'query_ids.json')
            }
        }

        self.batch_size = batch_size
        return

    # Used to combine the embeddings of all the tokens
    # Contriever model
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim = 1) / mask.sum(dim = 1)[..., None]
        return sentence_embeddings
    
    def filter_data(self, passages, queries, qrels):
        relevant_passage_ids = set()
        for qid in qrels:
            relevant_passage_ids.update(qrels[qid])
        passages = {passage_id: passages[passage_id] for passage_id in relevant_passage_ids}
        passage_ids = list(passages.keys())

        qrels = {qid: qrels[qid] for qid in queries if qid in qrels}
        
        qrels_ = {}
        queries_ = {}
        for qid in queries:
            if qid in qrels:
                qrels_[qid] = qrels[qid]
                queries_[qid] = queries[qid]

        query_ids = list(qrels_.keys())

        return {'passages': passages, 'queries': queries_, 'qrels': qrels_, 'passage_ids': passage_ids, 'query_ids': query_ids}

    def get_filtered_data(self, mode = 'train'):
        if mode not in ['train', 'dev']:
            raise ValueError('Invalid mode!')
        
        if self.passages == None:
            # If they are not loaded yet, load them
            raw_data = self.data_processor.get_data()
            self.passages, self.queries_train, self.queries_dev, self.qrels_train, self.qrels_dev = raw_data['passages'], raw_data['queries_train'], raw_data['queries_dev'], raw_data['qrels_train'], raw_data['qrels_dev']
        
        filtered_data = {
            'passage': None,
            'query': None
        }
        if mode == 'train':
            data = self.filter_data(passages = self.passages, queries = self.queries_train, qrels = self.qrels_train)
            filtered_data['passage'] = {
                'passages': data['passages'],
                'passage_ids': data['passage_ids']
            }
            filtered_data['query'] = {
                'queries': data['queries'],
                'qrels': data['qrels'],
                'query_ids': data['query_ids']
            }
        
        elif mode == 'dev':
            data = self.filter_data(passages = self.passages, queries = self.queries_dev, qrels = self.qrels_dev)

            # In case of development set, passages would be the entire collection (instead of the filtered ids using qrels)
            filtered_data['passage'] = {
                'passages': self.passages,
                'passage_ids': list(self.passages.keys())
            }
            filtered_data['query'] = {
                'queries': data['queries'],
                'qrels': data['qrels'],
                'query_ids': data['query_ids']
            }
        
        return filtered_data
            
    def compute_embeddings(self, type = 'passage', mode = 'train'):
        if type == None:
            print('Embedding type not provided, Not computing embeddings!')
            return
        
        if type not in ['passage', 'query']:
            raise ValueError('Invalid embedding type!')
        
        data = self.get_filtered_data(mode)[type]
        if type == 'query':
            data = data['queries']
        elif type == 'passage':
            data = data['passages']
        
        data_embeddings = []
        
        ids = data.keys()
        for i in tqdm(range(0, len(ids), self.batch_size), desc = f"Encoding {type}"):
            batch_data = [data[id] for id in ids[i:i + self.batch_size]]

            # Pad till the model's configured max_len (512)
            batch_inputs = self.tokenizer(batch_data, padding = True, truncation = True, return_tensors = 'pt')
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                outputs = self.model(batch_inputs["input_ids"], batch_inputs["attention_mask"])
                batch_embeddings = self.mean_pooling(outputs[0], batch_inputs['attention_mask'])
                data_embeddings.append(batch_embeddings)

        data_embeddings = torch.cat(data_embeddings, dim = 0)
    
        return data_embeddings, list(ids)
    
    def load_or_save_passage_embeddings(self, mode = 'train'):
        pass_embs_path = self.embeddings_path[mode]['passage_embs']
        pass_ids_path = self.embeddings_path[mode]['passage_ids']
        
        if os.path.exists(pass_embs_path) and os.path.exists(self.pass_ids_path):
            print("Loading cached passage embeddings & ids...")
            passage_embeddings = torch.load(pass_embs_path).to(device = self.device)
            self.emb_dim = passage_embeddings.shape[-1]

            with open(pass_ids_path, "r") as f:
                passage_ids = json.load(f)

            return {
                'embeddings':
                    {
                        'passage_embeddings': passage_embeddings
                    },
                'mappings': {
                        'passage_ids': passage_ids
                    }
                }
        
        passage_embeddings, passage_ids = self.compute_embeddings(type = 'passage', mode = mode)
        self.emb_dim = passage_embeddings.shape[-1]

        # Save embeddings to the appropriate path
        torch.save(passage_embeddings, pass_embs_path)

        # Save ID mappings
        with open(pass_ids_path, "w") as f:
            json.dump(passage_ids, f)

        print("Embeddings & Mappings saved.")

        return {
            'embeddings':
                {
                    'passage_embeddings': passage_embeddings
                },
            'mappings': {
                    'passage_ids': passage_ids
                }
            }
        
    
    def load_or_save_query_embeddings(self, mode = 'train'):
        query_embs_path = self.embeddings_path[mode]['query_embs']
        query_ids_path = self.embeddings_path[mode]['query_ids']
        
        if os.path.exists(query_embs_path) and os.path.exists(self.query_ids_path):
            print("Loading cached query embeddings & ids...")
            query_embeddings = torch.load(query_embs_path).to(device = self.device)
            self.emb_dim = query_embeddings.shape[-1]

            with open(query_ids_path, "r") as f:
                query_ids = json.load(f)

            return {
                'embeddings':
                    {
                        'query_embeddings': query_embeddings
                    },
                'mappings': {
                        'query_ids': query_ids
                    }
                }
        
        query_embeddings, query_ids = self.compute_embeddings(type = 'query', mode = mode)
        self.emb_dim = query_embeddings.shape[-1]

        # Save embeddings to the appropriate path
        torch.save(query_embeddings, query_embs_path)

        with open(query_ids_path, "w") as f:
            json.dump(query_ids, f)

        print("Embeddings & Mappings saved.")

        return {
            'embeddings':
                {
                    'query_embeddings': query_embeddings
                },
            'mappings': {
                    'query_ids': query_ids
                }
            }
    
    def load_or_save_embeddings(self, mode = 'train'):
        pass_embs_path = self.embeddings_path[mode]['passage_embs']
        query_embs_path = self.embeddings_path[mode]['query_embs']
        pass_ids_path = self.embeddings_path[mode]['passage_ids']
        query_ids_path = self.embeddings_path[mode]['query_ids']
        
        if os.path.exists(pass_embs_path) and os.path.exists(query_embs_path) and os.path.exists(self.pass_ids_path) and os.path.exists(self.query_ids_path):
            print("Loading cached embeddings & ids...")
            passage_embeddings = torch.load(pass_embs_path).to(device = self.device)
            query_embeddings = torch.load(query_embs_path).to(device = self.device)
            self.emb_dim = passage_embeddings.shape[-1]

            with open(pass_ids_path, "r") as f:
                passage_ids = json.load(f)

            with open(query_ids_path, "r") as f:
                query_ids = json.load(f)

            return {
                'embeddings':
                    {
                        'passage_embeddings': passage_embeddings,
                        'query_embeddings': query_embeddings
                    },
                'mappings': {
                        'passage_ids': passage_ids,
                        'query_ids': query_ids
                    }
                }
        
        passage_embeddings, passage_ids = self.compute_embeddings(type = 'passage', mode = mode)
        query_embeddings, query_ids = self.compute_embeddings(type = 'query', mode = mode)
        self.emb_dim = passage_embeddings.shape[-1]

        # Save embeddings to the appropriate path
        torch.save(passage_embeddings, pass_embs_path)
        torch.save(query_embeddings, query_embs_path)

        # Save ID mappings
        with open(pass_ids_path, "w") as f:
            json.dump(passage_ids, f)

        with open(query_ids_path, "w") as f:
            json.dump(query_ids, f)

        print("Embeddings & Mappings saved.")

        return {
            'embeddings':
                {
                    'passage_embeddings': passage_embeddings,
                    'query_embeddings': query_embeddings
                },
            'mappings': {
                    'passage_ids': passage_ids,
                    'query_ids': query_ids
                }
            }
    
    def get_emd_dim(self):
        if self.emb_dim is None:
            raise ValueError('Embedding dimension not found!')
        
        return self.emb_dim
        
