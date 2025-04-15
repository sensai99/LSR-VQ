import torch
from tqdm import tqdm

from utils import split_embedding_into_chunks
from vector_quantizer import Quantize

class VQHandler:
    def __init__(self, embedding_processor, quantizer = None, emb_dim = None, num_clusters = None, num_chunks = 32, batch_size = 128, device = 'cpu'):
        
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            self.quantizer = Quantize(dim = emb_dim, num_clusters = num_clusters)
        self.quantizer.training = True

        self.embedding_processor = embedding_processor
        train_embeddings = self.embedding_processor.load_or_save_embeddings(mode = 'train')['embeddings']
        dev_embeddings = self.embedding_processor.load_or_save_embeddings(mode = 'dev')['embeddings']
        
        self.train_query_embeddings = train_embeddings['query_embeddings'] 
        self.train_passage_embeddings = train_embeddings['passage_embeddings']
        self.dev_query_embeddings = dev_embeddings['query_embeddings']
        self.dev_passage_embeddings = dev_embeddings['passage_embeddings']
        
        # Number of chunks each emb to be divided into
        self.num_chunks = num_chunks
        self.device = device
        self.batch_size = batch_size

    def train(self):
        train_query_chunked_embs = split_embedding_into_chunks(self.train_query_embeddings, self.num_chunks)
        train_pass_chunked_embs = split_embedding_into_chunks(self.train_passage_embeddings, self.num_chunks)

        embeddings = torch.cat((train_query_chunked_embs, train_pass_chunked_embs), dim = 0)

        for i in tqdm(range(0, embeddings.shape[0], self.batch_size), desc = "Training codebook vectors"):
            batch_embs = embeddings[i:i + self.batch_size]
            _, code = self.quantizer(batch_embs)

        self.quantizer.training = False
        return self.quantizer
    
    def inference(self, type = 'passages', mode = 'dev'):
        
        embeddings = None
        if mode == 'dev':
            if type == 'passage':
                embeddings = self.dev_passage_embeddings
            elif type == 'query':
                embeddings = self.dev_query_embeddings
        
        if mode == 'train':
            if type == 'passage':
                embeddings = self.train_passage_embeddings
            elif type == 'query':
                embeddings = self.train_query_embeddings

        # Get the code book vectors for each passage in the devlopment set
        self.quantizer.training = False
        code_indices = []
        for i in tqdm(range(0, embeddings.shape[0], self.batch_size), desc = "Vector quantizing..."):
            batch_embs = embeddings[i:i + self.batch_size]
            batch_chunked_embs = split_embedding_into_chunks(batch_embs, self.num_chunks)
            _, code = self.quantizer(batch_chunked_embs)
            code = code.view(-1, self.num_chunks)
            code_indices.append(code)

        code_indices = torch.cat(code_indices, dim = 0)

        return code_indices


