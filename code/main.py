import torch
from transformers import AutoTokenizer, AutoModel

from dataset import DataProcessor
from embeddings import EmbeddingProcessor
from vector_quantizer_handler import VQHandler
from vector_quantizer import Quantize
from inverted_index import InvertedIndexHandler
from metrics import MetricsGenerator

'''
    - Load the raw dataset
'''
data_processor = DataProcessor(data_root_dir = '../data')
data = data_processor.get_data()
passages, queries_train, queries_dev, qrels_train, qrels_dev = data['passages'], data['queries_train'], data['queries_dev'], data['qrels_train'], data['qrels_dev']
data_processor.print_samples()

'''
    - Initialize the model(Contriever)
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device', device)

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(device)

'''
    - Load/Save the embeddings
'''
embedding_processor = EmbeddingProcessor(data_processor = data_processor, model = model, tokenizer = tokenizer, emb_root_dir = '../embeddings', batch_size = 128, device = device)

obj_train = embedding_processor.load_or_save_embeddings(mode = 'train')
obj_dev = embedding_processor.load_or_save_embeddings(mode = 'dev')

print('Train passage stats: ')
print('-> Number of passage embeddings: ', obj_train['embeddings']['passage_embeddings'].shape)
print('-> Number of passage ids: ', len(obj_train['mappings']['passage_ids']))
print('-> Number of query embeddings: ', obj_train['embeddings']['query_embeddings'].shape)
print('-> Number of query ids: ', len(obj_train['mappings']['query_ids']))

print('Dev passage stats: ')
print('-> Number of passage embeddings: ', obj_dev['embeddings']['passage_embeddings'].shape)
print('-> Number of passage ids: ', len(obj_dev['mappings']['passage_ids']))
print('-> Number of query embeddings: ', obj_dev['embeddings']['query_embeddings'].shape)
print('-> Number of query ids: ', len(obj_dev['mappings']['query_ids']))

'''
    - Train the Vector Quantizer
    - Vector quantize the embeddings & Get the code indices
'''
vocab_size = 6000
quantizer = Quantize(codebook_vector_dim = 48, num_clusters = vocab_size).to(device = device)
vq_hanlder = VQHandler(embedding_processor = embedding_processor, quantizer = quantizer, num_chunks = 16, batch_size = 2048, device = device)
quantizer = vq_hanlder.train(mode = 'train')

# Get the code indices for each passage
# These code indices are used to build the inverted index
# If mode is train, we would use the train passages for building the index (In that case, while evaluating using dev queries, train passages should also include the dev passages in it)
passage_code_indices = vq_hanlder.inference(type = 'passage', mode = 'dev').cpu().numpy()
print('\nPassage code indices shape (will be used for building the inverted index)', passage_code_indices.shape)

'''
    - Create the inverted index with the vector quantized indices
'''
inverted_index_handler = InvertedIndexHandler(embedding_processor = embedding_processor, vocab_size = vocab_size)

# Code indices of the passage set that you are working on
# Changes based on the mode (dev/train)
obj = inverted_index_handler.create_inverted_index(code_indices = passage_code_indices, mode = 'dev', index_type = 'csr')

'''
    - Generate Metrics using the inverted index built
'''
# Pass qrels to calculate the metrics
metrics_generator = MetricsGenerator(vq_handler = vq_hanlder, inverted_index_handler = inverted_index_handler, embedding_processor = embedding_processor, dev_qrels = qrels_dev, vocab_size = vocab_size)

# Get results for the VQ approach
results_vq = metrics_generator.get_metrics(method = 'vq', index_type = 'csr')
print('Results VQ', results_vq)

# Get results for the FAISS approach
results_faiss = metrics_generator.get_metrics(method = 'faiss', mode = 'dev')
print('Results FAISS', results_faiss)