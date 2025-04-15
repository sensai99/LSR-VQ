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
data_processor = DataProcessor(data_root_dir = 'data')
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
embedding_processor = EmbeddingProcessor(data_processor = data_processor, model = model, tokenizer = tokenizer, emb_root_dir = 'embeddings', batch_size = 128, device = device)
emd_dim = embedding_processor.get_emd_dim()

'''
    - Train the Vector Quantizer
    - Vector quantize the embeddings & Get the code indices
'''
quantizer = Quantize(dim = emd_dim, num_clusters = 256)
vq_hanlder = VQHandler(embedding_processor = embedding_processor, quantizer = quantizer, num_chunks = 32, device = device)
quantizer = vq_hanlder.train()

# Get the code indices for each passage in development set
# These code indices are used to build the inverted index
code_indices = vq_hanlder.inference(type = 'passage', mode = 'dev')


'''
    - Create the inverted index with the vector quantized indices
'''
inverted_index_handler = InvertedIndexHandler(embedding_processor = embedding_processor)

# Code indices of the passage set that you are working on
# Changes based on the mode (dev/train)
inverted_index_handler.create_inverted_index(code_indices = code_indices, mode = 'dev')


'''
    - Generate Metrics using the inverted index built
'''

# Pass qrels to calculate the metrics
qrels = {
    'train': qrels_train,
    'dev': qrels_dev
}

# Get the code indices for each query in the development set
# These code indices are used to calculate the scores and metrics
code_indices = vq_hanlder.inference(type = 'query', mode = 'dev')

metrics_generator = MetricsGenerator(inverted_index = inverted_index_handler, embedding_processor = embedding_processor, qrels = qrels)
metrics_generator.get_metrics()
