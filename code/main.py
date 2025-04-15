import torch
from transformers import AutoTokenizer, AutoModel

from dataset import DataProcessor
from embeddings import EmbeddingProcessor

data_processor = DataProcessor(data_root_dir = 'data')
data = data_processor.get_data()
passages, queries_train, queries_dev, qrels_train, qrels_dev = data['passages'], data['queries_train'], data['queries_dev'], data['qrels_train'], data['qrels_dev']

data_processor.print_samples()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device', device)

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(device)

emebdding_processor = EmbeddingProcessor(data_processor = data_processor, model = model, tokenizer = tokenizer, emb_root_dir = 'embeddings', batch_size = 128, device = device)
emebdding_processor.load_or_save_embeddings(mode = 'train')
emebdding_processor.load_or_save_embeddings(mode = 'dev')