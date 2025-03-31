# LASPAR - Neural LAtent SPArse Retrieval

LASPAR learns sparse, high-dimensional representations of queries and documents that enable efficient retrieval using an inverted index while maintaining competitive retrieval quality. This implementation supports training and evaluation on the MS MARCO passage ranking dataset.

## Key Features
* Sparse High-Dimensional Embeddings: Projects text from language model embeddings to a much higher dimensional sparse space
* Inverted Index Retrieval: Efficient retrieval using a custom inverted index implementation
* Regularization Techniques: Two types of regularization to enforce sparsity:
  * Equipartition regularization (from [KALE](https://dl.acm.org/doi/10.1145/3578337.3605131) paper)
  * FLOPS-based regularization (from [SPLADE](https://github.com/naver/splade) paper)
* Knowledge Distillation: Option to distill from a dense teacher model to improve retrieval quality
* Model Architecture Variants:
  * Original model: Token-level upscaling followed by addition
  * Enhanced model: Token attention mechanism before upscaling
 
## TODO
- [ ] Efficient inverted index implementation in Rust
- [ ] Add support for knowledge distillation from cross-encoder
 
## Key configuration options:
* hf_model_name: Base language model (default: "microsoft/MiniLM-L12-H384-uncased")
* d: Base LM embedding dimension (default: 384)
* D: Sparse embedding dimension / vocabulary size (default: 4M)
* sparse: Whether to use sparse retrieval (default: True)
* doc_regularizer: Type of regularization ("flops" or "equipartition")
* use_distillation: Whether to use knowledge distillation (default: True)
* query_lambda: L1 regularization weight for query embeddings
* doc_lambda: Regularization weight for document embeddings

## Model Architecture
LASPAR supports two types of sparse retrieval models:
1) **Token-level upscaling followed by addition** (EmbeddingRetrievalModel_Old):
  * Projects each token embedding to higher dimension first
  * Sums the projected embeddings across tokens
  * Applies log-ReLU transformation for sparsity
2) **Token attention followed by upscaling** (EmbeddingRetrievalModel):
  * Applies learned attention weights to aggregate token embeddings
  * Projects the aggregated representation to higher dimension
  * Applies log-ReLU transformation with a bias term for sparsity

## Retrieval Process
LASPAR uses an inverted index for efficient retrieval:
* During indexing, document embeddings are computed and stored in a sparse inverted index
* For retrieval, query embeddings are computed and matched against the index
* Only the top-k dimensions with highest weights are considered for both queries and documents
* Relevance scores are computed efficiently using the inverted index structure
