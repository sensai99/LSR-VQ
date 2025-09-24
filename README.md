# LSR-VQ

![Learned Sparse Retrieval with Vector Quantization](LSR_VQ.png)

Learned Sparse Retrieval with Vector Quantization

- Dataset: [MSMARCO](https://microsoft.github.io/msmarco/Datasets#passage-ranking-dataset) passage ranking dataset in full ranking setting.
- Access the entire folder setup [here](https://drive.google.com/drive/u/3/folders/1hFx3EKA1jqvvYu-qRvFgn19ju87jDkk4)
- Currently data is hosted on [Google Drive](https://drive.google.com/drive/folders/1LZxxAqjZJ8gpcAgM9XYGZ56MiydTQzsm?usp=drive_link)


This repository contains code for performing sparse retrieval using vector quantization techniques. It leverages transformer-based embeddings, quantizes and projects them into sparse codes, and evaluates retrieval performance on datasets using metrics like MRR, NDCG, and Recall.

## Structure

- code/Model.ipynb : Implementation of **Method A**: Symbolic Retrieval with Vector Quantization. Discretizes chunked dense embeddings into latent tokens and performs retrieval using symbolic indices such as BM25 over learned codes.
- code/Model_training.ipynb : Implementation and training of **Method B** : Sparse Projection with Vector Quantization (LSR VQ). This model projects quantized embeddings into high-dimensional sparse vectors using a learned transformation, enabling end-to-end training and more expressive scoring via sparse dot products.
- The code directory also contains major utility functions for handling embedding generation and storage, as well as utility classes for vector quantization, index construction, baselines and metric computation.
- **Dependencies**: Transformers, Torch, FAISS, ir_datasets, ranx.

## Dependencies

Install the required libraries using pip:

```bash
pip install torch ir_datasets wandb numpy scikit-learn sentence-transformers transformers tqdm scipy matplotlib rank-eval ranx
pip install faiss-cpu
```

## Evaluation

Retrieval performance is evaluated using:

- MRR@10
- NDCG@{10,100,1000}
- Recall@{10,100,1000}


