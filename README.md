# WaExt: Weight-aware Extension & Tasks for Evaluating Knowledge Graph Embeddings

WaExt is a toolset built on PyTorch and PyKEEN, designed for weight-aware evaluation and extension of Knowledge Graph Embedding (KGE) models. The toolkit provides a framework for weight-aware tasks that can extend and assess KGE models. Currently, the toolkit runs somewhat slowly, and we are working on optimizing the code to improve performance.

## Dependencies
1. PyKEEN, version=1.8.1  
2. PyTorch, version=1.10  

## Usage
To train and evaluate the model, use the following command:

```bash
python trains/train_walp_v2.py --base BASE --mode MODE --step EVALUATION_FREQUENCY --model MODEL_NAME --dataset DATASET_NAME --n_weight WEIGHT_OF_NEGATIVE_SAMPLES --dpct PERCENTAGE_OF_DATASET --epoch EPOCHES --train_batch TRAIN_BATCH_SIZE --eval_batch EVALUATION_BATCH_SIZE
```

- `--base_mode MODE`: Base of the exponential function  
- `--MODE`: Mode of activation function 
- `--step`: Frequency of evaluation  
- `--model`: Name of the model  
- `--dataset`: Name of the dataset  
- `--n_weight`: Weight for negative samples  
- `--dpct`: Percentage of the dataset  
- `--epoch`: Total number of training epochs  
- `--train_batch`: Batch size for training  
- `--eval_batch`: Batch size for evaluation

## TODO
1. Standardize variable names and add comments to key sections of the code for better readability and usability  
2. Perform extensive testing to eliminate potential bugs  
3. Implement a more efficient version to improve runtime performance
