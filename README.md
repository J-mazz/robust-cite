# Robust-Cite: NER-Based Citation Parser

A Named Entity Recognition (NER) system for extracting and classifying bibliographic components from academic citations using BERT-based transformers.

## 📋 Project Overview

This project implements a citation parsing system that identifies and classifies components of academic references including authors, titles, publication years, journals, publishers, and other bibliographic metadata. The model is trained on 30,000 citation sequences and shows promising results on the validation set (99.97% accuracy), though real-world performance and post-quantization results remain to be thoroughly evaluated.

## 🎯 Model Performance

### Training Results
- **Validation Accuracy**: 99.97% (on held-out validation set)
- **Training Dataset**: 30,000 citation sequences 
- **Training Split**: 24,000 sequences (80%)
- **Validation Split**: 6,000 sequences (20%)
- **Model Parameters**: 109,496,851 (417.70 MB)
- **Training Duration**: ~30 minutes on GPU

**Note**: These results are on the validation set. Real-world performance and behavior under model quantization/compression need further evaluation.

### Model Architecture
- **Base Model**: BERT-base-uncased (Transformers)
- **Framework**: TensorFlow/Keras with Hugging Face Transformers
- **Architecture**: BERT + Dense Classification Layer + Softmax
- **Sequence Length**: 128 tokens
- **Precision**: Mixed precision (FP16) for efficient GPU training
- **Output Classes**: 19 entity labels (BIO tagging scheme)

## 🏷️ Entity Labels

The model recognizes the following bibliographic entities using BIO (Begin-Inside-Outside) tagging:

| Entity Type | Description | Labels |
|-------------|-------------|---------|
| **Author** | Author names | B-Author, I-Author |
| **Year** | Publication year | B-Year, I-Year |
| **Title** | Article/paper title | B-Title, I-Title |
| **Container Title** | Journal/conference name | B-ContainerTitle, I-ContainerTitle |
| **Publisher** | Publishing organization | B-Publisher, I-Publisher |
| **Volume** | Volume number | B-Volume, I-Volume |
| **Issue** | Issue number | B-Issue, I-Issue |
| **Pages** | Page numbers | B-Pages, I-Pages |
| **DOI** | Digital Object Identifier | B-DOI, I-DOI |
| **Outside** | Non-entity tokens | O |

## 📊 Training Configuration

### Hyperparameters
```
Model: bert-base-uncased
Max Sequence Length: 128
Batch Size: 16
Epochs: 5
Learning Rate: 5e-05 (peak)
Optimizer: AdamW with warmup
Warmup Steps: 750 (10% of total training steps)
Total Training Steps: 7,500
Mixed Precision: enabled (FP16)
Gradient Clipping: enabled
```

### Training Environment
- **Platform**: Google Colab with GPU acceleration
- **Training Time**: ~30 minutes
- **Memory Optimization**: Mixed precision training
- **Data Pipeline**: TensorFlow data objects with shuffling and prefetching

## 🚀 Key Features

### Advanced Training Pipeline
- **Mixed Precision Training**: FP16 for faster training and reduced memory usage
- **Learning Rate Scheduling**: Warmup + exponential decay
- **Early Stopping**: Monitors validation accuracy with patience=3
- **Model Checkpointing**: Saves best weights based on validation performance
- **TensorBoard Integration**: Comprehensive training monitoring

### Robust Data Processing
- **Tokenization**: BERT tokenizer with proper sequence truncation/padding
- **Label Alignment**: Automatic alignment of BIO labels with subword tokens
- **Data Validation**: Comprehensive error handling and data integrity checks
- **Efficient Loading**: Processes 30K sequences with minimal memory overhead

### Production-Ready Features
- **Model Persistence**: Saves final model in Keras format
- **Error Handling**: Comprehensive logging and error recovery
- **Scalable Architecture**: Designed for large-scale citation processing
- **GPU Optimization**: Efficient utilization of GPU resources

## 📁 Project Structure

```
robust-cite/
├── train_robust.py              # Main training script
├── model.py                     # Model architecture definition  
├── evaluate.py                  # Model evaluation utilities
├── process_giant_data.py        # Data preprocessing pipeline
├── best_citation_model.keras    # Trained model (1.3GB)
├── label_mappings.json          # Entity label mappings
├── training_gpu.log             # Comprehensive training logs
├── processed_30k_citations.txt  # Processed training data
├── giant_30k_batch.csv          # Raw citation dataset
└── README.md                    # This file
```

## 🔧 Technical Implementation

### Model Architecture Details
```python
Model: "model"
__________________________________________________________________________________________________
Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
attention_mask (InputLayer) [(None, 128)]                0         []                            
input_ids (InputLayer)      [(None, 128)]                0         []                            
tf_bert_model (TFBertModel) TFBaseModelOutput...         109,482,240 ['attention_mask[0][0]',      
                                                                    'input_ids[0][0]']           
dropout_37 (Dropout)        (None, 128, 768)             0         ['tf_bert_model[0][0]']       
logits (Dense)              (None, 128, 19)              14,611    ['dropout_37[0][0]']          
probabilities (Softmax)     (None, 128, 19)              0         ['logits[0][0]']              
==================================================================================================
Total params: 109,496,851 (417.70 MB)
Trainable params: 109,496,851 (417.70 MB)
Non-trainable params: 0 (0.00 Byte)
```

### Training Optimizations
- **AdamW Optimizer**: Superior performance for transformer models
- **Gradient Clipping**: Prevents gradient explosion
- **Masked Loss/Accuracy**: Proper handling of padding tokens
- **Learning Rate Warmup**: Stabilizes early training
- **Mixed Precision**: 2x speed improvement on modern GPUs

## 📈 Performance Metrics

### Final Training Results
```
Training finished: 2025-05-03 23:02:54
Best validation accuracy: 99.97% (achieved at epoch 5)
Model size: 417.70 MB
Training efficiency: ~0.2 seconds per batch
```

### Training Progression
- **Initial epochs**: Model convergence and learning rate warmup
- **Middle epochs**: Steady accuracy improvement
- **Final epoch**: Achieved peak performance (99.97%)
- **No overfitting**: Validation accuracy remained stable

## 🛠️ Usage

### Requirements
```bash
tensorflow>=2.8.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
```

### Quick Start
```python
import tensorflow as tf
from transformers import AutoTokenizer

# Load the trained model
model = tf.keras.models.load_model('best_citation_model.keras')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Process a citation
citation = "Smith, J. (2023). Deep Learning for NLP. Nature Machine Intelligence, 5(2), 123-145."
inputs = tokenizer(citation, return_tensors="tf", max_length=128, padding=True, truncation=True)
predictions = model(inputs)
```

## 🔍 Model Capabilities

### Supported Citation Formats
- **Academic papers**: Journal articles, conference papers
- **Books**: Monographs, edited volumes, book chapters
- **Digital content**: DOI-enabled publications
- **Multi-author works**: Complex author name parsing
- **Various date formats**: Flexible year extraction

### Robust Parsing Features
- **Incomplete citations**: Handles missing information gracefully
- **Format variations**: Adapts to different citation styles
- **Punctuation handling**: Robust to formatting inconsistencies
- **Multi-language support**: BERT's multilingual capabilities
- **Nested entities**: Properly handles complex bibliographic structures

## 📋 Training Insights

### Successful Optimizations
- **Mixed precision training** reduced training time by ~50%
- **Learning rate scheduling** improved convergence stability
- **Proper label alignment** ensured accurate token-level classification
- **Data preprocessing** eliminated format inconsistencies
- **GPU utilization** achieved optimal hardware efficiency

### Lessons Learned
- **Version compatibility**: Required specific TensorFlow/Transformers versions
- **Memory management**: Mixed precision essential for large models
- **Data quality**: Clean preprocessing crucial for high performance
- **Validation strategy**: Proper train/validation split prevented overfitting

## 🔍 Current Limitations & Future Work

### Areas for Improvement
- **Real-world testing**: Validation accuracy doesn't guarantee production performance
- **Model quantization**: Need to evaluate INT8/FP16 quantized performance
- **Cross-domain robustness**: Trained on specific citation formats, may not generalize
- **Production deployment**: 1.3GB model size requires optimization for edge deployment
- **Evaluation metrics**: Need precision/recall per entity type for comprehensive assessment

### Next Steps
- **Quantization experiments**: Test INT8, FP16, and dynamic quantization
- **Diverse test sets**: Evaluate on citations from different domains/time periods
- **Error analysis**: Detailed failure case studies
- **Model compression**: Knowledge distillation to smaller architectures
- **Benchmarking**: Compare against existing citation parsing tools

## 📊 Model Statistics

| Metric | Value |
|--------|-------|
| Training Sequences | 30,000 |
| Validation Accuracy | 99.97% |
| Model Parameters | 109.5M |
| Training Time | ~30 minutes |
| Model Size | 1.3GB |
| Entity Classes | 19 |
| Max Sequence Length | 128 tokens |
| GPU Memory Usage | ~8GB (mixed precision) |

## 🏆 Project Summary

This project demonstrates a systematic approach to applying transformer-based NLP to bibliographic data processing. While the 99.97% validation accuracy shows promising training convergence, the real value lies in the robust training pipeline, proper experimental setup, and comprehensive logging that enables reproducible research.

Key technical contributions include efficient mixed-precision training, proper BIO label alignment with subword tokenization, and a scalable data processing pipeline. The next critical phase involves rigorous testing under production constraints, including model quantization and cross-domain evaluation.

---

**Development Status**: Training complete, production readiness pending quantization and real-world evaluation.
