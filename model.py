# model.py (No CRF Layer - Explicit Softmax Output for Manual Loss - Float32)

import tensorflow as tf
from tensorflow.keras import layers # Use keras layers directly
from transformers import TFAutoModel
import sys
import json
import os

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
try:
    transformer_base = TFAutoModel.from_pretrained(MODEL_NAME, from_pt=False)
    print(f"Successfully loaded TF weights for {MODEL_NAME}")
except OSError:
    print(f"Could not find TF weights for {MODEL_NAME}. Trying from PyTorch weights (from_pt=True)...")
    try:
        transformer_base = TFAutoModel.from_pretrained(MODEL_NAME, from_pt=True)
        print(f"Successfully loaded PyTorch weights for {MODEL_NAME}")
    except Exception as e:
        print(f"Failed to load model {MODEL_NAME} from both TF and PyTorch weights: {e}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading the model {MODEL_NAME}: {e}", file=sys.stderr)
    sys.exit(1)

# --- Labels (Same as before) ---
UNIQUE_LABELS = [
    'O', 'B-Author', 'I-Author', 'B-Year', 'I-Year', 'B-Title', 'I-Title',
    'B-ContainerTitle', 'I-ContainerTitle', 'B-Publisher', 'I-Publisher',
    'B-Volume', 'I-Volume', 'B-Issue', 'I-Issue', 'B-Pages', 'I-Pages',
    'B-DOI', 'I-DOI',
]
NUM_LABELS = len(UNIQUE_LABELS)
label2id = {label: i for i, label in enumerate(UNIQUE_LABELS)}
id2label = {i: label for i, label in enumerate(UNIQUE_LABELS)}
OUTSIDE_LABEL = 'O'

# --- Model Definition ---
def build_model(max_length=512):
    """
    Builds the Keras model for sequence labeling (BERT + Dense + Softmax).
    Outputs probabilities directly. Configured for Float32.
    """
    input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

    # BERT Base Model
   # Inside model.py, build_model function
    transformer_output = transformer_base({'input_ids': input_ids, 'attention_mask': attention_mask}) # Pass as dict
    sequence_output = transformer_output.last_hidden_state

    # Dropout
    sequence_output = layers.Dropout(0.1)(sequence_output)

    # Dense Layer to project to number of labels
    logits = layers.Dense(NUM_LABELS, name='logits')(sequence_output)

    # *** Explicit Softmax Layer - Removed dtype='float32' as it's default now ***
    probabilities = layers.Softmax(name='probabilities')(logits)

    # Define the model - outputs probabilities
    model = tf.keras.Model(
        inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
        outputs=probabilities # Output probabilities now
    )
    return model

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # (Same as before)
    print(f"Building model with {NUM_LABELS} labels based on '{MODEL_NAME}'...")
    model = build_model()
    model.summary()
    mapping_filename = 'label_mappings.json'
    # (Saving logic omitted for brevity)