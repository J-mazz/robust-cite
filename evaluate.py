# evaluate.py
# Script to evaluate the trained NER model.

import os
# --- Compatibility Notes ---
# Keep consistent with training script
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import numpy as np
import sys
import logging
import traceback
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import seqeval.metrics # For NER metrics calculation

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])

# --- Configuration (Should match training script where relevant) ---
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/apachecker/' # Adjust if needed
DATA_DIR = os.path.join(DRIVE_PROJECT_PATH, "processed_data")
CLEANED_DATA_FILE = os.path.join(DATA_DIR, "processed_30k_citations.txt")
CHECKPOINT_DIR = os.path.join(DRIVE_PROJECT_PATH, "ner_checkpoints_gpu")
FINAL_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "gpu_ner_model_final_best.weights.h5")

# --- Model & Training Params (Need to match model architecture and data prep) ---
MAX_LENGTH = 128
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
IGNORE_LABEL_ID = -100
EVAL_BATCH_SIZE = 32 # Can usually be larger than training batch size

# --- Path Verification and Setup ---
MODEL_PY_PATH = os.path.join(DRIVE_PROJECT_PATH, 'model.py')
if not os.path.isfile(MODEL_PY_PATH): sys.exit(f"Error: model.py not found at: {MODEL_PY_PATH}")
logging.info(f"Found model.py at: {MODEL_PY_PATH}")
project_dir = os.path.dirname(MODEL_PY_PATH)
if project_dir not in sys.path: sys.path.insert(0, project_dir)

# --- Import Model Definition ---
try:
    from model import build_model, MODEL_NAME, label2id, id2label, NUM_LABELS, OUTSIDE_LABEL
    logging.info(f"Imported definitions from model.py: {MODEL_NAME}, {NUM_LABELS} labels.")
except ImportError:
    logging.error(f"FATAL: Could not import from model.py at '{project_dir}'. Make sure it's accessible.")
    sys.exit(1)
except Exception as e:
    logging.error(f"FATAL: Error importing from model.py: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    if project_dir in sys.path: sys.path.remove(project_dir)


# --- Utility Functions (Copied/Adapted from train_robust.py) ---

def load_data_from_conll(filepath):
    """Loads sentences and tags from a CoNLL-style file (token\tTAG)."""
    # (Identical to function in train_robust.py)
    sentences, tags = [], []
    current_sentence, current_tags = [], []
    line_num, skipped_lines = 0, 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    if current_sentence:
                        if len(current_sentence) == len(current_tags):
                            sentences.append(current_sentence); tags.append(current_tags)
                        else: logging.warning(f"L~{line_num}: Skip sentence, len mismatch"); skipped_lines+=1
                        current_sentence, current_tags = [], []
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, tag = parts
                        valid_tag = tag if tag in label2id else OUTSIDE_LABEL
                        if tag != valid_tag: logging.debug(f"L{line_num}: Map tag '{tag}'->'{OUTSIDE_LABEL}'")
                        current_sentence.append(token); current_tags.append(valid_tag)
                    else: logging.warning(f"L{line_num}: Skip malformed line: '{line}'"); skipped_lines+=1
            if current_sentence: # Handle last sentence
                 if len(current_sentence) == len(current_tags): sentences.append(current_sentence); tags.append(current_tags)
                 else: logging.warning(f"EOF: Skip sentence, len mismatch"); skipped_lines+=1
    except FileNotFoundError: logging.error(f"FATAL: Data file not found: {filepath}"); sys.exit(1)
    except Exception as e: logging.error(f"FATAL: Error reading data file {filepath}: {e}"); traceback.print_exc(); sys.exit(1)
    if not sentences: logging.error(f"FATAL: No valid sequences loaded from {filepath}."); sys.exit(1)
    logging.info(f"Loaded {len(sentences)} sequences. Skipped {skipped_lines} lines/sentences.")
    return sentences, tags

def tokenize_and_align_labels(sentences, tags, max_length, tokenizer):
    """Tokenizes text and aligns labels to BERT subwords."""
    # (Identical to function in train_robust.py, but pass tokenizer)
    logging.info(f"Tokenizing {len(sentences)} sequences with max_length={max_length}...")
    tokenized_inputs = tokenizer(sentences, max_length=max_length, padding='max_length', truncation=True, is_split_into_words=True, return_tensors="tf", return_attention_mask=True, return_token_type_ids=False)
    logging.info("Aligning labels...")
    aligned_labels = []
    num_sequences = len(tags); skipped_align_count = 0
    for i in tqdm(range(num_sequences), desc="Aligning labels", unit="sequence"):
        label_list = tags[i]; word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None; label_ids = []; original_label_idx = 0; malformed = False
        for word_idx in word_ids:
            if word_idx is None: label_ids.append(IGNORE_LABEL_ID)
            elif word_idx != previous_word_idx:
                if original_label_idx < len(label_list): label_ids.append(label2id.get(label_list[original_label_idx], label2id[OUTSIDE_LABEL])); original_label_idx += 1
                else: logging.warning(f"Seq {i}: Ran out of labels."); label_ids.append(label2id[OUTSIDE_LABEL]); malformed = True
            else: label_ids.append(IGNORE_LABEL_ID)
            previous_word_idx = word_idx
        current_len = len(label_ids) # Pad or truncate labels to max_length
        if current_len < max_length: label_ids.extend([IGNORE_LABEL_ID] * (max_length - current_len))
        elif current_len > max_length: logging.warning(f"Seq {i}: Label list > max_length. Truncating."); label_ids = label_ids[:max_length]
        aligned_labels.append(label_ids);
        if malformed: skipped_align_count += 1
    if skipped_align_count > 0: logging.warning(f"Potential alignment issues for {skipped_align_count} sequences.")
    aligned_labels_tf = tf.constant(aligned_labels, dtype=tf.int32)
    # Shape check
    expected_shape=(len(sentences), max_length)
    if tokenized_inputs['input_ids'].shape!=expected_shape or aligned_labels_tf.shape!=expected_shape:
        logging.error(f"FATAL: Shape mismatch! Inputs:{tokenized_inputs['input_ids'].shape}, Labels:{aligned_labels_tf.shape}, Expected:{expected_shape}"); sys.exit(1)
    tokenized_inputs["labels"] = aligned_labels_tf
    logging.info("Tokenization and label alignment complete.")
    return tokenized_inputs # Return the dict including labels

def create_tf_dataset(encodings, labels, batch_size, is_training=False): # Default is_training=False for eval
    """Creates tf.data.Dataset for evaluation."""
    # (Adapted from train_robust.py: removed shuffle logic, uses passed batch_size)
    labels_int32 = tf.cast(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']},
        labels_int32
    ))
    if is_training: # Should not happen in this script, but keep for safety
        logging.warning("Shuffling dataset during evaluation - this is unusual.")
        buffer_size = tf.cast(tf.shape(labels_int32)[0], dtype=tf.int64)
        dataset = dataset.shuffle(buffer_size, seed=RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size) # Use EVAL_BATCH_SIZE
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    # 1. Load Model and Weights
    logging.info("Building model structure...")
    model = build_model(max_length=MAX_LENGTH)
    logging.info("Model structure built.")

    if os.path.exists(FINAL_WEIGHTS_PATH):
        logging.info(f"Loading weights from: {FINAL_WEIGHTS_PATH}")
        try:
            model.load_weights(FINAL_WEIGHTS_PATH)
            logging.info("Weights loaded successfully.")
        except Exception as e:
            logging.error(f"FATAL: Failed to load weights from {FINAL_WEIGHTS_PATH}: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        logging.error(f"FATAL: Saved weights not found at {FINAL_WEIGHTS_PATH}")
        sys.exit(1)

    # 2. Load and Prepare Evaluation Data (Using Validation Split)
    logging.info("Loading and splitting data for evaluation...")
    sentences, tags = load_data_from_conll(CLEANED_DATA_FILE)
    # Reproduce the exact same validation split
    _ , val_sentences, _ , val_tags = train_test_split(
        sentences, tags, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
    )
    logging.info(f"Using {len(val_sentences)} validation sequences for evaluation.")

    # --- Load Tokenizer ---
    logging.info(f"Loading tokenizer: {MODEL_NAME}")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    except Exception as e:
        logging.error(f"FATAL: Failed to load tokenizer '{MODEL_NAME}': {e}")
        sys.exit(1)

    # --- Preprocess Validation Data ---
    logging.info("Preprocessing validation data...")
    # Note: We need the val_encodings object later for word_ids
    val_encodings = tokenize_and_align_labels(val_sentences, val_tags, MAX_LENGTH, tokenizer)
    eval_dataset = create_tf_dataset(val_encodings, val_encodings['labels'], EVAL_BATCH_SIZE, is_training=False)
    logging.info("Validation dataset prepared.")

    # 3. Get Predictions
    logging.info(f"Running model.predict on validation data (batch size: {EVAL_BATCH_SIZE})...")
    predictions_raw = model.predict(eval_dataset, verbose=1)
    predicted_label_ids = np.argmax(predictions_raw, axis=-1)
    true_label_ids = val_encodings['labels'].numpy() # Get true labels as numpy array
    logging.info("Predictions generated.")

    # 4. Decode and Align Labels for Seqeval
    true_labels_list = []
    pred_labels_list = []

    logging.info("Decoding predictions and aligning sequences for seqeval...")
    # Loop through each sequence in the validation set
    for i in tqdm(range(len(true_label_ids)), desc="Decoding sequences"):
        true_seq_ids = true_label_ids[i]
        pred_seq_ids = predicted_label_ids[i]
        word_ids = val_encodings.word_ids(batch_index=i) # Get word IDs for this sequence

        true_labels_for_seq = []
        pred_labels_for_seq = []
        previous_word_idx = None

        for k, word_idx in enumerate(word_ids):
            # Check if it's a valid word token (not None) AND it's the first subword token for that word
            if word_idx is not None and word_idx != previous_word_idx:
                true_id = true_seq_ids[k]
                pred_id = pred_seq_ids[k]
                # Only include if it wasn't originally ignored (IGNORE_LABEL_ID)
                # This filters out padding and subsequent subword tokens
                if true_id != IGNORE_LABEL_ID:
                    true_labels_for_seq.append(id2label.get(true_id, OUTSIDE_LABEL))
                    pred_labels_for_seq.append(id2label.get(pred_id, OUTSIDE_LABEL))
            previous_word_idx = word_idx

        # Add the reconstructed sequences if they are not empty
        if true_labels_for_seq:
            true_labels_list.append(true_labels_for_seq)
            pred_labels_list.append(pred_labels_for_seq)

    logging.info("Decoding and alignment complete.")

    # 5. Calculate and Print Metrics using Seqeval
    if not true_labels_list or not pred_labels_list:
        logging.error("FATAL: No valid label sequences found after decoding. Cannot calculate metrics.")
        sys.exit(1)

    logging.info("Calculating seqeval classification report...")
    try:
        report = seqeval.metrics.classification_report(true_labels_list, pred_labels_list, digits=4)
        f1 = seqeval.metrics.f1_score(true_labels_list, pred_labels_list)
        precision = seqeval.metrics.precision_score(true_labels_list, pred_labels_list)
        recall = seqeval.metrics.recall_score(true_labels_list, pred_labels_list)
        accuracy = seqeval.metrics.accuracy_score(true_labels_list, pred_labels_list)


        print("\n--- Evaluation Results (Entity-Level using seqeval) ---")
        print(report)
        print("--- Overall Scores ---")
        print(f"Overall Precision (micro avg): {precision:.4f}")
        print(f"Overall Recall (micro avg):    {recall:.4f}")
        print(f"Overall F1 Score (micro avg):  {f1:.4f}")
        print(f"Overall Accuracy (token lvl):  {accuracy:.4f}") # Note: seqeval accuracy is token-level based on entities

    except Exception as e:
        logging.error(f"Error calculating seqeval metrics: {e}")
        traceback.print_exc()

    logging.info("Evaluation script finished.")

# --- End of Script ---