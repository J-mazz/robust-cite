# train_robust.py
# TensorFlow training script for NER using model.py and data in ./processed_data/
# Optimized for GPU with Mixed Precision, Warmup LR Schedule, and Manual Loss from Probabilities.

import os
# --- Compatibility Notes ---
# Force Keras 2 Behavior (Needed to resolve KerasTensor ValueError)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics
# --- ADDED: Mixed Precision Import ---
from tensorflow.keras import mixed_precision
from transformers import BertTokenizerFast, TFAutoModel # Use the fast tokenizer
import numpy as np
import sys
import logging
import traceback
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import math

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])

# --- Enable Mixed Precision for GPU ---
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
logging.info(f"GPU OPTIMIZATION: Enabled mixed precision policy: {policy.name}")

# --- Strategy (Default for Single GPU/CPU) ---
strategy = tf.distribute.get_strategy()
logging.info(f"Using default strategy: {strategy.__class__.__name__}")
gpus = tf.config.list_logical_devices('GPU')
if gpus:
    logging.info(f"Running on GPU(s): {gpus}")
else:
    logging.warning("No GPU found. Running on CPU.")
logging.info(f"Number of replicas (should be 1 for single GPU/CPU): {strategy.num_replicas_in_sync}")


# --- Configuration ---
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/apachecker/' # Adjust if needed
DATA_DIR = os.path.join(DRIVE_PROJECT_PATH, "processed_data")
CLEANED_DATA_FILE = os.path.join(DATA_DIR, "processed_30k_citations.txt")
CHECKPOINT_DIR = os.path.join(DRIVE_PROJECT_PATH, "ner_checkpoints_gpu") # Specific dir for GPU checkpoints
LOG_FILE = os.path.join(CHECKPOINT_DIR, "training_gpu.log")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --- Path Verification and Setup ---
MODEL_PY_PATH = os.path.join(DRIVE_PROJECT_PATH, 'model.py')
if not os.path.isfile(MODEL_PY_PATH): sys.exit(f"Error: model.py not found at: {MODEL_PY_PATH}")
logging.info(f"Found model.py at: {MODEL_PY_PATH}")
project_dir = os.path.dirname(MODEL_PY_PATH)
if project_dir not in sys.path: sys.path.insert(0, project_dir)

# --- Import Model Definition ---
try:
    from model import build_model, MODEL_NAME, label2id, id2label, NUM_LABELS, OUTSIDE_LABEL, UNIQUE_LABELS
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


# --- Model & Tokenizer Settings ---
MAX_LENGTH = 128
logging.info(f"Using MAX_LENGTH: {MAX_LENGTH}")

# --- Training Hyperparameters ---
EPOCHS = 5
BATCH_SIZE = 16 # Suitable for single GPU like T4
logging.info(f"Using Batch Size: {BATCH_SIZE} (suitable for single GPU)")
LEARNING_RATE = 5e-5 # Base learning rate
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1
GRADIENT_CLIP_VALUE = 1.0
ADAM_EPSILON = 1e-8
RANDOM_SEED = 42
IGNORE_LABEL_ID = -100


# --- Refined Logging Setup ---
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logging.info("-----------------------------------------------")
logging.info("Starting NER Training Script (GPU Optimized - train_robust.py)")
logging.info(f"Using model: {MODEL_NAME}")
logging.info(f"Data file path: {CLEANED_DATA_FILE}")
logging.info(f"Checkpoints directory: {CHECKPOINT_DIR}")
logging.info(f"Max Sequence Length: {MAX_LENGTH}")
logging.info(f"Batch Size: {BATCH_SIZE}")
logging.info(f"Epochs: {EPOCHS}")
logging.info(f"Learning Rate (Peak): {LEARNING_RATE}")
logging.info(f"Mixed Precision Policy: {policy.name}")
logging.info("-----------------------------------------------")


# --- Data Loading ---
def load_data_from_conll(filepath):
    """Loads sentences and tags from a CoNLL-style file (token\tTAG)."""
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

# --- Tokenization and Preprocessing ---
logging.info(f"Loading tokenizer: {MODEL_NAME}")
try:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
except Exception as e: logging.error(f"FATAL: Failed to load tokenizer '{MODEL_NAME}': {e}"); sys.exit(1)

def tokenize_and_align_labels(sentences, tags, max_length):
    """Tokenizes text and aligns labels to BERT subwords."""
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
    return tokenized_inputs


# --- Custom Loss and Metric ---

# --- Define Manual Masked Sparse Categorical Cross-Entropy Loss FROM PROBABILITIES --- <<< NEW
# Use this function if your model DEFINITELY outputs probabilities (from Softmax)
# and you want to bypass SparseCategoricalCrossentropy(from_logits=False)
def manual_loss_from_probs(y_true, y_pred_probs):
    """
    Manually calculates masked sparse categorical cross-entropy directly from probabilities.
    y_true: True labels (batch_size, seq_len), int32, -100 for ignored positions.
    y_pred_probs: Predicted probabilities from model's Softmax (batch_size, seq_len, num_labels), float32/float16.
    """
    # Ensure types - y_pred_probs might be float16 due to mixed precision policy
    y_true = tf.cast(y_true, dtype=tf.int32)
    # We expect probabilities, but loss calculation works better in float32
    y_pred_probs = tf.cast(y_pred_probs, dtype=tf.float32)

    # Create mask for non-ignored labels (shape: batch_size, seq_len)
    mask = tf.cast(y_true != IGNORE_LABEL_ID, dtype=tf.float32)

    # Clip probabilities to avoid log(0) - VERY IMPORTANT
    epsilon_ = tf.keras.backend.epsilon() # Use Keras backend epsilon for stability
    y_pred_probs_clipped = tf.clip_by_value(y_pred_probs, epsilon_, 1.0 - epsilon_)

    # Gather the probabilities corresponding to the true labels
    # Need shape (batch_size, seq_len) for gathered probs.
    # Create indices for gather_nd: (batch_index, seq_index, true_label_index)
    y_true_safe_for_gather = tf.maximum(y_true, 0) # Replace -100 with 0 for indexing
    batch_size = tf.shape(y_true)[0]
    seq_len = tf.shape(y_true)[1]
    batch_indices = tf.range(batch_size)
    seq_indices = tf.range(seq_len)
    # Create grids that align with y_true dimensions
    batch_grid, seq_grid = tf.meshgrid(batch_indices, seq_indices, indexing='ij')
    # Stack to create (batch_size, seq_len, 3) indices
    full_indices = tf.stack([batch_grid, seq_grid, y_true_safe_for_gather], axis=-1)

    # Gather the probabilities for the true class labels
    gathered_probs = tf.gather_nd(y_pred_probs_clipped, full_indices)

    # Calculate the negative log likelihood (cross-entropy)
    # Add epsilon inside log just in case clipping wasn't enough (defensive)
    negative_log_likelihood = -tf.math.log(gathered_probs + epsilon_)

    # Apply the mask (zero out losses for ignored positions)
    masked_loss_vals = negative_log_likelihood * mask # Renamed to avoid confusion

    # Compute the average loss over the non-masked items in the batch
    sum_masked_loss = tf.reduce_sum(masked_loss_vals)
    sum_mask = tf.reduce_sum(mask)
    average_loss = sum_masked_loss / tf.maximum(sum_mask, epsilon_) # Avoid division by zero

    return average_loss
# --- END of manual_loss_from_probs definition ---


# --- Masked Accuracy (Requires Logits for Argmax) ---
# NOTE: If model outputs probabilities, this accuracy function needs adjustment
#       or the model needs to also output logits. Assuming model.py *only* outputs probs for now.
#       THIS MIGHT REPORT INCORRECT ACCURACY if y_pred are probabilities.
#       A better approach if model only outputs probs is to calculate accuracy outside training loop.
#       For now, keep it as is, but be aware.
def masked_accuracy(y_true, y_pred): # y_pred will be probabilities here!
    y_true = tf.cast(y_true, tf.int32)
    # Get predicted labels (argmax over the last dimension - works on probs too)
    y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    mask = tf.not_equal(y_true, IGNORE_LABEL_ID)
    correct_predictions = tf.equal(y_true, y_pred_labels)

    # Consider only non-masked positions for accuracy calculation
    masked_correct = tf.logical_and(mask, correct_predictions)

    # Cast mask and masked_correct to float for division
    mask = tf.cast(mask, dtype=tf.float32)
    masked_correct = tf.cast(masked_correct, dtype=tf.float32)

    # Calculate accuracy: sum of correct predictions / sum of non-masked labels
    # Add epsilon for stability if mask sum is zero
    accuracy = tf.reduce_sum(masked_correct) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
    return accuracy


# --- Build Model ---
def build_training_model(max_length=MAX_LENGTH):
    logging.info("Building model for training (expecting Softmax output from model.py)...") # Adjusted log
    try:
        model = build_model(max_length=max_length) # From model.py
        logging.info("Model built successfully (using build_model from model.py).")
        return model
    except Exception as e: logging.error(f"FATAL: Error building model: {e}"); traceback.print_exc(); sys.exit(1)

# --- Custom Learning Rate Schedule Class Definition ---
class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies linear warmup followed by exponential decay."""
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, warmup_steps, staircase=False, name=None):
        super().__init__()
        self.peak_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = float(warmup_steps)
        self.staircase = staircase
        self.name = name
        self.exp_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.peak_learning_rate, decay_steps=decay_steps,
            decay_rate=decay_rate, staircase=staircase
        )

    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        warmup_lr = (self.peak_learning_rate / tf.maximum(self.warmup_steps, 1.0)) * step_float
        decay_lr = self.exp_decay_schedule(step_float - self.warmup_steps)
        learning_rate = tf.cond(step_float < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)
        return tf.maximum(learning_rate, 1e-9) # Add floor

    def get_config(self):
        return {
            "initial_learning_rate": self.peak_learning_rate, "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate, "warmup_steps": int(self.warmup_steps),
            "staircase": self.staircase, "name": self.name
        }
# --- End of LR Schedule Definition ---

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and Split Data
    sentences, tags = load_data_from_conll(CLEANED_DATA_FILE)
    train_sentences, val_sentences, train_tags, val_tags = train_test_split(sentences, tags, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    logging.info(f"Data split: {len(train_sentences)} train, {len(val_sentences)} validation.")

    # 2. Tokenize and Create tf.data.Dataset
    train_encodings = tokenize_and_align_labels(train_sentences, train_tags, MAX_LENGTH)
    val_encodings = tokenize_and_align_labels(val_sentences, val_tags, MAX_LENGTH)

    def create_tf_dataset(encodings, labels, is_training=True):
        labels_int32 = tf.cast(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((
            {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']},
            labels_int32
        ))
        if is_training:
            # Apply the buffer_size dtype fix
            buffer_size = tf.cast(tf.shape(labels_int32)[0], dtype=tf.int64) # Cast buffer_size to int64
            dataset = dataset.shuffle(buffer_size, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_tf_dataset(train_encodings, train_encodings['labels'], is_training=True)
    val_dataset = create_tf_dataset(val_encodings, val_encodings['labels'], is_training=False)
    logging.info("Created tf.data.Dataset objects.")

    # 3. Build and Compile Model
    model = build_training_model()
    model.summary(print_fn=logging.info)

    # --- Optimizer with Warmup Schedule ---
    steps_per_epoch = math.ceil(len(train_sentences) / BATCH_SIZE)
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(num_train_steps * WARMUP_RATIO)
    logging.info(f"Steps: steps_per_epoch={steps_per_epoch}, num_train_steps={num_train_steps}, num_warmup_steps={num_warmup_steps}")

    lr_schedule_with_warmup = WarmupExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=num_train_steps - num_warmup_steps,
        decay_rate=0.96, # Example decay rate
        warmup_steps=num_warmup_steps
    )

    # Optimizer definition - includes clipnorm here
    optimizer = optimizers.AdamW(
        learning_rate=lr_schedule_with_warmup, # Use the schedule
        weight_decay=0.01,
        epsilon=ADAM_EPSILON,
        clipnorm=GRADIENT_CLIP_VALUE # clipnorm added here
    )
    logging.info(f"Using AdamW optimizer with WarmupExponentialDecay schedule. Loss scaling for mixed precision handled by model.fit.")

    # Compile call - use the manual loss function
    model.compile(
        optimizer=optimizer,
        loss=manual_loss_from_probs, # <<< USE THE MANUAL LOSS FUNCTION
        metrics=[masked_accuracy]    # Keep masked accuracy (aware of potential issues if needed later)
    )
    logging.info("Model compiled with AdamW (schedule), MANUAL loss from probs, masked accuracy, grad clipping (in optimizer), mixed precision.")

    # 4. Setup Callbacks
    checkpoint_filepath = os.path.join(CHECKPOINT_DIR, "gpu_ner_model_epoch_{epoch:02d}_val_acc_{val_masked_accuracy:.4f}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=True, monitor='val_masked_accuracy',
        mode='max', save_best_only=True, verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_masked_accuracy', patience=EARLY_STOPPING_PATIENCE, mode='max',
        restore_best_weights=True, verbose=1
    )
    tensorboard_log_dir = os.path.join(CHECKPOINT_DIR, 'logs_gpu')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1, profile_batch='100,110') # Can adjust profile_batch range

    logging.info(f"Checkpoints monitor 'val_masked_accuracy', saving best weights to {CHECKPOINT_DIR}")
    logging.info(f"Early stopping monitor 'val_masked_accuracy', patience={EARLY_STOPPING_PATIENCE}")
    logging.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

    # 5. Train Model
    logging.info(f"Starting model training on GPU for up to {EPOCHS} epochs...")
    history = None
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback],
            verbose=1
        )
        logging.info("Training finished.")
        if early_stopping_callback.stopped_epoch > 0: logging.info(f"Early stopping triggered after epoch {early_stopping_callback.stopped_epoch + 1}")
    except Exception as e:
        logging.error(f"FATAL: Training error: {e}", exc_info=True)
        error_save_path = os.path.join(CHECKPOINT_DIR, "model_weights_on_error.weights.h5")
        try: model.save_weights(error_save_path); logging.info(f"Saved weights on error: {error_save_path}")
        except Exception as save_e: logging.error(f"Could not save weights after error: {save_e}")
        sys.exit(1)

    # 6. Save Final Best Model Weights
    try:
        final_model_path = os.path.join(CHECKPOINT_DIR, "gpu_ner_model_final_best.weights.h5")
        model.save_weights(final_model_path) # Weights should be the best ones due to restore_best_weights=True
        logging.info(f"Final best model weights saved to {final_model_path}")
        if history and history.history and 'val_masked_accuracy' in history.history:
            best_epoch_idx = np.argmax(history.history['val_masked_accuracy'])
            best_val_acc = history.history['val_masked_accuracy'][best_epoch_idx]
            logging.info(f"Best validation accuracy ({best_val_acc:.4f}) at epoch {best_epoch_idx + 1}.")
        else: logging.info("Could not determine best accuracy from history.")
    except Exception as e: logging.warning(f"Could not save final weights: {e}", exc_info=True)

    logging.info("Script execution completed.")

# --- End of Script ---