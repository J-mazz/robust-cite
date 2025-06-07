# apachecker/process_giant_data.py
# Version: Corrected Main Structure + Separator Logic
import csv
import sys
import os
import traceback # Import traceback for better error reporting
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from transformers import AutoTokenizer

# --- Configuration ---
INPUT_CSV_PATH = './DataPreprocessing/giant_100k_sample.csv'
OUTPUT_BIO_FILE = './DataPreprocessing/giant_100k_train_ContainerTitle.txt'
XML_COLUMN_INDEX = 3
TOKENIZER_NAME = 'bert-base-uncased'
OUTSIDE_LABEL = 'O'

# --- Separator Tokens (for BIO refinement) ---
SEPARATOR_TOKENS = {
    ',', '.', ';', ':', '&', 'and', 'et', 'al', 'al.'
}

# --- XML Tag to BIO Label Mapping ---
XML_TAG_TO_BASE_LABEL = {
    'author': 'Author',
    'issued': 'Year',
    'year': 'Year',
    'title': 'Title',
    'container-title': 'ContainerTitle', # Ensure matches model.py UNIQUE_LABELS
    'publisher': 'Publisher',
    'volume': 'Volume',
    'issue': 'Issue',
    'page': 'Pages',
    'url': 'DOI', # Assuming URL mostly contains DOI/link
}

# --- Global Tokenizer Initialization ---
print(f"Loading tokenizer: {TOKENIZER_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"\n--- FATAL ERROR: Could not load tokenizer '{TOKENIZER_NAME}' ---", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    sys.exit(1)

# --- Core XML Processing Functions ---

def process_node(node, current_base_label, tokens, bio_labels):
    """
    Recursively processes a BeautifulSoup node. Tokenizes text and assigns
    preliminary BIO labels based on inherited XML tag context.
    """
    global tokenizer
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            try:
                node_tokens = tokenizer.tokenize(text)
            except Exception as e:
                 print(f"Warning: Tokenizer failed on text segment: '{text[:50]}...'. Error: {e}. Skipping segment.", file=sys.stderr)
                 node_tokens = []
            if node_tokens:
                tokens.extend(node_tokens)
                label_to_assign = OUTSIDE_LABEL if current_base_label is None else current_base_label
                if label_to_assign == OUTSIDE_LABEL:
                     bio_labels.extend([OUTSIDE_LABEL] * len(node_tokens))
                else:
                    # Assign preliminary B-/I- (Refinement step handles final logic)
                    first_label = f"B-{label_to_assign}"
                    subsequent_label = f"I-{label_to_assign}"
                    current_node_bio_labels = [first_label] + [subsequent_label] * (len(node_tokens) - 1)
                    bio_labels.extend(current_node_bio_labels)
    elif isinstance(node, Tag):
        new_base_label = XML_TAG_TO_BASE_LABEL.get(node.name.lower(), None)
        label_for_children = new_base_label if new_base_label is not None else current_base_label
        # Use .children to iterate over direct children only (tags and strings)
        for child in node.children:
            process_node(child, label_for_children, tokens, bio_labels)

# THIS IS THE CORRECT VERSION OF THE FUNCTION (Defined ONCE at top level)
# --- REVISED XML Processing Function (with state tracking) ---
def process_xml_to_bio_final(xml_string, row_num_for_debug="N/A"):
    """
    Parses XML, generates tokens/prelim labels, refines BIO sequence
    (handling separators AND maintaining entity context), returns final lists.
    """
    tokens = []
    prelim_bio_labels = []
    if not xml_string or not xml_string.strip():
        return [], []
    try:
        # Use lxml parser, add dummy root tag
        soup = BeautifulSoup(f"<root>{xml_string}</root>", 'lxml')
        for element in soup.root.children:
             process_node(element, None, tokens, prelim_bio_labels)
    except Exception as e:
        print(f"Warning: Row {row_num_for_debug}: Failed to parse/process XML. Error: {e}. XML: {xml_string[:100]}...", file=sys.stderr)
        return [], []

    # --- State-Based BIO Refinement ---
    final_bio_labels = []
    last_active_base_label = None # Tracks the entity type we are logically inside

    if len(tokens) != len(prelim_bio_labels):
        print(f"Warning: Row {row_num_for_debug}: Token/prelim_label length mismatch BEFORE refinement ({len(tokens)} vs {len(prelim_bio_labels)}). XML: {xml_string[:100]}... Returning empty.", file=sys.stderr)
        return [], [] # Cannot proceed if lists are mismatched

    for i, current_prelim_label in enumerate(prelim_bio_labels):
        current_token = tokens[i]
        current_final_label = ""
        current_base_label = None # What entity type does this token *conceptually* belong to?

        # 1. Determine base label for this token from its preliminary label
        if current_prelim_label != OUTSIDE_LABEL:
            if current_prelim_label.startswith("B-") or current_prelim_label.startswith("I-"):
                 current_base_label = current_prelim_label[2:]
            else:
                 # This case should ideally not happen if process_node is correct
                 print(f"Warning: Row {row_num_for_debug}: Unexpected preliminary label format '{current_prelim_label}'. Treating as O.", file=sys.stderr)
                 current_base_label = None # Treat as O context

        # 2. Assign the final BIO label based on context and separators
        is_separator = current_token.lower() in SEPARATOR_TOKENS

        if is_separator:
            # Separators are always 'O', but they DO NOT break the active entity context
            current_final_label = OUTSIDE_LABEL
            # We keep 'last_active_base_label' as it was, allowing continuation after the separator
        elif current_base_label is None: # Token was originally OUTSIDE_LABEL
            current_final_label = OUTSIDE_LABEL
            last_active_base_label = None # Reset context, we are definitively outside an entity
        else: # Token belongs to an entity conceptually (current_base_label is not None)
            # Determine B- or I- based on comparison with the last active entity type
            if last_active_base_label is None or current_base_label != last_active_base_label:
                # Start of a new entity (or first after O / different type)
                current_final_label = f"B-{current_base_label}"
            else:
                # Continuation of the same entity type
                current_final_label = f"I-{current_base_label}"

            # Update the active entity type since this token belongs to one
            last_active_base_label = current_base_label

        final_bio_labels.append(current_final_label)

    # Final length check
    if len(tokens) != len(final_bio_labels):
         print(f"Warning: Row {row_num_for_debug}: Token/label length mismatch AFTER BIO refinement ({len(tokens)} vs {len(final_bio_labels)}). XML: {xml_string[:100]}... Returning empty.", file=sys.stderr)
         return [], []

    return tokens, final_bio_labels

    try:
        # Use lxml parser, add dummy root tag
        soup = BeautifulSoup(f"<root>{xml_string}</root>", 'lxml')
        for element in soup.root.children:
             process_node(element, None, tokens, prelim_bio_labels)
    except Exception as e:
        print(f"Warning: Row {row_num_for_debug}: Failed to parse/process XML. Error: {e}. XML: {xml_string[:100]}...", file=sys.stderr)
        return [], []

    # --- Refined BIO Logic with Separator Check ---
    final_bio_labels = []
    previous_final_label = OUTSIDE_LABEL
    if len(tokens) != len(prelim_bio_labels):
        print(f"Warning: Row {row_num_for_debug}: Token/prelim_label length mismatch BEFORE refinement ({len(tokens)} vs {len(prelim_bio_labels)}). XML: {xml_string[:100]}... Returning empty.", file=sys.stderr)
        return [], []

    for i, current_prelim_label in enumerate(prelim_bio_labels):
        current_token = tokens[i]
        current_final_label = ""

        # *** Check for Separators FIRST ***
        if current_token.lower() in SEPARATOR_TOKENS:
            current_final_label = OUTSIDE_LABEL
        # *** If not a separator, apply standard BIO logic ***
        elif current_prelim_label == OUTSIDE_LABEL:
            current_final_label = OUTSIDE_LABEL
        else:
            # Extract base label (e.g., "Author")
            if current_prelim_label.startswith("B-") or current_prelim_label.startswith("I-"):
                 current_base_label = current_prelim_label[2:]
            else:
                 print(f"Warning: Row {row_num_for_debug}: Unexpected preliminary label format '{current_prelim_label}'. Treating as O.", file=sys.stderr)
                 current_final_label = OUTSIDE_LABEL
                 current_base_label = None

            if current_base_label:
                previous_base_label = None
                if previous_final_label != OUTSIDE_LABEL and (previous_final_label.startswith("B-") or previous_final_label.startswith("I-")):
                    previous_base_label = previous_final_label[2:]

                # Assign B- if base label changes OR previous was O (or a separator forced to O)
                if current_base_label != previous_base_label:
                    current_final_label = f"B-{current_base_label}"
                else: # current_base_label == previous_base_label
                    current_final_label = f"I-{current_base_label}"
            else:
                 current_final_label = OUTSIDE_LABEL

        final_bio_labels.append(current_final_label)
        previous_final_label = current_final_label # Use the assigned final label for next comparison

    # Final length check
    if len(tokens) != len(final_bio_labels):
         print(f"Warning: Row {row_num_for_debug}: Token/label length mismatch AFTER BIO refinement ({len(tokens)} vs {len(final_bio_labels)}). XML: {xml_string[:100]}... Returning empty.", file=sys.stderr)
         return [], []

    return tokens, final_bio_labels

# --- Main Execution Function (Single, Corrected Version) ---
def main(input_path, output_path):
    """
    Main function to read CSV, process XML, and write BIO output.
    """
    print(f"Starting processing of '{input_path}'...")
    processed_count = 0
    skipped_count = 0
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: '{output_dir}'")
            os.makedirs(output_dir)
        print(f"Opening input file '{input_path}' and output file '{output_path}'...")
        with open(input_path, 'r', encoding='utf-8-sig') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            try:
                header = next(reader)
                print(f"Skipped header row: {header}")
            except StopIteration:
                print("Error: Input CSV file appears to be empty.", file=sys.stderr)
                return
            except csv.Error as e:
                print(f"Warning: Error reading header row: {e}. Attempting to continue.", file=sys.stderr)

            print("Starting main processing loop over CSV rows...")
            for i, row in enumerate(reader):
                row_num = i + 1
                try:
                    if not row:
                        skipped_count += 1
                        continue
                    if len(row) <= XML_COLUMN_INDEX:
                        print(f"Warning: Row {row_num} has only {len(row)} columns (expected >= {XML_COLUMN_INDEX + 1}). Skipping.", file=sys.stderr)
                        skipped_count += 1
                        continue

                    xml_string = row[XML_COLUMN_INDEX]
                    if not xml_string or not xml_string.strip():
                        skipped_count += 1
                        continue

                    # *** THIS IS THE CORRECT CALL to the top-level function ***
                    tokens, bio_labels = process_xml_to_bio_final(xml_string, row_num)

                    if tokens and bio_labels:
                        for token, label in zip(tokens, bio_labels):
                            if token: outfile.write(f"{token}\t{label}\n")
                        outfile.write("\n")
                        processed_count += 1
                    elif xml_string.strip(): # Count skip only if processing non-empty XML failed
                        skipped_count += 1

                except csv.Error as csv_e:
                    print(f"Warning: Row {row_num}: CSV parsing error: {csv_e}. Skipping row.", file=sys.stderr)
                    skipped_count += 1
                    continue
                except Exception as e_inner:
                     print(f"--- Error processing row {row_num} ---", file=sys.stderr)
                     print(f"XML Snippet: {str(xml_string)[:200]}...", file=sys.stderr)
                     traceback.print_exc()
                     print(f"--- Skipping row {row_num} due to error ---", file=sys.stderr)
                     skipped_count += 1
                     continue

                if row_num % 5000 == 0:
                    print(f"  ... processed {row_num} rows ({processed_count} citations written, {skipped_count} skipped/errors)...")
            print("Finished processing all rows.")

    except FileNotFoundError:
        print(f"\n--- FATAL ERROR: Input file not found ---", file=sys.stderr)
        print(f"Path: '{input_path}'", file=sys.stderr)
        sys.exit(1)
    except IOError as e_io:
         print(f"\n--- FATAL ERROR: File access problem ---", file=sys.stderr)
         print(f"Could not open/read/write file. Check permissions/path.", file=sys.stderr)
         print(f"Details: {e_io}", file=sys.stderr)
         sys.exit(1)
    except Exception as e_outer:
        print(f"\n--- FATAL ERROR: An unexpected error occurred during setup ---", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print(f"\n--- Processing Summary ---")
    print(f"Input file:          '{input_path}'")
    print(f"Output file:         '{output_path}'")
    print(f"Successfully written: {processed_count} citations.")
    print(f"Skipped/Errored rows: {skipped_count}")
    print(f"--- Processing complete ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        import lxml
    except ImportError:
        print("Warning: 'lxml' library not found. Consider 'pip install lxml'. Falling back to built-in 'xml' parser.", file=sys.stderr)

    # Call the main processing function using the global config paths
    main(INPUT_CSV_PATH, OUTPUT_BIO_FILE)