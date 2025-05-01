"""Dataset normalization and preprocessing for EmotionNet text classification"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.3.0"
__author__ = "OUAKIB Amine"

import argparse
import numpy as np
import pandas as pd
import os
from os import walk
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Text preprocessing parameters
MAX_SEQUENCE_LENGTH = 100  # Maximum length of text sequences
MAX_NUM_WORDS = 10000      # Maximum vocabulary size
EMBEDDING_DIM = 100        # Dimension of word embeddings
TEST_SPLIT = 0.0           # Percentage of data for testing
VAL_SPLIT = 0.2            # Percentage of training data for validation

# Emotion classes mapping (adjust according to your dataset)
EMOTION_MAP = {
    'joy': 0,
    'sadness': 1,
    'anger': 2,
    'fear': 3,
    'disgust': 4,
    'surprise':5
}

def extract_label(filename):
    """Extracts emotion label from filename or file content
    
    Parameters
    ----------
    filename : str
        The name of the file containing the text sample
        
    Returns
    -------
    int
        The integer label corresponding to the emotion
    """
    # Example implementation - adjust based on your dataset structure
    # Option 1: Label is encoded in filename (e.g., "joy_123.txt")
    emotion = filename.split('_')[0]
    return EMOTION_MAP.get(emotion, 0)  # Default to sadness if not found
    
    # Option 2: Label is in first line of file content
    # with open(filename, 'r') as f:
    #     first_line = f.readline().strip()
    # return EMOTION_MAP.get(first_line, 0)

def load_text_data(path):
    """Loads and preprocesses text data from directory
    
    Parameters
    ----------
    path : str
        Path to directory containing text files
        
    Returns
    -------
    tuple
        (text_sequences, labels, tokenizer) where:
        - text_sequences: Padded sequences of word indices
        - labels: Numeric emotion labels
        - tokenizer: Fitted Tokenizer object
    """
    texts = []
    labels = []
    
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            if filename.endswith('.txt'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(extract_label(filename))
    
    print(f"Loaded {len(texts)} samples with {len(set(labels))} emotion classes")
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences to uniform length
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    labels = np.array(labels)
    
    return data, labels, tokenizer

def normalize_data(path):
    """Main normalization function for text data
    
    Parameters
    ----------
    path : str
        Path to directory containing raw text files
        
    Returns
    -------
    tuple
        (train_data, val_data, test_data, tokenizer) where:
        - train_data: (sequences, labels) for training
        - val_data: (sequences, labels) for validation
        - test_data: (sequences, labels) for testing
        - tokenizer: Fitted Tokenizer object
    """
    # Load and preprocess raw text data
    data, labels, tokenizer = load_text_data(path)
    
    # Shuffle data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    # Split into train/val/test
    test_size = int(TEST_SPLIT * data.shape[0])
    val_size = int(VAL_SPLIT * (data.shape[0] - test_size))
    
    test_data = (data[:test_size], labels[:test_size])
    val_data = (data[test_size:test_size+val_size], labels[test_size:test_size+val_size])
    train_data = (data[test_size+val_size:], labels[test_size+val_size:])
    
    print(f"Dataset split:")
    print(f"- Training samples: {len(train_data[1])}")
    print(f"- Validation samples: {len(val_data[1])}")
    print(f"- Test samples: {len(test_data[1])}")
    
    return train_data, val_data, test_data, tokenizer

def save_data(data, labels, write_path, num_chunks, prefix):
    """Saves preprocessed data to multiple numpy files
    
    Parameters
    ----------
    data : numpy.ndarray
        Text sequences (num_samples x sequence_length)
    labels : numpy.ndarray
        Emotion labels (num_samples,)
    write_path : str
        Directory to save files
    num_chunks : int
        Number of files to split data into
    prefix : str
        Prefix for saved files ('train', 'val', or 'test')
    """
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    # Combine features and labels
    combined = np.column_stack((data, labels))
    
    # Split into chunks
    chunks = np.array_split(combined, num_chunks)
    
    for i, chunk in enumerate(chunks):
        filename = os.path.join(write_path, f"{prefix}_{i}.npy")
        np.save(filename, chunk)
        print(f"Saved {filename} with {len(chunk)} samples")

def save_tokenizer(tokenizer, write_path):
    """Saves tokenizer configuration for later use
    
    Parameters
    ----------
    tokenizer : Tokenizer
        Fitted Keras Tokenizer object
    write_path : str
        Directory to save tokenizer files
    """
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    import json
    
    # Save tokenizer config
    tokenizer_config = tokenizer.to_json()
    with open(os.path.join(write_path, 'tokenizer_config.json'), 'w') as f:
        f.write(tokenizer_config)
    
    # Save word index
    word_index = tokenizer.word_index
    with open(os.path.join(write_path, 'word_index.json'), 'w') as f:
        json.dump(word_index, f)
    
    print(f"Tokenizer saved to {write_path}")

def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description="Text data preprocessing for Emotion Classification"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-d", "--dataset",
        required=True,
        type=str,
        help="path to directory containing raw text files"
    )
    group.add_argument(
        "-w", "--write_path",
        required=True,
        type=str,
        help="path where to save the processed data"
    )
    group.add_argument(
        "-n", "--num_chunks",
        required=True,
        type=int,
        help="number of chunks to split the dataset into"
    )
    arguments = parser.parse_args()
    return arguments

def main(arguments):
    """Main preprocessing function"""
    # Normalize and split data
    train_data, val_data, test_data, tokenizer = normalize_data(arguments.dataset)
    
    # Save tokenizer first
    save_tokenizer(tokenizer, arguments.write_path)
    
    # Save data chunks
    save_data(train_data[0], train_data[1], arguments.write_path, 
             arguments.num_chunks, 'train')
    save_data(val_data[0], val_data[1], arguments.write_path, 
             max(1, arguments.num_chunks // 5), 'val')  # Smaller validation set
    save_data(test_data[0], test_data[1], arguments.write_path, 
             max(1, arguments.num_chunks // 5), 'test')  # Smaller test set
    
    print("Data preprocessing completed successfully")

if __name__ == "__main__":
    args = parse_args()
    main(args)