import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from collections import Counter
from utils import (sentences, train_data, val_data, english_vectorizer, turkish_vectorizer,
                   masked_loss, masked_acc, tokens_to_text)

import unittest

turkish_sentences, english_sentences = sentences

# Lets inspect raw senctences
print(f"English (to translate) sentence:\n\n{english_sentences[-5]}\n")
print(f"Turkish (translation) sentence:\n\n{turkish_sentences[-5]}")

# Get rid of raw sentences to save from memory
del turkish_sentences
del english_sentences
del sentences

# Lets inspect first 10 words of the english and turkish vocabulary
print(f"First 10 words of the english vocabulary:\n\n{english_vectorizer.get_vocabulary()[:10]}\n")
print(f"First 10 words of the turkish vocabulary:\n\n{turkish_vectorizer.get_vocabulary()[:10]}")
# ['', '[UNK]', '[SOS]', '[EOS]', '.', 'tom', 'to', 'i', 'the', 'you']
# ['', '[UNK]', '[SOS]', '[EOS]', '.', 'tom', 'bir', '?', ',', 'o']
# First 4 words are special words for the empty string, special token for unknown words, start, and end of sentence.

# Lets check vocab size
vocab_size = turkish_vectorizer.vocabulary_size()
print(f"Turkish vocabulary is made up of {vocab_size} words\n")
# We set max word size to 1200 in utils!!!

# We will define special words to ID so that it will help us to map
# This helps you convert from words to ids
word_to_id = tf.keras.layers.StringLookup(
    vocabulary=turkish_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]"
)

# This helps you convert from ids to words
id_to_word = tf.keras.layers.StringLookup(
    vocabulary=turkish_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]",
    invert=True,
)

unk_id = word_to_id("[UNK]")
sos_id = word_to_id("[SOS]")
eos_id = word_to_id("[EOS]")
baunilha_id = word_to_id("baunilha")

print(f"The id for the [UNK] token is {unk_id}")
print(f"The id for the [SOS] token is {sos_id}")
print(f"The id for the [EOS] token is {eos_id}")
print(f"The id for baunilha (vanilla) is {baunilha_id}\n")

# We arranged in batches of 64 (BATCH_SIZE = 64 in utils) examples. Well look at to first batch with TAKE method.
for (to_translate, sr_translation), translation in train_data.take(1):
    # Padding of 0 is added to match the size of all sentences in the batch

    # Sentence to translate
    print(f"Tokenized english sentence:\n{to_translate[0, :].numpy()}\n\n")
    # The shifted to the right translation - Teacher forcing
    print(f"Tokenized turkish sentence (shifted to the right):\n{sr_translation[0, :].numpy()}\n\n")
    # The translation
    print(f"Tokenized turkish sentence:\n{translation[0, :].numpy()}\n\n")

# We can use seq2seq model with LSTMs but it is not good for long sentence
# Basically, it is because first parts of the input will have very little effect on final vector passed to decoder
# Therefore, we come to attention mechanism to avoid this. We give decoder to access to all part of input sentence.

# Hidden state is produced at each timestep of encoder. These are all passed to the attention layer
# And each are given a score given the current activation (hidden state) of the decoder.

# I will be using Scaled Dot Product (this is better but practiced with basic) Attention. It is a type of attention mechanism that is used in transformer models.
# It is a type of self-attention mechanism. It is used to calculate the attention weights between the query and key.
# Attention (Q, K, V) = Softmax ( (QK^T) / sqrt(d_k) ) V
# d_k is the dimension of the key vectors. In our case, it is the dimension of the hidden state of the encoder.

VOCAB_SIZE = 12000
UNITS = 256 # The number of units in the LSTM layers (the same number will be used for all LSTM layers)

################ ENCODER ################
# Embedding: Define the appropriate input_dim and output_dim and let it know that you are using '0' as padding,
## It can be done by using the appropriate value for the mask_zero parameter.

# Bidirectional LSTM: