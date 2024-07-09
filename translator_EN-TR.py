import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from collections import Counter
from utils import (sentences, train_data, val_data, english_vectorizer, turkish_vectorizer,
                   masked_loss, masked_acc, tokens_to_text)

import TREN_unittest

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

# Bidirectional LSTM: We will implement bidrectional behavior for RNN-like layers with TF. We need to define type of layer and its paramaters.\
## In addition, we need to make sure we have appropriate number of units and LSTM returns full sequence not only last output.

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        """Initializes an instance of this class

        Args:
            vocab_size (int): Size of the vocabulary
            units (int): Number of units in the LSTM layer
        """
        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )

        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode="sum",
            layer=tf.keras.layers.LSTM(
                units=units,
                return_sequences=True
            ),
        )

    def call(self, context):
        """Forward pass of this layer

        Args:
            context (tf.Tensor): The sentence to translate

        Returns:
            tf.Tensor: Encoded sentence to translate
        """
        # Pass the context through the embedding layer
        x = self.embedding(context)

        # Pass the output of the embedding through the RNN
        x = self.rnn(x)
        return x

# Lets test the encoder
# Create an instance of  class
encoder = Encoder(VOCAB_SIZE, UNITS)

# Pass a batch of sentences to translate from english to turkish
encoder_output = encoder(to_translate)

print(f'Tensor of sentences in english has shape: {to_translate.shape}\n')
print(f'Encoder output has shape: {encoder_output.shape}')

TREN_unittest.test_encoder(Encoder)

################ CROSS ATTENTION ################
# MultiHeadAttention: Well define key dimension (size of the key and query tensors).
# We also need to set the number of heads to 1 since we use it for attention between two tensors (not multiple)

# We add will add layer for the shift to right translation since we do cross attention on decoder side.
# Layer normalization also will be performed for better stability of network.

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        """Initializes an instance of this class

        Args:
            units (int): Number of units in the LSTM layer
        """
        super().__init__()

        self.mha = (
            tf.keras.layers.MultiHeadAttention(
                key_dim=units,
                num_heads=1
            )
        )

        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, context, target):
        """Forward pass of this layer

        Args:
            context (tf.Tensor): Encoded sentence to translate
            target (tf.Tensor): The embedded shifted-to-the-right translation

        Returns:
            tf.Tensor: Cross attention between context and target
        """

        # Call the MH attention by passing in the query and value
        # For this case the query should be the translation and the value the encoded sentence to translate
        attn_output = self.mha(
            query=target,
            value=context
        )

        x = self.add([target, attn_output])
        x = self.layernorm(x)
        return x

# Lets test the cross attention
# Create an instance of class
attention_layer = CrossAttention(UNITS)

# The attention layer expects the embedded sr-translation and the context
# The context (encoder_output) is already embedded so we dont need to do this for sr_translation:
sr_translation_embed = tf.keras.layers.Embedding(VOCAB_SIZE, output_dim=UNITS, mask_zero=True)(sr_translation)

# Compute the cross attention
attention_result = attention_layer(encoder_output, sr_translation_embed)

print(f'Tensor of contexts has shape: {encoder_output.shape}')
print(f'Tensor of translations has shape: {sr_translation_embed.shape}')
print(f'Tensor of attention scores has shape: {attention_result.shape}')

TREN_unittest.test_encoder(CrossAttention)

################ DECODER ################
# Embeedding: