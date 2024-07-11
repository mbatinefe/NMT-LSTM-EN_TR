import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model
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

TREN_unittest.test_cross_attention(CrossAttention)


################ DECODER ################
# Embeedding: Input-Output and Maske zero for padding
# Pre-Attention LSTM: vanilla LSTM, number of units, returns full sequence.
# # Hidden state = Memory state
# # Cell state = Carry state
# Attention: Cross attention will be used
# Post-Attention = LSTM
# Dense Layer = every possible word in vocab -> and softtmax

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        """Initializes an instance of this class

        Args:
            vocab_size (int): Size of the vocabulary
            units (int): Number of units in the LSTM layer
        """
        super(Decoder, self).__init__()

        # The embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )

        # The RNN before attention
        self.pre_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True
        )

        # The attention layer
        self.attention = CrossAttention(units)

        # The RNN after attention
        self.post_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True
        )

        # The dense layer with logsoftmax activation
        self.output_layer = tf.keras.layers.Dense(
            units=vocab_size,
            activation=tf.nn.log_softmax
        )

    def call(self, context, target, state=None, return_state=False):
        """Forward pass of this layer

        Args:
            context (tf.Tensor): Encoded sentence to translate
            target (tf.Tensor): The shifted-to-the-right translation
            state (list[tf.Tensor, tf.Tensor], optional): Hidden state of the pre-attention LSTM. Defaults to None.
            return_state (bool, optional): If set to true return the hidden states of the LSTM. Defaults to False.

        Returns:
            tf.Tensor: The log_softmax probabilities of predicting a particular token
        """

        # Get the embedding of the input
        x = self.embedding(target)

        # Pass the embedded input into the pre attention LSTM
        # - The LSTM we defined earlier should return the output alongside the state (made up of two tensors)
        # - Pass in the state to the LSTM (needed for inference)
        x, hidden_state, cell_state = self.pre_attention_rnn(x, initial_state=state)

        # Perform cross attention between the context and the output of the LSTM (in that order)
        x = self.attention(context, x)

        # Do a pass through the post attention LSTM
        x = self.post_attention_rnn(x, initial_state=[hidden_state, cell_state])

        # Compute the logits
        logits = self.output_layer(x)

        if return_state:
            return logits, [hidden_state, cell_state]

        return logits

# Lets test the decoder
# Create an instance of class
decoder = Decoder(VOCAB_SIZE, UNITS)

# Notice that we don't need the embedded version of sr_translation since this is done inside the class
logits = decoder(encoder_output, sr_translation)

print(f'Tensor of contexts has shape: {encoder_output.shape}')
print(f'Tensor of right-shifted translations has shape: {sr_translation.shape}')
print(f'Tensor of logits has shape: {logits.shape}')

TREN_unittest.test_decoder(Decoder, CrossAttention)


################ TRANSLATOR ################
# LEts put all together into model
class Translator(tf.keras.Model):
    def __init__(self, vocab_size, units):
        """Initializes an instance of this class

        Args:
            vocab_size (int): Size of the vocabulary
            units (int): Number of units in the LSTM layer
        """
        super().__init__()

        # Define the encoder with the appropriate vocab_size and number of units
        self.encoder = Encoder(vocab_size, units)

        # Define the decoder with the appropriate vocab_size and number of units
        self.decoder = Decoder(vocab_size, units)

    def call(self, inputs):
        """Forward pass of this layer

        Args:
            inputs (tuple(tf.Tensor, tf.Tensor)): Tuple containing the context (sentence to translate) and the target (shifted-to-the-right translation)

        Returns:
            tf.Tensor: The log_softmax probabilities of predicting a particular token
        """

        # In this case inputs is a tuple consisting of the context and the target, unpack it into single variables
        context, target = inputs

        # Pass the context through the encoder
        encoded_context = self.encoder(context)

        # Compute the logits by passing the encoded context and the target to the decoder
        logits = self.decoder(encoded_context, target)

        return logits

# Lets check the translator

# Create an instance of class
translator = Translator(VOCAB_SIZE, UNITS)

# Compute the logits for every word in the vocabulary
logits = translator((to_translate, sr_translation))

print(f'Tensor of sentences to translate has shape: {to_translate.shape}')
print(f'Tensor of right-shifted translations has shape: {sr_translation.shape}')
print(f'Tensor of logits has shape: {logits.shape}')

TREN_unittest.test_translator(Translator, Encoder, Decoder)

################ TRAINING ################

def compile_and_train(model, epochs=40, steps_per_epoch=250):
    model.compile(optimizer="adam", loss=masked_loss, metrics=[masked_acc, masked_loss])

    history = model.fit(
        train_data.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )

    # Save the entire model to a TensorFlow SavedModel
    model.save('entire-model', save_format='tf')

    return model, history

# Train the model
trained_translator, history = compile_and_train(translator)

trained_translator.summary()

################ USING THE MODEL FOR INFERENCE ################

def generate_next_token(decoder, context, next_token, done, state, temperature=0.0):
    """Generates the next token in the sequence

    Args:
        decoder (Decoder): The decoder
        context (tf.Tensor): Encoded sentence to translate
        next_token (tf.Tensor): The predicted next token
        done (bool): True if the translation is complete
        state (list[tf.Tensor, tf.Tensor]): Hidden states of the pre-attention LSTM layer
        temperature (float, optional): The temperature that controls the randomness of the predicted tokens. Defaults to 0.0.

    Returns:
        tuple(tf.Tensor, np.float, list[tf.Tensor, tf.Tensor], bool): The next token, log prob of said token, hidden state of LSTM and if translation is done
    """
    # Get the logits and state from the decoder
    logits, state = decoder(context, next_token, state=state, return_state=True)

    # Trim the intermediate dimension
    logits = logits[:, -1, :]

    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)

    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    # Trim dimensions of size 1
    logits = tf.squeeze(logits)
    next_token = tf.squeeze(next_token)

    # Get the logit of the selected next_token
    logit = logits[next_token].numpy()

    # Reshape to (1,1) since this is the expected shape for text encoded as TF tensors
    next_token = tf.reshape(next_token, shape=(1, 1))

    # If next_token is End-of-Sentence token its done
    if next_token == eos_id:
        done = True

    return next_token, logit, state, done

# Process sentence to translate and encode

# Input sentence to be translated
eng_sentence = "I student"

# Conver to tensor
texts = tf.convert_to_tensor(eng_sentence)[tf.newaxis]

# Vectorize it and pass it through the encoder
context = english_vectorizer(texts).to_tensor()
context = encoder(context)

# Set state of decoder
# Next token is SOS - new start
next_token = tf.fill((1, 1), sos_id)

# Hidden and cell states, we will fill with random uniform samples
state = [tf.random.uniform((1, UNITS)), tf.random.uniform((1, UNITS))]

# Done flag until next token is EOS
done = False

# Generate next token
next_token, logit, state, done = generate_next_token(trained_translator.decoder, context, next_token, done, state, temperature=0.5)
print(f"Next token: {next_token}\nLogit: {logit:.4f}\nDone? {done}")

################ TRANSLATE ################
def translate(model, text, max_length=50, temperature=0.0):
    """Translate a given sentence from English to Turkish

    Args:
        model (tf.keras.Model): The trained translator
        text (string): The sentence to translate
        max_length (int, optional): The maximum length of the translation. Defaults to 50.
        temperature (float, optional): The temperature that controls the randomness of the predicted tokens. Defaults to 0.0.

    Returns:
        tuple(str, np.float, tf.Tensor): The translation, logit that predicted <EOS> token and the tokenized translation
    """
    # Lists to save tokens and logits
    tokens, logits = [], []

    # PROCESS THE SENTENCE TO TRANSLATE

    # Convert the original string into a tensor
    text = tf.convert_to_tensor(text)[tf.newaxis]

    # Vectorize the text using the correct vectorizer
    context = english_vectorizer(text).to_tensor()

    # Get the encoded context (pass the context through the encoder)
    # Hint: Remember you can get the encoder by using model.encoder
    context = model.encoder(context)

    # INITIAL STATE OF THE DECODER

    # First token should be SOS token with shape (1,1)
    next_token = tf.fill((1, 1), sos_id)

    # Initial hidden and cell states should be tensors of zeros with shape (1, UNITS)
    state = [tf.zeros((1, UNITS)), tf.zeros((1, UNITS))]

    # You are done when you draw a EOS token as next token (initial state is False)
    done = False

    # Iterate for max_length iterations
    for _ in range(max_length):

        # Generate the next token
        next_token, logit, state, done = generate_next_token(
            decoder=decoder,
            context=context,
            next_token=next_token,
            done=done,
            state=state,
            temperature=temperature
        )

        # If done then break out of the loop
        if done:
            break

        # Add next_token to the list of tokens
        tokens.append(next_token)

        # Add logit to the list of logits
        logits.append(logit)

    # Concatenate all tokens into a tensor
    tokens = tf.concat(tokens, axis=-1)

    # Convert the translated tokens into text
    translation = tf.squeeze(tokens_to_text(tokens, id_to_word))
    translation = translation.numpy().decode()

    return translation, logits[-1], tokens

# Translate the sentence

temp = 0.0
original_sentence = "I student"

translation, logit, tokens = translate(trained_translator, original_sentence, temperature=temp)
print(f"Temperature: {temp}\n\nOriginal sentence: {original_sentence}\nTranslation: {translation}\nTranslation tokens:{tokens}\nLogit: {logit:.3f}")

temp = 0.3
original_sentence = "I student"

translation, logit, tokens = translate(trained_translator, original_sentence, temperature=temp)
print(f"Temperature: {temp}\n\nOriginal sentence: {original_sentence}\nTranslation: {translation}\nTranslation tokens:{tokens}\nLogit: {logit:.3f}")

temp = 0.7
original_sentence = "I student"

translation, logit, tokens = translate(trained_translator, original_sentence, temperature=temp)
print(f"Temperature: {temp}\n\nOriginal sentence: {original_sentence}\nTranslation: {translation}\nTranslation tokens:{tokens}\nLogit: {logit:.3f}")

temp = 1.0
original_sentence = "I student"

translation, logit, tokens = translate(trained_translator, original_sentence, temperature=temp)
print(f"Temperature: {temp}\n\nOriginal sentence: {original_sentence}\nTranslation: {translation}\nTranslation tokens:{tokens}\nLogit: {logit:.3f}")

TREN_unittest.test_translate(translate, trained_translator)