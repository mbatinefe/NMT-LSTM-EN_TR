# NMT-LSTM-EN_TR

This model is acquired and modified from the DeepLearning.AI - Week 1 (Neural Machine Translation) course assignment-1.

## Features

- **Encoder-Decoder Architecture**: Utilizes LSTM networks for both the encoder and decoder components.
- **Attention Mechanism**: Incorporates attention to handle long sentences more effectively.
- **Decoding Strategies**:
  - *Greedy Decoding*: Selects the most probable word at each step.
  - *Minimum Bayes Risk (MBR) Decoding*: Aims to minimize expected loss, considering multiple translation hypotheses.

## Usage

### 1. Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/mbatinefe/NMT-LSTM-EN_TR.git
cd NMT-LSTM-EN_TR
```

### 2. Prepare the Dataset
- The dataset should include parallel English-Turkish sentence pairs.
- Preprocess and tokenize the data using the functions provided in utils.py.

### 3. Train the Model
Run the training script to build the encoder-decoder model with attention:
```bash
python translator_EN-TR.py
```
### 4. Evaluate the model
```bash
python TR-EN_unittest.py
```
### 5. Generate Translations
```bash
python translate_sentence.py --sentence "Can I become a good researcher?"
```

## RESULTS

### Translation Example

**English (to translate) sentence:**
You have the same camera as mine.

**Turkish (translation) sentence:**
Sen benimki ile aynı kameraya sahipsin.

### Vocabulary

**First 10 words of the English vocabulary:**
['', '[UNK]', '[SOS]', '[EOS]', '.', 'the', 'you', 'to', 'is', 'a']

**First 10 words of the Turkish vocabulary:**
['', '[UNK]', '[SOS]', '[EOS]', '.', 'bir', '?', 'bu', ',', 'o']

**Turkish vocabulary contains:** 12,000 words.
- The id for the [UNK] token is 1.
- The id for the [SOS] token is 2.
- The id for the [EOS] token is 3.
- The id for "baunilha" (vanilla) is 1.

### Tokenization

**Tokenized English sentence:**
[   2   94   26  681 1273   40 4567   15   32 4744 4751   11   3    0    0    0    0    0    0]

**Tokenized Turkish sentence (shifted to the right):**
[   2  250   72 1974   38    1    8  201   76  151 9880    6    0    0    0]

**Tokenized Turkish sentence:**
[ 250   72 1974   38    1    8  201   76  151 9880    6    3    0    0    0]

### Model Parameters

**Encoder output shape:** (64, 19, 256)  
**Decoder output shape:** (64, 15, 256)  
**Attention scores shape:** (64, 15, 256)  

### Model Summary

| Layer            | Output Shape     | Parameters |
|-------------------|------------------|------------|
| Encoder           | (multiple)       | 4,122,624  |
| Decoder           | (multiple)       | 7,470,304  |
| **Total params**  |                  | 11,592,928 |
| Trainable params  |                  | 11,592,928 |
| Non-trainable params |              | 0          |

---

### Samples

#### **Temperature: 0.0**
**Original sentence:** What's up bro  
**Translation:** gor kokteyl ayrlmak yapmadn uretildi parlyordu ...  

#### **Temperature: 0.3**
**Original sentence:** What's up bro  
**Translation:** uyuyamadn meshurdur tatsuyann ...  

#### **Temperature: 0.7**
**Original sentence:** What's up bro  
**Translation:** kulaga tadma elektrikle sorabilir olmayacak ...  

#### **Temperature: 1.0**
**Original sentence:** What's up bro  
**Translation:** yazmaktr dagtmak politikacdr yasarlar posterler ...



