# Word Prediction using Language Models

## Introduction
Predicting the next word in a sentence is a fundamental Natural Language Processing (NLP) task with various applications such as text autocompletion, spelling correction, and search engine optimization. This project explores different language model architectures for word prediction, focusing on:
- N-gram Language Models
- Long Short-Term Memory (LSTM) Networks
- Generative Pre-trained Transformer (GPT-2)

## Dataset
We used text data from **Project Gutenberg**, specifically:
- *The Republic of Plato* (1,427,843 characters)
- *The Complete Works of William Shakespeare* (5,535,477 characters)

The data was preprocessed as follows:
1. Tokenization using NLTK.
2. Cleaning by removing punctuations and special characters.
3. Lowercasing all text.
4. Filtering sentences to keep those with word counts between 5 and 100.
5. Adding start (`<s>`) and end (`</s>`) tokens.
6. Splitting the data into 80% training and 20% testing sets.

## Model Architectures
### 1. N-gram Language Model
- Implemented for n = 2, 3, and 5.
- Replaced low-frequency words with `<UNK>`.
- Applied optional smoothing.

### 2. LSTM-based Model
- Implemented using TensorFlow/Keras.
- Used a Stacked LSTM with Bidirectional LSTM layers:
  - Input Data → Embedding Layer → Bidirectional LSTM → LSTM → Dense Layer → Output
- Different sequence lengths (2, 3, 5) were tested.
- Softmax function was used for probability prediction.
- Trainable parameters: 353,166.

### 3. GPT-2 Fine-Tuned Model
- Implemented using OpenAI's GPT-2 via the Hugging Face library.
- Fine-tuned on our dataset to improve next-word prediction.
- Used Byte-Pair Encoding for tokenization.
- Evaluated using different decoding techniques:
  - Greedy Search
  - Beam Search
  - Random Sampling

## Evaluation
We used **perplexity** as the evaluation metric:
\[
PL(x) = \exp(-\frac{1}{T} \sum \log P(x_i | x_{<i}))
\]
- Lower perplexity indicates better model performance.
- Key results:
  - GPT-2 fine-tuned model performed best, especially on *The Republic of Plato* dataset.
  - LSTM outperformed the n-gram model in both training speed and accuracy.
  - Increasing sequence length improved performance across all models.

## Future Improvements
- Train models on variable-length sequences to capture more context.
- Utilize multi-core GPUs for faster training and fine-tuning.
- Extend to sentence generation and auto-completion.
- Fine-tune GPT-2 on user-specific datasets for personalized word prediction.
