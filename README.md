
Fake-News-Detection
===
# Dataset 

Data was collected from a variety of online news such as Reuters, The New York Times, The Washington Post, etc. Link to dataset is [here](https://drive.google.com/drive/folders/1mrX3vPKhEzxG96OCPpCeh9F8m_QKCM4z)

# Pre-processing
Make sure you have download the raw data, then use `os.getcwd()` to move to the current directory and run `preprocess.py` to process raw data. The processed will be saved in dataset.

# Methodology

We decided to do three kinds of model for detection task:
- Large Language Models: **BERT** (baseline) and **XLNet**
- Traditional Machine Learning: **Support Vector Machine** and **Logistic Regression**
- Deep Learning: **LSTM** and **GRU**

Each approach follows a distinct pipeline, enabling diverse methodologies and offering comparable insights across models. Most of models are trained on Nvidia Tesla 4.

## LLMs Pipeline
The pipeline includes: Fine-tuning (that's all 😁)

As LLMs have been pretrained, we didn't perform a pre-processed pipeline. Instead, we were putting raw data through the tokenizer and configuring parameters for fine-tuning on the dataset.

Models used in the experiments:

**bert-base-uncased:** 12 layers, 768 hidden size, 12 attention heads, ~110M parameters. Pretrained on lower-cased English.

**xlnet-base-cased:** 12 layers, 768 hidden size, 12 attention heads, ~117M parameters. Pretrained on cased English using permutation-based autoregressive modeling.

The parameters of BERT and XLNet is shown below:
|`epochs`|`learning_rate`|`max_length`|`batch_size`|
|--|--|--|--|
| 3 | 2e-5 | 64 | 8 |

## DL Pipeline
Pipeline of Deep Learning includes: Pre-processing -> Embedding -> Training -> Evaluation

In the pre-processing stage, several standard text normalization techniques are applied to clean the raw input text. These include:

- Converting all text to **lowercase** to ensure case insensitivity.
- **Removing numerical values** and **punctuation** to reduce noise.
- **Eliminating stop words** to retain only meaningful content.
- Performing **lemmatization** to reduce words to their base or dictionary form.

After cleaning, the text is tokenized using **TensorFlow's `Tokenizer`**, which is configured with `num_words=1821` to limit the vocabulary to the most frequent 1,821 tokens. This helps reduce dimensionality and mitigate overfitting. The resulting sequences are then **padded or truncated** to a fixed length of `maxlen=256` to ensure consistent input size for the neural network model. The parameters of GRU and LSTM is shown below:
||`epochs`|`learning_rate`|`input_dim`|`embedding_dim`|`hidden_dim`|
|--|--|--|--|--|--|
| LSTM | 10 | 0.0005 | 1821 | 128 | 256 |
| GRU | 8 | 0.0005 | 1821 | 128 | 512 |



## Traditional ML Pipeline
The pre-processing phase follows the same steps as used in the deep learning pipeline, including lowercasing, removing numbers and punctuation, eliminating stop words, and applying lemmatization.  After pre-processing, instead of using neural embeddings, the cleaned text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization from **scikit-learn**. The `TfidfVectorizer` is configured with `max_features=1821` to limit the vocabulary size to the 1,821 most informative terms, ensuring efficient and meaningful feature representation for classical machine learning models.



# Results
We used 4 metrics to benchmark our models: Accuracy, F1-Score, Precision, Recall. The result was given by below table:
|  | Accuracy | F1|Precision|Recall|
|--|--|--|--|--|
| BERT |  |
| XLNet |  |
| LSTM |  |
| GRU |  |
| SVM |  |
| Logistic Regression |  |
