import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.validation import check_is_fitted
import pickle
from typing import List, Tuple, Optional, Union    
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

np.random.seed(42)

# ========================================================================
def prepare_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Merge two subsets, and split into three sets
    Returns:
        Train, test, and validation set
    '''
    true_data = pd.read_csv('./DataSet_Misinfo_TRUE.csv')
    true_data['labels'] = 1  # Assign label 1 for true data

    fake_data = pd.read_csv('./DataSet_Misinfo_FAKE.csv')
    fake_data['labels'] = 0  # Assign label 0 for fake data

    full_data = pd.concat([true_data, fake_data], ignore_index=True)
    if 'Unnamed: 0' in full_data.columns:
        full_data.drop(columns=['Unnamed: 0'], inplace=True)

    full_data.drop_duplicates(inplace=True)
    full_data.dropna(subset=["text"], inplace=True)

    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42, stratify=train_data["labels"])  # 20% of the original data for validation

    return train_data, test_data, val_data


def process_text(text:str) -> List[str]:
    """
    Preprocess the single text data for machine learning or deep learning models.
    Args:
        text (str): The input text to preprocess.
    Returns:
        list: A list of preprocessed tokens.
    """
    if not isinstance(text, str):
        text = str(text)  # Ensure text is a string
    text = text.lower()  # 1. Lowercase
    text = re.sub(r'\d+', '', text)  # 2. Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # 3. Remove punctuation
    tokens = word_tokenize(text)  # 4. Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # 5. Stopword removal
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # 6. Lemmatization

    return tokens

def preprocess_data(data: pd.Series, merge:bool = True) -> Union[List[str], List[List[str]]]:
    """
    Preprocess the text data in the DataFrame.
    Args:
        data: The input is the list text data
    Returns:
        List of List of string or List of string: the token or sentences
    # """
    # if not isinstance(data, pd.Series):
    #     raise ValueError("Input data must be a pandas Series")
    processed_data = []
    for text in tqdm(data, desc="Preprocessing texts"):
        processed_tokens = process_text(text)
        processed_data.append(' '.join(processed_tokens) if merge else processed_tokens)  # Join tokens back into a string
    return processed_data


def process_and_save_token(data: pd.Series, path: str) -> List[List[str]]:
    '''
    Save token
    '''
    if os.path.exists(path):
        with open(path, 'rb') as f:
            processed_texts = pickle.load(f)
    else:
        processed_texts = preprocess_data(data.tolist(), merge=False)
        with open(path, 'wb') as f:
            pickle.dump(processed_texts, f)

    return processed_texts


def process_for_ml(texts: List[str], vectorizer = None) -> Tuple[TfidfVectorizer, List[List[float]]] :
    """
    Embedding using TF-IDF
    Args:
        texts: The processed sentence
    Returns:
        vectorizer: The embedder
        embedding vector
    """

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1821, ngram_range=(1, 1))  # Limit to 1821 features
        vectorizer.fit(texts)

    
    matrix = vectorizer.transform(texts)
    return vectorizer, [row.toarray().flatten().tolist() for row in matrix]


def process_for_dl(texts, tokenizer=None):
    """
    Preprocess the text data for deep learning models.
    Args:
        texts (list): A list of text strings to preprocess.
        
    Returns:
        tuple: A tuple containing the tokenizer and padded sequences.
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=1821, oov_token='<OOV>')

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    processed_sequences = pad_sequences(sequences, maxlen=256, padding='post')

    return tokenizer, processed_sequences.tolist()


def main():
    train_data, test_data, val_data = prepare_dataset()

    # preprocess
    train_texts = process_and_save_token(train_data['text'], path = './dataset/token/train.pkl')
    test_texts = process_and_save_token(test_data['text'], path = './dataset/token/test.pkl')
    val_texts = process_and_save_token(val_data['text'], path = './dataset/token/val.pkl')

    # convert list of string to list 
    train_texts = [' '.join(text) for text in train_texts]
    val_texts = [' '.join(text) for text in val_texts]
    test_texts = [' '.join(text) for text in test_texts]

    # Preprocess for ml
    vectorizer, ml_train_vectors = process_for_ml(train_texts)

    

    _, ml_test_vectors = process_for_ml(test_texts, vectorizer)
    _, ml_val_vectors = process_for_ml(val_texts, vectorizer)

    # save data
    with open('./dataset/ml/train_text.pkl', 'wb') as f:
        pickle.dump(ml_train_vectors, f)
    with open('./dataset/ml/val_text.pkl', 'wb') as f:
        pickle.dump(ml_val_vectors, f)
    with open('./dataset/ml/test_text.pkl', 'wb') as f:
        pickle.dump(ml_test_vectors, f)
    with open('./dataset/ml/train_labels.pkl', 'wb') as f:
        pickle.dump(train_data['labels'].values, f)
    with open('./dataset/ml/val_labels.pkl', 'wb') as f:
        pickle.dump(val_data['labels'].values, f)
    with open('./dataset/ml/test_labels.pkl', 'wb') as f:
        pickle.dump(test_data['labels'].values, f)

    # save vectorizer
    with open("./vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


    # # processed for dl
    # tokenizer, dl_train_sequences = process_for_dl(train_texts)
    # _, dl_val_sequences = process_for_dl(val_texts, tokenizer)
    # _, dl_test_sequences = process_for_dl(test_texts, tokenizer)

    # with open('./dataset/dl/train_text.pkl', 'wb') as f:
    #     pickle.dump(dl_train_sequences, f)
    # with open('./dataset/dl/val_text.pkl', 'wb') as f:
    #     pickle.dump(dl_val_sequences, f)
    # with open('./dataset/dl/test_text.pkl', 'wb') as f:
    #     pickle.dump(dl_test_sequences, f)
    # with open('./dataset/dl/train_labels.pkl', 'wb') as f:
    #     pickle.dump(train_data['labels'].values, f)
    # with open('./dataset/dl/val_labels.pkl', 'wb') as f:
    #     pickle.dump(val_data['labels'].values, f)
    # with open('./dataset/dl/test_labels.pkl', 'wb') as f:
    #     pickle.dump(test_data['labels'].values, f)

    # # save tokenizer
    # with open("./tokenizer.pkl", "wb") as f:
    #     pickle.dump(tokenizer, f)

if __name__ == "__main__":
    main()