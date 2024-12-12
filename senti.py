import os
import re
import string
import logging
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib


def output_logger(log_file: str = 'sentimentOutput.log') -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def check_nltk() -> None:
    nlkt_resources = ['vader_lexicon', 'punkt_tab', 'stopwords', 'wordnet']
    for resource in nlkt_resources:
        try:
            nltk.data.find(f'corpora/{resource}')
            logging.info(f"NLTK resource '{resource}' already exists.")
        except LookupError:
            logging.info(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> List[str]:
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)      
    text = re.sub(r'\S+@\S+', '', text)               
    text = re.sub(r'<.*?>', '', text)                  
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)                    
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens
def load_glove_embeddings(cache_file: str = 'glove_cache.pkl') -> Dict[str, np.ndarray]:
    if os.path.exists(cache_file):
        logging.info(f"Loading GloVe embeddings from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                embeddings_index = pickle.load(f)
            logging.info(f"Successfully loaded {len(embeddings_index)} word vectors from cache.")
        except Exception as e:
            logging.error(f"Failed to load cache file {cache_file}: {e}")
            raise
    else:
        logging.error(f"Cache file not found at '{cache_file}'. Please ensure the cache exists.")
        raise FileNotFoundError(f"Cache file '{cache_file}' does not exist.")
    return embeddings_index


def get_document_embedding_glove(tokens: List[str], embeddings: Dict[str, np.ndarray], vector_size: int = 300) -> np.ndarray:
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    logging.info(f"\n{model_name} Classification Report:")
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    logging.info(report)


def run_pipeline(data_path: str, glove_cache_path: str, output_dir: str = 'outputs') -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_logger(os.path.join(output_dir, 'sentimentOutput.log'))
    check_nltk()
    logging.info("Sentiment Analysis Pipeline Started.")
    sia = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    logging.info("Loading dataset...")
    data = pd.read_excel(data_path)
    data = data.dropna(subset=['review_text'])
    logging.info(f"After dropping missing 'review_text', {data.shape[0]} samples remain.")

    data['sentiment'] = ['positive' if r >= 7 else 'negative' for r in data['rating']]
    y = data['sentiment']
    X = data['review_text']

    logging.info("Preprocessing text data...")
    X_tokens = X.apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
    logging.info("Text preprocessing completed.")

    embeddings_index = load_glove_embeddings(cache_file=glove_cache_path)

    logging.info("Creating document embeddings...")
    X_embedded = X_tokens.apply(lambda tokens: get_document_embedding_glove(tokens, embeddings_index))
    X_embedded = np.vstack(X_embedded.values)
    logging.info(f"Document embeddings created with shape: {X_embedded.shape}")

    logging.info("Generating VADER sentiment scores...")
    vader_scores = X.apply(lambda x: sia.polarity_scores(x))
    vader_scores_df = pd.DataFrame(list(vader_scores))
    X_vader = vader_scores_df[['neg', 'neu', 'pos', 'compound']].values
    logging.info("VADER sentiment scores generated.")

    logging.info("Integrating VADER scores with GloVe embeddings...")
    X_final = np.hstack((X_embedded, X_vader))
    logging.info(f"Final feature set shape before scaling: {X_final.shape}")

    logging.info("Applying StandardScaler to features...")
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_final)
    logging.info("Feature scaling applied.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logging.info("Labels encoded.")

    logging.info("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logging.info(f"Training set size: {X_train.shape[0]}")
    logging.info(f"Testing set size: {X_test.shape[0]}")

    logging.info("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logging.info(f"After SMOTE: {X_train.shape[0]} training samples")

    models = {}
    
    models['LightGBM'] = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, max_depth=20, n_estimators=200, num_leaves=50, class_weight='balanced', random_state=42, verbose=-1)
    
    models['CatBoost'] = CatBoostClassifier(iterations=200, depth=10, learning_rate=0.1, l2_leaf_reg=3, class_weights=[1.0, 1.0], random_seed=42, verbose=0)
    
    models['XGBoost'] = XGBClassifier(scale_pos_weight=1.0, n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',random_state=4)
    
    models['BalancedRandomForest'] = BalancedRandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42)
    
    estimators = [('lgbm', models['LightGBM']), ('catboost', models['CatBoost'])]

    models['StackingClassifier'] = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)

    for model_name, model in models.items():
        logging.info(f"\nProcessing {model_name}...")

        logging.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        logging.info(f"{model_name} training completed.")

        logging.info(f"Evaluating {model_name}...")
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.decision_function(X_test) if hasattr(model, 'decision_function') else y_pred
        evaluate_model(y_test, y_pred, model_name)

        logging.info(f"Performing cross-validation for {model_name}...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_final, y_encoded, cv=skf, scoring='f1_macro', n_jobs=-1
        )
        logging.info(f"{model_name} Cross-Validated F1-Macro Scores: {cv_scores}")
        logging.info(f"Average F1-Macro Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    logging.info("\nAll processes completed successfully!")


if __name__ == "__main__":
    data_path = 'data/data.xlsx'          
    glove_cache_path = 'glove_cache.pkl' 
    output_dir = 'outputs'                
    os.makedirs(output_dir, exist_ok=True)
    output_logger(os.path.join(output_dir, 'sentimentOutput.log'))
    check_nltk()
    run_pipeline(data_path, glove_cache_path, output_dir)
