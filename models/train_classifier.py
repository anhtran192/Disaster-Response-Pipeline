import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier


nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """Load data from SQLite database"""
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, Y

def tokenize(text):
    """Tokenize and lemmatize text"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

class WordEmbeddingsExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract word embeddings using Word2Vec"""
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        pass  # Implement this method based on your Word2Vec model

# def build_model():
#     """Build and optimize ML pipeline"""
#     pipeline = Pipeline([
#         ('text_pipeline', Pipeline([
#             ('vect', CountVectorizer(tokenizer=tokenize)),
#             ('tfidf', TfidfTransformer())
#         ])),
#         ('clf', MultiOutputClassifier(SVC()))  # Support Vector Machines
#     ])

#     parameters = {
#         'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
#         'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
#         'text_pipeline__vect__max_features': (None, 5000, 10000),
#         'text_pipeline__tfidf__use_idf': (True, False),
#         'clf__estimator__C': [1, 10, 100],
#         'clf__estimator__gamma': [0.1, 0.01, 0.001],
#         'clf__estimator__kernel': ['linear', 'rbf']
#     }

#     cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3)
#     return cv

def build_model():
    """Build and optimize ML pipeline"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 5, 10]
    }
    
    # create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv



def evaluate_model(model, X_test, Y_test):
    """Evaluate model performance"""
    Y_pred = model.predict(X_test)
    for i, column in enumerate(Y_test.columns):
        print('Category:', column)
        print(classification_report(Y_test[column], Y_pred[:, i]))

def save_model(model, model_filepath):
    """Save model as a pickle file"""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...')
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)
        
        print('Saving model...')
        save_model(model, model_filepath)
        
        print('Model saved!')
    
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. '\
              '\n\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
