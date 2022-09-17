import sys
import re
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    Load the data from a database.

            Parameters:
                    database_filepath (str): File path of the database

            Returns:
                    X: Features
                    Y: Targets
                    category_names: Category names
    '''  
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con=engine)
    X = df.loc[:, 'message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenize the input text.

            Parameters:
                    text (str): Input text

            Returns:
                    clean_tokens (list): A list of clean tokens
    '''      
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build pipeline and grid search for the best model.

            Parameters:
                    None

            Returns:
                    cv: GridSearch object
    '''   
    pipeline = pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    parameters = {
        'clf__n_estimators': [10],
        'clf__min_samples_split': [2],
        #'clf__max_features': ['sqrt', 'log2'],
        #'clf__criterion' :['gini', 'entropy']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evalue the model.

            Parameters:
                     model: A sklearn model
                     X_test: Features in the test set
                     Y_test: Targets in the test set
                     category_name: Category names of the targets (multi-output)

            Returns:
                    cv: GridSearch object
    '''   
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    print(f"Best parameters: {model.best_params_}")
    print('overall performance:')
    print(classification_report(
            np.hstack(Y_test.values), np.hstack(Y_pred.values)))
    for col in category_names:
        print(f'classification report for {col} is:')
        print(classification_report(
            np.hstack(Y_test[col].values), np.hstack(Y_pred[col].values)))


def save_model(model, model_filepath):
    '''
    Save model in to pickle file.

            Parameters:
                     model: A sklearn model
                     model_filepath: File path where to save the model

            Returns:
                    None
    '''   
    
    joblib.dump(model.best_estimator_, model_filepath, compress=9)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()