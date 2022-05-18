import sys

# disable warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'], quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, classification_report


def load_data(database_filepath, table_name = 'Msg_Category_Tbl'):
    """Loads data from SQLite
    inputs:
        database_filepath (string): sqlite database's filepath.
        table_name (string): table name, default 'Msg_Category_Tbl'
    outputs:
        X (dataframe): messages
        Y (dataframe): Classification labels
        category_names (list): List of the category names for classification
    """
    # load data from SQLite
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM {}'.format(table_name), engine)
    
    # messages
    X = df['message']
    
    # classification labels
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    
    # category names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """Process text messages: Lemmatize, lower, remove blank, remove stop words
    
    inputs:
        text (String): text messages
       
    outputs:
        clean_tokens (list): clean tokens
    """
    # get list of stopwords in english
    stop_words = stopwords.words("english")
    
    # tokenize words
    tokens = word_tokenize(text)
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    
    # result
    clean_tokens = []
    for token in tokens:
        # Lemmatize + lower + remove blank
        clean = lemmatizer.lemmatize(token).lower().strip()
        # do not add stop word to result
        if clean not in stop_words:
            clean_tokens.append(clean)
    
    return clean_tokens


def build_model(use_grid_search_cv = False):
    """create model (and using GridSearchCV for tunning)
    inputs:
        use_grid_search_cv (string): using GridSearchCV for tunning, default = False
    outputs:
        cv (scikit-learn GridSearchCV): Grid search model object if use_grid_search_cv = True
        pipe (scikit-learn pipline): pipline if use_grid_search_cv = False
    """
    # create pipline with TF-IDF + AdaBoost
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    if use_grid_search_cv == 'True':
        parameters =  {'vect__ngram_range': ((1, 1), (1, 2)),
                      'clf__estimator__learning_rate': [0.5, 1.0, 1.5, 2.0],
                      'clf__estimator__n_estimators':[25, 50, 75]}
        # grid search
        cv = GridSearchCV(estimator=pipe, param_grid=parameters, scoring='f1_weighted', verbose=3, cv=3)
        
        return cv
    else:
        return pipe

def evaluate_model(model, X_test, Y_test, category_names):
    """test model with test data and print classification_report result
    inputs:
        model: the scikit-learn fitted model
        X_text (dataframe): input test set
        Y_test (dataframe): output test set
        category_names (list): the category names for classification
    """
    # using model to predict
    y_pred = model.predict(X_test)

    # print report
    report = classification_report(Y_test, y_pred, target_names=category_names)
    print(report)
    
    # because in this case (disaster) False Negative is extremely important 
    # eg: Someone actually needed help, but we predict false
    # so, F1-score is a better metric to evaluate model
    # comment  line 117 and uncomment these lines below to output F1-score only
    #print('\n\n==============================================================')
    #for i in range(Y_test.shape[1]):
    #    print('F1-score of %35s: %.2f' %(category_names[i], f1_score(Y_test.iloc[:, i].values, y_pred[:,i])))



def save_model(model, model_filepath):
    """save the model to disk
    inputs:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model
    """
    try:
        pickle.dump(model, open(model_filepath, 'wb'))
        print('Trained model saved!')
    except Exception as e:
        print('Failed to save model:')
        # print exception
        print(e)


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, use_grid_search_cv = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(use_grid_search_cv)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath) 
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterClean.db classifier.pkl False')


if __name__ == '__main__':
    main()
