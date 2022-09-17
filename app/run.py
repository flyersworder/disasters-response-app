import re
import json
import plotly
import pandas as pd
import swifter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.express as px
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
    tokens = [w for w in tokens if w.lower() not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    
    # genre counts
    genre_counts = pd.DataFrame(df.groupby('genre').count()['message']).reset_index()
    genre_counts.rename(columns={'message':'count'}, inplace=True)
    
    fig_genre = px.bar(genre_counts, x="genre", y="count", title='Distribution of Message Genres')

    # word frequency
    words = df['message'].swifter.apply(lambda x: tokenize(x))
    freq = pd.DataFrame(words.astype(str).str[1:-1].str.split(',', expand=True).stack(
        ).value_counts()).reset_index().head(20)
    freq.columns = ['word', 'count']

    fig_freq = px.bar(freq, x='word', y='count', title='Top 20 most frequent words')

    # category frequency
    categories = pd.DataFrame(df[df.columns[4:]].sum().sort_values()).reset_index()
    categories.columns = ['category', 'count']

    fig_cat = px.bar(categories, x='count', y='category', orientation='h', 
        width=1000, height=800, title='Category appearance (=1) frequency')

    # create visuals
    graphs = [fig_genre.to_dict(), fig_freq.to_dict(), fig_cat.to_dict()]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()