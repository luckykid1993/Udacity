import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from train_classifier import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterClean.db')
df = pd.read_sql('SELECT * FROM Msg_Category_Tbl', engine)

# load model
model = joblib.load('../models/classifier.pkl')

def clean_name(name):
    return name.replace('_', ' ').capitalize()
    
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # Distribution of messages/category
    msg_counts = Y.sum().sort_values(ascending=False)
    ctgr_names = list(map(clean_name, list(msg_counts.index)))
   
    # Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(map(clean_name, list(genre_counts.index)))
    
    # Top 10 responses
    top_10_counts = Y.sum(axis=0).sort_values(ascending=False)[0:10]
    top_10_names = list(map(clean_name, list(top_10_counts.index)))
    
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=ctgr_names,
                    y=msg_counts,
                )
            ],
           
            'layout': {
                'title': 'Distribution of Messages per Category',
                'yaxis': {
                    'title': "Number of Messages"
                    
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -30
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_10_names,
                    y=top_10_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Responses',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Response"
                }
            }
        }
    ]
    
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