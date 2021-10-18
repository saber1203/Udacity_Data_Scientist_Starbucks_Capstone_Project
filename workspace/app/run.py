import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import pickle


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/user_offer_matrix.db')
df = pd.read_sql_table('user_offer_matrix', engine)
portfolio=pd.read_pickle('data/portfolio.pkl')

# load model
model = pickle.load(open("models/recommendation.pkl", "rb"))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    offer_completed_counts = df.groupby('offer_id')['completed'].sum()
    offer_ids = list(offer_completed_counts.index)

    offer_completed_count=df.groupby('offer_id')['completed_count'].sum()
    offer_received_count=df.groupby('offer_id')['offer_id'].count()
    offer_completed_rate=offer_completed_count/offer_received_count
    offer_ids_rate = list(offer_completed_rate.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=offer_ids,
                    y=offer_completed_counts
                )
            ],

            'layout': {
                'title': 'Offer completed count by ids',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Offer id"
                }
            }
        },
        # GRAPH 2 - category graph    
        {
            'data': [
                Bar(
                    x=offer_ids_rate,
                    y=offer_completed_rate
                )
            ],

            'layout': {
                'title': 'Offer completed rate by ids',
                'yaxis': {
                    'title': "Completed Rate"
                },
                'xaxis': {
                    'title': "Offer id",
                    'tickangle': 35
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
    user_dict=eval(query)
    # use model to predict classification for query

    # create copy df from portfolio_new
    user_offer_info=portfolio.copy()

    # Assign the user info to the user_offer_info df
    for k,v in user_dict.items():
        user_offer_info.loc[:,k]=v

    # create X metric for prediction
    X=user_offer_info.drop(['offer_id'],axis=1)

    # Predict the value and assign the value to new completed column
    user_offer_info['pred_completed']=model.predict(X)
    user_offer_recs=user_offer_info[['offer_id','pred_completed']].sort_values(by='pred_completed',ascending=False)

    classification_labels = user_offer_recs['pred_completed']
    classification_results = dict(zip(user_offer_recs['offer_id'], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()