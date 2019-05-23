from flask import Flask, render_template
from data import Articles
#import codebase
from flask_jsonpify import jsonpify
import pdb
from ml_final import *

app = Flask(__name__)

Articles = Articles()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/articles')
def articles():
    return render_template("articles.html", articles = Articles)

#http://127.0.0.1:5000/recommendations/a7mTbEi2N8Zd-r-8jlReww
@app.route('/recommendations')
def closest_businesses_to(business = None, user = None, df = None):
    if business is not None:
        target = bus_values[bus_fmap[business]]
    if user is not None:
        target = user_values[u_fmap[user]]
    if df is None:
        df = bus_values
    best_restaurants = np.square(df - target[None,:]).sum(1).argsort()
    return best_restaurants
#
# @app.route('/recommendations/<business_id>')
# def recommendations(business_id):
#     print(business_id)
#     df = get_recommendations_for(business_id)
#     return render_template('recommendations.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
# @app.route('/recommendations/<business_id>')
# def get_recommendations_for(user_id = None, business_id = None):
#     if user_id is not None:
#         bids = closest_businesses_to(user = user_id)
#     else:
#         bids = closest_businesses_to(business = business_id)
#     bnames = [bus_invmap[b] for b in bids]
#     return restaurants_df.set_index('business_id').loc[
#         [b for b in bnames if b in restaurants_df['business_id'].values]]#.dropna()

if __name__ == '__main__':
    app.run(debug=True)
