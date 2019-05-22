from flask import Flask, render_template
from data import Articles
#import codebase
from flask_jsonpify import jsonpify
import pdb
# from ml_final import *

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

@app.route('/article/<id>/')
def article(id):
    return render_template("article.html", id=id)

#http://127.0.0.1:5000/recommendations/a7mTbEi2N8Zd-r-8jlReww
@app.route('/recommendations/<business_id>')
def recommendations(business_id):
    print(business_id)
    df = get_recommendations_for(business_id)
    return render_template('recommendations.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)





if __name__ == '__main__':
    app.run(debug=True)
