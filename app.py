from flask import Flask, render_template , request
from main import hybrid_recommendation, model

app = Flask(__name__)

@app.route("/", methods = ["GET","POST"])
def home():
    recommendations = []

    if request.method=="POST":
        user_id = request.form["userID"]
        movie_name = request.form["movie_name"]

        recommendations = hybrid_recommendation(user_id,movie_name)

    return render_template(
       "index.html",
       recommendations = recommendations
    )
model.load_weights("movie_model_weights")

if __name__== "__main__":
    app.run(debug=False)

