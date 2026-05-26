import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_recommenders as tfrs
import requests

API_KEY = "656559fa611141eaa638d10f6df78248"
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

df = pd.merge(ratings,movies,on="movieId")
print(df.head())

movies_df = df[["movieId","title","genres"]].drop_duplicates()
movies_df = movies_df.reset_index(drop=True)
print(movies_df.head())

cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies_df["genres"])
similarity = cosine_similarity(genre_matrix)
movies_indices = pd.Series(movies_df.index, index = movies_df["title"])

def recommend_movies(movie_title):
    idx = movies_indices[movie_title]
    sim_score = list(enumerate(similarity[idx]))
    sim_score = sorted(sim_score,key=lambda x: x[1],reverse=True)
    sim_score = sim_score[1:6]
    movies_indices_list = [i[0] for i in sim_score]
    return movies_df["title"].iloc[movies_indices_list]

print(recommend_movies("Toy Story (1995)"))

movies_tf = tf.data.Dataset.from_tensor_slices(
    movies_df["title"].values
)

ratings_tf = tf.data.Dataset.from_tensor_slices({
    "userId": ratings["userId"].astype(str).values,
    "movie_title": df["title"].values,
    "rating": ratings["rating"].values
})

userIds = ratings["userId"].astype(str).unique()
movie_titles = movies["title"].unique()

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=userIds, mask_token=None),
    tf.keras.layers.Embedding(len(userIds)+1,32)
    ])

movie_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary= movie_titles, mask_token=None),
    tf.keras.layers.Embedding(len(movie_titles)+1,32)
])

rating_model = tf.keras.Sequential([
    tf.keras.layers.Dense(256,activation="relu"),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1)
])

task = tfrs.tasks.Retrieval(
    metrics= tfrs.metrics.FactorizedTopK(
        movies_tf.batch(128).map(movie_model)
    )
)
ranking_task = tfrs.tasks.Ranking(
    loss= tf.keras.losses.MeanSquaredError(),
    metrics= [tf.keras.metrics.RootMeanSquaredError()]
)

class MovieModel(tfrs.Model):
    def __init__(self, ):
        super().__init__()

        self.user_model = user_model
        self.movie_model = movie_model
        self.retrieval_task = task
        self.rating_model = rating_model
        self.rating_task = ranking_task
    
    def compute_loss(self, features, training = False):
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movie_title"])

        ranking_prediction = self.rating_model(
            tf.concat([user_embeddings,movie_embeddings],axis=1)
        )
        
        retrieval_loss = self.retrieval_task(
            user_embeddings,
            movie_embeddings
        )

        rating_loss = self.rating_task(
            labels = features["rating"],
            predictions = ranking_prediction
        )

        return retrieval_loss + rating_loss

model = MovieModel()
model.compile(
    optimizer= tf.keras.optimizers.Adagrad(0.1)
)


index = tfrs.layers.factorized_top_k.BruteForce(
    model.user_model
)

index.index_from_dataset(
    movies_tf.batch(100).map(
        lambda title :(
            title,
            model.movie_model(title)
        )
    )
)
scores,titles = index(
    tf.constant([userIds]))
print("Recommendation for user1:")
for movie in titles[0,:10].numpy():
    print(movie.decode("utf-8")if isinstance(movie, bytes) else movie)



def predict_movie_rating(userID,movie_title):
    user_tensor = tf.constant([str(userID)])
    movie_tensor = tf.constant([movie_title])

    user_embedding = model.user_model(user_tensor)
    movie_embedding = model.movie_model(movie_tensor)

    predicted_rating = model.rating_model(
        tf.concat([user_embedding,movie_embedding],axis=1)
    )

    return predicted_rating.numpy()[0][0]

def fetch_movie_poster(movie_name):

    try:

        clean_name = movie_name.split("(")[0].replace(",", "").strip()

        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={clean_name}"

        response = requests.get(url)

        data = response.json()

        print(data)

        if data.get("results"):

            poster_path = data["results"][0].get("poster_path")

            print("Poster Path:", poster_path)

            if poster_path:

                full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"

                print("Full URL:", full_path)

                return full_path

        return "https://via.placeholder.com/300x450.png?text=No+Poster"

    except Exception as e:

        print("Poster Error:", e)

        return "https://via.placeholder.com/300x450.png?text=Error"

def hybrid_recommendation(userID,movie_title):
    print(f"\n Because you liked :{movie_title}\n")
    content_recom = recommend_movies(movie_title)
    print("Content Based Recommendation:\n")

    for movie in content_recom.values:
        print(movie)
    
    print("\nDeep Learning Recommendations:\n")
    scores,titles = index( 
        tf.constant([str(userID)]))
    
    recommendation_list = []

    for movie in titles[0, :10].numpy():

        movie_name = movie.decode("utf-8") if isinstance(movie, bytes) else movie

        predicted_rating = predict_movie_rating(userID, movie_name)

    poster = fetch_movie_poster(movie_name)

    recommendation_list.append({
        "title": movie_name,
        "rating": round(predicted_rating,2),
        "poster": poster
})
       

    return recommendation_list


if __name__ == "__main__":

    catched_train = ratings_tf.shuffle(100000).batch(8192)

    model.fit(catched_train, epochs=5)

    try:
        model.load_weights("movie_model_weights")
        print("Saved weights loaded successfully")
    except:
        print("No saved weights found")
    r_id = input("Enter User ID : ")
    movie_name = input("Enter Movie Name : ")

    recommendations = hybrid_recommendation(r_id, movie_name)

    for item in recommendations:
        print(item)