from flask import Flask, request, jsonify
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Always load files relative to THIS file location (app/ folder)
BASE_DIR = Path(__file__).resolve().parent

PRED_PATH = BASE_DIR / "pred_df.parquet"
MOVIE_MAP_PATH = BASE_DIR / "movie_map.parquet"

# Load artifacts (generated via notebooks)
pred_df = pd.read_parquet(PRED_PATH)
movie_map = pd.read_parquet(MOVIE_MAP_PATH)

# Ensure movie_id is numeric for clean joins/lookup
if "movie_id" in movie_map.columns:
    movie_map["movie_id"] = pd.to_numeric(movie_map["movie_id"], errors="coerce")

@app.route("/")
def home():
    return jsonify({
        "status": "Movie Recommendation API is running",
        "endpoints": ["/recommend?user_id=1&k=5"]
    })

@app.route("/recommend", methods=["GET"])
def recommend():
    # Inputs
    user_id = request.args.get("user_id", default=None, type=int)
    k = request.args.get("k", default=5, type=int)

    if user_id is None:
        return jsonify({"error": "Please provide user_id, e.g. /recommend?user_id=1"}), 400

    if k <= 0 or k > 50:
        return jsonify({"error": "k must be between 1 and 50"}), 400

    # --- Case A: pred_df is LONG format (user_id, movie_id, predicted_rating) ---
    if {"user_id", "movie_id", "predicted_rating"}.issubset(set(pred_df.columns)):
        user_rows = pred_df[pred_df["user_id"] == user_id]
        if user_rows.empty:
            return jsonify({"error": f"user_id {user_id} not found in pred_df"}), 404

        top_recs = (
            user_rows.merge(movie_map, on="movie_id", how="left")
            .sort_values("predicted_rating", ascending=False)
            .head(k)
        )

        out = top_recs[["movie_id", "title", "predicted_rating"]].to_dict(orient="records")
        return jsonify(out)

    # --- Case B: pred_df is MATRIX format (index=user_id, columns=movie_id) ---
    # This matches what your Notebook 04 screenshots suggest.
    if user_id not in pred_df.index:
        return jsonify({"error": f"user_id {user_id} not found in pred_df index"}), 404

    user_scores = pred_df.loc[user_id]

    # Make sure movie_id columns are numeric
    user_scores.index = pd.to_numeric(user_scores.index, errors="coerce")
    user_scores = user_scores.dropna()
    user_scores = user_scores.sort_values(ascending=False).head(k)

    recs = (
        pd.DataFrame({"movie_id": user_scores.index.astype(int), "predicted_rating": user_scores.values})
        .merge(movie_map, on="movie_id", how="left")
    )

    # If title missing, fill with placeholder
    if "title" not in recs.columns:
        recs["title"] = "Unknown title"

    out = recs[["movie_id", "title", "predicted_rating"]].to_dict(orient="records")
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
