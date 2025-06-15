# --------------------------  LIBRARIES  -----------------------------
# --------------------------------------------------------------------

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sentence_transformers import SentenceTransformer, util

# -----------------------  IMPORTING DATA  ---------------------------
# --------------------------------------------------------------------

# Load ratings data
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

# Load movie item data
items = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    names=["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL"] +
          [f"genre_{i}" for i in range(19)],
    usecols=range(24)
)

df = pd.merge(ratings, items, left_on="item_id", right_on="movie_id")

# ------------------------  NON-PERSONALIZED  ------------------------
# --------------------------------------------------------------------

def get_top_non_personalized(strategy="weighted"):
    """
    Returns top 10 non-personalized movie recommendations using a fixed strategy.
    Strategy options: 'editorial', 'top_n', 'average_rating', 'weighted', 'association'.
    """

    # Compute stats
    movie_counts = df.groupby('movie_id')['rating'].count()
    movie_means = df.groupby('movie_id')['rating'].mean()

    movie_stats = pd.DataFrame({
        "rating_count": movie_counts,
        "average_rating": movie_means
    })

    movie_stats = movie_stats.merge(items[["movie_id", "movie_title"]], on="movie_id")
    movie_stats.set_index("movie_id", inplace=True)

    if strategy == "editorial":
        # Simulated editorial list (hardcoded IDs)
        editorial_ids = [50, 172, 181, 100, 258, 1, 121, 174, 127, 7]
        editorial = movie_stats.loc[movie_stats.index.intersection(editorial_ids)]
        editorial = editorial[["movie_title"]].reset_index()
        editorial.index += 1
        return editorial[["movie_title"]].head(10)

    elif strategy == "top_n":
        result = movie_stats.sort_values(by="rating_count", ascending=False).head(10).reset_index()
        result.index += 1
        return result[["movie_title"]]

    elif strategy == "average_rating":
        filtered = movie_stats[movie_stats["rating_count"] >= 100]
        result = filtered.sort_values(by="average_rating", ascending=False).head(10).reset_index()
        result.index += 1
        return result[["movie_title"]]

    elif strategy == "weighted":
        C = movie_stats["average_rating"].mean()
        m = 100
        movie_stats["weighted_score"] = (
            (movie_stats["rating_count"] / (movie_stats["rating_count"] + m)) * movie_stats["average_rating"]
            + (m / (movie_stats["rating_count"] + m)) * C
        )
        result = movie_stats.sort_values(by="weighted_score", ascending=False).head(10).reset_index()
        result.index += 1
        return result[["movie_title"]]

    elif strategy == "association":
        from collections import Counter
        top_users = df[df["rating"] >= 4]["user_id"].value_counts().head(100).index
        high_rated = df[(df["user_id"].isin(top_users)) & (df["rating"] >= 4)]
        top_items = Counter(high_rated["item_id"]).most_common(10)
        top_item_ids = [i[0] for i in top_items]
        assoc = movie_stats.loc[movie_stats.index.intersection(top_item_ids)]
        assoc = assoc[["movie_title"]].reset_index()
        assoc.index += 1
        return assoc[["movie_title"]]

    else:
        raise ValueError("Invalid strategy. Choose from: 'editorial', 'top_n', 'average_rating', 'weighted', 'association'")

# -----------------------  USER & ITEM-BASED  ------------------------
# --------------------------------------------------------------------

# Create user-item matrix (entire dataset for now)
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')

def recommend_user_based(user_id, num_recommendations=10):
    """
    Recommend movies using User-Based Collaborative Filtering.

    Returns a pandas DataFrame with movie_title and estimated score.
    """
    # Fill missing with 0 for similarity calc
    user_similarity = cosine_similarity(user_item_matrix.fillna(0))
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Get most similar users (excluding self)
    similar_users = user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)

    weighted_ratings = pd.Series(dtype=float)

    for other_user, sim in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        weighted = other_ratings * sim
        weighted_ratings = weighted_ratings.add(weighted, fill_value=0)

    # Remove movies already rated by user
    seen_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].notna()].index
    recommendations = weighted_ratings.drop(index=seen_movies, errors='ignore')

    top_n = recommendations.sort_values(ascending=False).head(num_recommendations)
    titles = items.set_index("movie_id").loc[top_n.index]["movie_title"]

    return pd.DataFrame({
        "movie_title": titles.values,
    }, index=range(1, len(top_n) + 1))

def recommend_item_based(user_id, num_recommendations=10):
    """
    Recommend movies using Item-Based Collaborative Filtering.

    Returns a pandas DataFrame with movie_title and estimated score.
    """
    # Transpose to get item-user matrix
    item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    user_ratings = user_item_matrix.loc[user_id].dropna()
    weighted_scores = pd.Series(dtype=float)

    for item_id, rating in user_ratings.items():
        sim_scores = item_similarity_df[item_id] * rating
        weighted_scores = weighted_scores.add(sim_scores, fill_value=0)

    # Remove seen items
    weighted_scores = weighted_scores.drop(index=user_ratings.index, errors='ignore')

    top_n = weighted_scores.sort_values(ascending=False).head(num_recommendations)
    titles = items.set_index("movie_id").loc[top_n.index]["movie_title"]

    return pd.DataFrame({
        "movie_title": titles.values,
    }, index=range(1, len(top_n) + 1))

# -------------------------  CONTENT-BASED  --------------------------
# --------------------------------------------------------------------

def recommend_content_based(user_id, num_recommendations=10):
    """
    Recommends movies using content-based filtering with best practices.
    Uses genre taxonomy, weighted user profile, cosine similarity.
    Returns a top-N DataFrame with 1-based indexing.
    """

    # --- Setup movie feature matrix ---
    movie_features = items.set_index("movie_id")[["movie_title"] + [f"genre_{i}" for i in range(19)]]
    movie_features = movie_features.drop_duplicates(subset="movie_title")

    # --- Get user ratings ---
    user_ratings = df[df["user_id"] == user_id][["item_id", "rating"]]
    rated_movies = pd.merge(user_ratings, movie_features, left_on="item_id", right_index=True)

    if rated_movies.empty:
        return pd.DataFrame({"movie_title": ["No ratings by user"], "similarity_score": [None]})

    # --- Build weighted user profile (vector of genre preferences) ---
    genre_matrix = rated_movies[[f"genre_{i}" for i in range(19)]]
    weighted_genres = genre_matrix.T.dot(rated_movies["rating"])
    user_profile = weighted_genres.values.reshape(1, -1)

    # --- Normalize user profile vector ---
    user_profile = normalize(user_profile)

    # --- Prepare unseen movies ---
    seen_ids = user_ratings["item_id"].tolist()
    unseen_movies = movie_features[~movie_features.index.isin(seen_ids)]
    unseen_features = unseen_movies[[f"genre_{i}" for i in range(19)]].values

    # --- Normalize item genre vectors ---
    unseen_features = normalize(unseen_features)

    # --- Compute cosine similarity ---
    similarities = cosine_similarity(user_profile, unseen_features)[0]
    unseen_movies = unseen_movies.copy()
    unseen_movies["similarity_score"] = similarities

    # --- Return top N recommendations ---
    top_n = unseen_movies.sort_values(by="similarity_score", ascending=False).head(num_recommendations)

    return pd.DataFrame({
        "movie_title": top_n["movie_title"].values
    }, index=range(1, len(top_n) + 1))


# ---------------------  MATRIX FACTORISATION  -----------------------
# --------------------------------------------------------------------

def recommend_svd_sklearn(user_id, num_recommendations=10):
    """
    Recommend movies using Truncated SVD matrix factorization.
    Implements class best practices: user mean normalization,
    train/test split for RMSE, cosine similarity not used.
    
    Returns:
    - DataFrame with top-N recommended movie titles
    - RMSE on the test set
    """

    # 1. Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2. Build user-item matrix from training data
    train_matrix = train_df.pivot(index="user_id", columns="item_id", values="rating")

    # 3. Normalize ratings (subtract user mean)
    user_means = train_matrix.mean(axis=1)
    norm_matrix = train_matrix.sub(user_means, axis=0).fillna(0)

    # 4. Apply SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced = svd.fit_transform(norm_matrix)
    reconstructed = np.dot(reduced, svd.components_)

    # 5. Denormalize predictions
    predicted_ratings = pd.DataFrame(reconstructed, index=train_matrix.index, columns=train_matrix.columns)
    predicted_ratings = predicted_ratings.add(user_means, axis=0)

    # 6. RMSE Evaluation
    true, pred = [], []
    for _, row in test_df.iterrows():
        u, i, r = row["user_id"], row["item_id"], row["rating"]
        if u in predicted_ratings.index and i in predicted_ratings.columns:
            true.append(r)
            pred.append(predicted_ratings.loc[u, i])
    rmse = round(sqrt(mean_squared_error(true, pred)), 4)

    # 7. Recommend top-N unseen movies
    if user_id not in predicted_ratings.index:
        return pd.DataFrame({"movie_title": ["User not found"], "predicted_rating": [None]}), rmse

    user_row = predicted_ratings.loc[user_id]
    seen_items = train_matrix.loc[user_id][train_matrix.loc[user_id].notna()].index
    unseen_ratings = user_row.drop(seen_items, errors='ignore')

    top_n = unseen_ratings.sort_values(ascending=False).head(num_recommendations)
    titles = items.set_index("movie_id").loc[top_n.index]["movie_title"]

    result = pd.DataFrame({
        "movie_title": titles.values
    }, index=range(1, len(top_n) + 1))

    return result, rmse

# ------------------------  HYBRID APPROACH  -------------------------
# --------------------------------------------------------------------

def recommend_hybrid(user_id, num_recommendations=10, alpha=0.5):
    """
    Hybrid recommender combining SVD and content-based filtering.
    Returns top-N DataFrame with hybrid_score and prints SVD RMSE.
    """
    # ----- 1. Split + Matrix + Normalize -----
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_matrix = train_df.pivot(index="user_id", columns="item_id", values="rating")
    user_means = train_matrix.mean(axis=1)
    train_matrix_norm = train_matrix.sub(user_means, axis=0).fillna(0)

    # ----- 2. SVD -----
    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced = svd.fit_transform(train_matrix_norm)
    approx = np.dot(reduced, svd.components_)
    predicted_ratings = pd.DataFrame(approx, index=train_matrix.index, columns=train_matrix.columns)
    predicted_ratings = predicted_ratings.add(user_means, axis=0)

    # ----- 3. Evaluate RMSE -----
    true_ratings, pred_ratings = [], []
    for _, row in test_df.iterrows():
        u, i, r = row["user_id"], row["item_id"], row["rating"]
        if u in predicted_ratings.index and i in predicted_ratings.columns:
            true_ratings.append(r)
            pred_ratings.append(predicted_ratings.loc[u, i])
    rmse = sqrt(mean_squared_error(true_ratings, pred_ratings))
    print("📉 SVD RMSE:", round(rmse, 4))

    # ----- 4. Content-Based Profile -----
    movie_features = items.set_index("movie_id")[["movie_title"] + [f"genre_{i}" for i in range(19)]]
    movie_features = movie_features.drop_duplicates(subset="movie_title")

    user_ratings = df[df["user_id"] == user_id][["item_id", "rating"]]
    rated_movies = pd.merge(user_ratings, movie_features, left_on="item_id", right_index=True)
    genre_matrix = rated_movies[[f"genre_{i}" for i in range(19)]]
    user_profile = genre_matrix.T.dot(rated_movies["rating"])

    unseen_movies = movie_features[~movie_features.index.isin(user_ratings["item_id"])]
    content_features = unseen_movies[[f"genre_{i}" for i in range(19)]]
    content_scores = cosine_similarity([user_profile], content_features)[0]
    content_scores = pd.Series(content_scores, index=unseen_movies.index)

    # ----- 5. Combine Scores -----
    if user_id not in predicted_ratings.index:
        return pd.DataFrame({"movie_title": ["User not found"], "hybrid_score": [None]})

    # Collaborative scores from SVD
    svd_scores = predicted_ratings.loc[user_id].drop(user_ratings["item_id"].tolist(), errors="ignore")
    hybrid_scores = []

    for movie_id in content_scores.index:
        if movie_id in svd_scores:
            score = alpha * svd_scores[movie_id] + (1 - alpha) * content_scores[movie_id]
            hybrid_scores.append((movie_id, score))

    # ----- 6. Return Top N -----
    top_n = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
    movie_titles = items.set_index("movie_id").loc[[m[0] for m in top_n], "movie_title"].values
    scores = [m[1] for m in top_n]

    return pd.DataFrame({
        "movie_title": movie_titles
    }, index=range(1, len(scores) + 1))

# -------------------------  GENERATIVE AI  --------------------------
# --------------------------------------------------------------------

def recommend_genai(user_query, top_n=10):
    """
    Recommends movies based on a user's natural language query using semantic similarity.
    Uses all-MiniLM-L6-v2 to embed movie titles and the user's query.

    Parameters:
    - user_query (str): a natural language input like "I like thrillers with twists"
    - top_n (int): number of recommendations to return (default 10)

    Returns:
    - DataFrame: top-N recommended movies with similarity scores, 1-based index
    """

    # Load model (you may cache it globally in practice)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get movie titles as string list
    movie_descriptions = items["movie_title"].astype(str).tolist()

    # Encode user query and movie titles
    user_vector = model.encode(user_query, convert_to_tensor=True)
    movie_vectors = model.encode(movie_descriptions, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(user_vector, movie_vectors)

    # Get top N matches
    top_results = cosine_scores[0].topk(top_n)
    recommended_titles = [movie_descriptions[idx] for idx in top_results.indices.tolist()]

    # Build result DataFrame
    return pd.DataFrame({
        "movie_title": recommended_titles
    }, index=range(1, top_n + 1))

# ----------------------------  RESULTS  -----------------------------
# --------------------------------------------------------------------

from sklearn.decomposition import TruncatedSVD

def recommend_svd_sklearn_fold(train_df, user_id, num_recommendations=10):
    """
    Trains SVD using the given train_df and returns top-N recommendations for the user_id.
    """

    train_matrix = train_df.pivot(index="user_id", columns="item_id", values="rating")
    if user_id not in train_matrix.index:
        return pd.DataFrame({"movie_title": [], "movie_id": []})

    # Normalize: subtract user means
    user_means = train_matrix.mean(axis=1).fillna(0)
    norm_matrix = train_matrix.sub(user_means, axis=0).fillna(0)

    # Apply SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced = svd.fit_transform(norm_matrix)
    reconstructed = np.dot(reduced, svd.components_)

    predicted_ratings = pd.DataFrame(reconstructed, index=train_matrix.index, columns=train_matrix.columns)
    predicted_ratings = predicted_ratings.add(user_means, axis=0)

    # Get top-N unseen recommendations
    user_row = predicted_ratings.loc[user_id]
    seen = train_matrix.loc[user_id][train_matrix.loc[user_id].notna()].index
    unseen = user_row.drop(seen, errors='ignore').sort_values(ascending=False).head(num_recommendations)

    movie_titles = items.set_index("movie_id").loc[unseen.index]["movie_title"]

    return pd.DataFrame({
        "movie_id": unseen.index,
        "movie_title": movie_titles.values
    })

def recommend_hybrid_fold(train_df, user_id, num_recommendations=10):
    """
    Combines SVD and content-based recommendations using a weighted average.
    """

    # --- SVD Scores ---
    svd_recs = recommend_svd_sklearn_fold(train_df, user_id, num_recommendations * 2)
    svd_scores = pd.Series([1.0 - (i / len(svd_recs)) for i in range(len(svd_recs))], index=svd_recs["movie_id"])

    # --- Content Scores ---
    cb_recs = recommend_content_based(user_id, num_recommendations * 2)
    cb_recs = cb_recs.merge(items[["movie_id", "movie_title"]], on="movie_title", how="left")
    cb_scores = pd.Series([1.0 - (i / len(cb_recs)) for i in range(len(cb_recs))], index=cb_recs["movie_id"])

    # --- Combine and Rank ---
    combined_scores = (svd_scores.add(cb_scores, fill_value=0)) / 2
    top_n = combined_scores.sort_values(ascending=False).head(num_recommendations)
    top_titles = items.set_index("movie_id").loc[top_n.index]["movie_title"]

    return pd.DataFrame({
        "movie_id": top_n.index,
        "movie_title": top_titles.values
    })

def evaluate_models_on_folds():
    """
    Evaluates all models on u1-u5 folds and returns two DataFrames:
    - rmse_df: RMSE values for rating predictions
    - pr_df: Precision@10 and Recall@10 values for top-N recommendations
    """

    models_rmse = {
        "svd": [],
    }
    models_pr = {
        "user": [],
        "item": [],
        "content": [],
        "svd": [],
        "hybrid": []
    }

    for fold in range(1, 6):
        train_path = f"ml-100k/u{fold}.base"
        test_path = f"ml-100k/u{fold}.test"

        train_df = pd.read_csv(train_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
        test_df = pd.read_csv(test_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

        # --- RMSE for SVD ---
        train_matrix = train_df.pivot(index="user_id", columns="item_id", values="rating")
        user_means = train_matrix.mean(axis=1)
        norm_matrix = train_matrix.sub(user_means, axis=0).fillna(0)

        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=50, random_state=42)
        reduced = svd.fit_transform(norm_matrix)
        reconstructed = np.dot(reduced, svd.components_)

        predicted_ratings = pd.DataFrame(reconstructed, index=train_matrix.index, columns=train_matrix.columns)
        predicted_ratings = predicted_ratings.add(user_means, axis=0)

        true, pred = [], []
        for _, row in test_df.iterrows():
            u, i, r = row["user_id"], row["item_id"], row["rating"]
            if u in predicted_ratings.index and i in predicted_ratings.columns:
                true.append(r)
                pred.append(predicted_ratings.loc[u, i])

        rmse = round(sqrt(mean_squared_error(true, pred)), 4)
        models_rmse["svd"].append(rmse)

        # --- Precision & Recall @10 for all models ---
        def precision_recall_at_10(model_func):
            precisions = []
            recalls = []
            for user_id in test_df["user_id"].unique():
                relevant = test_df[(test_df["user_id"] == user_id) & (test_df["rating"] >= 4)]["item_id"].tolist()
                if not relevant:
                    continue
                try:
                    recs = model_func(user_id, num_recommendations=10)
                    recommended = items.set_index("movie_title").loc[recs["movie_title"]].index.tolist()
                except:
                    continue

                tp = len([item for item in recommended if item in relevant])
                precisions.append(tp / 10)
                recalls.append(tp / len(relevant))

            return round(np.mean(precisions), 4), round(np.mean(recalls), 4)

        for name, func in [
            ("user", recommend_user_based),
            ("item", recommend_item_based),
            ("content", recommend_content_based),
            ("svd", lambda uid, n=10: recommend_svd_sklearn_fold(train_df, uid, n)),
            ("hybrid", lambda uid, n=10: recommend_hybrid_fold(train_df, uid, n))
        ]:
            p, r = precision_recall_at_10(func)
            models_pr[name].append((p, r))

    # --- Build RMSE DataFrame ---
    rmse_df = pd.DataFrame(models_rmse, index=[f"Fold {i}" for i in range(1, 6)])

    # --- Build Precision/Recall DataFrame ---
    pr_rows = []
    for model, values in models_pr.items():
        for i, (p, r) in enumerate(values):
            pr_rows.append({"Model": model, "Fold": f"Fold {i+1}", "Precision@10": p, "Recall@10": r})
    pr_df = pd.DataFrame(pr_rows)

    return rmse_df, pr_df