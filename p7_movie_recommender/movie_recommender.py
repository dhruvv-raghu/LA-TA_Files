# movie_recommender.py
#
# A script to find similar users and recommend movies based on their ratings,
# using Euclidean distance and Pearson correlation as similarity metrics.

import numpy as np
import scipy.io

# --- DATA LOADING AND PREPARATION ---

def load_data(file_path):
    """Loads and preprocesses data from the .mat file."""
    data = scipy.io.loadmat(file_path)
    movies = [m[0] for m in data['movies'].flatten()]
    users_movies = data['users_movies']
    users_movies_sort = data['users_movies_sort']
    index_small = data['index_small'].flatten()
    trial_user = data['trial_user'].flatten()
    
    # Filter for users who have rated all 20 popular movies
    valid_users_mask = np.all(users_movies_sort > 0, axis=1)
    ratings_all_movies = users_movies[valid_users_mask]
    ratings_popular_movies = users_movies_sort[valid_users_mask]
    
    return movies, index_small, trial_user, ratings_all_movies, ratings_popular_movies

# --- RECOMMENDATION LOGIC ---

def find_recommendations(user_vector, all_ratings, popular_ratings, all_movies, popular_indices):
    """
    Finds movie recommendations for a given user vector using two methods.
    
    Args:
        user_vector (np.array): The 1D rating vector of the target user.
        all_ratings (np.array): The full rating matrix for valid users.
        popular_ratings (np.array): The rating matrix for only the popular movies.
        all_movies (list): A list of all movie titles.
        popular_indices (np.array): Indices of the popular movies.
        
    Returns:
        dict: A dictionary containing liked movies and recommendations.
    """
    
    # 1. Euclidean Distance Method
    euclidean_distances = np.linalg.norm(popular_ratings - user_vector, axis=1)
    closest_user_dist_idx = np.argmin(euclidean_distances)
    
    # 2. Pearson Correlation Method
    # Center the data by subtracting the mean rating of each user
    popular_ratings_cent = popular_ratings - np.mean(popular_ratings, axis=1, keepdims=True)
    user_vector_cent = user_vector - np.mean(user_vector)

    # Calculate correlation coefficients
    numerator = np.sum(popular_ratings_cent * user_vector_cent, axis=1)
    denominator = (np.linalg.norm(popular_ratings_cent, axis=1) * np.linalg.norm(user_vector_cent))
    pearson_coeffs = numerator / denominator
    closest_user_pearson_idx = np.argmax(pearson_coeffs)

    # 3. Generate Recommendations
    # Find movies the most similar users rated as 5
    rec_dist_indices = np.where(all_ratings[closest_user_dist_idx, :] == 5)[0]
    rec_pearson_indices = np.where(all_ratings[closest_user_pearson_idx, :] == 5)[0]

    recommendations = {
        "liked_movies": [all_movies[popular_indices[i]] for i, rating in enumerate(user_vector) if rating == 5],
        "euclidean": [all_movies[i] for i in rec_dist_indices],
        "pearson": [all_movies[i] for i in rec_pearson_indices]
    }
    
    return recommendations

def print_results(title, recommendations):
    """Prints the recommendations in a formatted way."""
    print(f"\n--- {title} ---")
    print("\nMovies you liked:")
    for movie in recommendations['liked_movies']:
        print(f"  - {movie}")
        
    print("\nRecommendations based on Euclidean Distance (Closest User):")
    for movie in recommendations['euclidean']:
        print(f"  - {movie}")

    print("\nRecommendations based on Pearson Correlation (Most Similar User):")
    for movie in recommendations['pearson']:
        print(f"  - {movie}")
    print("-" * 50)

# --- MAIN EXECUTION ---

def main():
    """Main function to run the recommender system."""
    try:
        file_path = '../materials/users_movies.mat'
        movies, index_small, trial_user, ratings_full, ratings_popular = load_data(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'.")
        print("Please ensure 'users_movies.mat' is in the top-level 'materials' folder.")
        return

    # --- Analysis for the provided 'trial_user' ---
    trial_user_recs = find_recommendations(trial_user, ratings_full, ratings_popular, movies, index_small)
    print_results("Analysis for Trial User", trial_user_recs)
    
    # --- Analysis for your own ratings ('myratings') ---
    #
    # <<< ACTION REQUIRED >>>
    # Edit the 'myratings' array below with your own ratings (1-5) for the
    # 20 popular movies listed when the script runs.
    #
    myratings = np.array([
        5, 1, 4, 3, 2, 5, 1, 4, 3, 5, # Your ratings for movies 1-10
        2, 5, 1, 3, 4, 5, 2, 1, 4, 3  # Your ratings for movies 11-20
    ])

    print("\nRunning analysis for your custom ratings ('myratings')...")
    print("\nRatings are based on the following popular movies:")
    for i in index_small:
        print(f"  - {movies[i]}")

    my_recs = find_recommendations(myratings, ratings_full, ratings_popular, movies, index_small)
    print_results("Analysis for Your Ratings", my_recs)

if __name__ == '__main__':
    main()
```