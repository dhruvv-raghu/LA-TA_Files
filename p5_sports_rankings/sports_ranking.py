# sports_rankings.py
#
# A script to calculate and compare sports team rankings using Colley's
# and Massey's linear algebra methods.

import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

# --- CONSTANTS ---
TEAMS = [
    'Baylor', 'Iowa State', 'University of Kansas', 'Kansas State',
    'University of Oklahoma', 'Oklahoma State', 'Texas Christian',
    'University of Texas Austin', 'Texas Tech', 'West Virginia'
]

# --- HELPER FUNCTIONS ---

def calculate_colley_ranks(scores_matrix):
    """Calculates team rankings using Colley's method."""
    num_teams = scores_matrix.shape[0]
    games_played = np.abs(scores_matrix)
    total_games = np.sum(games_played, axis=1)
    
    # Construct Colley's matrix and the right-hand side vector
    colley_matrix = 2 * np.eye(num_teams) + np.diag(total_games) - games_played
    right_side = 1 + 0.5 * np.sum(scores_matrix, axis=1)
    
    # Solve the linear system to get the ranks
    ranks = np.linalg.solve(colley_matrix, right_side)
    return ranks

def calculate_massey_ranks(differentials_matrix):
    """Calculates team rankings using Massey's method."""
    num_teams = differentials_matrix.shape[0]
    
    # Build the game matrix (P) and point differential vector (B)
    game_rows = []
    diff_values = []
    
    for j in range(num_teams):
        for k in range(j + 1, num_teams):
            if differentials_matrix[j, k] != 0:
                row = np.zeros(num_teams)
                row[j] = 1
                row[k] = -1
                game_rows.append(row)
                diff_values.append(differentials_matrix[j, k])

    P = np.array(game_rows)
    B = np.array(diff_values)
    
    # Create the normal system of linear equations (A = P.T * P, D = P.T * B)
    A = P.T @ P
    D = P.T @ B
    
    # Substitute the last row to enforce the constraint that sum(ranks) = 0
    # This ensures a unique solution.
    A[-1, :] = np.ones(num_teams)
    D[-1] = 0
    
    # Solve the system
    ranks = np.linalg.solve(A, D)
    return ranks

def print_rankings(title, ranks, teams):
    """Prints a formatted table of team rankings."""
    print(f"\n--- {title} ---")
    print(f"{'Rank':<5} {'Rating':<10} {'Team'}")
    print("-" * 40)
    
    # Get the sorted order of indices
    order = np.argsort(ranks)[::-1]
    
    for i, team_idx in enumerate(order):
        print(f"{i+1:<5} {ranks[team_idx]:<10.4f} {teams[team_idx]}")

# --- MAIN EXECUTION ---

def main():
    """Load data, calculate rankings, compare, and analyze sensitivity."""
    try:
        # 1. Load data from .mat files located in a 'materials' subdirectory
        scores_data = loadmat('materials/Scores.mat')['Scores']
        differentials_data = loadmat('materials/Differentials.mat')['Differentials']
    except FileNotFoundError:
        print("Error: Make sure 'Scores.mat' and 'Differentials.mat' are in a 'materials' subfolder.")
        return

    # 2. Calculate and display rankings for both methods
    ranks_colley = calculate_colley_ranks(scores_data)
    ranks_massey = calculate_massey_ranks(differentials_data)
    
    print_rankings("Colley's Method Rankings", ranks_colley, TEAMS)
    print_rankings("Massey's Method Rankings", ranks_massey, TEAMS)
    
    # 3. Compare the two ranking systems
    pearson_corr = np.corrcoef(ranks_colley, ranks_massey)[0, 1]
    spearman_corr, _ = spearmanr(ranks_colley, ranks_massey)
    
    print("\n--- Ranking Comparison ---")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman's rank correlation: {spearman_corr:.4f}")

    # 4. Sensitivity Analysis: What if one game's result was flipped?
    print("\n--- Sensitivity Analysis ---")
    print("Flipping the result of the game between the top two teams...")

    # Colley Analysis
    scores_updated = scores_data.copy()
    colley_order = np.argsort(ranks_colley)[::-1]
    team1_idx, team2_idx = colley_order[0], colley_order[1]
    
    # Flip the result in the scores matrix
    scores_updated[team1_idx, team2_idx] *= -1
    scores_updated[team2_idx, team1_idx] *= -1
    
    updated_ranks_colley = calculate_colley_ranks(scores_updated)
    print_rankings("Updated Colley Rankings", updated_ranks_colley, TEAMS)

    # Massey Analysis
    differentials_updated = differentials_data.copy()
    massey_order = np.argsort(ranks_massey)[::-1]
    team1_idx, team2_idx = massey_order[0], massey_order[1]

    # Flip the result in the differentials matrix
    differentials_updated[team1_idx, team2_idx] *= -1
    differentials_updated[team2_idx, team1_idx] *= -1

    updated_ranks_massey = calculate_massey_ranks(differentials_updated)
    print_rankings("Updated Massey Rankings", updated_ranks_massey, TEAMS)

if __name__ == '__main__':
    main()
