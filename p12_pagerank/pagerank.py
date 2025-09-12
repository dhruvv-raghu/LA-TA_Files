# pagerank.py
#
# A script to calculate webpage importance using the PageRank algorithm,
# demonstrating both power iteration and eigenvector methods.

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx

# --- CORE FUNCTIONS ---

def create_google_matrix(adj_matrix, alpha=0.15):
    """
    Creates the Google Matrix from an adjacency matrix.
    
    Args:
        adj_matrix (np.array): The adjacency matrix of the web graph.
        alpha (float): The damping factor (probability of a random jump).
        
    Returns:
        np.array: The Google Matrix.
    """
    num_pages = adj_matrix.shape[0]
    
    # Calculate the number of outgoing links for each page
    out_degrees = np.sum(adj_matrix, axis=1)
    
    # Create the transition matrix S
    S = np.zeros((num_pages, num_pages))
    for i in range(num_pages):
        if out_degrees[i] > 0:
            S[i, :] = adj_matrix[i, :] / out_degrees[i]
        else:
            # Handle dangling nodes: equal probability of jumping to any page
            S[i, :] = 1 / num_pages
            
    # Construct the Google Matrix
    J = np.ones((num_pages, num_pages)) / num_pages
    google_matrix = (1 - alpha) * S + alpha * J
    
    return google_matrix

def power_iteration(google_matrix, num_iterations=20):
    """
    Calculates the PageRank vector using the power iteration method.
    
    Returns:
        np.array: The PageRank vector.
    """
    num_pages = google_matrix.shape[0]
    # Start with a uniform rank vector
    rank_vector = np.ones(num_pages) / num_pages
    
    for _ in range(num_iterations):
        rank_vector = rank_vector @ google_matrix
        
    return rank_vector / np.sum(rank_vector) # Normalize the final vector

def eigenvector_method(google_matrix):
    """
    Calculates the PageRank vector by finding the principal eigenvector.
    
    Returns:
        np.array: The PageRank vector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(google_matrix.T)
    
    # Find the eigenvector corresponding to the eigenvalue closest to 1
    principal_eigenvector_index = np.argmax(eigenvalues.real)
    pagerank_vector = eigenvectors[:, principal_eigenvector_index].real
    
    # Normalize to be a probability distribution
    return pagerank_vector / np.sum(pagerank_vector)

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    try:
        # Load the adjacency matrix from the .mat file
        data = scipy.io.loadmat('../materials/AdjMatrix.mat')
        adj_matrix_full = data['AdjMatrix']
    except FileNotFoundError:
        print("Error: 'AdjMatrix.mat' not found. Ensure it is in the '../materials/' folder.")
        exit()

    # Create a smaller sub-network for analysis
    NUM_PAGES_SUBGRAPH = 500
    adj_matrix_small = adj_matrix_full[:NUM_PAGES_SUBGRAPH, :NUM_PAGES_SUBGRAPH].toarray()

    # --- 1. Visualize the Sub-Network ---
    print(f"Visualizing a subgraph of {NUM_PAGES_SUBGRAPH} webpages...")
    G = nx.from_numpy_array(adj_matrix_small, create_using=nx.DiGraph)
    plt.figure(figsize=(10, 10))
    # Use a spring layout for better visualization of network structure
    pos = nx.spring_layout(G, seed=42) 
    nx.draw(G, pos, with_labels=False, node_size=20, width=0.5, arrowsize=5)
    plt.title(f'Webgraph of the First {NUM_PAGES_SUBGRAPH} Nodes')
    plt.show()

    # --- 2. Calculate PageRank ---
    print("\nCalculating PageRank...")
    google_matrix = create_google_matrix(adj_matrix_small)
    
    # Method 1: Power Iteration
    pagerank_power_iter = power_iteration(google_matrix)
    
    # Method 2: Eigenvector Method
    pagerank_eigenvector = eigenvector_method(google_matrix)
    
    # Compare the results (they should be very close)
    print(f"Difference between methods: {np.linalg.norm(pagerank_power_iter - pagerank_eigenvector):.2e}")

    # --- 3. Analyze the Results ---
    # We'll use the eigenvector result for the final analysis
    page_max_rank_idx = np.argmax(pagerank_eigenvector)
    max_rank_score = pagerank_eigenvector[page_max_rank_idx]
    
    print(f"\nMost important page (by PageRank): Page #{page_max_rank_idx}")
    print(f"  - PageRank Score: {max_rank_score:.4f}")
    
    # For comparison, find the page with the most incoming links (in-degree)
    in_degrees = np.sum(adj_matrix_small, axis=0)
    page_max_links_idx = np.argmax(in_degrees)
    max_links_count = in_degrees[page_max_links_idx]
    
    print(f"\nPage with the most incoming hyperlinks: Page #{page_max_links_idx}")
    print(f"  - Number of Links: {int(max_links_count)}")

    # --- 4. Answer Key Questions ---
    print("\n--- Analysis Summary ---")
    if page_max_rank_idx == page_max_links_idx:
        print("The highest-ranking page IS the same as the page with the most hyperlinks.")
    else:
        print("The highest-ranking page IS NOT the same as the page with the most hyperlinks.")
        
    links_to_top_page = in_degrees[page_max_rank_idx]
    print(f"The highest-ranking page (Page #{page_max_rank_idx}) has {int(links_to_top_page)} incoming hyperlinks.")
