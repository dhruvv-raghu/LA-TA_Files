# 🌐 Project 12: PageRank Algorithm Implementation

This project contains a Python script (`pagerank.py`) that implements a simplified version of Google's **PageRank** algorithm. It demonstrates how to determine the importance of a webpage within a network by analyzing the link structure between pages.

The script uses concepts from linear algebra, including **graph theory**, **stochastic matrices**, and **eigenvector analysis**, to rank a subset of a web network.

---
## 🔬 The Core Concepts

### 1. The Web as a Directed Graph
The World Wide Web can be modeled as a massive directed graph, where each webpage is a **node** (or vertex) and each hyperlink from one page to another is a **directed edge**. The importance of a page is determined not just by how many links it has, but by the importance of the pages that link to it.

### 2. The Google Matrix (A Stochastic Matrix)
To model a user randomly clicking on links, we construct a special matrix called the **Google Matrix** ($G$). It's a type of **stochastic matrix**, where each entry $G_{ij}$ represents the probability of moving from page *i* to page *j*. It's built as follows:

-   First, we create a transition matrix $S$ where each row represents a webpage. If page *i* has $k$ outgoing links, then for each linked page *j*, $S_{ij} = 1/k$. All other entries are 0.
-   To handle pages with no outgoing links (dangling nodes), we modify the matrix so these pages have an equal probability of jumping to any other page in the network.
-   Finally, we introduce a **damping factor**, $\alpha$ (typically 0.15), which represents the probability that a user will stop following links and instead jump to a random page. The final Google Matrix is:
    $$
    G = (1-\alpha)S + \alpha \frac{J}{n}
    $$
    where $J$ is a matrix of all ones and $n$ is the total number of pages.

### 3. Finding the PageRank Vector
The PageRank of each page is given by the entries of a special vector. This vector is the **principal eigenvector** of the Google Matrix—specifically, the eigenvector corresponding to the eigenvalue of **1**. There are two common ways to find this vector:

1.  **Power Iteration Method:** We start with an initial guess for the rank vector (e.g., a uniform distribution) and repeatedly multiply it by the Google Matrix. After many iterations, this process is guaranteed to converge to the principal eigenvector.
2.  **Direct Eigenvector Calculation:** We can use numerical methods to directly compute the eigenvectors and eigenvalues of the Google Matrix and select the one corresponding to the eigenvalue of 1.

The resulting eigenvector contains the PageRank scores, where a higher value indicates a more important page.

---
## 🚀 How to Run the Script

### 1. Prerequisites
You need Python and the `numpy`, `scipy`, and `matplotlib` libraries. For a better graph visualization, `networkx` is also recommended.

### 2. Install Libraries
If you don't have them, open your terminal and run:
```bash
pip install numpy scipy matplotlib networkx
```

### 3. Data Files
1.  Make sure you have a folder named `materials`
2.  Use the `AdjMatrix.mat` file inside this `materials` folder.

The file structure should look like this:
```
.
├── p12_pagerank/
    ├── pagerank.py
    ├── README.md
└── materials/
    └── AdjMatrix.mat
```

### 4. Execute the Script
Run the script from your terminal:
```bash
python pagerank.py
```
The script will load the data, build and visualize a subgraph, calculate the PageRank vector using two different methods, and compare the results.