# 📊 Linear Algebra in Sports Rankings: Colley vs. Massey

This project contains a Python script (`sports_rankings.py`) that uses real game data to rank sports teams by implementing two well-known methods from linear algebra: **Colley's Method** and **Massey's Method**.

It serves as a practical example of how systems of linear equations can model complex relationships and provide insightful solutions.

---

## 🔬 The Ranking Methods Explained

Both methods aim to solve the system **Ax = b**, where **x** is the vector of team ratings we want to find. They differ in how they construct the matrix **A** and the vector **b**.

### 1. Colley's Method
Colley's method is based on a team's win-loss record and is elegantly designed to account for strength of schedule. It sets up a system of equations where each team's rating is derived from its number of wins, losses, and the ratings of its opponents. A key feature is that the resulting **Colley Matrix** is guaranteed to be invertible, meaning there is always a single, unique solution.

The right-side vector is calculated as: $b_i = 1 + \frac{1}{2}(w_i - l_i)$

### 2. Massey's Method
Massey's method uses the point differentials from games to create its rankings. The goal is to find a set of ratings **r** that best explains the observed point differentials. For a game between team *i* and team *j*, the model expects $r_i - r_j \approx \text{points}_i - \text{points}_j$.

This creates a large, overdetermined system of equations, which is solved using the **normal equations** ($A^T A x = A^T b$). This system is rank-deficient, so a final constraint (e.g., the sum of all ranks is zero) is added to ensure a unique solution.

---

## 🚀 How to Run the Script

### 1. Prerequisites
You need Python and the `numpy` and `scipy` libraries.

### 2. Install Libraries
If you don't have them, open your terminal and run:
```bash
pip install numpy scipy
```

### 3. Data Files
1.  Make sure you have a folder named `materials`.
2.  Use the `Scores.mat` and `Differentials.mat` files inside this `materials` folder.

The final file structure should look like this:
```
.
├── p5_sports_rankings/
│   └── sports_rankings.py
└── materials/
    ├── Scores.mat
    └── Differentials.mat
```

### 4. Execute the Script
Run the script from your terminal:
```bash
python sports_rankings.py
```
The script will load the data, calculate the rankings using both methods, print the results, and perform a sensitivity analysis.