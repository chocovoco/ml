# UNITS 5 \& 6: COMPLETE THEORY \& FORMULAS

## (No Numericals - Pure Theory)

\---

# UNIT 5: TREE-BASED METHODS

\---

## 5.1 BASICS OF DECISION TREES

### Definition

A Decision Tree is a supervised learning technique used for both classification and regression problems, where decisions are made by recursively splitting the dataset based on feature values (mostly preferred for classification problems).

### Core Idea: Divide and Conquer

* Break the dataset into smaller subsets
* Apply decision rules at each step
* Continue until a clear decision is reached

### Hierarchical Structure Representation

|Component|Description|
|-|-|
|**Internal Nodes**|Represent features of a dataset|
|**Branches**|Represent decision rules|
|**Leaf Nodes**|Represent the outcome/final prediction|

### Types of Nodes

|Node Type|Description|
|-|-|
|**Decision Nodes**|Used to make decisions; have multiple branches|
|**Leaf Nodes**|Output of decisions; do not contain further branches|
|**Root Node**|First node at the top; contains complete dataset; first split begins here|
|**Parent Node**|A node that divides into child nodes|
|**Child Node**|Nodes formed after a split|

### Other Components

|Component|Description|
|-|-|
|**Branch**|Connects one node to another; represents outcome of a condition|
|**Subtree**|Smaller section of the full tree|
|**Pruning**|Removing less useful branches to reduce overfitting|

### Working Process (Greedy Algorithm)

**Step-by-Step:**

1. **Start with entire dataset** - All training data becomes the root node
2. **Select best feature to split** - Choose the feature that best separates the data using criteria:

   * Information Gain
   * Gini Index
3. **Split the dataset** - Divide dataset based on selected feature; create child nodes
4. **Repeat recursively** - Apply to each subset; build tree downwards
5. **Stopping conditions** - Tree stops growing when:

   * All data belongs to same class
   * No features left
   * Minimum samples reached
   * Tree depth limit is reached
6. **Final output** - Leaf nodes get final prediction (majority class)

### Mathematical Representation

$$f(x) = \\begin{cases} c\_1 \& x \\in R\_1 \\ c\_2 \& x \\in R\_2 \\end{cases}$$

Each region $R\_i$ corresponds to a leaf node, and $c\_i$ is the prediction for that region.

### Making Predictions with Decision Tree

```
Algorithm predict(tree\_node, input):
    If current tree\_node is a leaf:
        return majority class of data points in leaf
    else:
        next\_node = child node whose feature value agrees with input
        return predict(next\_node, input)
```

### Exponentially Large Search Space

Finding the best tree is an **NP-hard problem** because there are exponentially many possible trees. Therefore, a **greedy algorithm** is used to find a "good" tree (not necessarily optimal).

\---

## 5.2 ATTRIBUTE SELECTION MEASURES

### 1\. Entropy

**Definition:** Measure of randomness or uncertainty in data. Lower entropy means less uniform distribution.

**Formula:**
$$\\text{Entropy}(S) = -\\sum\_{i=1}^{c} p\_i \\log\_2(p\_i)$$

Where:

* $c$ = number of classes
* $p\_i$ = proportion of class i in dataset

**Properties:**

* Entropy = 0 → all data belongs to same class (pure node)
* Entropy = 1 → maximum uncertainty (equal distribution)

### 2\. Information Gain (IG)

**Definition:** Information that can increase level of certainty after splitting.

**Formula:**
$$\\text{IG}(S, A) = \\text{Entropy}(S) - \\sum\_{v \\in \\text{Values}(A)} \\frac{|S\_v|}{|S|} \\cdot \\text{Entropy}(S\_v)$$

Where:

* $S$ = parent dataset
* $A$ = attribute/feature to split on
* $S\_v$ = subset where attribute A has value v
* $|S\_v|/|S|$ = weight of each child

**Interpretation:**

* Higher information gain → better split
* Measures reduction in entropy after split

### 3\. Gini Index

**Definition:** Calculated by subtracting the sum of squared probabilities of each class from one. Favor larger partitions.

**Formula for a node:**
$$\\text{Gini}(S) = 1 - \\sum\_{i=1}^{c} (p\_i)^2$$

**Formula for a split:**
$$\\text{Gini}*{\\text{split}}(S, A) = \\sum*{v \\in \\text{Values}(A)} \\frac{|S\_v|}{|S|} \\cdot \\text{Gini}(S\_v)$$

**Interpretation:**

* Lower Gini index → better split
* Gini = 0 → all data belongs to same class (pure node)
* Information Gain favors smaller partitions with distinct values
* Gini Index favors larger partitions

### Comparison: IG vs Gini

|Aspect|Information Gain|Gini Index|
|-|-|-|
|Based on|Entropy|Probability squared|
|Favors|Smaller partitions with distinct values|Larger partitions|
|Calculation|Uses log|Uses squared probabilities|
|Value range|0 to 1|0 to 1|

\---

## 5.3 REGRESSION TREES

### Definition

A regression tree is used when the output is a **continuous numeric value** (predict numbers instead of categories).

### How it Works

|Step|Description|
|-|-|
|1|Identifies data into smaller groups based on feature values|
|2|Each split tries to reduce prediction error|
|3|Final leaf node stores an average as expected numeric value|
|4|When new data comes, it follows path to a leaf node|
|5|That stored value becomes the prediction|

### Split Measure for Regression Trees

* **MSE (Mean Squared Error)** - Most common
* **MAE (Mean Absolute Error)**

### Applications

* Predicting house prices
* Predicting stock values
* Temperature forecasting
* Estimating delivery time

**Key Point:** Final output is always a **number or value**, not a category.

\---

## 5.4 CLASSIFICATION TREES

### Definition

A classification tree is used when the output is a **categorical label/class** (Yes/No, Spam/Not Spam, etc.)

### Characteristics

|Aspect|Description|
|-|-|
|Output type|Categorical (discrete)|
|Target variable|Discrete class|
|Split measures|Entropy, Gini Index|
|Leaf output|Class label (majority class)|
|Objective|Classification|

### Applications

* Email spam detection
* Student pass/fail prediction
* Disease diagnosis
* Customer churn prediction

\---

## 5.5 TREES vs LINEAR MODELS

|Aspect|Linear Models|Decision Trees|
|-|-|-|
|**Relationship**|Assumes linear relationship|Captures non-linear relationships|
|**Interpretability**|Easy (coefficients show impact)|Very easy (visual representation)|
|**Scaling needed**|Yes (feature scaling required)|No scaling needed|
|**Feature selection**|Manual (may need feature selection)|Automatic|
|**Data handling**|Works well with numerical data|Handles both categorical and numerical|
|**Overfitting**|Less prone (with regularization)|Prone (needs pruning)|
|**Prediction speed**|Fast|Fast|
|**Visualization**|Line/plane/hyperplane|Tree diagram|

### When to Use Which

|Scenario|Recommended|
|-|-|
|Linear relationship|Linear Model|
|Non-linear relationship|Decision Tree|
|Need interpretability|Decision Tree|
|Small dataset|Linear Model|
|Mixed data types|Decision Tree|
|Need feature scaling avoidance|Decision Tree|

\---

## 5.6 ADVANTAGES AND DISADVANTAGES OF DECISION TREES

### Advantages

|Advantage|Explanation|
|-|-|
|**Easy to understand and explain**|Intuitive tree structure, easy for non-experts|
|**Handles mixed data**|Can process both categorical and numerical data|
|**Visualizable**|Results can be displayed as actual trees|
|**No scaling needed**|No normalization or standardization required|
|**Automatic feature selection**|Algorithm selects best features for splits|
|**Fast prediction**|Only need to traverse tree path|
|**Captures non-linear relationships**|Can model complex patterns|
|**Handles large complex data**|Works effectively on complex datasets|

### Disadvantages

|Disadvantage|Explanation|
|-|-|
|**Prone to overfitting**|Without pruning, trees can memorize training data|
|**Unstable**|Small changes in data can produce completely different trees|
|**Not always optimal**|Greedy algorithm may not find globally optimal tree|
|**Biased**|Can be biased toward features with more levels|
|**Requires pruning**|Needs additional steps to prevent overfitting|

\---

## 5.7 PRUNING

### Definition

Pruning means removing less useful branches from the decision tree to reduce overfitting.

### Types of Pruning

|Type|Description|
|-|-|
|**Pre-pruning**|Stops tree growth early during training (before full tree built)|
|**Post-pruning**|Builds full tree first, then removes weak branches|

### Benefits of Pruning

* Reduces overfitting
* Better accuracy on test data
* Simpler model (easier to understand)
* Faster predictions

\---

## 5.8 ENSEMBLE LEARNING

### Definition

Ensemble learning is a machine learning technique in which **multiple models are combined together** to make a strong model.

### Core Principle

* A group of weak learners come together to form a strong learner
* Instead of depending on one decision tree, many trees work together
* One model may make mistakes, but other models can correct those mistakes
* Final answer is usually more accurate than a single model

### Types of Ensemble Learning

|Type|Approach|
|-|-|
|**Bagging**|Parallel (independent models)|
|**Boosting**|Sequential (one after another)|

### Ensemble Prediction Formula

For an ensemble of classifiers $f\_1(x), f\_2(x), ..., f\_T(x)$ with coefficients $w\_1, w\_2, ..., w\_T$:

$$\\hat{y} = \\text{sign}\\left(\\sum\_{t=1}^{T} w\_t f\_t(x)\\right)$$

For classification: Majority vote
For regression: Average of predictions

\---

## 5.9 BAGGING (Bootstrap Aggregating)

### Definition

Bagging is an ensemble method where many decision trees are created **independently**, each trained on a different random sample of the original dataset.

### Process

|Step|Description|
|-|-|
|1|Take random samples from original training dataset|
|2|**Sampling with replacement** - same record can be chosen more than once|
|3|Train one decision tree on each random sample|
|4|After training all trees, combine their predictions|

### Final Output Combination

|Task|Method|
|-|-|
|**Classification**|Most voted class (majority vote)|
|**Regression**|Average of all predictions|

### Benefits of Bagging

* Reduces overfitting
* Increases accuracy
* Predictions become more stable
* Reduces variance

\---

## 5.10 RANDOM FOREST

### Definition

Advanced version of Bagging. Creates many decision trees and combines them. One of the most popular ML algorithms.

### Process

|Step|Description|
|-|-|
|1|Take many random samples from original data (with replacement)|
|2|Train one decision tree on each sample|
|3|**Key difference from Bagging:** At each split, only consider a random subset of features (not all features)|
|4|This makes all trees different from each other|
|5|Combine final output (majority vote for classification, average for regression)|

### Why Random Subset of Features?

* Makes trees more diverse
* Reduces correlation between trees
* Improves overall ensemble performance

### Advantages of Random Forest

|Advantage|Explanation|
|-|-|
|**Better accuracy**|Usually outperforms single decision tree|
|**Reduces overfitting**|Ensemble method naturally prevents overfitting|
|**Handles noisy data**|Robust to outliers and noise|
|**Scales well**|Works well on large datasets|
|**Feature importance**|Can measure which features are most important|

### Disadvantages of Random Forest

|Disadvantage|Explanation|
|-|-|
|**Requires more memory**|Multiple trees stored|
|**More computation**|Training many trees takes time|
|**Less interpretable**|Harder to understand than single tree|
|**Slower prediction**|Must traverse all trees|

\---

## 5.11 BOOSTING

### Definition

Ensemble method where trees are built **sequentially** (one after another). Each new tree tries to correct mistakes made by previous trees.

### Process

|Step|Description|
|-|-|
|1|First tree is trained on data|
|2|Calculate errors made by first tree|
|3|Second tree focuses on wrong predictions (gives more weight to misclassified points)|
|4|Third tree corrects mistakes of second tree|
|5|Process continues for many iterations|

### Key Characteristics

* Models trained **sequentially**, not independently
* Each new model focuses on "hard" points (previous errors)
* More weight on difficult data points

### Benefits

* Reduces bias
* Improves accuracy
* Similar benefits to Random Forest

### Comparison: Bagging vs Boosting

|Aspect|Bagging|Boosting|
|-|-|-|
|**Approach**|Parallel (independent)|Sequential (dependent)|
|**Data sampling**|Bootstrap (with replacement)|Same data with different weights|
|**Model weights**|Equal|Different (based on performance)|
|**Focus**|Reduce variance|Reduce bias|
|**Overfitting risk**|Lower|Higher (if not careful)|

\---

## 5.12 ADABOOST (Adaptive Boosting)

### Definition

Boosting algorithm where many small weak models are combined to make a strong model. Gives more importance to difficult data points so the next model focuses on them.

### Analogy

If a student fails a test, the teacher gives more attention in the next class.

### Algorithm Steps

|Step|Description|
|-|-|
|**Initialization**|Start with same weight for all points: $\\alpha\_i = 1/N$|
|**For t = 1 to T**|Repeat for each classifier|
|**1. Learn classifier**|Learn $f\_t(x)$ with data weights $\\alpha\_i$|
|**2. Compute weighted error**|$error\_t = \\frac{\\sum\_{i: f\_t(x\_i) \\neq y\_i} \\alpha\_i}{\\sum\_{i=1}^N \\alpha\_i}$|
|**3. Compute coefficient**|$\\hat{w}\_t = \\frac{1}{2} \\ln\\left(\\frac{1 - error\_t}{error\_t}\\right)$|
|**4. Update weights**|Correct: $\\alpha\_i \\leftarrow \\alpha\_i \\cdot e^{-\\hat{w}\_t}$; Incorrect: $\\alpha\_i \\leftarrow \\alpha\_i \\cdot e^{+\\hat{w}\_t}$|
|**5. Normalize**|$\\alpha\_i \\leftarrow \\frac{\\alpha\_i}{\\sum\_{j=1}^N \\alpha\_j}$|

### Final Prediction

$$\\hat{y} = \\text{sign}\\left(\\sum\_{t=1}^{T} \\hat{w}\_t f\_t(x)\\right)$$

### Formula for Updating Weights

|Prediction|Multiply α by|Effect|
|-|-|-|
|Correct|$e^{-w\_t}$|Decreases importance|
|Incorrect|$e^{+w\_t}$|Increases importance|

### Advantages of AdaBoost

* Adaptive learning (focuses on difficult cases)
* Good accuracy
* Reduces overfitting (when properly tuned)

\---

## 5.13 XGBOOST (Extreme Gradient Boosting)

### Definition

Advanced, fast implementation of gradient boosting. One of the most powerful ML algorithms.

### Key Features

|Feature|Description|
|-|-|
|**Speed**|Highly optimized for fast computation|
|**Missing values**|Can handle missing values automatically|
|**Regularization**|Uses L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting|
|**Scalability**|Can handle very large datasets|
|**Parallel processing**|Optimized for parallel computation|

### Algorithm Steps (Theoretical)

|Step|Description|
|-|-|
|1|**Initialize ensemble** with single model (decision tree)|
|2|**Calculate errors** made by first model|
|3|**Train second model** to correct those errors|
|4|**Add second model** to ensemble|
|5|**Calculate new errors** of the combined ensemble|
|6|**Train third model** to correct new errors|
|7|**Add third model** to ensemble|
|8|**Repeat** for hundreds/thousands of iterations|
|9|**Make predictions** - weighted average of all models|
|10|**Tune hyperparameters** using cross-validation and grid search|

### Regularization Techniques in XGBoost

* **L1 regularization (Lasso)** - Can reduce some weights to zero
* **L2 regularization (Ridge)** - Shrinks weights but not to zero

### Advantages of XGBoost

|Advantage|Explanation|
|-|-|
|**Excellent accuracy**|Often wins Kaggle competitions|
|**Efficient for large data**|Highly scalable|
|**Handles missing values**|Automatic handling|
|**Prevents overfitting**|Built-in regularization|
|**Fast execution**|Optimized algorithms|

### Applications

* Kaggle competitions
* Industry ML projects
* Fraud detection
* Sales forecasting
* Credit scoring

\---

# UNIT 6: UNSUPERVISED LEARNING

\---

## 6.1 WHAT IS UNSUPERVISED LEARNING?

### Definition

Unsupervised learning is a type of machine learning where the model is trained using data that does **not contain labeled outputs**. Only input data is provided; no correct answers are given.

### Key Characteristics

|Aspect|Description|
|-|-|
|**Data Type**|Unlabeled (input only)|
|**Learning method**|Discovers hidden patterns, relationships, similarities|
|**Error correction**|None (outputs unknown)|
|**Accuracy**|Difficult to measure|
|**Human effort**|Requires less labeling effort|

### Why Unsupervised Learning is Used

* Large amounts of data available but labeled answers not available
* Discover patterns humans may not notice easily
* Explore data structure before applying supervised learning

### Applications

* Customer segmentation
* Fraud detection
* Anomaly detection
* Document clustering
* Recommendation systems
* Dynamic trend detection

### Challenges of Unsupervised Learning

* No ground truth to validate results
* Hard to evaluate performance objectively
* Results can be subjective
* Interpreting clusters may be difficult

\---

## 6.2 CLUSTERING BASICS

### What is a Cluster?

A cluster is a collection of data objects that are:

* **Similar (or related)** to one another within the same group
* **Dissimilar (or unrelated)** to objects in another group

### Cluster Analysis (Clustering, Data Segmentation)

Given a set of data points, partition them into groups (clusters) that are as similar as possible.

### Properties of Good Clustering

|Property|Description|
|-|-|
|**High intra-class similarity**|Cohesive within cluster (points in same cluster are similar)|
|**Low inter-class similarity**|Distinctive between clusters (different clusters are dissimilar)|

### Quality Function

* A separate quality function measures goodness of a cluster
* Hard to define "similar enough" or "good enough"
* Many similarity measures exist for different applications

### Typical Ways to Use Clustering

|Use|Description|
|-|-|
|**Stand-alone tool**|Get insight into data distribution|
|**Preprocessing step**|Intermediate step for another algorithm|
|**Collaborative filtering**|Recommendation systems|
|**Outlier detection**|Finding anomalous data points|

\---

## 6.3 SIMILARITY AND DISTANCE MEASURES

### Similarity Measure (Similarity Function)

|Aspect|Description|
|-|-|
|**Definition**|Real-valued function that quantifies similarity between two objects|
|**Meaning**|Measures how alike two data objects are|
|**Range**|Usually \[0,1]|
|**0 means**|No similarity|
|**1 means**|Completely similar|

### Dissimilarity (Distance Measure)

|Aspect|Description|
|-|-|
|**Definition**|Numerical measure of how different two data objects are|
|**Meaning**|Inverse of similarity|
|**Range**|\[0, ∞)|
|**0 means**|Completely similar (minimum dissimilarity)|

### Proximity

Refers to either similarity or dissimilarity (general term).

\---

## 6.4 MINKOWSKI DISTANCE (General Distance Formula)

### Formula

$$d(i,j) = \\sqrt\[p]{\\sum\_{f=1}^{l} |x\_{if} - x\_{jf}|^p}$$

Where:

* $i = (x\_{i1}, x\_{i2}, ..., x\_{il})$ and $j = (x\_{j1}, x\_{j2}, ..., x\_{jl})$ are two $l$-dimensional data objects
* $p$ is the order (L-p norm)

### Properties of Distance Measures

|Property|Formula|Meaning|
|-|-|-|
|**Positivity**|$d(i,j) > 0$ if $i \\neq j$, $d(i,i) = 0$|Distance is positive for different points, zero for same point|
|**Symmetry**|$d(i,j) = d(j,i)$|Distance from A to B equals distance from B to A|
|**Triangle Inequality**|$d(i,j) \\leq d(i,k) + d(k,j)$|Direct distance ≤ indirect path|

\---

## 6.5 SPECIAL CASES OF MINKOWSKI DISTANCE

### 1\. Manhattan Distance (L1 norm, City Block Distance)

**Condition:** $p = 1$

**Formula:**
$$d(i,j) = |x\_{i1} - x\_{j1}| + |x\_{i2} - x\_{j2}| + \\dots + |x\_{il} - x\_{jl}|$$

**Also called:** City block distance, L1 distance

**Example usage:** Hamming distance (number of bits different between binary vectors)

### 2\. Euclidean Distance (L2 norm)

**Condition:** $p = 2$

**Formula:**
$$d(i,j) = \\sqrt{|x\_{i1} - x\_{j1}|^2 + |x\_{i2} - x\_{j2}|^2 + \\dots + |x\_{il} - x\_{jl}|^2}$$

**Also called:** Straight line distance, L2 distance

**Most commonly used distance** in clustering algorithms.

### 3\. Supremum Distance (L∞ norm, Chebyshev Distance)

**Condition:** $p \\rightarrow \\infty$

**Formula:**
$$d(i,j) = \\max\_{f=1}^{l} |x\_{if} - x\_{jf}|$$

**Meaning:** Maximum difference between any component (attribute) of the vectors

**Also called:** L∞ distance, Chebyshev distance

### Summary Table

|Distance Name|p value|Formula|Use Case|
|-|-|-|-|
|Manhattan|p = 1|$\\sum \|x\_i - y\_i\|$|Grid-like paths|
|Euclidean|p = 2|$\\sqrt{\\sum (x\_i - y\_i)^2}$|Straight line distance|
|Supremum|p → ∞|$\\max \|x\_i - y\_i\|$|Maximum difference|

\---

## 6.6 COSINE SIMILARITY

### Definition

Measure of similarity between two vectors, commonly used for text/document similarity.

### Formula

$$\\cos(d\_1, d\_2) = \\frac{d\_1 \\cdot d\_2}{|d\_1| \\times |d\_2|}$$

Where:

* $d\_1 \\cdot d\_2$ = dot product of vectors
* $|d|$ = magnitude (length) of vector $= \\sqrt{\\sum\_{i=1}^{n} d\_i^2}$

### Properties

|Property|Value|
|-|-|
|Range|\[0, 1]|
|0 means|No similarity (orthogonal vectors)|
|1 means|Completely similar (identical direction)|

### Interpretation

* Cosine similarity measures the **angle** between two vectors, not their magnitude
* Used when only direction matters, not absolute values
* Common in text analysis (document similarity)

\---

## 6.7 PARTITIONING METHODS

### Definition

Partitioning methods discover grouping in the data by optimizing a specific objective function and iteratively improving the quality of partition.

### Characteristics

* Most common clustering approach
* Each data point belongs to **exactly one cluster**
* Requires iterative process
* Also called **K-Partition method**

### Problem Definition

Given $k$, find partition of $k$ clusters that optimize the chosen partition criteria.

### Objective Function Example

Sum of squared distances is minimized:
$$\\sum\_{j=1}^{k} \\sum\_{x \\in C\_j} |x - c\_j|^2$$

Where $c\_j$ is the centroid or medoid of cluster $C\_j$

### Complexity

* **Global optimal** requires exhaustive enumeration of all partitions (infeasible for large data)
* **Heuristic methods (Greedy algorithms)** are used instead

### Common Partitioning Methods

|Method|Description|
|-|-|
|**K-means**|Uses centroid (mean of points)|
|**K-medians**|Uses median|
|**K-medoids**|Uses actual data points as centers|

\---

## 6.8 K-MEANS CLUSTERING ALGORITHM

### Definition

One of the most popular clustering algorithms. Divides data into $k$ clusters, where $k$ is chosen in advance. Each cluster has a center point called **centroid**.

### Algorithm Steps (Pseudo-code)

```
Input: dataset, number of clusters k
Output: k clusters

1. Initialize k centroids randomly in the dataset
2. While max\_iterations not reached or centroids still moving:
   a. Assign each data point to its nearest centroid (using Euclidean distance)
   b. Calculate the mean of each cluster to get new centroid location
   c. Update location of each centroid to new location
3. Output the k clusters
```

### Detailed Steps

|Step|Description|
|-|-|
|**Step 1**|Choose number of clusters K|
|**Step 2**|Initialize K random centroids (can choose random data points)|
|**Step 3**|Assign each data point to nearest centroid using Euclidean distance|
|**Step 4**|Recalculate centroids by taking the mean of all points in each cluster|
|**Step 5**|Repeat Steps 3 and 4 until centroids stop changing (convergence)|

### Distance Calculation (Euclidean)

$$d(point, centroid) = \\sqrt{(x\_1 - c\_1)^2 + (x\_2 - c\_2)^2 + \\dots + (x\_n - c\_n)^2}$$

### Centroid Update Formula

$$c\_j = \\frac{1}{|C\_j|} \\sum\_{x \\in C\_j} x$$

Where $C\_j$ is the set of points in cluster $j$

### Advantages

|Advantage|Explanation|
|-|-|
|**Easy to implement**|Simple algorithm|
|**Fast**|Linear time complexity|
|**Simple to understand**|Intuitive concept|
|**Works well**|Effective for spherical clusters|

### Disadvantages

|Disadvantage|Explanation|
|-|-|
|**Requires K in advance**|Need to know number of clusters beforehand|
|**Sensitive to outliers**|Outliers can skew centroids|
|**Sensitive to initial centroids**|Different initializations can give different results|
|**Assumes spherical clusters**|Struggles with non-spherical shapes|
|**Assumes equal-sized clusters**|Biased toward equal-sized clusters|

### Effect of Outliers on K-Means

* Outliers can pull centroids away from true cluster centers
* May create clusters with just one outlier point
* Robust alternatives: K-medoids (uses actual data points instead of means)

\---

## 6.9 DIMENSION REDUCTION

### Definition

Process of reducing the number of features (dimensions) in a dataset while preserving important information.

### Why Dimension Reduction is Needed

|Problem|Explanation|
|-|-|
|**Slow training**|More features = more computation|
|**Irrelevant features**|Some features may not be useful|
|**Model confusion**|Too many features can confuse the model|
|**Visualization difficulty**|Hard to visualize high-dimensional data (need 2D/3D)|
|**Overfitting risk**|Model may memorize training data|
|**Sparsity**|Data points become spread out|

### Benefits of Dimension Reduction

|Benefit|Explanation|
|-|-|
|**Faster training**|Fewer features = quicker model training|
|**Less storage**|Smaller datasets use less memory|
|**Better visualization**|Can plot in 2D or 3D|
|**Removes noise**|Eliminates irrelevant features|
|**Reduces overfitting**|Simpler models generalize better|
|**May improve accuracy**|Removing irrelevant features can help|

\---

## 6.10 PRINCIPAL COMPONENT ANALYSIS (PCA)

### Definition

A linear method that transforms a dataset with many features into a smaller set of new features called **Principal Components** (PCs), while preserving as much of the original variance as possible.

### Core Concept

* PC1 captures the **highest variance**
* PC2 captures the **next highest variance** (uncorrelated with PC1)
* PC3, PC4, etc. capture remaining variance in decreasing order

### How PCA Works

|Step|Description|
|-|-|
|1|**Standardize** the range of continuous initial variables|
|2|**Compute covariance matrix** to identify correlations|
|3|**Compute eigenvectors and eigenvalues** to identify principal components|
|4|**Create feature vector** to decide which principal components to keep|
|5|**Recast data** along principal component axes|

### Mathematical Foundation

**Covariance Matrix:** Measures how much features vary together

$$\\text{Cov}(X,Y) = \\frac{\\sum (x\_i - \\bar{x})(y\_i - \\bar{y})}{n-1}$$

**Eigenvalues and Eigenvectors:**

* Eigenvectors define directions of principal components
* Eigenvalues represent amount of variance in each direction
* For matrix S: $S v = \\lambda v$ (where $\\lambda$ = eigenvalue, $v$ = eigenvector)

### Selecting Number of Components

* Keep components with highest eigenvalues (most variance explained)
* Often choose enough components to explain 90-95% of variance

### Example Interpretation

Features A, B, C → PC1 (combination of A+B) explains 91% variance → Add PC2 adds little improvement → Can keep only PC1

### Advantages of PCA

|Advantage|Explanation|
|-|-|
|**Reduces dimensionality**|Fewer features to process|
|**Removes redundancy**|Eliminates correlated features|
|**Removes noise**|Less important variation discarded|
|**Improves visualization**|Can plot first 2 or 3 components|
|**Handles multicollinearity**|Transforms correlated features into uncorrelated components|

### Disadvantages of PCA

|Disadvantage|Explanation|
|-|-|
|**Reduces interpretability**|PCs are combinations of original features (hard to explain)|
|**Linear method**|Cannot capture non-linear relationships|
|**Requires scaling**|Features must be standardized before applying|
|**Loses information**|Some variance (noise or signal) always lost|

\---

## 6.11 HIERARCHICAL CLUSTERING

### Definition

Hierarchical clustering creates a tree-like hierarchy of clusters. No need to specify number of clusters in advance.

### Two Approaches

|Approach|Direction|Starting point|Process|Efficiency|
|-|-|-|-|-|
|**Agglomerative**|Bottom-up|Each point as separate cluster|Merge most similar clusters|O(n³)|
|**Divisive**|Top-down|All points in one cluster|Split most heterogeneous cluster|More efficient|

\---

### Agglomerative Clustering (AGNES - AGglomerative NESTing)

**Process:**

1. Each object initially considered as a single-element cluster (leaf)
2. At each step, the two clusters that are most similar are combined into a new bigger cluster (node)
3. Procedure is iterated until all points are members of one single big cluster (root)

**Algorithm:**

1. Start with each point as separate cluster
2. Find pair of clusters with smallest distance
3. Merge them into one cluster
4. Update distance matrix
5. Repeat steps 2-4 until only one cluster remains

### Divisive Clustering (DIANA - DIvisive ANAlysis)

**Process:**

1. Begin with root (all objects in one cluster)
2. At each step, split the most heterogeneous cluster into two
3. Process is iterated until all objects are in their own cluster

### Similarity Measures Between Clusters

|Method|Description|Calculation|
|-|-|-|
|**Single link (nearest neighbor)**|Distance between nearest neighbors|$\\min{d(x,y): x \\in A, y \\in B}$|
|**Complete link (diameter)**|Distance between farthest points|$\\max{d(x,y): x \\in A, y \\in B}$|
|**Average link (group average)**|Average distance between all points|$\\frac{1}{|
|**Centroid link**|Distance between cluster centroids|$d(c\_A, c\_B)$|

### Dendrogram

**Definition:** A tree-like diagram that records the sequences of merges (in agglomerative) or splits (in divisive).

**Properties:**

* Height of merges/splits represents distance/similarity
* Can cut dendrogram at any level to get different number of clusters
* Provides complete hierarchical view of data

\---

## 6.12 RECOMMENDER SYSTEMS

### Definition

Software algorithms that recommend items to users based on their past behavior, preferences, and other factors.

### Why Recommender Systems are Important

|Benefit|Description|
|-|-|
|**Saves user time**|Finds relevant items quickly|
|**Improves user experience**|Personalized suggestions|
|**Increases sales**|More relevant product recommendations|
|**Discover new items**|Users find items they might not have found|
|**Increases engagement**|Keeps users on platform longer|

\---

### Types of Recommender Systems

#### 1\. Non-Personalized Recommendation Systems

**Definition:** Give the same recommendations to all users. Does not study personal interests or search history.

**Also called:** "One-size-fits-all" recommendation

**How it works:**

* Recommends items based on common popularity or current trends
* Uses most watched movies, festival offers, best-selling products, top trending videos

**When used:**

* New users (no history available)
* When not enough user data
* Seasonal/holiday recommendations

**Examples:**

* Best-seller lists
* Most popular items
* Trending videos
* Festival offers

#### 2\. Content-Based Recommendation Systems

**Definition:** Recommends items based on the content of the items themselves.

**Data used:**

* Item descriptions
* Tags
* User reviews
* Shared characteristics (genre, author, topic)

**How it works:**

* Finds items similar to what user previously liked
* Based on item attributes, not other users' behavior

**Example:**

* Recommend movie based on genre, cast, plot of previously watched movies

**When useful:**

* Not enough user data for collaborative filtering
* Users looking for items with specific attributes

**Advantages:**

* Personalized suggestions
* Matches user taste
* No need for data from other users

**Disadvantages:**

* May not suggest new types of items
* Needs sufficient user data
* May be slow for large catalogs
* Can lead to over-specialization (only recommending similar items)

\---

### Methods to Implement Recommender Systems

|Method|Description|Example|
|-|-|-|
|**Collaborative Filtering**|Based on behavior of similar users|If users A and B liked same movies, recommend A's movie to B|
|**Content-Based Filtering**|Based on item attributes|Recommend movies with same genre|
|**Hybrid Filtering**|Combines collaborative + content-based|Netflix: user behavior + movie genre|
|**Association Rule Mining**|Finds items bought together|"Market Basket Analysis" - If bread, then butter|
|**Matrix Factorization**|Decomposes user-item matrix to find hidden patterns|Predict missing ratings|
|**Deep Learning**|Neural networks for complex relationships|Advanced recommender systems|

\---

### Detailed Method Explanations

#### Collaborative Filtering

* Assumes users with similar choices in the past will like similar items
* Uses user-user or item-item similarities
* "Users who liked this also liked..."

#### Association Rule Mining (Market Basket Analysis)

* Finds items frequently purchased/used together
* Example: If customers buy bread and butter together, recommend butter when someone buys bread

#### Matrix Factorization

* Used when user-item rating data is available
* Decomposes large user-item matrix into smaller matrices
* Finds hidden patterns (latent factors)
* Predicts missing ratings
* Example: If a user has not rated a movie, system can estimate if user may like it

#### Deep Learning

* Uses neural networks to build advanced recommendation systems
* Can learn complex relationships between users and items
* More powerful but requires more data and computation

\---

### Evaluation Metrics for Recommender Systems

|Metric|Description|
|-|-|
|**Accuracy**|How well recommendations match user preferences (precision, recall, F1, MAP)|
|**Coverage**|Proportion of items the system can recommend|
|**Diversity**|Variety of recommended items (not all similar)|
|**Novelty**|How new or unexpected the recommendations are|

### Hybrid Techniques

|Technique|Description|
|-|-|
|**Weighted Hybrid**|Combine outputs of multiple algorithms with weights|
|**Switching Hybrid**|Switch between algorithms based on user context|
|**Feature Combination Hybrid**|Combine features from different algorithms into one vector|
|**Cascade Hybrid**|Use output of one algorithm as input to another|
|**Meta-level Hybrid**|Learn meta-model to combine algorithm outputs|

\---

## QUICK REFERENCE: IMPORTANT FORMULAS FOR UNIT 5 \& 6

### Unit 5 Formulas

|Formula|Purpose|
|-|-|
|$\\text{Entropy}(S) = -\\sum p\_i \\log\_2(p\_i)$|Measure uncertainty|
|$\\text{IG}(S,A) = \\text{Entropy}(S) - \\sum \\frac{|S\_v|
|$\\text{Gini}(S) = 1 - \\sum p\_i^2$|Measure impurity|
|$\\hat{y} = \\text{sign}\\left(\\sum w\_t f\_t(x)\\right)$|Ensemble prediction|
|$error\_t = \\frac{\\sum\_{i: f\_t(x\_i) \\neq y\_i} \\alpha\_i}{\\sum \\alpha\_i}$|AdaBoost weighted error|
|$\\hat{w}\_t = \\frac{1}{2}\\ln\\left(\\frac{1-error\_t}{error\_t}\\right)$|AdaBoost coefficient|

### Unit 6 Formulas

|Formula|Purpose|
|-|-|
|$d(i,j) = \\sqrt\[p]{\\sum|x\_{if} - x\_{jf}|
|$d(i,j) = \\sum|x\_i - y\_i|
|$d(i,j) = \\sqrt{\\sum (x\_i - y\_i)^2}$|Euclidean distance|
|$d(i,j) = \\max|x\_i - y\_i|
|$\\cos(d\_1,d\_2) = \\frac{d\_1 \\cdot d\_2}{\|d\_1\|\|d\_2\|}$|Cosine similarity|
|$c\_j = \\frac{1}{|C\_j|

\---

*End of Units 5 \& 6 - Complete Theory \& Formulas (No Numericals)*

