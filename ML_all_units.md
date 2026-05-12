# Complete Machine Learning Notes - All Units

## UNIT 1: MACHINE LEARNING FUNDAMENTALS

\---

### 1.1 What is Machine Learning?

**Definition (Tom Mitchell, 1998):**

> "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E."

**PTE Framework:** A well-defined learning task is given by `<P, T, E>`

|Component|Meaning|Example (Spam Filter)|
|-|-|-|
|**Task (T)**|Activity to be performed|Classify emails as spam/not spam|
|**Experience (E)**|Data the system learns from|Previously labeled emails|
|**Performance (P)**|Measure to evaluate success|Accuracy of classification|

\---

### 1.2 Traditional Programming vs Machine Learning

|Traditional Programming|Machine Learning|
|-|-|
|Programmer writes explicit rules and logic|Computer learns patterns from data|
|Rules are fixed|Model improves with more data|
|Data + Program → Output|Data + Output → Program|

\---

### 1.3 ML Terminology

|Term|Definition|Example|
|-|-|-|
|**Dataset**|Collection of examples/records|Student marks table|
|**Feature**|Input variable describing aspect of data|Age, salary, height|
|**Label/Target**|Output variable the model tries to predict|Pass/Fail, Price, Disease type|
|**Training Data**|Portion used to train the model (usually 60-80%)|60% of dataset|
|**Validation Data**|Used to tune model hyperparameters|20% of dataset|
|**Testing Data**|Used to evaluate final model performance|20% of dataset|
|**Model**|Mathematical representation of learning|y = mx + c|
|**Algorithm**|Step-by-step procedure to train model|Gradient Descent|
|**Model Complexity**|Number of input parameters/features|Low: linear regression, High: deep neural networks|

**Generalization:** Ability of a model to perform well on unseen data.

\---

### 1.4 Types of Machine Learning

#### Supervised Learning

```
Input (x) + Output (y) → Model → Predictions
```

|Aspect|Description|
|-|-|
|Data Type|Labeled (input + output)|
|Learning|Maps inputs to known outputs|
|Error Correction|Guided by error correction|
|Accuracy|Easily measurable|
|Human Effort|Requires more labeling effort|

**Examples:**

* Regression (continuous output): House price prediction, Temperature forecasting
* Classification (discrete output): Spam detection, Disease diagnosis

#### Unsupervised Learning

```
Input (x) only → Model → Hidden Patterns
```

|Aspect|Description|
|-|-|
|Data Type|Unlabeled (only input)|
|Learning|Discovers hidden patterns|
|Error Correction|No error correction (outputs unknown)|
|Accuracy|Difficult to measure|
|Human Effort|Requires less effort|

**Examples:**

* Clustering: Customer segmentation, Document categorization
* Association: Market basket analysis

\---

### 1.5 Underfitting and Overfitting

|Condition|Underfitting|Good Fit|Overfitting|
|-|-|-|-|
|**Definition**|Model too simple to capture pattern|Model captures true pattern|Model memorizes training data|
|**Bias**|High|Low|Low|
|**Variance**|Low|Low|High|
|**Training Error**|High|Low|Very Low (close to 0)|
|**Testing Error**|High|Low|High|
|**Complexity**|Low|Optimal|High|
|**Example**|Straight line on nonlinear data|Balanced curve|Wavy curve through every point|

\---

### 1.6 Sources of Error: Noise, Bias, Variance

#### Noise

**Definition:** Random variation in observed data that cannot be explained by the model or true function.

**Formula:**
$$\\epsilon\_i = y\_i - f\_w^{(true)}(x\_i)$$

**Characteristics:**

* Noise is random and unpredictable
* Noise cannot be reduced by improving the model
* Noise exists in all real-world datasets
* Noise represents **irreducible error**

**Causes of Noise:**

* Measurement errors
* Missing variables
* Environmental factors
* Data collection errors
* Random real-world variations

**Example:** Two houses with same size may have different prices due to location or condition.

#### Bias

**Definition:** Error due to overly simple assumptions in the model. Bias measures the difference between the true function and the average prediction of the model across all possible training datasets.

**Formula:**
$$\\text{Bias}(x) = f\_w^{(true)}(x) - \\bar{f}\_w(x)$$

Where:

* $f\_w^{(true)}(x)$ = true function (actual correct relationship)
* $\\bar{f}\_w(x)$ = average prediction of the model

**Properties of High Bias:**

* Model cannot capture real pattern
* Both training and testing errors are high
* Leads to underfitting
* Low complexity → High bias

#### Variance

**Definition:** Error due to model being too sensitive to training data. Variance measures how much the model's predictions change when trained on different datasets.

**Formula:**
$$\\text{Variance}(x) = f\_w^{(train\_i)}(x) - \\bar{f}\_w(x)$$

**Properties:**

* Small difference → Low variance
* Large difference → High variance
* Low complexity → Low variance
* High complexity → High variance
* Models remain stable for low variance

**Variance in High Complexity Model:**

* Model fits training data very closely
* Predictions change significantly with different training datasets
* Model becomes sensitive to training data
* Leads to overfitting

\---

### 1.7 Bias-Variance Trade-off

**Key Relationship:**

* Bias and variance move in opposite directions as model complexity changes
* Low complexity: High bias, Low variance → Underfitting
* High complexity: Low bias, High variance → Overfitting
* Optimal point: Balance that minimizes total error

```
Total Error = Bias² + Variance + Irreducible Error
```

|Model Complexity|Bias|Variance|Training Error|Testing Error|
|-|-|-|-|-|
|Low (Underfitting)|High|Low|High|High|
|Good Fit|Low|Low|Low|Low|
|High (Overfitting)|Low|High|Low|High|

\---

### 1.8 Model Selection

**Definition:** Process of choosing the best model that gives accurate predictions on both training and testing data.

**Goal:** Choose model with right balance between bias and variance:

* Not too simple → avoids high bias
* Not too complex → avoids high variance

**Characteristics of Ideal Model:**

* Low bias
* Low variance
* Low training error
* Low testing error
* Good generalization

\---

### 1.9 Real-World Applications of ML

|Application|ML Task|Example|
|-|-|-|
|Email Filtering|Classification|Gmail Spam Filter, Outlook Junk Mail|
|Recommendation Systems|Collaborative Filtering|Netflix, Amazon, YouTube|
|Medical Diagnosis|Classification|Cancer detection from X-rays/MRI|
|Stock Market|Regression/Prediction|Algorithmic trading|
|Image Recognition|Classification|Face detection, Object recognition|
|Natural Language Processing|Classification|Sentiment analysis, Chatbots|

\---

## UNIT 2: EXPLORATORY DATA ANALYSIS (EDA)

\---

### 2.1 What is EDA?

Exploratory Data Analysis is the process of analyzing and understanding data before building a machine learning model.

**EDA helps to:**

* Understand data structure
* Detect errors
* Handle missing values
* Detect outliers
* Prepare data for model training

**Why EDA is needed:**
Raw data is usually:

* Incomplete
* Noisy
* Unstructured
* Contains errors

\---

### 2.2 Missing Value Treatment

**Missing values occur due to:**

1. Data collection issues
2. Sensor failures
3. Incomplete surveys
4. Data corruption

**Why Missing Values are a Problem:**

* ML algorithms cannot handle missing values directly
* Can lead to biased models
* Incorrect mean and variance calculations
* Treatment improves data quality and performance

#### Method 1: Mean Imputation

**Formula:**
$$\\text{Mean} = \\frac{\\sum x}{n}$$

**Used for:** Numerical data with normal distribution, no extreme outliers

**Example:** Values: 10, 20, NaN, 40

* Mean = (10 + 20 + 40)/3 = 23.3
* Replace NaN → 23.3

**Advantages:** Simple and fast
**Limitations:** Offsets data distribution, reduces variance

#### Method 2: Median Imputation

**Used for:** Numerical data that is skewed or contains outliers

**Example:** Values: 10, 20, NaN, 40, 50

* Median = 30
* Replace NaN → 30

**Advantages:** Not affected by extreme values

#### Method 3: Mode Imputation

**Used for:** Categorical data

**Example:** Gender: M, F, M, NaN, M

* Mode = M
* Replace NaN → M

#### Method 4: Random Sample Imputation

**Process:**

1. Divide dataset into NaN and non-NaN values
2. Randomly select values from non-NaN group
3. Replace NaN positions with selected values

**Example:**
Original: 10, NaN, 20, NaN, 30
Without NaN: 10, 20, 30
Random selection: NaN → 20, NaN → 10
Result: 10, 20, 20, 10, 30

**Advantages:** Preserves original data distribution, maintains variance
**Limitations:** Introduces randomness, may add noise

#### Method 5: End of Tail Distribution Imputation

**Used for:** Skewed data

**Process:**

* Positive skew → use right tail values
* Negative skew → use left tail values

**Example (Positive Skew):**
Dataset: 10, 12, 15, NaN, 20, 100
Tail value = 100
Replace NaN → 100

**Why used:** Mean may not represent skewed data properly; tail values preserve skewness

#### Method 6: Arbitrary Value Imputation

Replace missing value with fixed value like: 0, -1, 999

**Example:**
Salary: 20000, NaN, 30000
Replace NaN → -1

**Why used:** When missing values have special meaning; helps model distinguish missing values

#### Method 7: Forward Fill (FFill)

Replace missing value with previous available value

**Used for:** Time-series or sequential data

**Example:** Values: 10, 20, NaN, 40
Replace NaN → 20

#### Method 8: Backward Fill (BFill)

Replace missing value with next available value

**Used for:** Time-series or sequential data

**Example:** Values: 10, 20, NaN, 40
Replace NaN → 40

**Limitation:** May introduce future information into past records

\---

### 2.3 Handling Categorical Data

**Problem:** ML models work only on numerical data; categorical data must be converted to numerical form.

#### Types of Categorical Features

|Type|Definition|Example|Encoding Method|
|-|-|-|-|
|**Nominal**|Categories with no natural order/ranking|Colors (R,G,B), Cities, Gender|One-Hot Encoding|
|**Ordinal**|Categories with meaningful order/ranking|Size (S,M,L), Rating (Low,Medium,High)|Label Encoding / Ordering|

#### One-Hot Encoding (For Nominal Data)

Each category becomes a separate binary column (0 or 1)

**Example - Gender:**

* Male → (1, 0)
* Female → (0, 1)

|Gender|Male|Female|
|-|-|-|
|M|1|0|
|F|0|1|

**Advantages:**

* No false ordering introduced
* Suitable for nominal data

**Limitations:**

* Increases number of features
* Curse of dimensionality

#### Curse of Dimensionality

**Definition:** Problems that arise when number of features becomes very large.

**Effects:**

* Increased memory usage
* Increased computation time
* Increased model complexity
* Risk of overfitting
* Data becomes sparse

**Example:** If a column has 100 categories, one-hot encoding creates 100 columns.

**Sparsity:** When number of features increases, data points become more spread out (sparse), making outliers easier to detect.

#### Label Encoding (For Ordinal Data)

Map categories to integers based on natural order.

**Example - Rating:**

* Low → 1
* Medium → 2
* High → 3

**Advantage:** Preserves ranking information

#### Handling Missing Values in Categorical Data

1. **Mode Imputation:** Replace with most frequent category
2. **Random Imputation:** Replace with random category
3. **Create New Category:** Replace with "Unknown"

\---

### 2.4 Outliers

**Definition:** Outliers are observations that are unusually large or small compared to the rest of the data. They lie far away from the majority of observations.

**Example:** Salary dataset: 20000, 22000, 25000, 27000, 30000, 500000

* 500000 is an outlier

**Causes of Outliers:**

1. Data entry error (typing 500 instead of 50)
2. Measurement error
3. Natural variation
4. Rare events

**Why Outlier Detection is Needed:**

* Improves model accuracy
* Removes incorrect values
* Improves data quality
* Outliers may affect mean and standard deviation

\---

### 2.5 Outlier Detection Methods

#### Method 1: Box Plot (Tukey's Box Plot Method)

**Components:**

* Q1 (25th quartile)
* Q3 (75th quartile)
* IQR = Q3 - Q1

**Formulas:**

* Inner fence: \[Q1 - 1.5×IQR, Q3 + 1.5×IQR]
* Outer fence: \[Q1 - 3×IQR, Q3 + 3×IQR]

**Interpretation:**

* Values outside inner fence → Possible outliers
* Values outside outer fence → Probable outliers

**Advantages:**

* Robust to outliers and skewed data
* Simple to compute
* Works well for real-world data

**Disadvantages:**

* Not suitable for small datasets
* Ignores overall data distribution shape

#### Method 2: Z-Score Method

**Step 1: Variance Formula**
$$\\text{Var} = \\frac{\\sum(x - \\bar{x})^2}{n}$$

**Step 2: Standard Deviation Formula**
$$\\sigma = \\sqrt{\\text{Var}} = \\sqrt{\\frac{\\sum(x - \\bar{x})^2}{n}}$$

**Step 3: Z-Score Formula**
$$Z = \\frac{x - \\mu}{\\sigma}$$

**Empirical Rule (68-95-99.7 Rule):**

* Within ±1σ: 68% of data
* Within ±2σ: 95% of data
* Within ±3σ: 99.7% of data

**Outlier Condition:**
$$|Z| > 3$$

Value outside μ ± 3σ is considered an outlier (only 0.3% of values lie outside this range)

**Example:**
Dataset: 10, 12, 14, 15, 18, 100
μ = 28.17
Z for 100 = (100 - 28.17)/σ → If Z > 3 → Outlier

**Advantages:**

* Uses mean and standard deviation
* Works well for normally distributed data
* Detects extreme values accurately

**Limitations:**

* Works only for normally distributed data
* Affected by extreme values
* Not suitable for skewed data

**For Skewed Data:** Use log transformation to convert to normal distribution

#### Comparison: IQR vs Z-Score

|Aspect|IQR Method|Z-Score Method|
|-|-|-|
|Uses|Quartiles|Mean and standard deviation|
|Best for|Skewed data|Normal distribution|
|Affected by outliers|No|Yes|

\---

### 2.6 Outlier Treatment

#### Method 1: Mean/Median/Random Imputation

Replace outliers with appropriate values (preserves dataset size)

**Mean Imputation:** Replace with mean (for normal distribution)
**Median Imputation:** Replace with median (for skewed data) - preferred as median not affected by outliers
**Random Imputation:** Replace with randomly selected normal values

#### Method 2: Trimming

Remove outliers completely from dataset (reduces dataset size)

**When to use:**

* Outlier is clearly incorrect
* Outlier due to data entry error
* Outlier affects model performance

**Advantages:** Improves model performance, removes extreme values
**Disadvantage:** Data loss occurs

#### Method 3: Capping

Replace outlier with maximum or minimum normal value (tail value)

**Example:**
Data: 10, 12, 14, 15, 200
Upper normal limit = 15
Replace 200 → 15
Final data: 10, 12, 14, 15, 15

\---

### 2.7 Feature Engineering

**Definition:** Process of transforming raw data into useful features so that machine learning models can perform better.

**Includes:**

* Transforming features
* Scaling features
* Creating new features
* Making features comparable

**Why Feature Engineering is Needed:**
Different features have different ranges, causing statistical imbalance.

**Example:**
MTT Marks: 2, 3, 4, 5 (small range)
TEE Marks: 40, 60, 70, 80 (large range)
→ TEE becomes dominant → Model gives more importance to larger values

**Solution:** Apply Feature Scaling

\---

### 2.8 Feature Scaling Techniques

#### 1\. Absolute Maximum Scaling

**Formula:**
$$X\_{new} = \\frac{X}{|X\_{max}|}$$

**Steps:** Find maximum absolute value, divide all values by that maximum

**Example:** Values: 40, 60, 80
Max = 80
Result: 40→0.5, 60→0.75, 80→1
Range: 0 to 1

#### 2\. Min-Max Scaling

**Formula:**
$$X\_{new} = \\frac{X - X\_{min}}{X\_{max} - X\_{min}}$$

**Example:** Values: 40, 60, 80
min=40, max=80
Result: 40→0, 60→0.5, 80→1

#### 3\. Normalization

**Formula:**
$$X\_{new} = \\frac{X - \\text{mean}}{X\_{max} - X\_{min}}$$

**Purpose:** Makes features comparable, reduces imbalance

#### 4\. Standardization (Z-Score Scaling)

**Formula:**
$$X\_{new} = \\frac{X - \\text{mean}}{\\text{standard deviation}}$$

**Result:** Mean becomes 0, standard deviation becomes 1

**Used in:** SVM, Logistic Regression, Neural Networks

#### 5\. Robust Scaling

**Formula:**
$$X\_{new} = \\frac{X - \\text{median}}{IQR}$$

**Where:** IQR = Q3 - Q1

**Advantage:** Works well when outliers are present

#### Summary Table

|Technique|Formula|Range|Best for|
|-|-|-|-|
|Absolute Max Scaling|X /|X\_max||
|Min-Max Scaling|(X-X\_min)/(X\_max-X\_min)|0 to 1|No outliers|
|Normalization|(X-mean)/(X\_max-X\_min)|Variable|Imbalanced features|
|Standardization|(X-mean)/σ|Variable|Normal distribution|
|Robust Scaling|(X-median)/IQR|Variable|Data with outliers|

\---

### 2.9 Feature Selection

**Definition:** Process of selecting a subset of relevant and meaningful features from original dataset.

**Need for Feature Selection:**

* Reduces dimensionality of data
* Improves model performance
* Reduces overfitting
* Decreases training time and computational cost
* Makes models easier to interpret

#### Method 1: Forward Selection

Start with no features, add features one by one.

**Steps:**

1. Start with empty model
2. Add one feature at a time
3. Check model performance
4. Keep feature if performance improves
5. Repeat

**Example:** Features A, B, C
Try: Model with A → Model with A+B → Model with A+B+C
Select best combination

#### Method 2: Backward Selection

Start with all features, remove features one by one.

**Steps:**

1. Start with all features
2. Remove one feature
3. Check performance
4. Remove feature that reduces performance least
5. Repeat

**Example:**
All features (A+B+C) → 90% accuracy
Remove A: (B+C) → 89% accuracy
Remove B: (A+C) → 70% accuracy
Remove C: (A+B) → 88% accuracy
Remove feature with least impact → Remove A
Final model: B+C

#### Method 3: Principal Component Analysis (PCA)

**Definition:** Technique that reduces number of features by creating new features called Principal Components.

**Purpose:**

* Reduce dimensionality
* Remove redundancy
* Reduce noise

**Example:** Features A, B, C
Create PC1 = combination of A+B
Create PC2 = combination of remaining info
Model with PC1 → 91% accuracy
Model with PC1+PC2 → 91% accuracy
PC2 adds little improvement → Select PC1

**Advantages:** Reduces dimensionality, redundancy, noise
**Limitations:** Reduces interpretability

#### Method 4: Correlation Analysis

**Definition:** Measures relationship between features. If two features are highly correlated, remove one.

**Example:** Height and weight → highly correlated → remove one

**Process:**
Features: A, B, C
Check correlation: A-B highly correlated, A-C low, B-C low
Remove B → Final features: A+C

**Advantages:** Simple and fast
**Limitation:** Only captures linear relationships

\---

## UNIT 3: REGRESSION

\---

### 3.1 What is Regression?

**Definition:** Supervised ML algorithm used to predict continuous values.

**Examples:**

* House price prediction
* Temperature forecasting
* Stock price prediction

**Key Concept:** Model learns relationship between independent variables (x) and dependent variable (y)

\---

### 3.2 Cost Functions / Loss Functions

#### Mean Squared Error (MSE)

$$\\text{MSE} = \\frac{1}{n}\\sum\_{i=1}^{n}(y\_i - \\hat{y}\_i)^2$$

#### Root Mean Squared Error (RMSE)

$$\\text{RMSE} = \\sqrt{\\frac{\\sum\_{i=1}^{n}(y\_i - \\hat{y}\_i)^2}{n}}$$

#### Mean Absolute Error (MAE)

$$\\text{MAE} = \\frac{1}{n}\\sum\_{i=1}^{n}|y\_i - \\hat{y}\_i|$$

#### R-squared (Coefficient of Determination)

$$R^2 = 1 - \\frac{SS\_{RES}}{SS\_{TOT}} = 1 - \\frac{\\sum\_i(y\_i - \\hat{y}\_i)^2}{\\sum\_i(y\_i - \\bar{y})^2}$$

\---

### 3.3 Simple Linear Regression

**Equation:**
$$y = w\_0 + w\_1x$$

Where:

* $y$ = predicted value
* $w\_0$ = intercept (bias)
* $w\_1$ = slope (weight)
* $x$ = input feature

#### Least Squares Method (Analytical Approach)

**Slope Formula:**
$$w\_1 = \\frac{\\sum\_{i=1}^{n}(x\_i - \\bar{x})(y\_i - \\bar{y})}{\\sum\_{i=1}^{n}(x\_i - \\bar{x})^2}$$

**Alternative Slope Formula:**
$$w\_1 = \\frac{n\\sum xy - (\\sum x)(\\sum y)}{n\\sum x^2 - (\\sum x)^2}$$

**Intercept Formula:**
$$w\_0 = \\bar{y} - w\_1\\bar{x}$$

**Alternative Intercept Formula:**
$$w\_0 = \\frac{\\sum y - w\_1\\sum x}{n}$$

#### Gradient Descent Method (Iterative Approach)

**Algorithm:**

```
repeat until convergence {
    w₀ = w₀ - α \* (1/m) \* Σ(h\_w(x⁽ⁱ⁾) - y⁽ⁱ⁾)
    w₁ = w₁ - α \* (1/m) \* Σ(h\_w(x⁽ⁱ⁾) - y⁽ⁱ⁾) \* x⁽ⁱ⁾
}
```

**Where:**

* $α$ = learning rate (hyperparameter)
* $m$ = number of training examples
* $h\_w(x) = w\_0 + w\_1x$ = hypothesis

**Update Rules:**
$$w\_{i+1} = w\_i - \\eta \\frac{\\partial \\text{Cost}}{\\partial w\_i}$$

**Gradient Calculation:**
$$\\frac{\\partial \\text{Cost}}{\\partial w\_0} = -\\frac{2}{n}\\sum\_{i=1}^{n}(y\_i - \\hat{y}\_i)$$

$$\\frac{\\partial \\text{Cost}}{\\partial w\_1} = -\\frac{2}{n}\\sum\_{i=1}^{n}(y\_i - \\hat{y}\_i) \\cdot x\_i$$

\---

### 3.4 Multiple Linear Regression

**Equation:**
$$y = w\_0 + w\_1x\_1 + w\_2x\_2 + \\dots + w\_nx\_n$$

Or in matrix form:
$$y = w^T x$$

#### Normal Equation for Multiple Regression

$$W = (X^T X)^{-1} X^T Y$$

This is used for small datasets.

**For large datasets:** Use Gradient Descent

#### Assumptions of Multiple Linear Regression:

1. Linear relationship between variables
2. No strong correlation between independent variables
3. Errors are normally distributed
4. Constant variance (homoscedasticity)

**Advantages:**

* Models real-world problems better
* More accurate predictions using multiple features
* Flexible and scalable

**Disadvantages:**

* Requires more data
* Can overfit if too many features

\---

### 3.5 Polynomial Regression

**Definition:** Used when relationship between independent and dependent variables is non-linear. Fits a curved line by introducing higher-degree terms.

**General Equation:**
$$y = w\_0 + w\_1x + w\_2x^2 + w\_3x^3 + \\dots + w\_nx^n$$

**Degrees:**

* 0 degree: y = constant
* 1 degree: y = mx + c (straight line)
* 2 degree: y = ax² + bx + c (parabola/curve)

**Cost Function for Polynomial Regression:**
$$\\text{MSE} = \\frac{1}{n}\\sum\_{i=1}^{n}\\left(\\sum\_{j=0}^{k} w\_j x\_i^j - y\_i\\right)^2$$

**Why Polynomial Regression is needed:**

* Linear regression underfits non-linear data
* Captures curves and complex patterns
* More flexible than linear regression

**Matrix Form for Polynomial Regression:**
Design matrix includes powers of x:

|x₀=1|x₁ = x|x₂ = x²|y|
|-|-|-|-|
|1|x₁|x₁²|y₁|
|1|x₂|x₂²|y₂|
|1|x₃|x₃²|y₃|

Then use: $W = (X^T X)^{-1} X^T Y$

\---

## UNIT 4: CLASSIFICATION

\---

### 4.1 What is Classification?

**Definition:** Process that takes input features and converts them into useful predictions (labels/categories).

**Example:**

* Email → Spam/Not Spam
* News → Sports/Finance/Technology
* Reviews → Positive/Negative

**Input (x):** Features like words or numbers
**Output (y):** Class labels (discrete, not continuous)

$$y = f(x)$$

\---

### 4.2 Types of Classification

|Type|Description|Example|
|-|-|-|
|**Binary Classification**|Only 2 classes|Spam/Not Spam, Pass/Fail|
|**Multi-class Classification**|More than 2 classes|News: Education/Finance/Technology|
|**Multi-label Classification**|Multiple classes simultaneously|Image containing cat AND dog|

\---

### 4.3 Confusion Matrix

A table showing comparison between actual and predicted values.

||Predicted Positive (1)|Predicted Negative (0)|
|-|-|-|
|**Actual Positive (1)**|**True Positive (TP)**|**False Negative (FN)**|
|**Actual Negative (0)**|**False Positive (FP)**|**True Negative (TN)**|

#### Definitions:

|Term|Meaning|Error Type|
|-|-|-|
|**True Positive (TP)**|Correctly predicted positive|-|
|**True Negative (TN)**|Correctly predicted negative|-|
|**False Positive (FP)**|Wrongly predicted positive|Type I Error|
|**False Negative (FN)**|Wrongly predicted negative|Type II Error|

**Example - Spam Detection:**

* Email is spam → predicted as spam → TP
* Email is spam → predicted as not spam → FN

\---

### 4.4 Performance Metrics

#### Accuracy

$$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$

* Overall correctness of model
* Use when dataset is balanced

#### Precision

$$\\text{Precision} = \\frac{TP}{TP + FP}$$

* Out of all predicted positives, how many were correct
* Focus: Prediction quality (avoid false positives)
* Example (Spam): High precision → very few normal emails marked as spam

#### Recall (Sensitivity / True Positive Rate)

$$\\text{Recall} = \\frac{TP}{TP + FN}$$

* Out of actual positives, how many were correctly detected
* Focus: Capturing all positives (avoid false negatives)
* Example (Disease): High recall → fewer missed patients

#### F1-Score

$$\\text{F1} = \\frac{2 \\times \\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$

* Harmonic mean of precision and recall
* Balances both metrics
* Use when both precision and recall are important

#### Specificity (True Negative Rate)

$$\\text{Specificity} = \\frac{TN}{TN + FP}$$

* Out of actual negatives, how many were correctly identified

#### Summary Table

|Metric|Formula|Focus|
|-|-|-|
|Accuracy|(TP+TN)/(Total)|Overall correctness|
|Precision|TP/(TP+FP)|Avoid false positives|
|Recall|TP/(TP+FN)|Avoid false negatives|
|F1-Score|2×(P×R)/(P+R)|Balance P and R|
|Specificity|TN/(TN+FP)|Correctly identify negatives|

#### One-Line Definitions (Exam Ready)

* **Accuracy:** Ratio of correct predictions to total predictions
* **Precision:** Correct positive predictions out of predicted positives
* **Recall:** Correct positives out of actual positives
* **F1-score:** Harmonic mean of precision and recall
* **Specificity:** Correct negatives out of actual negatives

#### When to Use What

|Scenario|Recommended Metric|
|-|-|
|Spam detection|Precision (don't mark good emails as spam)|
|Disease detection|Recall (don't miss positive patients)|
|Imbalanced data|F1-score|
|False alarms matter|Specificity|

#### Numerical Example

Given: Total emails = 100, Spam = 20, Normal = 80

**Case 1: Very Strict Model**

* Predicted spam = 10 (all correct)
* TP = 10, FP = 0, FN = 10
* Precision = 10/(10+0) = 1.0
* Recall = 10/(10+10) = 0.5

**Case 2: Very Loose Model**

* Predicted spam = 30 (20 correct, 10 wrong)
* TP = 20, FP = 10, FN = 0
* Precision = 20/(20+10) = 0.66
* Recall = 20/(20+0) = 1.0

**Key Insight:**

* **Precision = Be correct when saying YES**
* **Recall = Don't miss any YES**
* **Specificity = Be correct when saying NO**

\---

### 4.5 ROC Curve (Receiver Operating Characteristic Curve)

**Definition:** Graph used to evaluate classification model at different thresholds. Shows how well the model separates classes.

**What is Plotted:**

* X-axis: **False Positive Rate (FPR)**
* Y-axis: **True Positive Rate (TPR)**

**Formulas:**
$$TPR = \\text{Recall} = \\frac{TP}{TP + FN}$$
$$FPR = \\frac{FP}{FP + TN}$$

**Threshold Concept:**
Model gives probabilities (0.2, 0.7, etc.)

* If probability > threshold → Positive
* If probability < threshold → Negative

**Effect of Changing Threshold:**

* Lower threshold → More positives predicted → TPR ↑ (good), FPR ↑ (bad)
* Higher threshold → Fewer positives predicted → TPR ↓, FPR ↓

**Interpretation:**

* Top-left curve → Best model
* Diagonal line → Random guessing (AUC = 0.5)
* Below diagonal → Worst model

\---

### 4.6 AUC (Area Under the ROC Curve)

**Definition:** Area under the ROC curve. Measures total area under the ROC graph.

**Range:** $0 \\leq AUC \\leq 1$

**Interpretation:**

|AUC Value|Meaning|
|-|-|
|1.0|Perfect model|
|0.9+|Excellent|
|0.8|Good|
|0.7|Fair|
|0.5|Random guessing|
|< 0.5|Poor model|

**Key Insight:**

* Higher AUC → better model
* AUC considers all thresholds
* Better than just accuracy

\---

### 4.7 k-Fold Cross Validation

**Definition:** Technique to evaluate model performance more reliably by splitting data into k equal parts (folds) and training/testing multiple times.

**How it Works:**

1. Divide dataset into **k equal folds**
2. Repeat k times:

   * Use 1 fold as test data
   * Use remaining (k-1) folds as training data
3. Compute accuracy (or any metric) each time
4. Take **average of all results**

**Example (k=5):**

* Fold 1: Test, Folds 2-5: Train
* Fold 2: Test, Folds 1,3,4,5: Train
* Fold 3: Test, Folds 1,2,4,5: Train
* Fold 4: Test, Folds 1,2,3,5: Train
* Fold 5: Test, Folds 1,2,3,4: Train

**Final Score:**
$$\\text{Final Score} = \\frac{\\text{Score}\_1 + \\text{Score}\_2 + \\dots + \\text{Score}\_k}{k}$$

**Why Use It:**

* Reduces overfitting risk
* Uses entire dataset efficiently
* Gives more reliable performance
* Less bias
* Works well for small datasets

**Common k values:** k = 5 or k = 10

**Example:**
Accuracies: 80%, 85%, 78%, 82%, 84%
Final Accuracy = (80+85+78+82+84)/5 = 81.8%

\---

### 4.8 Linear Classifier

**Definition:** Separates data using a straight line (decision boundary).

**Mathematical Form:**
$$\\text{Score}(x) = w^T x + b$$

**Decision Rule:**

* If Score(x) ≥ 0 → Class +1
* If Score(x) < 0 → Class -1

**Where:**

* w = weights
* b = bias
* In 2D → line
* In higher dimensions → hyperplane

\---

### 4.9 Why Not Use Linear Regression for Classification?

**Problem:** Linear model output: $(-\\infty, +\\infty)$, but we need probability in \[0,1]

**Solution: Link Function**
Converts $(-\\infty, +\\infty) \\rightarrow (0,1)$

$$\\hat{P}(y = +1 \\mid x) = g(w^T x)$$

Where g() is a link function that squeezes the real line into \[0,1].

\---

### 4.10 Logistic Regression

#### Sigmoid Function (Logistic Function)

**Formula:**
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

**What it does:** Converts any value from $(-\\infty, +\\infty) \\rightarrow (0,1)$

**Key Values:**

|z (input)|Output σ(z)|
|-|-|
|+∞|≈ 1|
|0|0.5|
|-∞|≈ 0|

**Shape:** S-shaped curve (sigmoid)

#### Logistic Regression Model

**Step 1: Linear Score**
$$\\text{Score}(x\_i) = w^T h(x\_i)$$

**Step 2: Apply Sigmoid**
$$P(y = +1 \\mid x\_i, w) = \\sigma(\\text{Score}(x\_i)) = \\frac{1}{1 + e^{-w^T h(x\_i)}}$$

**Decision Rule:**

* If probability ≥ 0.5 → Class +1
* If probability < 0.5 → Class 0

**Why 0.5?** Middle point of sigmoid, corresponds to $w^T x = 0$

**Key Insight:** Logistic regression creates non-linear probability curve but linear decision boundary.

#### Binary Cross Entropy (Log Loss) Cost Function

For a single example:
$$\\text{cost}(y\_{hat}, y) = \\begin{cases}
-\\log(g(y\_{hat})) \& \\text{if } y = 1 \\
-\\log(1 - g(y\_{hat})) \& \\text{if } y = 0
\\end{cases}$$

Combined formula:
$$\\text{cost}(y\_{hat}, y) = -y \\cdot \\log(g(y\_{hat})) - (1-y) \\cdot \\log(1 - g(y\_{hat}))$$

**Total Cost:**
$$\\text{Cost}(W) = \\frac{1}{m} \\sum\_{i=1}^{m} \[-y\_i \\cdot \\log(g(y\_{hat\_i})) - (1-y\_i) \\cdot \\log(1 - g(y\_{hat\_i}))]$$

#### Gradient for Logistic Regression

$$\\frac{d\\text{cost}}{dw} = (\\hat{y} - y) \\cdot X$$

**Update Rule:**
$$\\beta\_j = \\beta\_j - \\alpha (\\hat{y} - y) x\_j$$

\---

### 4.11 Naive Bayes Classifier

**Definition:** Probabilistic classification algorithm based on Bayes' theorem.

**Key Assumption:** All features are independent of each other given the class (this is "naive" - not true in real world but works well).

#### Bayes' Theorem

$$P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}$$

Where:

* $P(A)$ = Prior probability (probability of event A)
* $P(B)$ = Evidence (probability of event B)
* $P(B|A)$ = Likelihood (probability of B given A)
* $P(A|B)$ = Posterior probability (probability of A given B)

**For Classification:**
$$P(\\text{Class} | \\text{Features}) = \\frac{P(\\text{Features} | \\text{Class}) \\cdot P(\\text{Class})}{P(\\text{Features})}$$

**Advantages:**

* Simple and easy to implement
* Works well with large datasets
* Fast training and prediction
* Performs well with text data
* Requires less training data
* Not very sensitive to irrelevant features
* Handles both continuous and discrete data

**Disadvantages:**

* Assumes feature independence (usually false)
* Less accurate for complex problems
* Sensitive to data distribution
* Cannot capture feature relationships
* Zero probability problem (if not handled with Laplace smoothing)

**Applications:**

* Text classification
* Spam detection
* Sentiment analysis

\---

### 4.12 Support Vector Machine (SVM)

**Definition:** Supervised ML algorithm for classification and regression. Finds optimal hyperplane that separates data points of different classes.

**Core Idea:** Plot each data item as a point in n-dimensional space (n = number of features). Perform classification by finding hyperplane that differentiates classes well.

**Objective:** Maximize distance between hyperplane and nearest data points (the margin).

#### SVM Terminology

|Term|Definition|
|-|-|
|**Hyperplane**|Decision boundary that separates classes (exists in multiple dimensions)|
|**Support Vectors**|Data points closest to the hyperplane; pass through marginal hyperplane|
|**Margin**|Distance between marginal plane and actual plane|
|**Linear Separable**|Samples can be separated through hyperplane|
|**Non-linear Separable**|Samples cannot be separated through hyperplane|
|**SVM Kernel**|Converts low-dimensional data into high-dimensional data to make it separable|

#### Mathematical Formulation

**Decision Boundary:**
$$w^T x + b = 0$$

**Margin Boundaries:**

* $w^T x + b = +1$ (for positive class)
* $w^T x + b = -1$ (for negative class)

**Margin Width:**
$$\\text{Margin} = \\frac{2}{|w|}$$

**SVM Objective:**
$$\\text{Maximize Margin} = \\frac{2}{|w|}$$
Equivalent to:
$$\\text{Minimize } |w|^2$$

#### Finding the Hyperplane

For support vectors $x\_1$ and $x\_2$:
$$w^T x\_1 + b = 1$$
$$w^T x\_2 + b = -1$$

Subtracting:
$$w^T(x\_1 - x\_2) = 2$$

**Optimal Function:**
$$f(w^*, b^*) = \\frac{1}{2} |w|^2$$

#### Hard Margin vs Soft Margin

|Type|Description|
|-|-|
|**Hard Margin**|Assumes data is perfectly separable; points must lie outside margin|
|**Soft Margin**|Allows some misclassification for non-linearly separable data|

#### Kernel Functions

Used to transform data into higher-dimensional space to make it linearly separable.

**Common Kernels:**

* Linear kernel
* Polynomial kernel
* RBF (Radial Basis Function) kernel
* Sigmoid kernel

\---

### 4.13 Neural Networks

**Definition:** Computational model inspired by the structure and function of the human brain.

#### Basic Building Blocks

**Artificial Neuron (Perceptron):**

* Receives inputs, processes them, produces output
* Consists of 2 parts: Summer function and Activation function

**Components of a Neural Network:**

|Component|Function|
|-|-|
|**Input Layer**|Receives initial data/features to be processed|
|**Hidden Layers**|Intermediate layers between input and output; data flow not directly observable|
|**Output Layer**|Final layer that produces model's output|
|**Weights**|Numerical values associated with connections between neurons|
|**Bias**|Constant term added to weighted sum, helps shift decision boundary|
|**Activation Function**|Applied to neuron output to introduce non-linearity|

#### Neuron Mathematical Model

**Summer Function:**
$$z = \\sum\_{i=0}^{k} w\_i x\_i$$

**Activation Function:**
$$y = f(z)$$

Where f() is an activation function (sigmoid, ReLU, tanh, etc.)

#### Structure

```
Input Layer → Hidden Layer(s) → Output Layer
```

* Each neuron connected to neurons in next layer (fully connected)
* Data flows forward (feedforward) or both directions (backpropagation)

#### Types of Perceptron

**1. Single-Layer Perceptron:**

* Simplest ANN type
* Feed-forward network with threshold transfer
* Can only learn linearly separable patterns
* Main objective: Analyze linearly separable objects with binary outcomes

**2. Multi-Layer Perceptron (MLP):**

* Similar to single-layer but has more hidden layers
* Can learn both linear and non-linear patterns
* Implements logic gates (AND, OR, XOR, XNOR, NOR)

#### Stages in MLP

1. **Forward Stage:** Data flows from input layer to output layer; activation functions applied at each layer
2. **Backward Stage (Back-propagation):** Weights and biases modified based on error; error propagated backward from output to input

#### Forward Propagation

Process through which data passes through layers of neurons from input layer to output layer.

#### Back-propagation

**Definition:** Supervised learning algorithm for training neural networks.

**Process:**

1. Calculate total cost (error between actual and predicted values)
2. Propagate error back into network
3. Update each weight using gradient descent

**Weight Update Rule:**
$$\\Delta w\_i = \\eta \\cdot (y\_i - \\hat{y}\_i) x\_i$$
$$w\_i' = w\_i + \\Delta w\_i$$
$$b = b + \\eta \\cdot (y\_i - \\hat{y}\_i)$$

Where:

* $\\eta$ = learning rate
* $y\_i$ = actual output
* $\\hat{y}\_i$ = predicted output

#### Activation Functions

|Function|Formula|Output Range|
|-|-|-|
|**Step Function**|$f(z) = 1 \\text{ if } z > t, 0 \\text{ else}$|{0,1}|
|**Sign Function**|$f(z) = 1 \\text{ if } z > 0, -1 \\text{ if } z < 0, 0 \\text{ if } z = 0$|{-1,0,1}|
|**Sigmoid**|$f(z) = \\frac{1}{1 + e^{-z}}$|(0,1)|
|**ReLU**|$f(z) = \\max(0, z)$|\[0,∞)|
|**Leaky ReLU**|$f(z) = \\max(0.01z, z)$|(-∞,∞)|
|**Tanh**|$f(z) = \\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$|(-1,1)|

#### Example: Logic Gates with Perceptron

**AND Gate:**
Weights: w₁ = 0.7, w₂ = 0.6, Threshold = 1

|A|B|A∧B|Weighted Sum|Output|
|-|-|-|-|-|
|0|0|0|0|0|
|0|1|0|0.6|0|
|1|0|0|0.7|0|
|1|1|1|1.3|1|

**OR Gate:**
Weights: w₁ = 1.1, w₂ = 1.1, Threshold = 1

\---

## UNIT 5: TREE-BASED METHODS

\---

### 5.1 Decision Trees

**Definition:** Supervised learning technique for classification and regression. Decisions are made by recursively splitting dataset based on feature values.

**Structure:**

* **Root Node:** First node at top, contains complete dataset
* **Decision Nodes (Internal Nodes):** Intermediate nodes with conditions/questions
* **Leaf Nodes (Terminal Nodes):** Final nodes giving output (no further splitting)
* **Branches:** Connect nodes, represent decision outcomes
* **Parent Node:** Node that divides into child nodes
* **Child Node:** Nodes formed after split
* **Subtree:** Smaller section of full tree
* **Pruning:** Removing less useful branches to reduce overfitting

**Core Idea:** Divide and conquer - break dataset into smaller subsets, apply decision rules at each step, continue until clear decision is reached.

#### Decision Tree Learning Algorithm

**Steps:**

1. Start with entire dataset (becomes root)
2. Select best feature to split (using Information Gain or Gini Index)
3. Split dataset based on selected feature; create child nodes
4. Repeat recursively on each subset
5. Stop when:

   * All data belongs to same class
   * No features left
   * Minimum samples reached
   * Tree depth limit reached
6. Leaf nodes get final prediction (majority class)

#### Attribute Selection Measures

**1. Entropy**

Measure of randomness or uncertainty.

$$\\text{Entropy}(S) = -\\sum\_{i=1}^{c} p\_i \\log\_2(p\_i)$$

Where:

* $p\_i$ = proportion of class i in dataset
* Lower entropy → less uniform distribution

**2. Information Gain (IG)**

Information that can increase level of certainty after splitting.

$$\\text{IG}(S, A) = \\text{Entropy}(S) - \\sum\_{v \\in \\text{Values}(A)} \\frac{|S\_v|}{|S|} \\cdot \\text{Entropy}(S\_v)$$

Where:

* Entropy(S) = entropy of parent
* Weighted average of children entropies

**3. Gini Index**

Calculated by subtracting sum of squared probabilities from one.

$$\\text{Gini}(S) = 1 - \\sum\_{i=1}^{c} (p\_i)^2$$

**For a split:**
$$\\text{Gini}*{\\text{split}}(S, A) = \\sum*{v \\in \\text{Values}(A)} \\frac{|S\_v|}{|S|} \\cdot \\text{Gini}(S\_v)$$

* Lower Gini index → better split
* Favors larger partitions (unlike IG which favors smaller partitions with distinct values)

#### Example: Gini Index Calculation

For a node with 12 total samples (7 Yes, 5 No):
$$p\_{\\text{Yes}} = \\frac{7}{12}, \\quad p\_{\\text{No}} = \\frac{5}{12}$$
$$\\text{Gini} = 1 - \\left\[\\left(\\frac{7}{12}\\right)^2 + \\left(\\frac{5}{12}\\right)^2\\right] = 1 - \\left\[0.340 + 0.174\\right] = 0.486$$

#### Regression Tree vs Classification Tree

|Aspect|Classification Tree|Regression Tree|
|-|-|-|
|Output type|Categorical|Continuous|
|Target variable|Discrete class|Numerical value|
|Split measure|Entropy, Gini|MSE, MAE|
|Leaf output|Class label|Numeric value (average)|
|Objective|Classification|Value prediction|
|Example|Spam detection|House price prediction|

#### Trees vs Linear Models

|Aspect|Linear Models|Decision Trees|
|-|-|-|
|Relationship|Assumes linear|Captures non-linear|
|Interpretability|Easy (coefficients)|Very easy (visual)|
|Scaling needed|Yes|No|
|Feature selection|Manual|Automatic|
|Overfitting|Less prone|Prone (needs pruning)|

#### Advantages of Decision Trees

* Easy to understand and explain
* Can handle both categorical and numerical data
* Results can be visualized
* No need for scaling/normalization
* Automatic feature selection
* Fast prediction time
* Handles mixed data

#### Disadvantages

* Prone to overfitting (without pruning)
* Can be unstable (small data changes = different tree)
* Not always optimal (greedy algorithm)

#### Pruning

**Definition:** Removing unnecessary branches to reduce overfitting.

|Type|Description|
|-|-|
|**Pre-pruning**|Stops tree growth early during training|
|**Post-pruning**|Builds full tree first, removes weak branches later|

**Benefits:** Reduces overfitting, better accuracy on test data, simpler model

\---

### 5.2 Ensemble Learning

**Definition:** Combining multiple models to create a strong model.

**Why Ensemble:**

* One model may make mistakes, other models can correct them
* Final answer is usually more accurate than single model

#### Main Types: Bagging and Boosting

|Aspect|Bagging|Boosting|
|-|-|-|
|Approach|Parallel (independent models)|Sequential (one after another)|
|Data sampling|Bootstrap samples (with replacement)|Same data with different weights|
|Model weights|Equal|Different (based on performance)|
|Focus|Reduce variance|Reduce bias|
|Overfitting risk|Lower|Higher (if not careful)|

\---

### 5.3 Bagging (Bootstrap Aggregating)

**Definition:** Ensemble method where many decision trees are created independently, each trained on a different random sample of the original dataset.

**Process:**

1. Take random samples from original dataset (with replacement - same record can be chosen multiple times)
2. Train one decision tree on each sample
3. For prediction:

   * Classification: Majority vote
   * Regression: Average value

**Benefits:**

* Reduces overfitting
* Increases accuracy
* Predictions become more stable

\---

### 5.4 Random Forest

**Definition:** Advanced version of Bagging. Creates many decision trees and combines them. One of the most popular ML algorithms.

**Process:**

1. Take many random samples from original data (with replacement)
2. Train one decision tree on each sample
3. At each split, only consider a random subset of features (not all features)
4. This makes all trees different from each other
5. Final output:

   * Classification: Most voted class
   * Regression: Average value

**Advantages:**

* Better accuracy than single tree
* Reduces overfitting
* Handles noisy data well
* Works well on large datasets

**Disadvantages:**

* Requires more memory and computation
* Less interpretable than single tree

\---

### 5.5 Boosting

**Definition:** Ensemble method where trees are built sequentially. Each new tree tries to correct mistakes made by previous trees.

**Process:**

1. First tree trained on data
2. Calculate errors made by first tree
3. Second tree focuses on wrong predictions (gives more weight to misclassified points)
4. Third tree corrects mistakes of second tree
5. Continue for many iterations

**Benefits:** Same as Random Forest (reduces variance, improves accuracy)

\---

### 5.6 AdaBoost (Adaptive Boosting)

**Definition:** Boosting algorithm where many small weak models are combined to make a strong model.

**Key Feature:** Gives more importance (higher weight) to difficult data points so next model focuses on them.

**Process:**

1. Start with same weight for all points: $\\alpha\_i = 1/N$
2. For $t = 1$ to $T$:

   * Learn $f\_t(x)$ with data weights $\\alpha\_i$
   * Compute weighted error:
$$\\text{error}*t = \\frac{\\sum*{i: f\_t(x\_i) \\neq y\_i} \\alpha\_i}{\\sum\_{i=1}^N \\alpha\_i}$$
   * Compute coefficient:
$$\\hat{w}\_t = \\frac{1}{2} \\ln\\left(\\frac{1 - \\text{error}\_t}{\\text{error}\_t}\\right)$$
   * Update weights:

     * Correct prediction: $\\alpha\_i \\leftarrow \\alpha\_i \\cdot e^{-\\hat{w}\_t}$
     * Incorrect prediction: $\\alpha\_i \\leftarrow \\alpha\_i \\cdot e^{+\\hat{w}\_t}$
   * Normalize weights: $\\alpha\_i \\leftarrow \\frac{\\alpha\_i}{\\sum\_{j=1}^N \\alpha\_j}$
3. Final prediction:
$$\\hat{y} = \\text{sign}\\left(\\sum\_{t=1}^T \\hat{w}\_t f\_t(x)\\right)$$

**Example Analogy:** If a student fails a test, the teacher gives more attention in next class.

**Advantages:**

* Adaptive learning
* Good accuracy
* Reduces overfitting

\---

### 5.7 XGBoost (Extreme Gradient Boosting)

**Definition:** Advanced, fast implementation of gradient boosting.

**Key Features:**

* Can handle missing values automatically
* Uses regularization techniques (L1, L2) to reduce overfitting
* Highly scalable for large datasets

**XGBoost Algorithm Steps:**

1. Initialize ensemble with single model (usually a decision tree)
2. Calculate errors made by first model
3. Train second model to correct those errors
4. Add second model to ensemble
5. Calculate combined model errors
6. Train third model to correct new errors
7. Repeat for set number of iterations
8. Final prediction: Weighted average of all models
9. Tune hyperparameters (cross-validation, grid search)

**Advantages:**

* Excellent accuracy
* Efficient for large data
* Handles missing values
* Regularization prevents overfitting

**Applications:**

* Kaggle competitions
* Industry ML projects
* Fraud detection
* Sales forecasting

\---

## UNIT 6: UNSUPERVISED LEARNING

\---

### 6.1 What is Unsupervised Learning?

**Definition:** Type of machine learning where model is trained using data that does not contain labeled outputs. Only input data is provided, no correct answers are given.

**Goal:** System discovers hidden patterns, relationships, similarities, or structures in the data by itself.

**Why it's used:**

* Large amounts of data available but labeled answers not available
* Discover patterns humans may not notice easily

**Applications:**

* Customer segmentation
* Fraud detection
* Anomaly detection
* Document clustering

|Supervised vs Unsupervised|
|-|
|Uses labeled data|
|Maps inputs to known outputs|
|Error correction possible|
|Accuracy measurable|
|More labeling effort|

\---

### 6.2 Distance Measures

Used to measure similarity between data points.

#### Minkowski Distance (General Formula)

$$d(i,j) = \\sqrt\[p]{\\sum\_{f=1}^{l} |x\_{if} - x\_{jf}|^p}$$

Where $p$ is the order (L-p norm).

#### Special Cases

|p value|Distance Name|Formula|Use Case|
|-|-|-|-|
|p = 1|Manhattan (City Block)|$d = \\sum \|x\_{if} - x\_{jf}\|$|Grid-like paths|
|p = 2|Euclidean|$d = \\sqrt{\\sum (x\_{if} - x\_{jf})^2}$|Straight line distance|
|p → ∞|Supremum (Chebyshev)|$d = \\max\_f \|x\_{if} - x\_{jf}\|$|Maximum difference|

**Properties of Distance Measures:**

1. Positivity: $d(i,j) > 0$ if $i \\neq j$, $d(i,i) = 0$
2. Symmetry: $d(i,j) = d(j,i)$
3. Triangle Inequality: $d(i,j) \\leq d(i,k) + d(k,j)$

#### Cosine Similarity

Used for text/document similarity.

**Formula:**
$$\\cos(d\_1, d\_2) = \\frac{d\_1 \\cdot d\_2}{|d\_1| \\times |d\_2|}$$

Where:

* $d\_1 \\cdot d\_2$ = dot product
* $|d|$ = magnitude of vector

**Range:** 0 to 1 (0 = no similarity, 1 = completely similar)

**Example:**
$d\_1 = (5, 0, 3, 0, 2, 0, 0, 2, 0, 0)$
$d\_2 = (3, 0, 2, 0, 1, 1, 0, 1, 0, 1)$

* Dot product = $5×3 + 3×2 + 2×1 + 2×1 = 25$
* $|d\_1| = \\sqrt{25 + 9 + 4 + 4} = 6.48$
* $|d\_2| = \\sqrt{9 + 4 + 1 + 1 + 1} = 4.12$
* $\\cos = 25/(6.48×4.12) = 0.94$

\---

### 6.3 Partitioning Methods

**Definition:** Discover grouping in data by optimizing an objective function and iteratively improving partition quality.

**Characteristics:** Each data point belongs to exactly one cluster.

**Common Methods:**

* K-means (uses centroid - mean)
* K-medians (uses median)
* K-medoids (uses actual data points as centers)

\---

### 6.4 K-Means Clustering

**Definition:** Popular clustering algorithm that divides data into K clusters, where K is chosen in advance. Each cluster has a center point called centroid.

**Steps:**

1. Choose number of clusters K
2. Initialize K random centroids
3. Assign each data point to nearest centroid (using Euclidean distance)
4. Recalculate centroids (mean of all points in cluster)
5. Repeat steps 3-4 until centroids stop changing (stabilization)

**Advantages:**

* Easy to implement
* Fast
* Simple

**Disadvantages:**

* Requires K in advance
* Sensitive to outliers
* Sensitive to initial centroid selection

#### K-Means Numerical Example

Given points: A1(2,10), A2(2,5), A3(6,4), A4(5,8), A5(8,4), A6(7,5)
Initial centers: C1 = A1(2,10), C2 = A4(5,8)

**Iteration 1 - Assign points:**
Calculate distances from each point to C1 and C2:

|Point|Dist to C1 (2,10)|Dist to C2 (5,8)|Assigned Cluster|
|-|-|-|-|
|A2(2,5)|√\[(2-2)²+(5-10)²]=5|√\[(2-5)²+(5-8)²]=4.24|C2|
|A3(6,4)|√\[(6-2)²+(4-10)²]=7.21|√\[(6-5)²+(4-8)²]=4.12|C2|
|A5(8,4)|√\[(8-2)²+(4-10)²]=8.49|√\[(8-5)²+(4-8)²]=5|C2|
|A6(7,5)|√\[(7-2)²+(5-10)²]=7.07|√\[(7-5)²+(5-8)²]=3.61|C2|

**Recalculate centroids:**

* C1: Only A1(2,10) → C1 = (2,10)
* C2: A2(2,5), A3(6,4), A4(5,8), A5(8,4), A6(7,5)
Mean x = (2+6+5+8+7)/5 = 28/5 = 5.6
Mean y = (5+4+8+4+5)/5 = 26/5 = 5.2
C2 = (5.6, 5.2)

Continue until centroids stabilize.

\---

### 6.5 Dimension Reduction

**Need for Dimension Reduction:**

* Training becomes slow with too many features
* Some features may be irrelevant or redundant
* Too many features confuse model
* High-dimensional data difficult to visualize
* Risk of overfitting (model memorizes training data)

**Benefits:**

1. Faster training
2. Less storage and memory
3. Better visualization (2D or 3D)
4. Removes noise
5. Reduces overfitting
6. Sometimes improves accuracy

\---

### 6.6 Principal Component Analysis (PCA)

**Definition:** Linear method to convert dataset with many features into smaller set of new features called Principal Components (PCs).

**Goal:** Keep maximum possible information (variance) from original dataset.

**Key Concept:** PCA measures variance (how much data values spread).

* **First Principal Component (PC1):** Captures highest variance
* **Second Principal Component (PC2):** Captures next highest variance (uncorrelated with PC1)
* And so on...

**PCA Steps:**

1. Standardize the range of continuous variables
2. Compute covariance matrix to identify correlations
3. Compute eigenvectors and eigenvalues to identify principal components
4. Create feature vector (decide which PCs to keep)
5. Recast data along principal component axes

**Example:**
Features: A, B, C

* PC1 (combination of A+B) → 91% accuracy
* PC2 (remaining info) → adds very little improvement
* → Select only PC1

**Uses:**

* Reduces dimensionality
* Removes redundancy and noise
* Helps with multicollinearity

**Disadvantage:** Reduces interpretability (PCs are combinations of original features)

\---

### 6.7 Hierarchical Clustering

#### Agglomerative Clustering (Bottom-up)

**Process:**

* Each object initially considered as single-element cluster (leaf)
* At each step, merge two most similar clusters
* Continue until all points are in one big cluster (root)

**Algorithm (AGNES - AGglomerative NESTing):**

1. Start with each point as separate cluster
2. Find pair of clusters with smallest distance
3. Merge them into one cluster
4. Update distance matrix
5. Repeat steps 2-4 until only one cluster remains

#### Divisive Clustering (Top-down)

**Process:**

* Begin with root (all objects in one cluster)
* At each step, split most heterogeneous cluster
* Continue until each object is in its own cluster

#### Dendrogram

Tree-like diagram recording sequences of merges or splits. Shows hierarchical relationships between clusters.

#### Distance Measures for Clusters

|Method|Description|
|-|-|
|**Single link**|Distance between nearest neighbors (min distance)|
|**Complete link**|Distance between farthest points (max distance)|
|**Average link**|Average distance between all points in clusters|
|**Centroid link**|Distance between cluster centroids|

#### Example: Agglomerative Clustering

Student marks: 10, 32, 8, 20, 5

**Step 1:** Create proximity matrix (find smallest distance)
**Step 2:** Merge closest points
**Step 3:** Update proximity matrix
**Step 4:** Repeat until one cluster

\---

### 6.8 Recommender Systems

**Definition:** Software algorithms that recommend items to users based on past behavior, preferences, and other factors.

**Why Recommender Systems are Important:**

* Save user time
* Improve user experience
* Increase sales for companies
* Help users discover new items
* Keep users engaged on platform

#### Types of Recommender Systems

**1. Non-Personalized Recommendation Systems:**

* Same recommendations to all users (one-size-fits-all)
* Does not study personal interests or search history
* Based on general popularity or current trends
* Examples: Best-seller lists, most popular items, trending videos, festival offers
* Used for new users or when not enough data

**2. Content-Based Recommendation Systems:**

* Recommends items based on content of items themselves
* Uses item descriptions, tags, user reviews
* Finds items similar to what user previously liked
* Example: Recommend movie based on genre, cast, plot of previously watched movies

#### Methods to Implement Recommender Systems

|Method|Description|Example|
|-|-|-|
|**Collaborative Filtering**|Based on behavior of similar users|If users A and B liked same movies, recommend A's movie to B|
|**Content-Based Filtering**|Based on item attributes|Recommend similar genre movies|
|**Hybrid Filtering**|Combines collaborative + content-based|Netflix uses user behavior + movie genre|
|**Association Rule Mining**|Finds items bought together (Market Basket Analysis)|If bread, then butter recommendation|
|**Matrix Factorization**|Decomposes user-item matrix to find hidden patterns|Predict missing ratings for unrated movies|
|**Deep Learning**|Uses neural networks for complex relationships|Advanced recommendation systems|

#### Evaluation Metrics

|Metric|Description|
|-|-|
|**Accuracy**|How well recommendations match user preferences (precision, recall, F1, MAP)|
|**Coverage**|Proportion of items the system can recommend|
|**Diversity**|Variety of recommended items (not all similar)|
|**Novelty**|How new or unexpected recommendations are|

#### Matrix Factorization

**How it works:**

* Decomposes user-item interaction matrix into two lower-dimensional matrices
* User matrix: represents user preferences for latent features
* Item matrix: represents item strengths for latent features
* Predict missing interactions using dot product

\---

### 6.9 Hybrid Machine Learning Techniques

|Technique|Description|
|-|-|
|**Weighted Hybrid**|Combine outputs of multiple algorithms with weights|
|**Switching Hybrid**|Switch between algorithms based on user context|
|**Feature Combination Hybrid**|Combine features from different algorithms|
|**Cascade Hybrid**|Use output of one algorithm as input to another|
|**Meta-level Hybrid**|Learn meta-model to combine algorithm outputs|

\---



\---

## QUICK REFERENCE: FORMULAS SUMMARY

### Regression

|Formula|Purpose|
|-|-|
|$y = w\_0 + w\_1x$|Simple Linear Regression|
|$y = w\_0 + w\_1x + w\_2x^2$|Polynomial Regression (degree 2)|
|$y = w^T x$|Multiple Linear Regression|
|$W = (X^T X)^{-1} X^T Y$|Normal Equation|
|MSE = (1/n) Σ(y - ŷ)²|Mean Squared Error|
|R² = 1 - SS\_RES/SS\_TOT|Coefficient of Determination|

### Classification Metrics

|Formula|Purpose|
|-|-|
|Accuracy = (TP+TN)/(TP+TN+FP+FN)|Overall correctness|
|Precision = TP/(TP+FP)|Prediction quality|
|Recall = TP/(TP+FN)|Finding all positives|
|F1 = 2×P×R/(P+R)|Harmonic mean|
|Specificity = TN/(TN+FP)|Correct negatives|

### Logistic Regression

|Formula|Purpose|
|-|-|
|σ(z) = 1/(1+e^{-z})|Sigmoid function|
|P(y=1 \| x) = 1/(1+e^{-w^T x})|Probability prediction|
|Cost = -\[y log(ŷ) + (1-y) log(1-ŷ)]|Binary Cross Entropy|

### SVM

|Formula|Purpose|
|-|-|
|w^T x + b = 0|Decision boundary|
|w^T x + b = ±1|Margin boundaries|
|Margin = 2/|(incomplete)|
|Minimize|(incomplete)|

### Neural Networks

|Formula|Purpose|
|-|-|
|z = Σ w\_i x\_i + b|Weighted sum|
|Δw = η(y - ŷ)x|Weight update|
|f(z) = max(0,z)|ReLU activation|

### Distance Measures

|Formula|Purpose|
|-|-|
|d = √Σ(x\_i - y\_i)²|Euclidean distance|
|d = Σ \|x\_i - y\_i\||Manhattan distance|
|cos = (a·b)/(\|a\|\|b\|)|Cosine similarity|

### Decision Trees

|Formula|Purpose|
|-|-|
|Entropy = -Σ p\_i log₂(p\_i)|Randomness measure|
|Gini = 1 - Σ p\_i²|Impurity measure|
|IG = Entropy(parent) - Σ (|S\_v|

### Feature Scaling

|Formula|Purpose|
|-|-|
|X\_new = (X-X\_min)/(X\_max-X\_min)|Min-Max Scaling|
|X\_new = (X-mean)/σ|Standardization|
|X\_new = (X-median)/IQR|Robust Scaling|

\---



