# 🔢 Handling Missing Numerical Data with Imputation

> **Relearning Machine Learning in Public | Day 25**

Missing values are one of the most common problems in real-world datasets.

Instead of deleting incomplete observations, this project explores three techniques for replacing missing numerical values:

- **Mean Imputation**
- **Median Imputation**
- **Arbitrary Value Imputation**

The goal is not just to learn how to fill `NaN` values.

It is to understand:

> **What happens to the data after we fill them?**

This notebook explores how different imputation strategies affect variance, feature distributions, outliers, and the information available to a machine learning model.

> **Imputation does not recover missing information. It replaces that uncertainty with an assumption.**

---

# 📌 What You'll Learn

This notebook covers:

- What data imputation is
- The difference between univariate and multivariate imputation
- How Mean Imputation works
- How Median Imputation works
- When to use the mean and when to use the median
- Why outliers matter when choosing an imputation strategy
- How imputation affects variance and feature distributions
- What Arbitrary Value Imputation is
- Why missingness itself can sometimes be useful information
- How to implement these techniques using Scikit-Learn's `SimpleImputer`
- Why fitting an imputer on training data is important
- How to avoid data leakage during preprocessing

---

# 🧩 The Missing Data Problem

A machine learning dataset may contain missing numerical values because data was never collected, a user chose not to provide it, or a system failed to capture it.

For example:

| Customer | Age | Annual Income |
|---|---:|---:|
| A | 24 | 450000 |
| B | NaN | 620000 |
| C | 31 | NaN |
| D | 28 | 510000 |

This leaves us with a preprocessing decision:

```text
                 Missing Values
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
       Remove                      Impute
      the Data                 the Missing Value
          │                           │
          ▼                           ▼
 Complete Case              Replace with a Value
    Analysis
```

This project focuses on the second path.

---

# 🔍 What is Imputation?

**Imputation** is the process of replacing missing values with estimated or predefined values instead of deleting the affected observations.

Consider:

| Experience |
|---:|
| 2 |
| 5 |
| NaN |
| 8 |
| 6 |

With Mean Imputation:

\[
\text{Mean} = \frac{2 + 5 + 8 + 6}{4} = 5.25
\]

The missing value becomes:

| Experience |
|---:|
| 2 |
| 5 |
| **5.25** |
| 8 |
| 6 |

> **The true missing value has not been recovered. It has been estimated using a chosen strategy.**

---

# 🧠 Univariate vs Multivariate Imputation

| Type | Description | Examples |
|---|---|---|
| **Univariate Imputation** | Uses information from the same feature containing missing values | Mean, Median, Arbitrary Value |
| **Multivariate Imputation** | Uses relationships between multiple features | KNN Imputer, Iterative Imputer |

This notebook focuses on **Univariate Imputation**.

```text
Feature: Age
     │
     ▼
Look only at
observed Age values
     │
     ▼
Calculate or choose
replacement value
     │
     ▼
Fill missing values
```

---

# 📊 Technique 1: Mean Imputation

Mean Imputation replaces every missing value with the **average of the observed values**.

Example:

| Salary |
|---:|
| 8 |
| 10 |
| 12 |
| NaN |
| 15 |

\[
\frac{8 + 10 + 12 + 15}{4} = 11.25
\]

After imputation:

| Salary |
|---:|
| 8 |
| 10 |
| 12 |
| **11.25** |
| 15 |

## When is Mean Imputation Useful?

Mean Imputation is generally most suitable when the feature:

- Is numerical
- Has a reasonably symmetric distribution
- Does not contain extreme outliers
- Has a relatively small proportion of missing values

```text
        Symmetric Distribution

              /\
            /    \
          /        \
--------/------------\--------
             Mean
```

---

# 📉 Why Outliers Matter

The mean is sensitive to extreme values.

Consider:

| Property Price |
|---:|
| 150 |
| 180 |
| 200 |
| NaN |
| 850 |

The mean is:

\[
\frac{150 + 180 + 200 + 850}{4} = 345
\]

The extreme value pulls the average upward, even though most observations lie between 150 and 200.

This is where Median Imputation becomes useful.

---

# 📊 Technique 2: Median Imputation

Median Imputation replaces missing values with the **middle value** of the observed data.

For:

```text
150, 180, 200, 850
```

The median is:

\[
\frac{180 + 200}{2} = 190
\]

After Median Imputation:

| Property Price |
|---:|
| 150 |
| 180 |
| 200 |
| **190** |
| 850 |

Unlike the mean, the median is much less affected by extreme observations.

---

# ⚖️ Mean vs Median Imputation

| Situation | Preferred Strategy | Why |
|---|---|---|
| Approximately symmetric distribution | **Mean** | Represents central tendency well |
| Skewed distribution | **Median** | Less affected by skewness |
| Significant outliers | **Median** | Robust to extreme values |

```text
Numerical Feature
       │
       ▼
Inspect Distribution
       │
       ├── Approximately Symmetric
       │           │
       │           ▼
       │      Mean Imputation
       │
       └── Skewed / Outliers Present
                   │
                   ▼
              Median Imputation
```

---

# 🛠️ Implementing Mean and Median Imputation

Scikit-Learn provides `SimpleImputer`.

### Mean Imputation

```python
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy='mean')

X_train = mean_imputer.fit_transform(X_train)
X_test = mean_imputer.transform(X_test)
```

### Median Imputation

```python
from sklearn.impute import SimpleImputer

median_imputer = SimpleImputer(strategy='median')

X_train = median_imputer.fit_transform(X_train)
X_test = median_imputer.transform(X_test)
```

---

# 🚨 Avoiding Data Leakage

The imputer should learn replacement values only from the training data.

```text
Training Data
      │
      ▼
      Fit
      │
      ▼
Learn Mean / Median
      │
      ├───────────────┐
      ▼               ▼
Transform         Transform
Training Data      Test Data
```

The correct workflow is:

```python
imputer.fit(X_train)

X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

> **The test set should remain unseen while preprocessing decisions are being learned.**

---

# 📉 How Mean and Median Imputation Affect the Data

Imputation preserves observations, but it can also modify the statistical properties of a feature.

## 1. Variance Shrinks

When multiple missing values are replaced with exactly the same number, they introduce no new variation.

```text
Original Data

  •      •        •   •
      •      •


After Imputation

  •      •  ●●●●  •   •
      •      •
           ↑
      Imputed Value
```

As more observations are concentrated at one point, the overall spread can decrease.

## 2. The Distribution Can Change

Repeatedly inserting the same value can create an artificial spike.

```text
Original Distribution

       /\
     /    \
   /        \


After Imputation

       /|\
     /  |  \
   /    |    \
        ↑
  Repeated Imputed Value
```

## 3. Relationships Between Features Can Be Affected

Reduced variation can influence:

- Covariance
- Correlation
- Relationships between features

The impact can become more significant as the proportion of missing values increases.

---

# 📌 The 5% Rule of Thumb

A practical guideline is that Mean or Median Imputation is generally more reliable when the percentage of missing values is relatively small.

> **If a feature contains less than approximately 5% missing values, simple statistical imputation is less likely to significantly alter its distribution.**

This is not a universal law.

The decision should still consider:

- Feature distribution
- Presence of outliers
- Amount of missing data
- Nature of the missingness

---

# 🎯 Technique 3: Arbitrary Value Imputation

Arbitrary Value Imputation takes a different approach.

Instead of estimating the missing value, it replaces it with a predefined constant outside the natural range of the feature.

Examples:

- `-1`
- `999`
- `-999`

For example:

| Age |
|---:|
| 22 |
| 28 |
| NaN |
| 35 |
| 40 |

After imputation:

| Age |
|---:|
| 22 |
| 28 |
| **-1** |
| 35 |
| 40 |

The unrealistic value is deliberate.

It allows a model to distinguish between:

```text
Real Value
    vs
Originally Missing Value
```

---

# 🧠 Why Preserve Missingness?

Sometimes the fact that a value is missing can itself contain useful information.

```text
Customer Income
        │
        ▼
Customer chooses not
to disclose income
        │
        ▼
Missingness may itself
be a useful signal
```

Replacing every missing value with the mean can hide this information.

An arbitrary value makes the missingness visible to the model.

---

# 🛠️ Implementing Arbitrary Value Imputation

```python
from sklearn.impute import SimpleImputer

arbitrary_imputer = SimpleImputer(
    strategy='constant',
    fill_value=-1
)

X_train = arbitrary_imputer.fit_transform(X_train)
X_test = arbitrary_imputer.transform(X_test)
```

The `constant` strategy uses the value supplied through `fill_value`.

---

# ⚠️ Choosing an Arbitrary Value

The chosen value should not overlap with legitimate observations.

| Feature | Possible Arbitrary Value |
|---|---|
| Age | `-1` |
| Number of Purchases | `-1` |
| Annual Income | `-999999` |

If `0` can naturally occur in a feature, using `0` as the missing-value indicator would make it impossible to distinguish a real value from an imputed one.

---

# 📊 How Arbitrary Value Imputation Affects the Distribution

Arbitrary Value Imputation can preserve the missingness signal, but it can also introduce strong distortion.

Repeatedly inserting an extreme value can create:

- A visible spike in the distribution
- Artificial outliers
- Large changes in variance
- Changes in relationships with other variables

This technique should therefore be used intentionally rather than automatically.

---

# ⚖️ Comparing the Three Techniques

| Technique | Best Used When | Main Strength | Main Limitation |
|---|---|---|---|
| **Mean Imputation** | Distribution is approximately symmetric | Simple and fast | Sensitive to outliers |
| **Median Imputation** | Data is skewed or contains outliers | Robust to extreme values | Repeated values reduce variation |
| **Arbitrary Value Imputation** | Missingness itself may contain useful information | Preserves missingness signal | Can introduce artificial outliers |

> **There is no universally best imputation strategy.**

The right technique depends on the feature, its distribution, and the nature of the missingness.

---

# 📊 Complete Workflow

```text
                Raw Dataset
                     │
                     ▼
          Identify Missing Values
                     │
                     ▼
             Select Feature
                     │
                     ▼
          Inspect Distribution
                     │
          ┌──────────┴───────────┐
          │                      │
          ▼                      ▼
   Symmetric Data          Skewed / Outliers
          │                      │
          ▼                      ▼
      Mean Imputation       Median Imputation
          │                      │
          └──────────┬───────────┘
                     │
                     ▼
       Is Missingness Itself Useful?
                     │
             ┌───────┴───────┐
             │               │
            Yes              No
             │               │
             ▼               ▼
      Arbitrary Value    Validate Result
         Imputation
             │
             ▼
      Compare Distribution
      and Statistical Impact
             │
             ▼
       Continue ML Pipeline
```

---

# 📈 What Should Be Validated After Imputation?

After filling missing values, inspect whether the feature has changed significantly.

Useful checks include:

1. **Distribution** — compare before and after imputation.
2. **Variance** — check whether the spread changed substantially.
3. **Outliers** — identify artificial extreme values.
4. **Relationships** — consider effects on covariance and correlation.

The goal is not simply:

```text
No More NaN Values
```

The goal is:

```text
No More NaN Values
        +
Minimal Unnecessary Distortion
        +
Useful Information Preserved
```

---

# ✅ Advantages of Simple Imputation

- Preserves observations instead of automatically deleting rows
- Easy to implement with `SimpleImputer`
- Computationally efficient
- Can be reused consistently on new data
- Allows different strategies for different features

---

# ⚠️ Limitations of Simple Imputation

### 1. The True Value Is Not Recovered

Imputation provides an estimate or placeholder, not the actual missing value.

### 2. Variance Can Decrease

Repeated replacement values reduce variation.

### 3. Distributions Can Be Distorted

A large number of identical replacement values can create artificial spikes.

### 4. Relationships Can Change

Covariance and correlation between variables may be affected.

### 5. Arbitrary Values Can Create Artificial Outliers

Extreme replacement values can substantially change a feature's statistical properties.

---

# 💡 Key Learnings

- Imputation is not the same as recovering the original missing information.
- Mean Imputation is generally more suitable for approximately symmetric numerical features.
- Median Imputation is more robust when features are skewed or contain outliers.
- Replacing many values with the same number can reduce variance.
- Imputation can alter the shape of a distribution.
- Simple imputation can affect relationships between variables.
- Arbitrary Value Imputation can preserve information about the fact that a value was missing.
- The arbitrary value should not overlap with legitimate observations.
- Always learn preprocessing parameters from training data only.
- Use `fit()` on training data and `transform()` on unseen data to avoid leakage.
- The best imputation strategy depends on the feature, distribution, and missingness mechanism.

---

# 🧰 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# 📂 Project Structure

```text
Handling Missing Data/
│
├── handling_missing_numerical_data.ipynb
└── README.md
```

---

# 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/PRIYANSHUSETHI/ML-FOUNDATIONS-REBUILDING.git
```

Open the notebook using:

- Jupyter Notebook
- JupyterLab
- VS Code
- Google Colab

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then open:

```text
handling_missing_numerical_data.ipynb
```

and run the cells sequentially.

---

# 📝 Medium Article

This notebook is part of my **Relearning Machine Learning in Public** series.

### Mean, Median & Arbitrary Value Imputation Explained: Simple Yet Powerful Techniques for Handling Missing Data

🔗 [Read the full article on Medium](https://priyanshu20032002.medium.com/mean-median-arbitrary-value-imputation-explained-simple-yet-powerful-techniques-for-handling-70449a1a6662?sharedUserId=priyanshu20032002)

---

# ⭐ The Bigger Lesson

Handling missing values is not just about making a dataset usable.

Every decision changes the data.

```text
Missing Value
      │
      ▼
Choose a Strategy
      │
      ├── Mean
      ├── Median
      └── Arbitrary Value
              │
              ▼
      Validate the Impact
              │
              ▼
      Preserve Useful Information
              │
              ▼
       Build Better Models
```

A better question than:

> **"How do I get rid of these missing values?"**

is:

> **"What assumptions am I introducing when I replace them?"**

> **Good imputation is not about filling every blank with a number. It's about preserving as much useful information as possible while introducing as little unnecessary distortion as possible.**

---

### 🌱 Part of the "Relearning Machine Learning in Public" Series

This repository documents my journey of revisiting and strengthening the foundations of Machine Learning — one concept, notebook, and project at a time.

If you find this repository useful, feel free to ⭐ **star the repository** and follow along with the journey.
