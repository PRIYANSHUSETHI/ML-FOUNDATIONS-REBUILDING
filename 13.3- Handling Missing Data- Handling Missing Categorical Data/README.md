# 🏷️ Handling Missing Categorical Data


Missing values are one of the most common problems encountered when working with real-world datasets.

But not all missing values should be handled in the same way.

For **numerical variables**, we can use techniques such as Mean, Median, or Arbitrary Value Imputation.

Categorical variables are different.

We cannot calculate the average of:

```text
Delhi
Mumbai
Jaipur
```

or the median of:

```text
Good
Typical
Poor
```

Instead, categorical imputation requires us to think about **frequency, category distributions, and the information contained in missingness itself**.

This project explores two important strategies for handling missing categorical data:

- **Most Frequent (Mode) Imputation**
- **Missing Category Imputation**

The goal is not simply to eliminate `NaN` values.

It is to understand:

> **What assumption are we making when we replace a missing categorical value?**

---

# 📌 What You'll Learn

This notebook covers:

- Why categorical data requires a different approach to missing-value imputation
- What Most Frequent / Mode Imputation is
- How Mode Imputation works
- When Most Frequent Imputation is appropriate
- The assumptions behind Mode Imputation
- How Mode Imputation can introduce artificial bias
- What Missing Category Imputation is
- Why missingness itself can sometimes be informative
- The difference between MCAR and MNAR in the context of categorical imputation
- How missing-value percentage affects the choice of strategy
- Why a dominant category matters
- How imputation can affect the feature-target relationship
- How to implement both techniques using Scikit-Learn's `SimpleImputer`
- Why the imputer should be fitted only on training data
- How to avoid data leakage during preprocessing
- How the House Prices dataset illustrates the difference between the two approaches

---

# 🧩 The Missing Categorical Data Problem

Suppose we have a categorical feature:

| Garage Quality |
|---|
| TA |
| Gd |
| Missing |
| TA |
| Missing |
| TA |

What should we do with the missing values?

One possibility is to replace them with the most common category:

```text
TA
Gd
Missing  →  TA
TA
Missing  →  TA
TA
```

This is **Most Frequent Imputation**.

Another possibility is to preserve the fact that the value was missing:

```text
TA
Gd
Missing
TA
Missing
TA
```

Here, `Missing` becomes an explicit category.

This is **Missing Category Imputation**.

The two approaches may produce a dataset without `NaN` values, but they make very different assumptions about what the missing observations represent.

---

# 🔍 What Is Categorical Imputation?

**Categorical imputation** is the process of replacing missing values in categorical features with either:

1. An existing category selected according to some rule, or
2. A new category representing the missing state.

For categorical data, the most common approaches explored in this project are:

```text
                    Missing Categorical Data
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
       Most Frequent                  Missing Category
          Imputation                    Imputation
                 │                         │
                 ▼                         ▼
       Replace with Mode          Create "Missing"
                                      Category
```

The important question is not:

> **"Which technique is easier?"**

It is:

> **"What does the missingness represent in this dataset?"**

---

# 📊 Technique 1: Most Frequent Imputation

Most Frequent Imputation, also called **Mode Imputation**, replaces every missing value with the category that occurs most frequently.

For example:

| Color |
|---|
| Blue |
| Blue |
| Red |
| Missing |
| Blue |

The mode is:

```text
Blue
```

Therefore:

| Color |
|---|
| Blue |
| Blue |
| Red |
| **Blue** |
| Blue |

The underlying assumption is:

> **The missing observations are likely to belong to the most common category.**

This makes the technique extremely simple and computationally inexpensive.

---

# 🧠 When Does Most Frequent Imputation Make Sense?

Most Frequent Imputation works best when several conditions are satisfied.

## 1. Missingness Is Low

A practical guideline from this study is that missingness should ideally be around **5% or less**.

If only a small number of observations are missing, replacing them with the mode is less likely to substantially change the feature's distribution.

---

## 2. Missingness Is MCAR

MCAR stands for:

**Missing Completely At Random**

The missingness has no systematic relationship with the observed or unobserved data.

In such a situation, using the dominant category as a replacement can be a reasonable simplification.

---

## 3. One Category Clearly Dominates

This is particularly important.

Consider:

```text
TA       82%
Gd        7%
Fa        6%
Ex        5%
```

Here, `TA` clearly dominates.

Replacing a small number of missing observations with `TA` is unlikely to dramatically change the categorical distribution.

But consider:

```text
TA       30%
Gd       28%
Fa       22%
Ex       20%
```

There is no meaningful dominant category.

Replacing missing values with `TA` would artificially increase the representation of one category.

---

# ⚖️ Advantages and Limitations

| Advantages | Limitations |
|---|---|
| Simple to implement | Can over-represent the mode |
| Computationally inexpensive | Can introduce artificial bias |
| Easy to deploy | Can distort category frequencies |
| Easy to reproduce | Can alter the feature-target relationship |
| Works well with low missingness and a dominant category | Can cause covariance shift when used inappropriately |

The biggest concern is that Mode Imputation can make the model believe that many observations belong to a category simply because we chose that category as the replacement.

---

# 📉 How Mode Imputation Can Distort the Data

Suppose a categorical feature originally looks like:

```text
TA       ████████████████████
Gd       █████
Fa       ████
Missing  ██
```

After Most Frequent Imputation:

```text
TA       ██████████████████████
Gd       █████
Fa       ████
```

The `TA` category becomes artificially larger.

When the proportion of missing values is small, this change may be negligible.

When the proportion is large, the distortion can become substantial.

This is particularly problematic when the categorical feature has an important relationship with the target variable.

---

# 🧠 Technique 2: Missing Category Imputation

Missing Category Imputation takes a fundamentally different approach.

Instead of guessing which existing category the missing observation belongs to, we create a new category.

For example:

| Fireplace Quality |
|---|
| Gd |
| TA |
| Missing |
| Missing |
| Fa |

The missing value is not replaced with `Gd`, `TA`, or `Fa`.

Instead:

```text
Missing → "Missing"
```

The missing state becomes an explicit category that the model can learn from.

---

# 💡 Why Can Missingness Be Useful Information?

Sometimes the absence of a value is not random.

Consider a housing dataset with:

```text
Garage Quality
```

If this feature is missing, it may be because the house **does not have a garage**.

Replacing the missing value with:

```text
TA = Typical/Average
```

would incorrectly tell the model that the house has an average-quality garage.

Creating a separate:

```text
Missing
```

category preserves the information that the value was absent.

The model can then determine whether that missingness has predictive value.

> **Sometimes the missing value is the feature.**

---

# 🔎 When Should Missing Category Imputation Be Preferred?

This approach becomes particularly useful when:

### High Missingness

The study uses **greater than approximately 10% missingness** as a practical signal that creating a separate category may be safer.

### Missingness May Be MNAR

MNAR stands for:

**Missing Not At Random**

Here, the fact that the value is missing may itself be related to the underlying data or the target.

In such situations, replacing missing values with an existing category can hide potentially useful information.

### There Is No Dominant Category

If the observed categories have relatively similar frequencies, choosing the mode can arbitrarily increase one category's representation.

Creating a separate `Missing` category avoids that distortion.

---

# 🏠 Case Study: House Prices Dataset

The notebook uses the **Advanced Regression: House Prices** dataset to examine how different imputation strategies affect categorical features and their relationship with the target variable, `SalePrice`.

Two categorical variables provide an excellent comparison:

```text
GarageQual
     vs.
FireplaceQu
```

The amount and structure of missingness are very different for the two features.

---

# 🔬 Scenario A: GarageQual

For `GarageQual`:

- Missingness was approximately **5%**
- The dominant category was **TA (Typical/Average)**

This is a situation where Most Frequent Imputation is relatively appropriate.

The missing proportion is small, and there is a clearly dominant category.

After replacing the missing values with `TA`, the distribution of `SalePrice` remained relatively stable.

The PDF comparison suggested that the relationship between the feature and target was largely preserved.

---

# 🔥 Scenario B: FireplaceQu

The situation changes dramatically for `FireplaceQu`.

For this feature:

- Missingness was approximately **50%**
- The mode was **Gd (Good)**
- However, `Gd` did not dominate the other categories by a large margin

Now imagine filling approximately half of the dataset with:

```text
Gd = Good
```

This would dramatically increase the representation of `Gd`.

The resulting feature would no longer resemble the original categorical distribution.

The study found a substantial **covariance shift** when Most Frequent Imputation was applied in this situation.

The resulting `SalePrice` PDF diverged sharply from the original distribution.

In other words, the imputation created a relationship that was not supported by the original data.

---

# 📊 What the Case Study Demonstrates

The two scenarios illustrate an important principle:

```text
                  Missing Categorical Data
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Low Missingness            High Missingness
       + Dominant Mode            + No Dominant Mode
              │                         │
              ▼                         ▼
       Most Frequent             Missing Category
         Imputation                Imputation
              │                         │
              ▼                         ▼
       Small Distortion          Preserve Missingness
```

The technique should therefore be selected based on the **characteristics of the missingness**, rather than simply applying the same method to every categorical feature.

---

# 🛠️ Implementing Most Frequent Imputation

Scikit-Learn provides the `SimpleImputer` class for implementing categorical imputation.

The Most Frequent strategy is specified using:

```python
strategy='most_frequent'
```

Example:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

imputer.fit(X_train)

X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

The imputer learns the most frequent category from the training data.

That learned value is then used to transform both the training and test datasets.

---

# 🛠️ Implementing Missing Category Imputation

To create an explicit missing category, `SimpleImputer` can use the `constant` strategy together with a custom `fill_value`.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy='constant',
    fill_value='Missing'
)

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

The missing observations are replaced with:

```text
Missing
```

instead of an existing category.

---

# 🚨 Avoiding Data Leakage

One of the most important implementation details is **when the imputer is fitted**.

The imputer should learn its replacement value from the **training data only**.

Incorrect workflow:

```text
Entire Dataset
      │
      ▼
Calculate Mode
      │
      ▼
Split into Train / Test
```

The test set has influenced the preprocessing decision.

That is a form of **data leakage**.

The correct workflow is:

```text
                 Dataset
                    │
                    ▼
              Train / Test Split
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
          Training        Test
             │             │
             ▼             │
            Fit            │
             │             │
             ▼             │
       Learn Mode          │
             │             │
             ├─────────────┘
             │
             ▼
          Transform
          Train + Test
```

In code:

```python
imputer.fit(X_train)

X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

> **The test set should remain unseen while preprocessing decisions are being learned.**

---

# 🧱 A Note About `SimpleImputer`

`SimpleImputer` returns a **NumPy array** after transformation.

If your downstream workflow relies on Pandas column names, it is useful to convert the transformed result back into a DataFrame before continuing with subsequent preprocessing steps such as One-Hot Encoding.

This becomes particularly important in a larger production preprocessing pipeline.

---

# ⚖️ Most Frequent vs Missing Category Imputation

| Aspect | Most Frequent | Missing Category |
|---|---|---|
| Basic idea | Replace with mode | Create a new category |
| Missingness | Prefer low | Often useful when high |
| Practical guideline | Around <5% | Around >10% |
| Missingness mechanism | MCAR | MNAR can be informative |
| Dominant category | Important | Not required |
| Preserves missingness signal | ❌ | ✅ |
| Can over-represent a category | ✅ | ❌ |
| Implementation | `strategy='most_frequent'` | `strategy='constant'` |
| Main risk | Artificial bias / covariance shift | Adds a new category |
| Best question to ask | "Is the mode a reasonable replacement?" | "Could missingness itself be meaningful?" |

---

# 📊 Decision Framework

A practical decision framework from this study is:

```text
                 Categorical Feature
                         │
                         ▼
                Measure Missingness
                         │
             ┌───────────┴───────────┐
             │                       │
          Low (<5%)              High (>10%)
             │                       │
             ▼                       ▼
      Is there a clear        Is missingness potentially
      dominant category?       informative / MNAR?
             │                       │
        ┌────┴────┐             ┌────┴────┐
        │         │             │         │
       Yes        No           Yes        No
        │         │             │         │
        ▼         ▼             ▼         ▼
   Most Frequent  Consider   Missing    Consider
    Imputation   Missing     Category   dataset-specific
                 Category   Imputation   strategy
```

The thresholds above should be treated as **practical guidelines rather than absolute laws**.

The underlying distribution and meaning of the missingness still matter.

---

# 📈 What Should Be Validated After Imputation?

Removing all `NaN` values does not automatically mean the preprocessing step was successful.

The transformed dataset should be compared with the original data.

Useful checks include:

### 1. Category Distribution

Compare the proportion of each category before and after imputation.

```text
Original Distribution
        ↓
Imputation
        ↓
New Distribution
        ↓
Compare Category Proportions
```

### 2. Feature-Target Relationship

For categorical features, investigate whether the relationship with the target has changed.

In the House Prices case study, the effect was examined using the distribution of `SalePrice`.

### 3. Distribution Shift

Compare the target distribution before and after imputation.

A significant change can indicate that the imputation strategy has introduced distortion.

### 4. Missingness Signal

For Missing Category Imputation, determine whether the newly created `Missing` category has a meaningful relationship with the target.

The objective is not to assume that missingness is predictive.

The objective is to **give the model the opportunity to learn whether it is**.

---

# ⚠️ Common Mistakes

### 1. Applying Mode Imputation Automatically

Just because a categorical column has missing values does not mean its mode is a good replacement.

### 2. Ignoring the Percentage of Missing Values

Replacing 2% of observations with the mode is very different from replacing 50%.

### 3. Ignoring Category Frequencies

A mode is much safer when one category clearly dominates.

If categories have similar frequencies, the mode may be an arbitrary choice.

### 4. Destroying Information About Missingness

If missingness is meaningful, replacing it with an existing category can hide a potentially useful signal.

### 5. Fitting the Imputer Before the Train-Test Split

This can allow information from the test set to influence preprocessing.

Always fit preprocessing transformations using the training data.

---

# ⚖️ Advantages vs Limitations

| Strategy | Advantages | Limitations |
|---|---|---|
| **Most Frequent** | Simple, fast, easy to deploy | Can over-represent the mode |
| **Most Frequent** | Works well with low missingness | Can introduce artificial bias |
| **Most Frequent** | Easy to implement with `SimpleImputer` | Can cause covariance shift |
| **Missing Category** | Preserves missingness information | Introduces an additional category |
| **Missing Category** | Avoids forcing missing values into an existing class | May not always be appropriate if missingness is genuinely random |
| **Missing Category** | Allows the model to learn whether missingness matters | Requires careful interpretation |

---

# 💡 Key Learnings

This project reinforced several important lessons about categorical missing data:

- Categorical data cannot be imputed using numerical concepts such as mean or median.
- **Mode Imputation** replaces missing values with the most frequent category.
- Most Frequent Imputation is most suitable when missingness is relatively low and a category clearly dominates.
- A practical guideline from this study is to consider Most Frequent Imputation when missingness is around **5% or less**.
- Missing Category Imputation treats missingness as an explicit category rather than guessing an existing value.
- A practical guideline from this study is to consider a separate missing category when missingness is **greater than approximately 10%**.
- Missingness can itself contain predictive information.
- MNAR situations are particularly important because the reason a value is missing may be related to the underlying phenomenon being modeled.
- Replacing a large number of missing values with the mode can substantially alter category frequencies.
- Artificially increasing the frequency of one category can distort the feature-target relationship.
- The House Prices case study demonstrates why the same imputation strategy should not automatically be applied to every categorical feature.
- `GarageQual` illustrates a case where low missingness and a dominant mode made Most Frequent Imputation relatively effective.
- `FireplaceQu` illustrates a case where high missingness and the lack of a dominant category made Most Frequent Imputation inappropriate.
- Always fit the imputer on the training data only.
- After imputation, validate the resulting distributions and feature-target relationships.
- The goal of imputation is not simply to eliminate `NaN` values.

> **Good imputation is about preserving useful information while introducing as little unnecessary distortion as possible.**

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# 📂 Project Structure

```text
Handling Missing Categorical Data/
│
├── handling_missing_categorical_data.ipynb
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
handling_missing_categorical_data.ipynb
```

and run the cells sequentially.

---

# 📝 Medium Article

This notebook is part of my **Relearning Machine Learning in Public** series.

### Most Frequent vs. Missing Category Imputation Explained: Choosing the Right Strategy for Missing Categorical Data

🔗 [Read the full article on Medium](https://medium.com/@priyanshu20032002/most-frequent-vs-missing-category-imputation-explained-choosing-the-right-strategy-for-missing-5038e413454a?sharedUserId=priyanshu20032002)

---

# ⭐ The Bigger Lesson

Handling missing categorical data may look like a simple preprocessing task.

But every imputation strategy introduces an assumption.

```text
                 Missing Category
                        │
                        ▼
               Understand the
               Missingness
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Low Missingness       High / Informative
       + Dominant Mode           Missingness
             │                     │
             ▼                     ▼
      Most Frequent          Missing Category
        Imputation             Imputation
             │                     │
             └──────────┬──────────┘
                        ▼
                Validate the Impact
                        │
                        ▼
             Preserve Useful Signal
                        │
                        ▼
                 Build Better
                    Models
```

The better question is not:

> **"How do I fill these missing categorical values?"**

It is:

> **"What assumption am I making by filling them this way?"**

Sometimes the best decision is to make an educated guess.

Sometimes the better decision is to admit that we don't know — and let **"Missing"** become part of the data.

---

### 🌱 Part of the "Relearning Machine Learning in Public" Series

This repository documents my journey of revisiting and strengthening the foundations of Machine Learning — one concept, notebook, and project at a time.

If you find this repository useful, feel free to ⭐ **star the repository** and follow along with the journey.
