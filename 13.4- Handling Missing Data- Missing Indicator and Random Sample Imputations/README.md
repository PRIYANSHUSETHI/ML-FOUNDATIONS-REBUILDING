# Advanced Missing Data Imputation

This folder contains the notes and practical Python work for **advanced
techniques for handling missing data** as part of the **Relearning
Machine Learning in Public** series.

The topic moves beyond simple Mean, Median, and categorical-value
imputation and focuses on techniques that either preserve more of the
original data distribution or allow the model to learn from the fact
that a value was missing.

## 📌 Topics Covered

This topic focuses on three ideas:

1.  **Random Sample Imputation**
2.  **Missing Indicator**
3.  **Automatic Imputation Strategy Selection using GridSearchCV**

The accompanying notebook provides hands-on implementations and
experiments for Random Sample Imputation and Missing Indicators. The
Medium article also discusses how imputation strategies can be treated
as hyperparameters and selected using GridSearchCV.

------------------------------------------------------------------------

## 🧠 1. Random Sample Imputation

### What is it?

Random Sample Imputation replaces each missing value with a randomly
selected value from the **non-missing observations of the same
feature**.

Instead of replacing every missing value with a single statistic such as
the mean or median, the technique samples actual observations from the
feature.

For example:

``` text
Original:
22, 35, 28, NaN, 41, NaN, 19

After Random Sample Imputation:
22, 35, 28, 35, 41, 22, 19
```

The imputed values already existed in the original feature.

### Why use it?

Mean and median imputation can create an artificial concentration around
a single value, especially when a feature has many missing observations.
Random Sample Imputation attempts to preserve the original distribution
and variance by sampling from the existing observations.

### Notebook experiment

The notebook first applies Random Sample Imputation to the numerical
**Age** feature from the Titanic dataset.

The data contains:

-   `Age`
-   `Fare`
-   `Survived`

The notebook checks the percentage of missing values and finds that
approximately **19.87% of Age values are missing**.

After the train-test split, an `Age_imputed` column is created for
comparison. Missing training values are sampled from the available
non-missing `Age` values, while the test-set replacements are also
sampled from the training distribution.

The notebook then compares:

-   The distribution of `Age` before and after imputation.
-   The variance before and after imputation.
-   The covariance between `Fare`, `Age`, and `Age_imputed`.
-   Box plots of the original and imputed feature.

### Important observation

The original variance of `Age` in the notebook is approximately:

``` text
204.35
```

After Random Sample Imputation, it becomes approximately:

``` text
210.10
```

This is considerably closer to the original variance than what we would
generally expect from an approach that repeatedly inserts a single
central value.

The density plots also show that the original and imputed distributions
remain broadly similar.

### Important limitation

Random Sample Imputation treats the feature independently. It does not
use information from other features when selecting a replacement value.

For example, if `Age` and `Fare` have an important relationship,
randomly selecting an age does not guarantee that the selected age is
appropriate for the corresponding fare. Therefore, while the technique
can preserve the distribution of an individual feature, it does not
necessarily preserve relationships between features.

------------------------------------------------------------------------

## 🏠 Random Sample Imputation for Categorical Variables

Random Sample Imputation can also be applied to categorical features.

The notebook demonstrates this using the House Prices dataset with:

-   `GarageQual`
-   `FireplaceQu`
-   `SalePrice`

The missing-value proportions are approximately:

``` text
GarageQual     5.55%
FireplaceQu   47.26%
SalePrice      0.00%
```

The notebook randomly samples existing categories from the non-missing
observations and uses them to replace missing categories.

The resulting category distributions are then compared with the original
distributions.

For `GarageQual`, the original and imputed category proportions remain
very similar. This demonstrates that random sampling can also preserve
the distribution of categorical variables reasonably well.

The notebook additionally compares the relationship between
`FireplaceQu` categories and `SalePrice` before and after imputation
using KDE plots.

### Key takeaway

Random Sample Imputation can be used for both numerical and categorical
variables, but it works best when the proportion of missing values is
not excessively high and when preserving the feature's distribution is
important.

------------------------------------------------------------------------

# 🔎 2. Missing Indicator

## What is a Missing Indicator?

Sometimes the fact that a value is missing is itself useful information.

A **Missing Indicator** creates an additional binary feature that
records whether the original value was missing.

For example:

``` text
Age     Age_NA
22        0
35        0
NaN       1
28        0
NaN       1
```

Here:

-   `0` → the original value was present
-   `1` → the original value was missing

The original feature can then be imputed using another strategy, while
the indicator gives the model information about the original
missingness.

## Why is this useful?

Missingness is not always random.

For example, a customer choosing not to provide income information may
behave differently from a customer who provides it. Similarly, a medical
test that was not performed can itself contain information about a
patient.

A Missing Indicator allows the model to learn these patterns instead of
completely removing the information contained in the missingness.

------------------------------------------------------------------------

## 🧪 Notebook Experiment

The notebook again uses the Titanic dataset with:

-   `Age`
-   `Fare`
-   `Survived`

A Logistic Regression model is first trained after applying standard
`SimpleImputer` without a missing indicator.

The resulting test accuracy is:

``` text
0.6145
```

The notebook then creates a Missing Indicator using Scikit-Learn's
`MissingIndicator` class.

The indicator identifies which observations originally contained missing
values. For the Titanic example, this primarily captures missingness in
the `Age` feature.

The indicator is then added as an additional feature, after which the
missing values are imputed using `SimpleImputer`.

The Logistic Regression model is trained again.

The resulting accuracy increases to approximately:

``` text
0.6313
```

This experiment demonstrates an important idea: **the missingness
pattern itself can sometimes provide predictive information**.

------------------------------------------------------------------------

## 🛠️ Scikit-Learn Shortcut

Instead of creating the indicator separately with `MissingIndicator`,
Scikit-Learn provides a much simpler approach through `SimpleImputer`.

``` python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    add_indicator=True
)
```

Setting `add_indicator=True` tells `SimpleImputer` to both perform the
imputation and add missing-indicator features.

This is generally much cleaner when building a preprocessing pipeline.

------------------------------------------------------------------------

# ⚙️ 3. Automatic Selection of the Imputation Strategy

Different datasets can benefit from different imputation strategies.

For numerical variables, Mean Imputation may work better in one dataset
while Median Imputation may work better in another. Similarly,
categorical variables may perform better with Most Frequent or Constant
Imputation.

Instead of manually choosing the strategy, we can treat the imputation
method as a **hyperparameter** and allow cross-validation to evaluate
different options.

This is where **GridSearchCV** becomes useful.

A parameter grid can contain multiple imputation strategies, for
example:

``` python
param_grid = {
    "preprocessor__num__imputer__strategy": [
        "mean", "median"
    ],
    "preprocessor__cat__imputer__strategy": [
        "most_frequent", "constant"
    ]
}
```

GridSearchCV can then evaluate different combinations using
cross-validation and identify the combination that performs best
according to the chosen evaluation metric.

The important idea is that **preprocessing decisions can also be
optimized**, rather than being selected purely through intuition.

> **Note:** The current notebook focuses primarily on Random Sample
> Imputation and Missing Indicators. The GridSearchCV discussion is
> covered in the accompanying article and represents the next step
> toward building fully optimized preprocessing pipelines.

------------------------------------------------------------------------

# 📊 Comparing the Techniques

  -----------------------------------------------------------------------
  Technique         Main Idea         Main Strength     Main Limitation
  ----------------- ----------------- ----------------- -----------------
  **Random Sample   Replace missing   Preserves         Does not preserve
  Imputation**      values with       distribution and  relationships
                    randomly selected variance better   between features
                    existing                            
                    observations                        

  **Missing         Add a feature     Allows the model  May add
  Indicator**       showing whether   to learn from     unnecessary
                    the original      missingness       features when
                    value was missing                   missingness is
                                                        not informative

  **GridSearchCV    Compare multiple  Data-driven       Requires
  for Imputation**  imputation        strategy          additional
                    strategies        selection         computation
                    automatically                       
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🧩 Key Concepts Learned

### Random Sample Imputation

-   Uses existing non-missing observations as replacement values.
-   Can be applied to numerical and categorical features.
-   Helps preserve the original distribution.
-   Generally preserves variance better than repeatedly inserting a
    single value.
-   Does not account for relationships between different features.
-   Should be used carefully when the percentage of missing values is
    very high.

### Missing Indicator

-   Converts missingness into an additional feature.
-   Uses `0` for observed values and `1` for missing values.
-   Can be useful when missingness itself contains information.
-   Can be implemented separately with `MissingIndicator`.
-   Can also be combined directly with imputation using
    `SimpleImputer(add_indicator=True)`.

### GridSearchCV

-   Allows preprocessing choices to be treated as hyperparameters.
-   Can compare different numerical and categorical imputation
    strategies.
-   Uses cross-validation to identify the best-performing combination.
-   Helps replace manual trial-and-error with a systematic approach.

------------------------------------------------------------------------

# 📁 Files in This Folder

  ---------------------------------------------------------------------------------------------
  File                                                      Description
  --------------------------------------------------------- -----------------------------------
  `missing_indicator_and_random_sample_imputations.ipynb`   Practical notebook covering Random
                                                            Sample Imputation for numerical and
                                                            categorical variables, Missing
                                                            Indicators, and Logistic Regression
                                                            experiments.

  `README.md`                                               Overview of the concepts,
                                                            experiments, important
                                                            observations, and techniques
                                                            covered in this topic.
  ---------------------------------------------------------------------------------------------

The notebook expects the relevant datasets, including the Titanic
`train.csv` and House Prices `house-train.csv`, to be available in the
working directory used when running the notebook.

------------------------------------------------------------------------

# 📖 Related Article

The concepts and experiments from this topic are explained in greater
detail in the accompanying Medium article:

**Advanced Missing Data Imputation: Random Sample Imputation, Missing
Indicators & Automatic Strategy Selection**

https://medium.com/@priyanshu20032002/advanced-missing-data-imputation-random-sample-imputation-missing-indicators-automatic-strategy-d8c4236dd446

------------------------------------------------------------------------

# 🚀 What's Next?

The techniques covered here still treat each feature largely
independently.

The next step is to move toward **multivariate imputation**, where
information from other features is used to estimate missing values.

This leads to techniques such as:

-   **KNN Imputer**
-   **Iterative Imputer**
-   **MICE (Multiple Imputation by Chained Equations)**

These methods allow us to move beyond simply looking at one feature at a
time and start using relationships between variables to make more
informed imputations.

------------------------------------------------------------------------

## Relearning Machine Learning in Public

This repository is part of my journey of rebuilding my Machine Learning
foundations from the ground up through consistent study, implementation,
experimentation, and public documentation.

The goal isn't just to remember algorithms, but to understand **why they
work, when they should be used, and how to implement them in practice**.
