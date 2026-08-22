# 🧹 Complete Case Analysis for Handling Missing Data

> **Relearning Machine Learning in Public | Day 24**

Missing values are one of the most common problems in real-world datasets.

Before training a machine learning model, we need to decide what to do with observations that contain missing information.

One possible approach is simply to remove them.

This project explores **Complete Case Analysis (CCA)**, also known as **Listwise Deletion** — a technique that removes observations containing missing values from the selected features.

But the important question is not:

> **How do we delete missing values?**

The real question is:

> **When is deleting data actually the right decision?**

Using a real-world job candidate dataset, this notebook explores the assumptions behind Complete Case Analysis, applies it selectively to features with relatively low missingness, and validates whether removing observations significantly changes the underlying data.

> **The goal is not simply to remove missing values. The goal is to remove them without changing the story the data is trying to tell.**

---

# 📌 What You'll Learn

This notebook covers:

- What Complete Case Analysis is
- Why blindly deleting missing values can be dangerous
- The concept of **Missing Completely At Random (MCAR)**
- The practical **5% rule of thumb**
- How to identify features suitable for Complete Case Analysis
- How to use Pandas' `dropna()` function
- Why the `subset` parameter is important
- How to compare dataset size before and after CCA
- How to validate numerical distributions using Probability Density Functions
- How to compare categorical proportions before and after deletion
- The advantages and limitations of Complete Case Analysis

---

# 🔍 What is Complete Case Analysis?

Complete Case Analysis is one of the simplest techniques for handling missing values.

The idea is straightforward:

```text
Dataset
   │
   ▼
Check for Missing Values
   │
   ▼
Does a selected feature contain NaN?
   │
   ├── Yes → Remove the entire observation
   │
   └── No  → Keep the observation
```

Only observations with complete information across the selected features are retained.

For example:

### Before Complete Case Analysis

| Customer | Age | Income | Experience |
|---|---:|---:|---:|
| A | 24 | 450000 | 2 |
| B | 31 | NaN | 6 |
| C | 27 | 620000 | 3 |
| D | NaN | 510000 | 4 |
| E | 35 | 780000 | 9 |

After applying Complete Case Analysis:

### After Complete Case Analysis

| Customer | Age | Income | Experience |
|---|---:|---:|---:|
| A | 24 | 450000 | 2 |
| C | 27 | 620000 | 3 |
| E | 35 | 780000 | 9 |

Rows containing missing values are removed completely.

No values are estimated.

No artificial values are introduced.

The incomplete observations are simply discarded.

---

# 🧠 The Core Assumption: MCAR

Complete Case Analysis works best when the missing data is:

## **Missing Completely At Random (MCAR)**

MCAR means that the probability of a value being missing is unrelated to:

- The missing value itself
- Other variables in the dataset
- The target variable

A useful way to think about this is through random sampling.

Imagine a dataset containing 1,000 observations.

If we randomly remove 50 observations:

```text
Original Dataset
1000 Observations
        │
        │ Randomly remove 50
        ▼
Remaining Dataset
950 Observations
```

The remaining 950 observations should still reasonably represent the original population because the removed observations were selected randomly.

This is the basic intuition behind MCAR.

However, things become problematic when missingness is related to the data itself.

For example:

```text
High-income employees
        │
        ▼
More likely to hide their salary
        │
        ▼
Salary values are missing
        │
        ▼
Complete Case Analysis removes them
        │
        ▼
Remaining data underrepresents high-income employees
```

In this case, deleting the missing observations can introduce bias.

This is why the **reason behind missingness matters just as much as the amount of missing data**.

---

# 📊 The 5% Rule of Thumb

A common practical guideline is to consider Complete Case Analysis when the percentage of missing values in a feature is relatively small.

A frequently used rule of thumb is:

> **If a feature contains less than approximately 5% missing values, Complete Case Analysis may be a reasonable option.**

A simple decision framework looks like this:

| Missing Percentage | Possible Approach |
|---|---|
| **Less than 5%** | Complete Case Analysis may be reasonable |
| **5–20%** | Investigate the missingness carefully |
| **20–50%** | Consider imputation or other approaches |
| **Very high missingness** | Consider whether the feature should be removed |

These are not strict rules.

The final decision should depend on:

- Dataset size
- Importance of the feature
- Amount of data lost
- Reason for missingness
- Impact on the distribution
- The machine learning problem being solved

---

# 📂 Dataset

This notebook uses a **Data Science Job Candidate dataset** containing:

```text
19,158 observations
```

The dataset contains a combination of:

- Numerical variables
- Categorical variables
- Demographic information
- Education-related information
- Work experience
- Company information
- Training-related information

Several features contain missing values, making it a useful dataset for exploring different missing-data handling strategies.

---

# 🔎 Step 1: Exploring Missing Values

Before removing anything, the first step is to identify:

1. Which features contain missing values
2. How much data is missing from each feature
3. Whether applying Complete Case Analysis would result in excessive data loss

The general workflow is:

```text
Raw Dataset
     │
     ▼
Identify Missing Values
     │
     ▼
Calculate Missing Percentage
     │
     ├── Low Missingness
     │         │
     │         ▼
     │    Candidate for CCA
     │
     └── High Missingness
               │
               ▼
      Consider Other Strategies
```

The important lesson here is that we should **not blindly apply `dropna()` to the entire dataset**.

A single feature with a large amount of missing data could cause us to remove a significant proportion of the observations.

---

# 🎯 Features Selected for Complete Case Analysis

Instead of applying CCA to every column, the notebook selectively targets features with relatively low levels of missingness.

The selected variables include:

- `city_development_index`
- `training_hours`
- `experience`
- `enrolled_university`
- `education_level`

Features with substantially higher missingness, such as:

- `gender`
- `company_size`
- `company_type`

are excluded from the Complete Case Analysis step.

This prevents unnecessary loss of observations.

The idea is:

```text
Low Missingness Features
        │
        ▼
Apply CCA
        │
        ▼
Remove Selected Incomplete Rows


High Missingness Features
        │
        ▼
Do Not Include in CCA
        │
        ▼
Handle Separately Later
```

---

# 🛠️ Implementing Complete Case Analysis

The simplest way to perform Complete Case Analysis in Pandas is:

```python
df.dropna()
```

However, this removes every row containing a missing value anywhere in the dataset.

That can be unnecessarily aggressive.

Instead, we can use the `subset` parameter.

```python
df_clean = df.dropna(
    subset=[
        'city_development_index',
        'experience',
        'training_hours',
        'enrolled_university',
        'education_level'
    ]
)
```

Now, Pandas only checks the specified features.

This gives us much more control over which observations are removed.

---

# 📉 Impact on Dataset Size

After applying Complete Case Analysis:

```text
Original Dataset
19,158 observations
        │
        ▼
Complete Case Analysis
        │
        ▼
17,182 observations
```

Approximately:

```text
1,976 observations removed
```

This represents roughly:

```text
10% of the original dataset
```

At first glance, losing nearly 2,000 observations may seem significant.

But dataset size alone does not tell us whether CCA was a good or bad decision.

The next question is much more important:

> **Did removing those observations change the underlying distribution of the data?**

---

# 📈 Step 2: Validating Numerical Features

For numerical features, the notebook compares the distributions before and after Complete Case Analysis.

The workflow looks like this:

```text
Original Dataset
       │
       ├── Distribution of Feature
       │
       ▼
Apply CCA
       │
       ▼
Cleaned Dataset
       │
       └── Distribution of Same Feature
```

The distributions can then be compared using visualisation techniques such as:

- Probability Density Functions
- Histograms
- Distribution plots

If the distributions overlap closely, it suggests that removing the incomplete observations has not dramatically changed the feature.

The notebook compares numerical variables including:

- `city_development_index`
- `experience`
- `training_hours`

The resulting distributions remain largely stable after Complete Case Analysis, supporting the idea that the removed observations did not substantially distort these variables.

---

# 🧩 Step 3: Validating Categorical Features

Numerical distributions are not enough.

Complete Case Analysis can also unintentionally affect categorical variables.

For example, imagine the following situation:

### Before CCA

| Education Level | Percentage |
|---|---:|
| Graduate | 60% |
| Masters | 25% |
| High School | 15% |

After CCA:

| Education Level | Percentage |
|---|---:|
| Graduate | 45% |
| Masters | 40% |
| High School | 15% |

Although the dataset may now contain no missing values in the selected features, the proportions have changed significantly.

This could indicate that CCA disproportionately removed observations from certain groups.

To investigate this, the notebook compares the relative proportions of categories before and after deletion.

Conceptually:

```text
Original Dataset
      │
      ▼
Calculate Category Ratios
      │
      ▼
Apply CCA
      │
      ▼
Calculate New Category Ratios
      │
      ▼
Compare
```

When the proportions remain similar, it provides additional evidence that the deletion process has not disproportionately affected particular categories.

---

# 📊 Complete Case Analysis Workflow

The entire approach followed in this notebook can be summarised as:

```text
                Raw Dataset
                     │
                     ▼
          Identify Missing Values
                     │
                     ▼
       Calculate Missing Percentage
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Low Missingness        High Missingness
          │                     │
          ▼                     ▼
 Apply Complete Case      Handle Separately
       Analysis          (Imputation / Drop)
          │
          ▼
     Cleaned Dataset
          │
          ├───────────────────┐
          │                   │
          ▼                   ▼
Compare Numerical     Compare Categorical
 Distributions          Proportions
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
         Validate Data Integrity
                    │
                    ▼
       Continue ML Pipeline
```

---

# ✅ Advantages of Complete Case Analysis

### 1. Simple to Implement

CCA can often be implemented with a single line of code:

```python
df.dropna()
```

or with greater control:

```python
df.dropna(subset=selected_columns)
```

### 2. No Artificial Data

Unlike imputation techniques, CCA does not estimate or invent values.

The remaining observations contain only original values.

### 3. Computationally Efficient

No complex statistical modelling is required.

This makes CCA fast and easy to implement.

### 4. Can Preserve Distributions

When the MCAR assumption is reasonably satisfied and only a small amount of data is removed, the remaining dataset can retain a distribution similar to the original data.

---

# ⚠️ Limitations of Complete Case Analysis

### 1. Data Loss

Every incomplete observation is removed.

When missingness exists across multiple features, the total amount of data lost can become much larger than expected.

### 2. Risk of Bias

If the missing values are not random, removing observations can create a biased dataset.

The model may then learn from a population that no longer represents the real world.

### 3. Loss of Potentially Useful Information

Sometimes, the fact that a value is missing is itself meaningful.

Deleting the entire observation removes that information.

### 4. Production Challenges

A model trained using Complete Case Analysis still needs a strategy for handling missing values when new data arrives.

If a production pipeline receives an observation containing a missing value, additional preprocessing is required before the model can make a prediction.

---

# ⚖️ Advantages vs Limitations

| Advantages | Limitations |
|---|---|
| Simple to implement | Can cause significant data loss |
| Computationally inexpensive | Depends on assumptions about missingness |
| Does not create artificial values | Can introduce bias |
| Preserves original values in retained rows | Missingness may itself contain useful information |
| Can preserve distributions when appropriate | Requires additional handling for production data |

---

# 💡 Key Learnings

This project reinforced several important lessons about handling missing data:

- Missing values should never be removed blindly.
- Understanding **why** data is missing is just as important as knowing **how much** is missing.
- Complete Case Analysis works best when missingness behaves approximately like **MCAR**.
- The **5% rule** is a useful guideline, not an absolute law.
- Using `dropna()` with `subset` provides more control than dropping every row containing a missing value.
- Always compare the dataset before and after preprocessing.
- Numerical distributions can be validated using PDFs and histograms.
- Categorical variables should be validated by comparing category proportions.
- Preserving dataset size is not enough — we also need to preserve the underlying characteristics of the data.
- A simple preprocessing technique can still have major consequences if applied without validation.

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 📂 Project Structure

```text
Handling Missing Data/
│
├── handling_missing_data_(part_1).ipynb
└── README.md
```

---

# 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/PRIYANSHUSETHI/ML-FOUNDATIONS-REBUILDING.git
```

Navigate to the project directory and open the notebook using:

- Jupyter Notebook
- JupyterLab
- VS Code
- Google Colab

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

Then open:

```text
handling_missing_data_(part_1).ipynb
```

and run the cells sequentially.

---

# 📝 Medium Article

This notebook is part of my **Relearning Machine Learning in Public** series.

### Complete Case Analysis Explained: When Deleting Missing Data Is Actually the Right Choice in Machine Learning

Read the full article on Medium:

https://medium.com/@priyanshu20032002/complete-case-analysis-explained-when-deleting-missing-data-is-actually-the-right-choice-in-a6e1e169d53d

---

# ⭐ The Bigger Lesson

Handling missing data is not simply a preprocessing task.

It's a decision about **what information we keep and what information we throw away**.

Complete Case Analysis may only require one line of code:

```python
df.dropna()
```

But deciding whether that one line is appropriate requires understanding the data, investigating the missingness, measuring the information loss, and validating the result.

```text
Missing Data
      ↓
Understand Why It Is Missing
      ↓
Measure the Amount of Missingness
      ↓
Choose an Appropriate Strategy
      ↓
Validate the Impact
      ↓
Build a More Reliable Model
```

> **Good preprocessing is not about making a dataset look clean. It's about making sure the dataset remains trustworthy.**

---

### 🌱 Part of the "Relearning Machine Learning in Public" Series

This repository documents my journey of revisiting and strengthening the foundations of Machine Learning — one concept, notebook, and project at a time.

If you find this repository useful, feel free to ⭐ **star the repository** and follow along with the journey.
