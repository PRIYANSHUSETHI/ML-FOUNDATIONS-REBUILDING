# Handling Mixed Variables in Machine Learning

> **Relearning Machine Learning in Public | Day 22**

This project explores how to handle **mixed variables** during feature engineering — features that contain both numerical and categorical information.

Using the **Titanic dataset**, this notebook demonstrates how to identify different types of mixed variables and transform them into clean, meaningful features that can be used effectively in machine learning pipelines.

The core idea is simple:

> **Don't force messy data into a single type. Decompose it into the information it actually contains.**

---

## 📌 What You'll Learn

- What mixed variables are
- Two common forms of mixed variables
- Handling numerical and categorical values mixed across rows
- Using `pd.to_numeric()` with `errors="coerce"`
- Separating numerical and categorical information
- Extracting information from combined strings
- Using string operations and Regular Expressions
- Handling `Cabin` and `Ticket` variables
- Reducing high-cardinality categorical features
- Imputing values after feature extraction
- Turning messy variables into model-ready features

---

## 🔍 What Are Mixed Variables?

Mixed variables generally appear in two forms.

### 1. Mixed Types Across Different Rows

A single column may contain both numerical values and categorical labels.

For example:

```text
1
2
3
Alone
5
```

The numerical values carry quantitative information, while `"Alone"` represents a categorical state.

Instead of converting the entire column into strings or discarding the categorical information, we separate the two.

### 2. Multiple Types Inside a Single Cell

A single value may contain both categorical and numerical information.

For example:

```text
C85
B52
C123
```

Here:

```text
C85
│ └── Numerical component → 85
└──── Categorical component → C
```

The same idea applies to more complicated identifiers such as:

```text
A/5 21171
PC 17599
```

Extracting the meaningful components allows us to create cleaner features.

---

# 🛠️ Approach 1: Separating Mixed Types Across Rows

For values such as:

```text
1
2
Alone
5
```

we first isolate the numerical values using `pd.to_numeric()`.

```python
df["number_numerical"] = pd.to_numeric(
    df["number"],
    errors="coerce",
    downcast="integer"
)
```

The important part is:

```python
errors="coerce"
```

Valid numerical values remain numbers, while values that cannot be converted become `NaN`.

This allows us to identify the categorical values separately.

```python
df["number_categorical"] = np.where(
    df["number_numerical"].isnull(),
    df["number"],
    np.nan
)
```

The original mixed feature is therefore transformed into two homogeneous features:

```text
number_numerical
number_categorical
```

Missing values can then be handled according to the meaning of the data.

For example, in the Titanic dataset, `"Alone"` can logically correspond to `0` companions.

---

# 🔤 Approach 2: Extracting Information from Combined Strings

Some variables contain multiple pieces of information inside a single string.

For example:

```text
Cabin → C85
```

can be separated into:

```text
Deck → C
Room → 85
```

This can be done using regular expressions:

```python
df["cabin_cat"] = df["Cabin"].str.extract(r"([A-Za-z]+)")

df["cabin_num"] = df["Cabin"].str.extract(r"(\d+)")
```

The result is:

| Original | Category | Numerical |
|----------|----------|-----------|
| C85 | C | 85 |
| C123 | C | 123 |
| B52 | B | 52 |

This is more useful than treating every individual cabin number as a separate category.

---

# 🎫 Ticket Feature Extraction

Ticket identifiers can also contain both categorical and numerical components.

Examples:

```text
A/5 21171
PC 17599
```

The notebook extracts the numerical component:

```python
df["ticket_num"] = df["Ticket"].apply(
    lambda x: x.split()[-1]
)

df["ticket_num"] = pd.to_numeric(
    df["ticket_num"],
    errors="coerce",
    downcast="integer"
)
```

and separates the categorical prefix:

```python
df["ticket_cat"] = df["Ticket"].apply(
    lambda x: x.split()[0]
)

df["ticket_cat"] = np.where(
    df["ticket_cat"].str.isdigit(),
    np.nan,
    df["ticket_cat"]
)
```

This transforms a messy identifier into separate features that can be processed independently.

---

# 📉 Why Does This Matter?

One of the biggest problems with variables such as `Cabin` and `Ticket` is **high cardinality**.

If every unique cabin or ticket number is treated as a separate category, the model may have to deal with hundreds of distinct values.

By extracting broader categories, we can reduce this complexity.

For example:

```text
C85   → C
C123  → C
B52   → B
```

Instead of learning from hundreds of individual cabin identifiers, the model can learn patterns associated with broader groups such as **Deck A, B, C, etc.**

This turns potentially noisy information into more useful features.

---

# 🚢 Dataset

The notebook uses the **Titanic dataset** to demonstrate mixed-variable handling.

The main variables explored are:

- `Number`
- `Cabin`
- `Ticket`

These provide good examples of both major forms of mixed data:

```text
Mixed across rows
        ↓
Number → 1, 2, 3, "Alone"

Mixed within cells
        ↓
Cabin → C85
Ticket → A/5 21171
```

---

# 🧰 Technologies Used

- Python
- NumPy
- Pandas
- Regular Expressions
- Jupyter Notebook

---

# 📂 Project Structure

```text
Handling Mixed Variables/
│
├── handling_mixed_variables.ipynb
└── README.md
```

---

# 📝 Medium Article

This notebook is part of my **Relearning Machine Learning in Public** series.

I documented the intuition, techniques, and implementation in the accompanying Medium article:

### Handling Mixed Variables in Machine Learning: Extracting Hidden Information from Messy Data

🔗 [Read the full article on Medium](https://priyanshu20032002.medium.com/handling-mixed-variables-in-machine-learning-extracting-hidden-information-from-messy-data-e60c63ee1120)

---

# 💡 Key Takeaways

- Mixed variables can contain both **numerical and categorical information**.
- Mixed types can occur **across different rows** or **inside individual cells**.
- `pd.to_numeric(errors="coerce")` is useful for separating numerical values from categorical labels.
- Regular expressions and string operations can extract meaningful components from combined identifiers.
- Splitting `Cabin` and `Ticket` into separate features can reduce unnecessary complexity.
- High-cardinality variables can often be transformed into more meaningful, lower-cardinality categories.
- Missing values should be handled after the relevant information has been extracted.
- Good feature engineering is often about **revealing information that already exists in the raw data**.

---

## ⭐ The Bigger Lesson

A messy feature is not necessarily a useless feature.

Sometimes the problem isn't that the data lacks information.

**The information is simply packed together in a form the model cannot understand.**

Feature engineering is the process of unpacking it.
