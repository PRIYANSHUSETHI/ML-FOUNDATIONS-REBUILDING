# Before the Model: Understanding Your Data the Right Way

Understanding your dataset is one of the most important—and often overlooked—steps in any machine learning project.

Before selecting algorithms, engineering features, or tuning hyperparameters, it's essential to ask the right questions about the data itself.

This repository is part of my **Relearning Machine Learning in Public** series and focuses on developing the habit of exploring and understanding a dataset before moving on to Exploratory Data Analysis (EDA) or model building.

---

## 📖 Related Blog

Read the complete article here:

**Before the Model: Understanding Your Data the Right Way**  
🔗 https://priyanshu20032002.medium.com/before-the-model-understanding-your-data-the-right-way-1e48cd4bf728

---

## 🎯 Project Objective

The objective of this notebook is **not** to train a machine learning model.

Instead, it demonstrates the fundamental questions every data scientist should ask when working with a new dataset. These initial checks help uncover the dataset's structure, quality, and characteristics, laying the groundwork for every stage of the machine learning pipeline.

---

## 📂 Dataset

This project uses the **Titanic Dataset**, one of the most widely used datasets for learning machine learning and data analysis.

**Target Variable**

| Column | Description |
|---------|-------------|
| **Survived** | 0 → Did Not Survive, 1 → Survived |

---

# ❓ Questions Explored

Instead of jumping straight into model building, this notebook answers seven fundamental questions about the dataset.

## 1. How big is the dataset?

Understanding the size of the dataset helps estimate its complexity and influences decisions around preprocessing, model selection, and computational requirements.

### Function Used

```python
df.shape
```

---

## 2. What does the data actually look like?

Viewing random observations provides a better representation of the dataset than simply looking at the first few rows.

### Functions Used

```python
df.head()

df.sample(5)
```

---

## 3. What is the data type of every feature?

Knowing whether features are numerical, categorical, or text is essential before preprocessing and feature engineering.

### Function Used

```python
df.info()
```

---

## 4. Are there missing values?

Missing values are common in real-world datasets and must be identified before training machine learning models.

### Function Used

```python
df.isnull().sum()
```

---

## 5. What does the data look like mathematically?

Summary statistics provide valuable insights into the distribution and spread of numerical features.

### Function Used

```python
df.describe()
```

---

## 6. Are there duplicate records?

Duplicate rows can bias a model by giving repeated observations more importance than intended.

### Function Used

```python
df.duplicated().sum()
```

---

## 7. How are numerical features related?

Correlation analysis helps identify relationships between numerical variables and the target variable.

### Functions Used

```python
df.corr(numeric_only=True)

df.corr(numeric_only=True)['Survived']
```

---

# 🛠️ Pandas Functions Used

| Function | Purpose |
|----------|---------|
| `pd.read_csv()` | Load the dataset into a Pandas DataFrame |
| `shape` | Returns the number of rows and columns |
| `head()` | Displays the first few observations |
| `sample()` | Displays random observations |
| `info()` | Displays column information, data types, non-null counts, and memory usage |
| `isnull().sum()` | Counts missing values in each column |
| `describe()` | Generates descriptive statistics for numerical columns |
| `duplicated().sum()` | Counts duplicate rows |
| `corr()` | Computes correlations between numerical features |

---

# 📌 Key Takeaways

After completing this notebook, you will be able to:

- Understand the overall structure of a dataset
- Identify numerical and categorical features
- Detect missing values
- Identify duplicate records
- Interpret descriptive statistics
- Analyze correlations between numerical variables
- Develop the habit of understanding data before building machine learning models

---

# 📁 Project Structure

```
.
├── Understanding_your_Data_Asking_basic_questions.ipynb
├── titanic.csv
├── README.md
```

---

# 💻 Technologies Used

- Python
- Pandas
- Jupyter Notebook

---

# 🚀 What's Next?

This notebook concludes the **Data Understanding** stage of the machine learning workflow.

The next step is **Exploratory Data Analysis (EDA)**, where we'll move beyond tables and numerical summaries to visualize the data through charts and plots. EDA helps uncover hidden patterns, identify outliers, study feature distributions, and develop a much stronger intuition about the dataset before preprocessing and model development.

---

# 🌱 Relearning Machine Learning in Public

This repository is part of my **Relearning Machine Learning in Public** journey, where I'm revisiting machine learning from first principles while documenting everything I learn through projects, blogs, and code.

The goal isn't just to build models—it's to understand **why** every step in the machine learning workflow exists.

If you find these repositories helpful, consider ⭐ starring them and following along as the series progresses.

---

# 🤝 Connect with Me

If you enjoyed this project or would like to follow my journey in Machine Learning, Data Science, and AI, let's connect!

- 💼 **LinkedIn**  
  https://www.linkedin.com/in/priyanshu-sethi/

- ✍️ **Medium**  
  https://priyanshu20032002.medium.com/

- 💻 **GitHub**  
  https://github.com/priyanshu20032002

I'm documenting my learning journey through hands-on projects, detailed technical articles, and practical implementations. Feedback, suggestions, and collaborations are always welcome!

⭐ **If this repository helped you, consider giving it a star!**
It motivates me to continue building and sharing my learning in public.
