# 📊 Univariate Exploratory Data Analysis

<div align="center">

### Part of **Relearning ML in Public** 🚀
*Rebuilding Machine Learning foundations from scratch and documenting the journey in public.*

</div>

---

## 🗺️ Machine Learning Roadmap

```
✔ Problem Framing

✔ Supervised Learning

✔ Unsupervised Learning

✔ Tensors

✔ Exploratory Data Analysis
      ├── ✔ Univariate Analysis  ← You are here
      ├── ⏳ Bivariate Analysis
      └── ⏳ Multivariate Analysis

⬜ Data Cleaning

⬜ Feature Engineering

⬜ Feature Selection

⬜ Model Building

⬜ Model Evaluation

⬜ Model Deployment
```

---

## 📖 About this Repository

Before selecting a machine learning algorithm, tuning hyperparameters, or optimizing model performance, there's a much more fundamental question to answer:

> **Do you actually understand your data?**

That's the purpose of **Exploratory Data Analysis (EDA).**

This repository explores the **first stage of EDA — Univariate Analysis**, where every feature is studied independently to understand its characteristics before investigating relationships between variables.

Using the famous **Titanic dataset**, this notebook demonstrates how to:

- understand the distribution of individual features
- distinguish between categorical and numerical variables
- identify skewness and outliers
- summarize variables using descriptive statistics
- build intuition before moving to preprocessing or model building

Rather than treating EDA as a collection of plotting functions, the goal is to understand **why each visualization exists, what question it answers, and how it influences the rest of the machine learning workflow.**

---

# 📚 Resources

### 📄 Companion Article

**Exploring Data Before Building Models: A Practical Guide to Univariate EDA**

🔗 https://priyanshu20032002.medium.com/exploring-data-before-building-models-a-practical-guide-to-univariate-eda-1d61a08e9e9b

### 📓 Notebook

`EDA_Using_Univariate_Analysis.ipynb`

### 📊 Dataset

Titanic Dataset (`train.csv`)

https://www.kaggle.com/c/titanic/data

---

# 🎯 What You'll Learn

After working through this notebook, you'll be able to:

- ✅ Understand the role of Exploratory Data Analysis in the ML pipeline
- ✅ Differentiate between categorical and numerical variables
- ✅ Perform univariate analysis on any structured dataset
- ✅ Choose the right visualization for different feature types
- ✅ Interpret charts instead of simply generating them
- ✅ Detect distributions, skewness, spread, and potential outliers
- ✅ Build intuition before feature engineering and model training

---

# 🛠️ Topics Covered

### Dataset Exploration

- Importing libraries
- Loading data with Pandas
- Inspecting the dataset
- Understanding feature types

---

### Categorical Variable Analysis

- Count Plots
- Pie Charts
- Frequency Distribution
- Percentage Distribution

Applied to:

- Survived
- Passenger Class
- Sex
- Embarked

---

### Numerical Variable Analysis

- Histograms
- Distribution (KDE) Plots
- Box Plots
- Descriptive Statistics

Applied to:

- Age
- Fare
- SibSp
- Parch

---

# ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 📂 Repository Structure

```
.
├── EDA_Using_Univariate_Analysis.ipynb
├── train.csv
├── README.md
```

---

# 💡 Key Takeaways

- Exploratory Data Analysis is about **understanding data before building models.**
- Different variable types require different analytical techniques.
- Good visualizations answer questions rather than simply displaying information.
- Outliers should be investigated—not automatically removed.
- A strong univariate analysis creates the foundation for every subsequent stage of the machine learning workflow.

---

# 🔜 What's Next?

Understanding variables individually is only the beginning.

The next repositories in this series will cover:

### 📈 Bivariate Analysis

Exploring relationships between **two variables**.

Examples:

- Does age influence survival?
- Does passenger class affect ticket fare?

---

### 📊 Multivariate Analysis

Understanding how **multiple variables interact simultaneously** to uncover deeper patterns hidden within the data.

Together, these complete the core Exploratory Data Analysis workflow before moving into feature engineering and model building.

---

# 🤝 Let's Connect

If you found this repository useful:

⭐ Star the repository

📖 Read more on Medium

https://priyanshu20032002.medium.com/

💼 Connect on LinkedIn

https://www.linkedin.com/in/priyanshusethi/

---

<div align="center">

### 🚀 Thanks for following my **Relearning ML in Public** journey!

If this repository helped you learn something new, consider giving it a ⭐.

</div>
