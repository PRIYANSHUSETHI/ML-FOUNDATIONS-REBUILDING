# Deconstructing Exploratory Data Analysis: Understanding Relationships Through Bivariate and Multivariate Analysis

This repository contains the Python notebook accompanying my Medium article on **Bivariate and Multivariate Exploratory Data Analysis (EDA)** as part of my **Relearning Machine Learning in Public** series.

The goal of this notebook is not just to demonstrate how to create visualizations using Seaborn and Matplotlib, but to understand **why** each visualization exists, **what questions it answers**, and **when it should be used** during exploratory data analysis.

Unlike univariate analysis, where we examine one variable at a time, this notebook focuses on uncovering relationships between multiple variables—an essential step before feature engineering and machine learning model building.

---

## 📖 Medium Article

**Deconstructing Exploratory Data Analysis: Understanding Relationships Through Bivariate and Multivariate Analysis**

https://priyanshu20032002.medium.com/deconstructing-exploratory-data-analysis-understanding-relationships-through-bivariate-and-cc4b335d4344

---

## 📌 What You'll Learn

This notebook covers the fundamentals of bivariate and multivariate exploratory data analysis, including:

- Numerical vs Numerical analysis
- Numerical vs Categorical analysis
- Categorical vs Categorical analysis
- Time-series visualization basics
- Correlation analysis
- Understanding distributions across multiple groups
- Choosing the right visualization based on the business question

Rather than generating every possible graph, the notebook focuses on the most commonly used visualizations that help uncover meaningful patterns in data.

---

## 📊 Visualizations Covered

- Scatter Plot
- Scatter Plot with Hue, Style & Size
- Bar Plot
- Grouped Bar Plot
- Box Plot
- Box Plot with Hue
- Distribution Plot
- Kernel Density Estimation (KDE)
- Heatmap
- Cluster Map
- Pair Plot
- Line Plot

For every visualization, the notebook explores:

- What information it represents
- When to use it
- What patterns to look for
- Real-world applications

---

## 📂 Datasets Used

Since different visualizations require different types of variables, multiple well-known datasets have been used throughout the notebook.

### Tips Dataset
Used for understanding relationships between:
- Total Bill
- Tip Amount
- Gender
- Smoking Status
- Party Size
- Day
- Time

### Titanic Dataset
Used for analysing:
- Survival
- Passenger Class
- Age
- Fare
- Gender
- Family Information

### Iris Dataset
Used for exploring relationships between multiple numerical features and understanding pair plots.

### Flights Dataset
Used for demonstrating line plots and analysing trends in time-series data.

These datasets are simple, beginner-friendly, and widely used for learning data visualization techniques.

---

## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 🎯 Key Takeaways

By the end of this notebook, you'll understand how to:

- Explore relationships between variables
- Select the appropriate visualization for different types of data
- Interpret graphs beyond their appearance
- Identify trends, correlations, clusters, and outliers
- Build intuition for asking better questions during exploratory data analysis

More importantly, you'll see that **EDA is not about creating charts—it's about understanding the story your data is trying to tell.**

---

## 🚀 About the Series

This notebook is part of my **Relearning Machine Learning in Public** series, where I'm rebuilding my understanding of Machine Learning from first principles while documenting everything I learn through blogs, code, and practical implementations.

The aim is to create beginner-friendly resources that focus on intuition before implementation.

If you find this repository helpful, consider ⭐ starring it and following the series.

Happy Learning! 🚀
