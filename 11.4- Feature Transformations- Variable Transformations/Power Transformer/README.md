# Function Transformers Explained: Applying Mathematical Transformations the Right Way in Scikit-Learn

This repository is part of my **Relearning Machine Learning in Public** series, where I'm rebuilding my Machine Learning foundations from scratch by revisiting core concepts, implementing them in Python, and documenting everything I learn.

This project focuses on **Function Transformers**—one of the simplest yet most effective feature engineering techniques for transforming numerical features using mathematical functions. You'll learn when these transformations are useful, how they affect different machine learning algorithms, and how to integrate them into Scikit-Learn preprocessing pipelines using `FunctionTransformer`.

---

## 📌 What You'll Learn

- Why feature transformations are important
- Understanding skewed distributions
- Why some algorithms benefit from normally distributed features
- How to evaluate feature distributions before transforming them
- Using PDF plots, Skewness, and QQ Plots
- Different mathematical transformations
  - Log Transformation
  - Square Root Transformation
  - Square Transformation
  - Reciprocal Transformation
- Using `FunctionTransformer` in Scikit-Learn
- Integrating FunctionTransformer with `ColumnTransformer`
- Comparing model performance before and after transformations
- Best practices for applying feature transformations

---

## 📂 Project Structure

```text
Function Transformers/
│
├── Function Transformers.ipynb
├── README.md
```

---

## 📖 Mathematical Transformations Covered

### 1. Log Transformation
Used primarily for highly right-skewed features.

- Compresses large values
- Reduces skewness
- Helps stabilize variance
- `np.log1p()` safely handles zero values

### 2. Square Root Transformation
A milder transformation than logarithms.

Useful when data is only moderately skewed.

### 3. Square Transformation
Typically used for left-skewed distributions.

Expands larger values to improve symmetry.

### 4. Reciprocal Transformation
Applies an inverse relationship using:

`1 / x`

Useful in specific scenarios where inversion improves the feature distribution.

---

## 📊 Distribution Evaluation Techniques

This project demonstrates:

- Probability Density Function (PDF)
- Skewness
- Quantile-Quantile (QQ) Plot

These tools help determine whether a feature actually requires transformation.

---

## ⚙️ FunctionTransformer in Scikit-Learn

Scikit-Learn's `FunctionTransformer` allows mathematical transformations to become part of reusable preprocessing pipelines.

Examples include:

- `np.log1p`
- `np.sqrt`
- Custom mathematical functions

The notebook also demonstrates how to combine `FunctionTransformer` with `ColumnTransformer` to selectively transform only specific features.

---

## 🚢 Case Study

The Titanic dataset is used to demonstrate:

- transforming the **Fare** feature
- leaving the **Age** feature unchanged
- visualizing distributions before and after transformation
- evaluating model performance using cross-validation

This illustrates why not every numerical feature should be transformed.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-Learn

---

## 📝 Medium Article

**Function Transformers Explained: Applying Mathematical Transformations the Right Way in Scikit-Learn**

https://priyanshu20032002.medium.com/function-transformers-explained-applying-mathematical-transformations-the-right-way-in-f40b47b9e878

---

## 🚀 Repository

https://github.com/PRIYANSHUSETHI/ML-FOUNDATIONS-REBUILDING

---

## ⭐ Key Takeaways

- Always inspect the distribution before applying transformations.
- Different transformations solve different types of skewness.
- FunctionTransformer makes preprocessing modular and reproducible.
- Linear models generally benefit more from feature transformations than tree-based models.
- Validate preprocessing decisions using cross-validation rather than relying on a single train-test split.

---

If you found this project helpful, consider giving the repository a ⭐ and feel free to connect with me on LinkedIn to follow my **Relearning Machine Learning in Public** journey.
