# Feature Scaling in Machine Learning

Imagine you're trying to decide whether two people are similar.

One feature is **Age**, ranging from 18 to 60.

Another is **Annual Salary**, ranging from ₹20,000 to ₹20,00,000.

Even if age is just as important as salary, many machine learning algorithms won't see it that way. They'll naturally give far more importance to salary simply because its numerical values are much larger.

This isn't because salary is more meaningful.

It's because the algorithm only sees numbers.

That's exactly the problem **Feature Scaling** solves.

This repository is part of my **Relearning Machine Learning in Public** series, where I'm rebuilding my understanding of machine learning from first principles—focusing not just on *how* things work, but *why* they work.

---

# 📖 Medium Article

I explore these concepts in much greater detail in my accompanying Medium article:

### **Feature Scaling in Machine Learning: Why Your Features Need to Speak the Same Language**

🔗 **Read here:**

https://priyanshu20032002.medium.com/feature-scaling-in-machine-learning-why-your-features-need-to-speak-the-same-language-bc862dee1412

---

# 🚀 What You'll Learn

Instead of simply applying `StandardScaler()` or `MinMaxScaler()`, this repository explains the intuition behind feature scaling.

You'll learn:

- Why feature scaling is necessary
- Which machine learning algorithms actually require scaling
- Why some algorithms are completely unaffected by it
- The mathematical intuition behind Standardization
- The mathematical intuition behind Normalization
- When to choose one technique over the other
- How to correctly apply scaling without introducing data leakage

---

# 📂 Repository Contents

```
Feature Scaling/
│
├── Feature_Scaling_Standardization.ipynb
├── Feature_Scaling_Normalization.ipynb
└── README.md
```

---

# 📚 Topics Covered

## Understanding Feature Scaling

- What feature scaling is
- Why varying feature magnitudes create problems
- Real-world intuition
- Distance-based algorithms and optimization

---

## Standardization (Z-Score Scaling)

Learn how Standardization transforms data by centering it around the mean.

Topics include:

- Z-score intuition
- Mathematical formula
- Mean = 0
- Standard deviation = 1
- Why negative values appear
- Advantages and limitations
- Distribution before and after scaling

Implementation includes:

- `StandardScaler`
- `fit()`
- `transform()`
- `fit_transform()`

You'll also see how Scikit-Learn stores the learned parameters using:

- `mean_`
- `scale_`

---

## Normalization (Min-Max Scaling)

Understand how Min-Max Scaling compresses feature values into a fixed range.

Topics include:

- Mathematical intuition
- Scaling values between 0 and 1
- Advantages
- Limitations
- Why normalization is sensitive to outliers
- Common applications

Implementation includes:

- `MinMaxScaler`
- `fit()`
- `transform()`
- `fit_transform()`

---

## Standardization vs Normalization

A practical comparison covering:

- Mathematical differences
- Output ranges
- Effect of outliers
- Suitable machine learning algorithms
- When each technique should be preferred

---

## Best Practices

One of the most overlooked parts of preprocessing is avoiding **data leakage**.

This repository demonstrates the correct workflow:

1. Split the dataset
2. Fit the scaler only on the training data
3. Transform both training and testing data using the same scaler

Understanding *why* this order matters is just as important as writing the code.

---

# 🛠️ Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

# 🎯 Key Takeaways

After going through these notebooks, you'll understand:

✅ Why feature scaling is often essential

✅ Which algorithms require scaling (and which don't)

✅ The intuition behind Standardization and Normalization

✅ How both techniques work mathematically

✅ Their advantages, disadvantages, and ideal use cases

✅ How to implement them correctly using Scikit-Learn

Most importantly, you'll understand **why feature scaling isn't just another preprocessing step—it's about ensuring every feature gets a fair opportunity to contribute to the learning process.**

---

# 🔜 What's Next?

Scaling isn't always enough.

Many real-world datasets contain heavily skewed features where simply changing the scale doesn't solve the problem.

In the next repository, we'll explore **Feature Transformation** techniques like:

- Log Transformation
- Reciprocal Transformation
- Square Root Transformation
- Power Transformations

and learn how changing a feature's distribution can improve machine learning models even further.

---

# 🤝 Let's Connect

If you found this repository helpful or have suggestions, I'd love to hear from you!

- **LinkedIn:** https://www.linkedin.com/in/priyanshu-sethi/
- **Medium:** https://priyanshu20032002.medium.com

Happy Learning! 🚀
