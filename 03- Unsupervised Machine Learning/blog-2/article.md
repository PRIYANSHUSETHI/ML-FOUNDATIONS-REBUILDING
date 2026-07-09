# Deconstructing Unsupervised Learning: Discovering Patterns Without Labels

> **Part 2 – Finding Structure in Chaos**

Medium Profile: https://medium.com/@priyanshu20032002

---

## 📖 Overview

This repository accompanies **Part 2** of my three-part series on **Unsupervised Learning**, written as part of my **Relearning ML in Public** journey.

In the first article, we explored *why* unsupervised learning exists and how it differs fundamentally from supervised learning.

This second part answers the natural follow-up question:

> **If hidden structure already exists inside the data, how can a machine actually discover it?**

To answer that, we explore the intuition behind clustering and dimensionality reduction—two of the most fundamental ideas in unsupervised learning.

Rather than focusing on implementation details, this article builds an understanding of **how these algorithms think**, the assumptions they make, and when they are most effective.

---

## 🎯 Learning Objectives

By the end of this article, you'll understand:

- What clustering really means
- How K-Means discovers hidden groups
- The intuition behind the K-Means objective function
- Choosing the right number of clusters
- Limitations of K-Means
- Why DBSCAN views clusters differently
- The Curse of Dimensionality
- How Principal Component Analysis (PCA) simplifies complex datasets
- When dimensionality reduction becomes necessary

---

## 📚 Read the Article

📖 **Medium**

🔗 https://medium.com/@priyanshu20032002/deconstructing-unsupervised-learning-discovering-patterns-without-labels-2bbf3b3c2c0d

---

## 🖼 Figures

| Figure | Description |
|---------|-------------|
| Figure 1 | Customer data before and after clustering |
| Figure 2 | K-Means clustering process |
| Figure 3 | Comparison of K-Means and DBSCAN on non-spherical data |
| Figure 4 | High-dimensional data projected onto principal components using PCA |

---

## 💡 Key Takeaways

- Clustering is about discovering natural groupings in data—not forcing observations into predefined categories.
- K-Means works well when clusters are compact and roughly spherical but struggles with irregular shapes.
- DBSCAN identifies clusters based on density, making it effective for arbitrarily shaped clusters and noisy datasets.
- High-dimensional data becomes increasingly difficult to interpret and analyze.
- PCA reduces dimensionality while preserving as much information as possible, making visualization and downstream learning easier.
- Every clustering algorithm makes different assumptions about what a "cluster" looks like, so there is no universally best approach.

---

## 📂 Folder Structure

```text
blog-2-finding-structure-in-chaos/
│
├── article.md
```

---

## 📖 Series Navigation

⬅️ **Previous:** **Part 1 – Learning to See Without Being Told**

https://medium.com/@priyanshu20032002/deconstructing-unsupervised-learning-discovering-patterns-without-labels-0814856b899b

➡️ **Next:** **Part 3 – Beyond Groups: Representations and Relationships**

https://medium.com/@priyanshu20032002/28d50264f32e

---

## 🌐 Connect With Me

- **Medium:** https://medium.com/@priyanshu20032002
- **LinkedIn:** *https://www.linkedin.com/in/priyanshu-sethi-bitsh/*

---

## ⭐ Support the Project

If you found this repository helpful, consider **starring** the project.

It helps others discover the series and motivates me to continue documenting my journey of **Relearning Machine Learning in Public**.

---

> *The best way to understand an algorithm isn't to memorize its steps—it's to understand the assumptions it makes about the data.*
