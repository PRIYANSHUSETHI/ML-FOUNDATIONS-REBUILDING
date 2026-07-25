# Column Transformer with Scikit-Learn

This repository demonstrates how to use **Scikit-Learn's ColumnTransformer** to preprocess datasets containing multiple types of features in a single, unified workflow.

Instead of manually applying different preprocessing techniques to individual columns and combining the outputs, **ColumnTransformer** allows us to define all preprocessing steps in one place, resulting in cleaner, more scalable, and production-ready machine learning pipelines.

This project is part of my **Relearning Machine Learning in Public** series, where I'm revisiting machine learning concepts from first principles while documenting my learning journey through code, articles, and notes.

---

## 📖 Medium Article

Read the complete explanation here:

**ColumnTransformer Explained: The Right Way to Pre-process Mixed Data in Scikit-Learn**

🔗 https://priyanshu20032002.medium.com/columntransformer-explained-the-right-way-to-pre-process-mixed-data-in-scikit-learn-ff8cd27f72f5

---

## 📂 Files in this Repository

- **Column_Transformer.ipynb** – Complete implementation and walkthrough of ColumnTransformer.
- **Technical Specification.pdf** – Detailed notes explaining the intuition, architecture, workflow, and best practices behind ColumnTransformer.
- **Covid_toy.csv** - data for the project
- **README.md** – Project overview.

---

## 📌 Concepts Covered

- Why manual preprocessing becomes difficult for mixed datasets
- Understanding the need for ColumnTransformer
- Applying different preprocessing techniques to different columns
- Using `SimpleImputer` for missing values
- Applying `OrdinalEncoder` to ordinal categorical features
- Applying `OneHotEncoder` to nominal categorical features
- Understanding the `transformers` parameter
- Using the `remainder='passthrough'` parameter
- Executing preprocessing with a single `fit_transform()` call
- Preventing data leakage by maintaining consistent preprocessing
- Building cleaner and more maintainable preprocessing pipelines

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Google Colab

---

## 🚀 Key Takeaways

- Different columns often require different preprocessing techniques.
- ColumnTransformer applies transformations to selected columns simultaneously.
- Multiple preprocessing steps can be combined into a single object.
- The `remainder` parameter controls what happens to untouched columns.
- A single `fit_transform()` call generates a model-ready feature matrix.
- The same preprocessing workflow can be consistently applied to both training and test data.
- ColumnTransformer integrates seamlessly with Scikit-Learn Pipelines for production-ready workflows.

---

## 📚 Learning Series

This project is part of my **Relearning Machine Learning in Public** initiative, where I'm rebuilding my understanding of Machine Learning by combining theory with hands-on implementation.

Follow along as I continue exploring:

- Feature Engineering
- Feature Scaling
- Feature Selection
- Feature Extraction
- Machine Learning Algorithms
- Model Evaluation
- End-to-End ML Pipelines

---

If you found this repository useful, feel free to ⭐ the repository and connect with me on LinkedIn to follow my learning journey!
