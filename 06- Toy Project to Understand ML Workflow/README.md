# Before the Model: Understanding the Machine Learning Workflow

> *"Before I started learning Machine Learning models, I wanted to understand what an actual Machine Learning project looks like from start to finish."*

Most Machine Learning tutorials begin with an algorithm.

Train a model. Make predictions. Calculate accuracy. But that's only one small piece of the puzzle.

Before a model ever learns from data, there's an entire workflow that prepares the problem, cleans the data, structures the inputs, evaluates the results, and helps us understand whether the model learned anything useful.

This project was my attempt to understand **that workflow**.

Not to build the smartest model. Not to achieve state-of-the-art accuracy.

Simply to answer one question:

> **What actually happens inside a Machine Learning project?**

---

## What You'll Find in This Project

Using a simple Student Placement dataset, this notebook walks through the complete lifecycle of a supervised Machine Learning project.

- 📥 Importing the data
- 🔍 Exploring and understanding the dataset
- 🧹 Basic preprocessing
- 🎯 Selecting features and target variables
- ✂️ Splitting the data into training and testing sets
- ⚖️ Scaling the features
- 🤖 Training a Logistic Regression model
- 📈 Making predictions
- ✅ Evaluating model performance
- 📊 Visualizing the decision boundary

The goal isn't to master Logistic Regression.

The goal is to understand **where every piece fits into the larger Machine Learning pipeline.**

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## What This Project Taught Me

One realization completely changed how I look at Machine Learning.

The model isn't where Machine Learning begins. It begins with understanding the problem. Then comes the data. Then exploration. Then preprocessing.

Only after all of that do we finally reach the algorithm.

Instead of seeing Machine Learning as *"pick an algorithm and train it,"* I now see it as a structured workflow where every step builds on the previous one.

---
| Step | What happens | In the notebook |
|---|---|---|
| 1. Reading the Data | Load the dataset before doing anything else | `pd.read_csv('placement.csv')` |
| 2. Exploratory Data Analysis | Check shape, dtypes, and plot CGPA vs IQ to see the two classes visually | `df.shape`, `df.info()`, scatter plot colored by placement |
| 3. Data Pre-processing | Drop the column that isn't useful for prediction | `df = df.iloc[:, 1:]` |
| 4. Choosing Inputs and Outputs | Separate what the model already knows (CGPA, IQ) from what it should predict (placement) | `X = df.iloc[:,0:2]`, `y = df.iloc[:,-1]` |
| 5. Splitting the Dataset | Never test on the data the model trained on | `train_test_split(X, y, test_size=0.1)` |
| 6. Scaling the Features | Put features on a comparable scale so no single feature dominates just because of its range | `StandardScaler()` |
| 7. Training the Model | Fit a model to the training split | `LogisticRegression().fit(X_train, y_train)` |
| 8. Making Predictions | Generate the model's best guesses on unseen data | `clf.predict(X_test)` |
| 9. Evaluating Performance | Compare predictions against the actual labels | `accuracy_score(y_test, y_pred)` |
| 10. Visualizing the Decision Boundary | See how the model actually separates the two classes | `plot_decision_regions()` from `mlxtend` |

---

## Medium Article

I wrote a detailed article explaining the ideas behind this project and the thought process that went into building it.

📖 **Before the Model: Understanding the Machine Learning Workflow**

🔗 https://priyanshu20032002.medium.com/before-the-model-understanding-the-machine-learning-workflow-7f5e6dfe78e3

---

## Relearning Machine Learning in Public

This repository is part of my journey of rebuilding my Machine Learning foundations from first principles.

Rather than rushing into advanced models, I'm trying to understand **why things work** before learning **how to implement them**.

Each project builds on the previous one, gradually moving from intuition to implementation.

---

## What's Next?

This project only scratches the surface.

Future projects in this series will explore topics like:

- Understandin the data
- EDA
- Handling Missing Data
- Feature Engineering
- Evaluation Metrics
- Model Selection
- Bias-Variance Tradeoff
- Cross Validation
- Regression & Classification Algorithms
- Ensemble Methods
- Neural Networks
- Deep Learning

One concept, one project, one article at a time.

---

## Let's Connect

If you're also learning Machine Learning or enjoy learning from first principles, I'd love to connect.

- **LinkedIn:** https://www.linkedin.com/in/priyanshu-sethi/
- **Medium:** https://priyanshu20032002.medium.com/
