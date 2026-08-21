# ⏰ Working with Date & Time Features in Machine Learning

> **Relearning Machine Learning in Public | Day 23**

Dates and timestamps often look like simple pieces of information, but a single timestamp can contain information about years, months, days, weekdays, weekends, quarters, hours, minutes, seconds, and elapsed time.

This project explores how to transform raw date and time variables into **meaningful features that machine learning models can learn from**.

> **The goal is not to create more features. The goal is to reveal the information hidden inside time.**

---

## 📌 What You'll Learn

- Convert date columns into Pandas `datetime` objects
- Extract year, month, month name, and day
- Extract the day of the week and identify weekends
- Extract week numbers, quarters, and semesters
- Work with timestamps containing both date and time
- Extract hours, minutes, and seconds
- Isolate the time component from a timestamp
- Calculate elapsed time between events
- Work with `Timedelta` and `np.timedelta64`
- Apply practical best practices for temporal feature engineering

---

# 🔍 Why Do Date & Time Features Matter?

Consider:

```text
2024-07-15 08:45:12
```

Instead of treating this as one opaque value, it can be decomposed into:

```text
Year        → 2024
Month       → 7
Day         → 15
Weekday     → Monday
Quarter     → 3
Hour        → 8
Minute      → 45
Second      → 12
```

This process is known as **Date and Time Feature Engineering** or **Temporal Feature Engineering**.

---

# 1️⃣ Converting Strings into Datetime Objects

Before extracting features, Pandas must recognise the column as a date or timestamp.

```python
df['date'] = pd.to_datetime(df['date'])
```

This unlocks Pandas' powerful:

```python
.dt
```

accessor.

---

# 📅 2️⃣ Extracting Primary Date Components

## 🗓️ Year

```python
df['year'] = df['date'].dt.year
```

## 📆 Month

```python
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.month_name()
```

Months can help capture seasonality, holidays, tourism patterns, weather effects, and recurring business cycles.

## 📍 Day of the Month

```python
df['day'] = df['date'].dt.day
```

This may capture salary cycles, subscription renewals, and beginning-of-month or end-of-month behaviour.

---

# 📊 3️⃣ Extracting Behavioural and Business Features

## 📅 Day of the Week

```python
df['day_of_week'] = df['date'].dt.dayofweek
```

| Day | Value |
|---|---:|
| Monday | 0 |
| Tuesday | 1 |
| Wednesday | 2 |
| Thursday | 3 |
| Friday | 4 |
| Saturday | 5 |
| Sunday | 6 |

For readable names:

```python
df['day_name'] = df['date'].dt.day_name()
```

## 🏖️ Identifying Weekends

```python
df['is_weekend'] = np.where(
    df['date'].dt.day_name().isin(['Saturday', 'Sunday']),
    1,
    0
)
```

Useful for e-commerce, food delivery, streaming, travel, and entertainment data.

## 🔢 Week Number

```python
df['week'] = df['date'].dt.isocalendar().week
```

## 📈 Quarter

```python
df['quarter'] = df['date'].dt.quarter
```

Useful for financial analysis, sales forecasting, business reporting, and seasonal modelling.

## 🎓 Semester

```python
df['semester'] = np.where(
    df['date'].dt.quarter.isin([1, 2]),
    1,
    2
)
```

---

# ⏰ 4️⃣ Working with Time Features

## 🕗 Hour

```python
df['hour'] = df['date'].dt.hour
```

Useful for identifying rush hours, peak shopping periods, meal-time demand, and customer activity patterns.

## ⏱️ Minute

```python
df['minute'] = df['date'].dt.minute
```

## ⏲️ Second

```python
df['second'] = df['date'].dt.second
```

## 🕒 Extracting Only the Time

```python
df['time_only'] = df['date'].dt.time
```

---

# ⌛ 5️⃣ Calculating Elapsed Time

Often, the most useful feature is not the timestamp itself but **how much time has passed since an event occurred**.

```python
from datetime import datetime

current_time = datetime.now()
df['duration'] = current_time - df['date']
```

The resulting values are Pandas `Timedelta` objects.

Extract elapsed days:

```python
df['days_elapsed'] = df['duration'].dt.days
```

Elapsed-time features can be especially useful in churn prediction, fraud detection, predictive maintenance, recommendation systems, and customer analytics.

---

# 🔄 6️⃣ Working with `np.timedelta64`

## 📅 Months Passed

```python
df['months_passed'] = (
    df['duration'] / np.timedelta64(1, 'M')
).astype(int)
```

## ⏱️ Minutes Passed

```python
df['minutes_passed'] = (
    df['duration'] / np.timedelta64(1, 'm')
).astype(int)
```

### ⚠️ Important: Unit Codes Are Case-Sensitive

```text
'M' → Months
'm' → Minutes
```

---

# 🧠 Putting It All Together

```text
Raw Timestamp
      │
      ├── Date Features
      │   ├── Year
      │   ├── Month
      │   ├── Day
      │   ├── Day of Week
      │   ├── Weekend
      │   ├── Week Number
      │   ├── Quarter
      │   └── Semester
      │
      ├── Time Features
      │   ├── Hour
      │   ├── Minute
      │   └── Second
      │
      └── Elapsed-Time Features
          ├── Days Passed
          ├── Months Passed
          └── Minutes Passed
```

> **A single raw feature can contain multiple layers of useful information.**

---

# 🌍 Real-World Applications

### 🛒 E-Commerce
Month, day of the week, weekend, and days since last purchase.

### 🍔 Food Delivery
Hour, day of the week, and weekend.

### 🏦 Banking and Finance
Day of the month, month, quarter, and time since the previous transaction.

### 🔧 Predictive Maintenance
Days since inspection, hours since previous failure, and time since maintenance.

### 💬 Messaging and Social Platforms
Hour, minute, day of the week, and time since the previous message.

---

# ⚠️ Best Practices

1. **Always convert to datetime first** using `pd.to_datetime()`.
2. **Use vectorised `.dt` operations** instead of manually iterating through rows.
3. **Don't extract every possible feature**—create features relevant to the problem.
4. **Use domain knowledge** to decide which temporal patterns matter.
5. **Focus on meaning, not just more columns.**

```text
Raw Data
    ↓
Understand the Problem
    ↓
Identify Hidden Information
    ↓
Extract Relevant Features
    ↓
Build Better Models
```

---

# 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Datetime
- Jupyter Notebook

---

# 📂 Project Structure

```text
Handling Date and Time Variables/
│
├── handling_date_and_time_variables.ipynb
└── README.md
```

---

# 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/PRIYANSHUSETHI/ML-FOUNDATIONS-REBUILDING.git
```

Open the notebook using Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

Install the required libraries:

```bash
pip install pandas numpy
```

---

# 📝 Medium Article

This notebook is part of my **Relearning Machine Learning in Public** series.

### Working with Date & Time Features in Machine Learning: Turning Timestamps into Valuable Features

🔗 [Read the full article on Medium](https://priyanshu20032002.medium.com/working-with-date-time-features-in-machine-learning-turning-timestamps-into-valuable-features-e652a5267b5e)

---

# 💡 Key Takeaways

- Raw dates and timestamps contain more information than they initially appear to.
- Convert date columns using `pd.to_datetime()` before feature extraction.
- The Pandas `.dt` accessor makes temporal feature extraction simple.
- Useful date features include year, month, day, weekday, week number, quarter, semester, and weekend indicators.
- Useful time features include hour, minute, and second.
- Elapsed time can often be more informative than the original timestamp.
- `Timedelta` objects can be converted into useful numerical features.
- `np.timedelta64` unit codes are case-sensitive: `'M'` represents months, while `'m'` represents minutes.
- The best temporal features depend on the problem, domain, and behaviour being modelled.

---

# ⭐ The Bigger Lesson

A timestamp is not just a timestamp.

It can contain information about:

```text
Seasonality
      +
Human Behaviour
      +
Business Cycles
      +
Recency
      +
Patterns Over Time
```

But a machine learning model cannot automatically understand these relationships.

We need to expose them.

> **Feature engineering is not always about creating new information. Sometimes, it's about uncovering the information that was already there.**

---

### 🌱 Part of the "Relearning Machine Learning in Public" Series

This repository documents my journey of revisiting and strengthening the foundations of Machine Learning—one concept, notebook, and project at a time.

If you find this repository useful, feel free to ⭐ **star the repository** and follow along with the journey.
