# Loading CSV Numeric Data in NumPy

---

## 1. Topic Overview

CSV = Comma Separated Values file.

It stores tabular data like:

```
age,height,weight
25,170,70
30,165,65
```

NumPy loads CSV into an **ndarray**.

Why this exists:

* ML models need numeric matrix input
* CSV is common data format

**Analogy (one only):**
CSV is like a text spreadsheet.
NumPy converts it into a numeric matrix.

---

## 2. Core Theory

NumPy reads CSV using:

```
np.loadtxt()
np.genfromtxt()
```

Both convert text → numeric array.

Internally:

* File is read line by line
* Split by delimiter (comma)
* Converted to dtype
* Stored in continuous memory

If rows have unequal length → error.

---

## 3. Main Methods

---

## Method 1: `np.loadtxt()` (Strict)

Use when:

* File is clean
* No missing values
* All numeric

---

### Example CSV file (`data.csv`)

```
1,2,3
4,5,6
7,8,9
```

---

### Code

```python
import numpy as np

data = np.loadtxt("data.csv", delimiter=",")

print(data)
print("Shape:", data.shape)
print("ndim:", data.ndim)
print("dtype:", data.dtype)
```

### Output

```
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]

Shape: (3, 3)
ndim: 2
dtype: float64
```

Explanation:

* 3 rows
* 3 columns
* Default dtype is float64
* Always 2D if multiple rows

---

## Method 2: `np.genfromtxt()` (Flexible)

Use when:

* Missing values exist
* Mixed formatting
* Need more control

---

### Example with missing value

```
1,2,3
4,,6
7,8,9
```

---

### Code

```python
data = np.genfromtxt("data.csv", delimiter=",", filling_values=0)

print(data)
print(data.shape)
```

Output:

```
[[1. 2. 3.]
 [4. 0. 6.]
 [7. 8. 9.]]
(3, 3)
```

Explanation:

* Missing value replaced with 0
* Shape remains consistent

---

## 4. Important Parameters

### delimiter

```python
delimiter=","
```

Separates columns.

---

### skip_header

```python
np.loadtxt("data.csv", delimiter=",", skiprows=1)
```

Used when first row contains column names.

---

### dtype

```python
np.loadtxt("data.csv", delimiter=",", dtype=np.float32)
```

Controls memory usage.

---

## 5. Validate Shape (Very Important)

After loading:

```python
print(data.shape)
```

Expected ML input format:

```
(n_samples, n_features)
```

Example:

If CSV contains:

```
100 rows
5 columns
```

Then:

```
Shape: (100, 5)
```

If shape is `(100,)`, something is wrong.

---

## 6. Common Errors

### 1. Header present but not skipped

Error:

```
could not convert string to float
```

Fix:

```
skiprows=1
```

---

### 2. Single column CSV

File:

```
1
2
3
4
```

Load:

```python
data = np.loadtxt("data.csv")
print(data.shape)
```

Output:

```
(4,)
```

This is 1D.

If ML expects 2D:

```python
data = data.reshape(-1, 1)
```

---

### 3. Unequal columns

File:

```
1,2
3,4,5
```

This breaks `loadtxt`.

Must fix CSV first.

---

## 7. Performance & Best Practice

* Use `dtype=np.float32` for ML
* Always validate shape
* Always print dtype
* Avoid object dtype
* Use `genfromtxt` only if needed (slower)

---

## 8. Real Data Science Example

Assume CSV:

```
age,height,weight
25,170,70
30,165,65
```

Load properly:

```python
data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

print("Shape:", data.shape)
print("First sample:", data[0])
```

Output:

```
Shape: (2, 3)
First sample: [ 25. 170.  70.]
```

Meaning:

* 2 samples
* 3 features

Correct for ML.

---

# Final Rule for ML

After loading CSV, always run:

```python
print(data.shape)
print(data.ndim)
print(data.dtype)
```

If shape is not `(n_samples, n_features)`, fix it before training.

---