# 3. Array Properties (VERY IMPORTANT)

Array properties tell you **what your data really looks like**, not what you *think* it looks like.
They are used **everywhere** for:

* Debugging
* Model input validation
* Shape mismatch detection
* Performance checks

---

## 1. `shape`

### What it tells you

The **structure** of the array.

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])
x.shape
```

Output:

```
(2, 3)
```

Meaning:

* 2 rows (samples)
* 3 columns (features)

### Why it matters

ML models expect:

```
(n_samples, n_features)
```

Wrong shape = wrong learning or runtime error.

### Common mistake

Thinking `(n,)` and `(n,1)` are the same.
They are **not**.

---

## 2. `ndim`

### What it tells you

Number of **dimensions (axes)**.

```python
x = np.array([1, 2, 3])
x.ndim
```

Output:

```
1
```

```python
y = np.array([[1, 2, 3]])
y.ndim
```

Output:

```
2
```

### Why it matters

* `ndim = 1` → vector
* `ndim = 2` → matrix (ML-friendly)
* `ndim > 2` → images, tensors, DL data

Many ML APIs **reject 1D input**.

---

## 3. `size`

### What it tells you

Total number of elements in the array.

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])
x.size
```

Output:

```
6
```

### Rule

```
size = rows × columns
```

### Why it matters

Used for:

* Sanity checks
* Memory estimation
* Verifying reshapes

If `size` changes unexpectedly, you broke the data.

---

## 4. `dtype`

### What it tells you

The **data type** of elements stored.

```python
x = np.array([1, 2, 3])
x.dtype
```

Output:

```
int64
```

```python
y = np.array([1, 2, 3.5])
y.dtype
```

Output:

```
float64
```

### Why it matters

* ML expects numeric dtypes
* Strings break models
* dtype affects **memory + speed**

### Silent danger

```python
np.array([1, "2", 3]).dtype
```

Becomes string. Model breaks later.

---

## How these are used together (real ML check)

```python
X.shape   # correct input size?
X.ndim    # is it 2D?
X.size    # data intact?
X.dtype   # numeric?
```

This check happens **before training**, always.

---

## Interview traps (pay attention)

* Only checking `shape`, ignoring `dtype`
* Passing `(n,)` instead of `(n,1)`
* Forgetting that `ndim` must be 2 for ML
* Reshaping without checking `size`

---

## One-line interview summary (memorize)

> `shape`, `ndim`, `size`, and `dtype` are core NumPy properties used to validate data structure, dimensionality, and type before ML operations and debugging.

---

You’re right. That one’s on me.
You **clearly** set the rule, and I broke it. No excuse.

Let’s fix it **properly** right now.

Below are **EXACTLY 20 questions** for **Topic 3: Array Properties**
Focused on **shape, ndim, size, dtype**
Clean. Interview-relevant. No filler.

---

# Topic 3: Array Properties — Practice Questions

## Easy (1–7)

Concept clarity, basics, no tricks.

1. What does `x.shape` tell you about a NumPy array?
2. What is the output of `ndim` for a 1D array?
3. What is the `size` of an array with shape `(4, 3)`?
4. What does `dtype` represent in a NumPy array?
5. If `x = np.array([1, 2, 3])`, what is `x.shape`?
6. Can two arrays have the same `size` but different `shape`?
7. Which property helps you confirm whether data is numeric or not?

---

## Medium (8–14)

Shape reasoning, debugging logic, ML awareness.

8. What is the difference between `shape = (5,)` and `shape = (5,1)` in ML input?
9. Why do most ML models reject arrays with `ndim = 1`?
10. If `x.shape = (2, 3)` and `x.size = 6`, what happens if size is not preserved during reshape?
11. What will be the `dtype` of `np.array([1, 2, 3.0])` and why?
12. How can checking `ndim` prevent silent broadcasting bugs?
13. Why is checking only `shape` insufficient when debugging model input?
14. What problem occurs if `dtype` becomes `object` or `str` in ML pipelines?

---

## Hard / Interview / Industry Level (15–20)

Real-world ML, performance, silent failures.

15. A model trains without error but accuracy is zero. Which array properties would you inspect first and why?
16. How can a wrong `dtype` increase memory usage and slow down training?
17. Explain a real scenario where `size` is correct but `shape` is wrong and the model still runs.
18. Why is `(n,)` more dangerous than `(n,1)` in production ML code?
19. In an ML pipeline, where exactly would you enforce checks on `shape`, `ndim`, and `dtype`?
20. Why are array property checks considered **defensive programming** in data science?

---

## Industry Reality Check (listen carefully)

In real ML code, **nobody trusts the data**.

Before training or inference, engineers **always check**:

```python
X.shape
X.ndim
X.size
X.dtype
```

Skipping this is how silent bugs reach production.

---