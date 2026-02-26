# 1. Basic Introduction to NumPy

## What NumPy Actually Is

**NumPy = Numerical Python**

It is a library built for:

* Fast numerical computation
* Efficient array operations
* Vectorized math
* Memory-efficient data storage

Core object:
👉 `ndarray` (N-dimensional array)

Not list.
Not tuple.
Not DataFrame.

Everything revolves around arrays.

---

## Why NumPy Exists

Python lists are:

* Flexible
* Easy
* But slow for math
* Not memory efficient

NumPy arrays are:

* Fixed type
* Stored in contiguous memory
* Fast because operations are implemented in C
* Support vectorization

If you're still thinking in loops, you're not thinking in NumPy.

---

## Example: Python List vs NumPy

Python list addition:

```python
a = [1,2,3]
b = [4,5,6]
a + b
```

Output:

```
[1,2,3,4,5,6]
```

Concatenation, not math.

NumPy:

```python
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b
```

Output:

```
[5,7,9]
```

Element-wise operation.

That is the power.

---

# 2. Applications of NumPy

NumPy is not just for arrays. It is the base of the entire data ecosystem.

### 🔹 1. Data Science

* Data cleaning
* Statistical calculations
* Feature engineering

Used under:

* Pandas
* Scikit-learn

---

### 🔹 2. Machine Learning

All ML models ultimately work on numerical arrays.

Used under:

* TensorFlow
* PyTorch

Even deep learning tensors are basically advanced ND arrays.

---

### 🔹 3. Scientific Computing

* Matrix multiplication
* Linear algebra
* Eigenvalues
* Simulations

Used in:

* Physics simulations
* Finance modeling
* Optimization problems

---

### 🔹 4. Image Processing

Images are just 3D arrays.

Example:

* Height × Width × Channels

Let’s visualize:

![Image](https://ars.els-cdn.com/content/image/1-s2.0-S0167404821000717-gr3.jpg)

![Image](https://www.researchgate.net/publication/364516028/figure/fig1/AS%3A11431281091193667%401666332191163/Representation-of-a-grayscale-image-left-as-a-2D-grid-object-middle-The-grid-is-a.png)

![Image](https://www.researchgate.net/publication/267210444/figure/fig6/AS%3A295732335661069%401447519491773/A-three-dimensional-RGB-matrix-Each-layer-of-the-matrix-is-a-two-dimensional-matrix-of.png)

![Image](https://miro.medium.com/1%2A8k6Yk6MhED2SxF2zLctG7g.png)

Each pixel = 3 values (R,G,B)

So a 224x224 RGB image:

```
(224, 224, 3)
```

That’s a 3D array.

---

## Axis

Axis = direction of operation.

This destroys beginners.

For a 2D array:

```
axis=0 → column-wise
axis=1 → row-wise
```

Example:

```
[[1,2,3],
 [4,5,6]]
```

Sum:

* axis=0 → [5,7,9]
* axis=1 → [6,15]

If you mix this up, your ML pipeline will silently break.

---

# Real World Dataset Example

Let’s say student marks:

| Student | Math | Science | English |
| ------- | ---- | ------- | ------- |
| A       | 90   | 85      | 88      |
| B       | 70   | 75      | 72      |

As NumPy:

```
(2, 3)
```

Now:

* Mean per student → axis=1
* Mean per subject → axis=0

This is exactly how ML feature matrices work:

```
(n_samples, n_features)
```

If you don’t deeply understand this shape structure, ML will confuse you later.

---

# Hard Truth

Most beginners think they “know NumPy” because they can:

* Create arrays
* Do slicing
* Use reshape

That’s surface-level knowledge.

Real understanding means:

* You can mentally track shape changes
* You understand broadcasting rules
* You know memory layout implications
* You avoid loops automatically
* You question dtype choices

You’re building your applied-numpy repo. Good.

But don’t just write notes.
Make mistakes.
Break shapes.
Test edge cases.

---