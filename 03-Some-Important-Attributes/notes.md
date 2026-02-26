# NumPy ndarray properties and attributes

# 1️⃣ `ndim` → Number of Dimensions

## What it tells you

How many axes the array has.

* Scalar → 0D
* Vector → 1D
* Matrix → 2D
* Image batch → 4D

## Example

```python
import numpy as np

a = np.array([10, 20, 30])
b = np.array([[1, 2], [3, 4]])

print(a.ndim)  # 1
print(b.ndim)  # 2
```

## Why it matters

In ML:

* Features matrix → 2D `(samples, features)`
* Image → 3D `(height, width, channels)`
* Batch of images → 4D `(batch, h, w, c)`

If you don’t track `ndim`, you will mess up model inputs.

---

# 2️⃣ `shape` → Structure of Data

## What it tells you

Size of array along each axis.

```python
b = np.array([[1, 2], [3, 4]])
print(b.shape)
```

Output:

```
(2, 2)
```

Means:

* 2 rows
* 2 columns

---

## Real Example (Dataset Style)

Suppose:

```python
data = np.random.rand(100, 5)
```

`shape = (100, 5)`

Meaning:

* 100 samples
* 5 features

In ML, this is standard training data format.

If shape is wrong, model crashes.

---

# 3️⃣ `size` → Total Number of Elements

```python
print(b.size)
```

For `(2,2)` → size = 4

Formula:

```
size = product of shape values
```

Example:

```
(100, 5) → 500 elements
```

---

## Why Important?

Memory calculation:

```
Total Memory = size × dtype size
```

Large arrays = large RAM usage.

---

# 4️⃣ `dtype` → Data Type

## What it tells you

Type of elements inside array.

```python
a = np.array([1, 2, 3])
print(a.dtype)
```

Output:

```
int64
```

---

## Why This Is Critical

* `float32` → Deep Learning (saves memory)
* `float64` → Scientific computing
* `int` → indexing
* `bool` → masks

If dtype is wrong:

* Computation slower
* Memory wasted
* Overflow errors possible

Example:

```python
np.array([1,2,3], dtype=np.float32)
```

---

# 5️⃣ `itemsize` → Bytes per Element

```python
a = np.array([1,2,3], dtype=np.int32)
print(a.itemsize)
```

Output:

```
4
```

Means:

* Each element uses 4 bytes

---

## Why Important?

If:

```
size = 1,000,000
itemsize = 8
```

Memory = 8MB

Scaling matters.

---

# 6️⃣ `nbytes` → Total Memory Used

```python
print(a.nbytes)
```

Formula:

```
nbytes = size × itemsize
```

If you work with large image datasets, this matters a lot.

---

# 7️⃣ `T` → Transpose

Flips axes.

```python
m = np.array([[1,2,3],[4,5,6]])
print(m.shape)
print(m.T.shape)
```

Before:

```
(2, 3)
```

After transpose:

```
(3, 2)
```

---

## Real ML Use

If:

```
X shape = (samples, features)
```

Sometimes math requires:

```
(features, samples)
```

Transpose fixes alignment for matrix multiplication.

---

# 8️⃣ `reshape()` → Change Structure Without Changing Data

```python
a = np.arange(12)
b = a.reshape(3,4)
```

Now:

```
(3,4)
```

Important rule:

```
New shape product must equal old size
```

---

## Real Example

Image of 28×28 pixels:

```
(28,28)
```

Flatten for ML:

```
(784,)
```

Using:

```python
img.reshape(-1)
```

---

# 9️⃣ `flatten()` vs `ravel()`

Both convert to 1D.

Difference:

* `flatten()` → copy
* `ravel()` → view (memory shared)

Memory understanding starts here.

---

# 🔟 `strides` (Advanced but Important)

Tells how many bytes to jump in memory to move along each axis.

Example:

```python
a = np.array([[1,2],[3,4]])
print(a.strides)
```

This explains how NumPy walks through memory.

Most beginners ignore this. That’s why they don’t understand performance.

---

# Real-World Example: Image Dataset

Suppose:

```
batch = np.random.rand(32, 224, 224, 3)
```

Properties:

* `ndim` → 4
* `shape` → (32,224,224,3)
* `size` → 32×224×224×3
* `dtype` → float64 (by default)
* `nbytes` → huge memory
* `T` → meaningless unless axes specified

This is how deep learning frameworks structure data.

---

# 🚨 Common Beginner Mistakes

1. Ignoring shape before operations
2. Mixing int and float dtype
3. Using reshape incorrectly
4. Not checking memory size
5. Confusing 1D and 2D arrays

Example mistake:

```python
np.array([1,2,3]).shape
```

Output:

```
(3,)
```

NOT `(1,3)`

This breaks matrix multiplication later.

---

# 🧠 Production-Level Thinking

Before writing NumPy code, always ask:

* What is input shape?
* What is output shape?
* What dtype is required?
* How much memory will this consume?
* Is this a view or copy?

If you don’t think like this, you are still thinking like a Python beginner, not a numerical engineer.

---