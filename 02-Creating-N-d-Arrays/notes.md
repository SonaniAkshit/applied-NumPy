# Creating-N-d-Arrays

# 1️⃣ `np.array()`

### 🔹 Purpose

Convert Python list/tuple into a NumPy array.

### 🔹 Why it exists

Because Python lists are slow and don’t support vectorized math properly.

### 🔹 Syntax

```python
np.array(object, dtype=None)
```

### 🔹 Example

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
```

Output:

```
array([1, 2, 3, 4])
```

2D example:

```python
arr2 = np.array([[1,2,3],
                 [4,5,6]])
```

Shape → (2, 3)

### 🔹 When to use

* Converting raw data
* Small manual arrays
* Testing

### 🔹 When NOT to use

* When you need specific initialization (zeros, random, etc.)
* When working with large uninitialized arrays

---

# 2️⃣ `np.zeros()`

### 🔹 Purpose

Create array filled with zeros.

### 🔹 Syntax

```python
np.zeros(shape, dtype=float)
```

### 🔹 Example

```python
np.zeros((3, 4))
```

Output shape → (3, 4)

### 🔹 Why useful?

* Initialize weights
* Preallocate memory
* Avoid dynamic resizing

### 🔹 Use case

Machine learning weight initialization (sometimes)

---

# 3️⃣ `np.ones()`

### 🔹 Purpose

Create array filled with ones.

### 🔹 Syntax

```python
np.ones(shape, dtype=float)
```

### 🔹 Example

```python
np.ones((2, 3))
```

Output:

```
[[1. 1. 1.]
 [1. 1. 1.]]
```

### 🔹 Use case

* Masks
* Bias initialization
* Mathematical scaling

---

# 4️⃣ `np.empty()`

### 🔹 Purpose

Create array without initializing values.

### 🔹 Syntax

```python
np.empty(shape)
```

### 🔹 Example

```python
np.empty((2, 2))
```

⚠️ Values will be random garbage memory.

### 🔹 Why use?

* Faster than zeros
* When you immediately overwrite values

### 🔹 Beginner mistake

Thinking it initializes with zeros. It does NOT.

---

# 5️⃣ `np.arange()`

### 🔹 Purpose

Range with step size.

### 🔹 Syntax

```python
np.arange(start, stop, step)
```

### 🔹 Example

```python
np.arange(0, 10, 2)
```

Output:

```
[0 2 4 6 8]
```

### 🔹 Difference from Python `range()`?

Returns NumPy array.
Works with float steps.

Example:

```python
np.arange(0, 1, 0.2)
```

⚠️ Floating precision errors possible.

---

# 6️⃣ `np.linspace()`

### 🔹 Purpose

Create evenly spaced numbers between two limits.

### 🔹 Syntax

```python
np.linspace(start, stop, num)
```

### 🔹 Example

```python
np.linspace(0, 1, 5)
```

Output:

```
[0.   0.25 0.5  0.75 1.  ]
```

### 🔹 Why better than arange?

You control number of values, not step size.

### 🔹 Used in:

* Plotting
* Signal processing
* ML feature scaling

---

# 7️⃣ `np.eye()`

### 🔹 Purpose

Identity matrix.

### 🔹 Syntax

```python
np.eye(n)
```

### 🔹 Example

```python
np.eye(3)
```

Output:

```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

### 🔹 Use case

* Linear algebra
* Matrix multiplication
* Neural network math

---

# 8️⃣ `np.random` Module

### `np.random.rand()`

Uniform random [0,1)

```python
np.random.rand(2, 3)
```

### `np.random.randn()`

Normal distribution

```python
np.random.randn(2, 2)
```

### `np.random.randint()`

Random integers

```python
np.random.randint(1, 10, size=(3, 3))
```

### 🔹 Used in

* ML weight initialization
* Simulations
* Testing

---

# 9️⃣ `np.full()`

### 🔹 Purpose

Fill array with specific value.

### 🔹 Syntax

```python
np.full(shape, fill_value)
```

### 🔹 Example

```python
np.full((2,3), 7)
```

Output:

```
[[7 7 7]
 [7 7 7]]
```

---

# 1️⃣0️⃣ `np.reshape()` (Important for Creation Thinking)

### 🔹 Purpose

Change shape without changing data.

```python
arr = np.arange(6)
arr.reshape(2,3)
```

Original shape → (6,)
New shape → (2,3)

Total elements must match.

---

# 🚀 Real World Example

Imagine student marks dataset:

```python
marks = np.array([
    [80, 85, 90],
    [70, 75, 78],
    [88, 92, 95]
])
```

Shape → (3 students, 3 subjects)

Now:

* `np.zeros((1000, 3))` → preparing space for 1000 students
* `np.random.randint(0, 101, size=(1000, 3))` → simulate marks
* `np.linspace(0, 100, 11)` → grading scale

This is practical usage.

---

# 🔥 Critical Engineering Thinking

Before creating any array ask:

1. What shape do I need?
2. What dtype do I need?
3. Will this scale to 10M rows?
4. Do I need initialized values?
5. Is memory contiguous?

If you skip these, you are coding blindly.

---

# ⚠️ Common Beginner Mistakes

* Forgetting tuple for shape → `np.zeros(2,3)` ❌
* Mixing list and array operations
* Using loops instead of vectorization
* Ignoring dtype (int vs float)
* Floating precision errors with `arange()`

---

# ⚡ Final Advice

If you don’t visualize shape in your head before writing code,
you are not thinking in NumPy.