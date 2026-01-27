
## Practice(Answer)

### Easy

1. Why was NumPy created?

> NumPy was created to enable **fast and efficient numerical computation in Python**.
Python lists are slow and memory-inefficient for math, so NumPy provides array-based computation using optimized C code and contiguous memory.


2. What does ndarray stand for?
> `ndarray` stands for **N-dimensional array**.

3. Why are Python lists slow for math?
> Python lists are slow for math because they store numbers as separate Python objects, use scattered memory, and require Python-level loops for operations, which adds significant overhead.


4. Output:

   ```python
   np.array([1, 2, 3]) + 1
   ```
> **Output:**
`array([2, 3, 4])`
**Reason:**
NumPy performs element-wise addition using vectorized computation, so `1` is added to each element of the array.


5. Why does list multiplication behave differently?
> List multiplication behaves differently because lists are containers, not numerical structures.
The `*` operator repeats the list instead of performing element-wise arithmetic, unlike NumPy arrays which support vectorized math.


6. Why does NumPy enforce a single dtype?
> NumPy enforces a single dtype to allow contiguous memory storage and fast vectorized operations.
A fixed dtype removes per-element type checking, which improves performance and memory efficiency.


7. Is NumPy optional for data science?
> No, NumPy is not optional for data science.
It is the numerical foundation of Pandas, machine learning libraries, and efficient data processing, and is required to understand, debug, and scale real data science workflows.


### Medium

8. Explain contiguous memory in your own words
> Contiguous memory means data elements are stored next to each other in one continuous block of memory.
This allows faster access by the CPU and enables efficient vectorized operations in NumPy.


9. Why is vectorization faster than loops?
> Vectorization is faster than loops because operations run in optimized C code on entire arrays at once, avoiding Python’s per-iteration overhead and repeated type checking.


10. What happens internally in `arr + 5`?
>Internally, NumPy applies the addition in optimized C code, adding `5` to every element of the array in a single vectorized operation without using a Python loop.


11. How can dtype issues affect ML models?
> Dtype issues can cause loss of precision, unexpected type casting, and increased memory usage, which may lead to incorrect model calculations, slower training, or model failures during preprocessing and training.


12. Why does Pandas depend on NumPy?
> Pandas depends on NumPy because NumPy provides fast, memory-efficient array storage and numerical operations.
Pandas Series and DataFrames are built on top of NumPy arrays, so all core computations rely on NumPy.


13. What is shape and why does it matter?
> Shape describes the dimensions of an array, such as the number of rows and columns.
It matters because NumPy operations, broadcasting, and ML models require compatible shapes, and incorrect shapes lead to errors or wrong results.


14. Give one example where ignoring shape breaks code
> If you try to add two arrays with incompatible shapes, the code breaks.

Example:

```python
a = np.array([[1, 2, 3]])   # shape (1, 3)
b = np.array([[1], [2]])   # shape (2, 1)

a + b
```

> This fails because the shapes are not compatible for broadcasting, showing how ignoring shape can break code.


### Hard

15. Why is NumPy written in C internally?
> NumPy is written in C internally to achieve high performance.
C allows low-level memory control and fast numerical operations, which makes NumPy much faster than pure Python for numerical computation.


16. How does CPU cache help NumPy speed?
> CPU cache helps NumPy because data stored in contiguous memory can be loaded into cache efficiently.
This reduces slow memory access and allows the CPU to process many values quickly in sequence.

17. Why are shape bugs common in ML pipelines?
> Shape bugs are common in ML pipelines because data passes through many transformations and models expect specific input dimensions.
A small shape mismatch can silently propagate or cause errors, making shape management critical.

18. Explain one real-world NumPy use case
> One real-world NumPy use case is feature normalization in machine learning.
Numerical features are converted into NumPy arrays and scaled using vectorized operations before being fed into ML models.


19. Why is `import numpy as np` a standard?
> `import numpy as np` is a standard because it is short, readable, and used consistently across the Python and data science ecosystem.
Using the same alias improves code clarity, collaboration, and compatibility with documentation and examples.


20. What happens if you treat NumPy like a list?
> If you treat NumPy like a list, you end up using loops, ignoring shape and dtype, and losing performance.
This leads to slower code, hard-to-debug errors, and incorrect behavior in data science and ML workflows.


