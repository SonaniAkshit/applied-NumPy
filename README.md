# Applied NumPy

This repository documents **my hands-on learning progress with NumPy**.

I built this repo while learning NumPy from the ground up, focusing on **understanding concepts by implementing them**, not just reading theory.
Each topic is broken into its own folder with **notes I wrote for myself** and **practical code experiments**.

This is not a polished course or library. It’s a **learning-first repository**.

---

## Why I Created This Repo

* To understand NumPy deeply instead of memorizing syntax
* To move from Python lists to array-based thinking
* To build a solid foundation for data science and machine learning
* To keep a clean, revisitable record of what I actually learned

If something exists here, it’s because I needed it to understand NumPy better.

---

## Repository Structure

Each folder represents **one concept I studied**, in the order I learned it.

```id="s3t1h0"
applied-numpy/
│
├── 01-Introduction-to-Numpy
├── 02-Creating-N-d-Arrays
├── 03-Some-Important-Attributes
├── 04-Python-Lists-Vs-Numpy-Arrays
├── 05-Indexing-Slicing-and-Iteration
├── 06-Numpy-Operations
├── 07-Reshaping-Numpy-arrays
├── 08-Fancy-indexing-in-Numpy
├── 09-Indexing-with-Boolean-Arrays
├── 10-Plotting-graphs-using-Numpy
├── 11-Broadcasting
└── 12-Some-important-numpy-functions
```

Every folder contains:

* **Notes** written in my own words
* **Code implementations** I ran and tested myself

---

## What I Covered (In Learning Order)

### 1. Introduction to NumPy

Started by understanding what NumPy actually solves and why it’s used instead of Python lists for numerical work.

### 2. Creating N-D Arrays

Learned how NumPy represents data using 1D, 2D, and higher-dimensional arrays and different ways to create them.

### 3. Important Array Attributes

Explored how arrays store shape, size, dimensions, and data types and why these matter.

### 4. Python Lists vs NumPy Arrays

Compared behavior, performance, and limitations. This was where array-based thinking started to click.

### 5. Indexing, Slicing, and Iteration

Practiced accessing and modifying data properly without relying on slow Python loops.

### 6. NumPy Operations

Worked through arithmetic, mathematical, and statistical operations directly on arrays.

### 7. Reshaping Arrays

Learned how to change array shapes and layouts without losing data.

### 8. Fancy Indexing

Experimented with selecting data using index arrays instead of simple slices.

### 9. Boolean Indexing

Used conditions to filter data, similar to how real datasets are handled.

### 10. Plotting Using NumPy

Generated numerical data using NumPy and used it for plotting and analysis.

### 11. Broadcasting

Understood how NumPy handles operations between arrays of different shapes and why it’s powerful.

### 12. Important NumPy Functions

Collected frequently used functions that kept showing up during practice and experiments.

---

## How I Use This Repo

* I revisit folders when concepts feel weak
* I modify the code to test edge cases
* I use this as a reference before moving to pandas, ML, and deep learning

This repo grows only when my understanding grows.

---

## Requirements

* Python 3.x
* NumPy

```bash id="w_1fnn"
pip install numpy
```

---

## Final Note

This repository represents **learning in progress**, not finished expertise.

The focus is on **thinking correctly with arrays**, because without that, everything built on top of NumPy breaks.
