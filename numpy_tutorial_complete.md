# 🎓 NumPy Complete Tutorial: From Absolute Basics to Advanced Mastery

**Welcome, Future NumPy Master!** 🚀

This tutorial will take you from zero to hero in NumPy. Think of me as your friendly mentor sitting next to you, explaining everything step-by-step with real-life analogies, visual mental models, deep architectural insights, and practical, runnable code.

Whether you are writing your first line of Python or building high-performance deep learning models, this guide is designed to give you both intuitive understanding and complete mastery over NumPy's internals.

---

## 📚 Table of Contents

1. [What is NumPy and Why It Exists](#1-what-is-numpy-and-why-it-exists)
2. [Installing, Importing, and Environment Setup](#2-installing-importing-and-environment-setup)
3. [NumPy Arrays vs Python Lists (Architecture & Memory Layout)](#3-numpy-arrays-vs-python-lists-architecture--memory-layout)
4. [Creating Arrays (From Basics to Advanced Grids)](#4-creating-arrays-from-basics-to-advanced-grids)
5. [Array Properties, Data Types, and Memory Anatomy](#5-array-properties-data-types-and-memory-anatomy)
6. [Indexing, Slicing, and Advanced Fancy Indexing](#6-indexing-slicing-and-advanced-fancy-indexing)
7. [Array Operations & Universal Functions (ufuncs)](#7-array-operations--universal-functions-ufuncs)
8. [Broadcasting Mechanics Deep Dive](#8-broadcasting-mechanics-deep-dive)
9. [Mathematical, Statistical, and Aggregation Functions](#9-mathematical-statistical-and-aggregation-functions)
10. [Reshaping, Transposing, and Dimension Manipulation](#10-reshaping-transposing-and-dimension-manipulation)
11. [Joining, Stacking, and Splitting Arrays](#11-joining-stacking-and-splitting-arrays)
12. [Copy vs View and Memory Layout (Strides & Cache Locality)](#12-copy-vs-view-and-memory-layout-strides--cache-locality)
13. [Boolean Indexing, Masking, and Conditional Selection](#13-boolean-indexing-masking-and-conditional-selection)
14. [Sorting, Searching, and Counting](#14-sorting-searching-and-counting)
15. [Random Number Generation (Modern Generator API vs Legacy)](#15-random-number-generation-modern-generator-api-vs-legacy)
16. [Linear Algebra and Tensor Operations (np.linalg & np.einsum)](#16-linear-algebra-and-tensor-operations-nplinalg--npeinsum)
17. [Structured Arrays and Record Arrays](#17-structured-arrays-and-record-arrays)
18. [File I/O and Out-of-Core Big Data (np.memmap)](#18-file-io-and-out-of-core-big-data-npmemmap)
19. [Performance Optimization, Vectorization, and Cache Friendly Code](#19-performance-optimization-vectorization-and-cache-friendly-code)
20. [NumPy with Real-World End-to-End Case Studies](#20-numpy-with-real-world-end-to-end-case-studies)
21. [Common Mistakes and Pitfalls (Beginner to Advanced)](#21-common-mistakes-and-pitfalls-beginner-to-advanced)
22. [When to Use NumPy and When Not To (Ecosystem Decision Guide)](#22-when-to-use-numpy-and-when-not-to-ecosystem-decision-guide)
23. [Quick Reference Cheat Sheet & Final Summary](#23-quick-reference-cheat-sheet--final-summary)

---

## 1. What is NumPy and Why It Exists

### 🤔 WHY

**Real-Life Analogy:**
Imagine you are managing a warehouse with 1,000,000 numbered packages:
- **Standard Python List:** Like writing down each package number on an individual post-it note and sticking them randomly across 100 different rooms. To find, inspect, or add 5 to each number, a worker must run to room 1, inspect the post-it, check what kind of object it is, perform the math, run to room 2, and repeat 1,000,000 times. Painfully slow! 🐢
- **NumPy Array:** Like stacking all 1,000,000 packages in one single, uninterrupted conveyor belt, identically sized and packed shoulder-to-shoulder. An automated machine scans the entire belt at hardware speed in a single pass. Super fast! ⚡

**What Problem NumPy Solves:**
Python is a dynamically typed, interpreted language. Every number in Python is not just a raw number in memory; it is a full **`PyObject`** struct containing:
1. Reference count (`ob_refcnt`, 8 bytes)
2. Type pointer (`ob_type`, 8 bytes)
3. Size descriptor (`ob_size`, 8 bytes)
4. Actual value (digit array, 4-8 bytes)
A simple integer in Python takes **28 bytes** of RAM instead of 4 or 8 bytes! Furthermore, iterating over a Python list requires the CPython interpreter to evaluate bytecode instructions and verify types on every single iteration.

NumPy solves this by providing:
- **Unboxed, homogeneous data:** Raw C-level numbers packed contiguously into memory.
- **Vectorized C-loops:** Mathematical loops run directly in compiled C/Fortran, bypassing interpreter overhead.
- **SIMD (Single Instruction, Multiple Data):** Modern CPUs process 4 to 8 numbers simultaneously in hardware vector registers (AVX, SSE, NEON).
- **GIL (Global Interpreter Lock) Bypass:** During heavy numerical routines, NumPy releases the Python GIL, allowing true multi-core parallel computing.

### ⏰ WHEN

**Use NumPy when:**
- Working with large numerical datasets (from hundreds to hundreds of millions of numbers)
- Performing linear algebra, matrix manipulations, or tensor operations
- Processing signals, audio, computer vision images, or video frames
- Doing data science, statistics, finance, and machine learning
- Serving as the foundational engine beneath **Pandas**, **SciPy**, **Scikit-Learn**, **PyTorch**, and **TensorFlow**

**Avoid NumPy when:**
- Storing truly heterogeneous, mixed-type records that change dynamically (use standard Python dictionaries or dataclasses)
- Performing heavy string manipulation or text parsing (use Python strings or Polars/Pandas)
- Managing small collections (< 50 elements) where the C-API overhead outweighs the calculation time

### 🔧 HOW

**Example 1: Benchmark — Python List vs NumPy Array (Vectorized Speed)**
```python
import time
import numpy as np

N = 10_000_000

# 1. Pure Python approach
py_list = list(range(N))
start = time.perf_counter()
py_result = [x * 2 + 1 for x in py_list]
py_time = time.perf_counter() - start
print(f"Python list comprehension: {py_time:.4f} seconds")

# 2. NumPy vectorized approach
np_arr = np.arange(N)
start = time.perf_counter()
np_result = np_arr * 2 + 1
np_time = time.perf_counter() - start
print(f"NumPy vectorized array:   {np_time:.4f} seconds")
print(f"Speedup: {py_time / np_time:.1f}x faster!")

# Typical Output:
# Python list comprehension: 0.6842 seconds
# NumPy vectorized array:   0.0152 seconds
# Speedup: 45.0x faster!
```

**Example 2: Memory Footprint Comparison**
```python
import sys
import numpy as np

# 1,000,000 integers
n_items = 1_000_000
py_list = list(range(n_items))
np_arr = np.arange(n_items, dtype=np.int64)

# Memory of Python list includes the list container pointers PLUS each integer object
list_pointer_size = sys.getsizeof(py_list)
integer_object_size = sys.getsizeof(py_list[0])
total_py_mem = list_pointer_size + (n_items * integer_object_size)

# Memory of NumPy array is simply its raw data buffer
total_np_mem = np_arr.nbytes

print(f"Python list memory: {total_py_mem / (1024**2):.2f} MB")
print(f"NumPy array memory: {total_np_mem / (1024**2):.2f} MB")
print(f"Memory reduction:   {total_py_mem / total_np_mem:.1f}x less memory!")

# Typical Output:
# Python list memory: 34.33 MB
# NumPy array memory: 7.63 MB
# Memory reduction:   4.5x less memory!
```

### 🏋️ Practice Question
**Q:** Why does `arr * 2` run faster in NumPy than `[x * 2 for x in my_list]` in Python, even though both execute on the exact same CPU?
**A:** In Python, the interpreter must fetch each pointer, unwrap the `PyObject` structure, verify its type, extract the value, allocate a new `PyObject` for the result, and append its pointer to a new list. In NumPy, all numbers sit side-by-side in raw contiguous memory; a single compiled C-loop iterates over the addresses directly, leveraging CPU cache lines and SIMD registers to multiply several numbers per clock cycle without interpreter overhead.

---

## 2. Installing, Importing, and Environment Setup

### 🤔 WHY

NumPy is not bundled inside the Python Standard Library to ensure Python remains lightweight and to allow NumPy to be updated independently with optimized C, C++, and Fortran compilers.

### ⏰ WHEN

Set up NumPy at the start of any data science, quantitative, machine learning, or scientific computing project.

### 🔧 HOW

#### 2.1 Installation Methods

Depending on your environment manager:

```bash
# Standard pip installation
pip install numpy

# Using uv (extremely fast modern installer)
uv pip install numpy

# Using Conda / Mamba (often pre-configured with Intel MKL or OpenBLAS)
conda install numpy
```

#### 2.2 Verifying and Inspecting Your Installation

Always import NumPy under its universal alias `np`:

```python
import numpy as np

# Verify version
print("NumPy Version:", np.__version__)

# Inspect hardware acceleration libraries (OpenBLAS, MKL, BLIS)
np.show_config()
```

#### 2.3 Key Notes for NumPy 2.0+
If you are using NumPy 2.0 or newer:
- Many legacy aliases like `np.float_`, `np.int_`, or `np.bool_` have been streamlined to standard Python types `float`, `int`, `bool`, or explicit bit-width types like `np.float64`, `np.int64`.
- The default behavior of functions like `np.array(..., copy=False)` is safer and more consistent.

### 🏋️ Practice Question
**Q:** Why is the convention `import numpy as np` universally adopted across the global Python data science community instead of `from numpy import *`?
**A:** `from numpy import *` pollutes the namespace with hundreds of functions and overwrites Python built-ins (such as `sum`, `min`, `max`, `abs`, and `round`). The `np.` prefix clearly communicates to anyone reading your code that a high-performance vector operation is taking place.

---

## 3. NumPy Arrays vs Python Lists (Architecture & Memory Layout)

### 🤔 WHY

Understanding the internal memory architecture of a NumPy array is the single most valuable mental model you can acquire. It explains why slicing is fast, why reshaping doesn't copy data, and why data types matter.

### ⏰ WHEN

Keep this mental model in mind whenever you care about execution speed, RAM consumption, and data integrity.

### 🔧 HOW

#### 3.1 Memory Layout Visualized

Let us look at what memory physically looks like when storing three numbers `[10, 20, 30]`:

```
========================================================================
Python List Memory Architecture: Fragmented Pointers
========================================================================
PyListObject:
[ Pointer 0 ] ------> PyLongObject at 0x7FFF01: [ refcount | type | 10 ]
[ Pointer 1 ] ------> PyLongObject at 0x7FFF98: [ refcount | type | 20 ]
[ Pointer 2 ] ------> PyLongObject at 0x7FFF12: [ refcount | type | 30 ]
(Notice: Objects are scattered across RAM; reading them causes CPU cache misses!)

========================================================================
NumPy ndarray Memory Architecture: Single Contiguous Block
========================================================================
PyArrayObject:
[ Header: shape=(3,), strides=(8,), dtype=int64, data_ptr=0x7FFF00 ]
                                                      |
                                                      v
Contiguous Buffer: [ 0000000a | 00000014 | 0000001e ]  (Raw 8-byte ints)
                   0x7FFF00   0x7FFF08   0x7FFF10
(Notice: Data is contiguous; 64-byte CPU cache lines load all elements at once!)
```

#### 3.2 Key Architectural Differences Table

| Feature | Python List | NumPy `ndarray` |
| :--- | :--- | :--- |
| **Data Types** | Heterogeneous (mixed types allowed) | Homogeneous (all elements share identical `dtype`) |
| **Memory Buffer** | Array of 64-bit pointers to heap objects | Continuous sequential memory block of raw bytes |
| **Memory Overhead** | High (~28-36 bytes per integer element) | Minimal (e.g. exactly 1, 2, 4, or 8 bytes per element) |
| **CPU Cache Locality** | Poor (pointer chasing causes L1/L2 misses) | Outstanding (spatial locality saturates cache lines) |
| **Math Operations** | Requires explicit loops / comprehensions | Vectorized element-wise expressions |
| **Resizing** | Dynamic and fast via amortized `.append()` | Fixed size; resizing requires memory reallocation |
| **Dimensionality** | Nested lists of lists (`[[1, 2], [3, 4]]`) | Native N-dimensional tensors with coordinate indexing |

**Example: Demonstrating Homogeneity**
```python
import numpy as np

# A Python list happily stores strings, floats, ints, and booleans together
py_list = [42, "hello", 3.14, True]
print("Python list:", py_list)

# A NumPy array will coerce all elements into the most general common type
np_arr = np.array([42, "hello", 3.14, True])
print("NumPy coerced array:", np_arr)
print("Array dtype:", np_arr.dtype)

# Output:
# Python list: [42, 'hello', 3.14, True]
# NumPy coerced array: ['42' 'hello' '3.14' 'True']
# Array dtype: <U32 (All elements converted into fixed-width Unicode strings!)
```

### 🏋️ Practice Question
**Q:** What is "CPU cache locality" and why does it make NumPy arrays significantly faster than Python lists when summing 10,000,000 numbers?
**A:** Modern CPUs do not fetch individual bytes from RAM; they fetch 64-byte blocks called **cache lines** into ultra-fast L1/L2 CPU caches. Because NumPy array elements sit next to each other, loading one element pulls the next 7 elements (for 64-bit numbers) into the cache automatically. With a Python list, consecutive pointers lead to arbitrary memory addresses scattered across the heap, forcing the CPU to repeatedly wait for slow system RAM (cache misses).

---

## 4. Creating Arrays (From Basics to Advanced Grids)

### 🤔 WHY

Arrays can represent different structures: raw measurements, image grids, identity matrices, neural network weights, or coordinate systems. NumPy provides specialized creation routines optimized for speed and memory efficiency.

### ⏰ WHEN

Choose the right creation routine for the job:
- Have existing data? Use `np.array()`.
- Initializing a buffer you will fill later? Use `np.empty()` or `np.zeros()`.
- Working with mathematical matrices? Use `np.eye()`, `np.diag()`, or `np.triu()`.
- Plotting functions or generating simulation grids? Use `np.linspace()` or `np.meshgrid()`.

### 🔧 HOW

#### 4.1 Basic Creation with `np.array()`

```python
import numpy as np

# 1D Vector
v = np.array([1, 2, 3, 4], dtype=np.float32)

# 2D Matrix (2 rows, 3 columns)
m = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.int32)

# 3D Tensor (2 matrices, each 2x3)
t = np.array([[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]]])

print(f"1D shape: {v.shape}, dtype: {v.dtype}")
print(f"2D shape: {m.shape}, dtype: {m.dtype}")
print(f"3D shape: {t.shape}, dtype: {t.dtype}")

# Output:
# 1D shape: (4,), dtype: float32
# 2D shape: (2, 3), dtype: int32
# 3D shape: (2, 2, 3), dtype: int64
```

#### 4.2 Constant Value Arrays (`zeros`, `ones`, `full`) and their `*_like` variants

```python
import numpy as np

# Filled with zeros
z = np.zeros((2, 4), dtype=int)

# Filled with ones
o = np.ones((3, 2), dtype=float)

# Filled with any arbitrary constant value
f = np.full((2, 3), fill_value=7.5)

# Matching the exact shape and dtype of an existing array
sample = np.array([[1, 2], [3, 4]], dtype=np.float32)
z_like = np.zeros_like(sample)  # Shape (2, 2), float32
f_like = np.full_like(sample, fill_value=-1.0)

print("Zeros:\n", z)
print("Full:\n", f)
print("Full-like sample:\n", f_like)

# Output:
# Zeros:
#  [[0 0 0 0]
#  [0 0 0 0]]
# Full:
#  [[7.5 7.5 7.5]
#  [7.5 7.5 7.5]]
# Full-like sample:
#  [[-1. -1.]
#  [-1. -1.]]
```

#### 4.3 Uninitialized Memory with `np.empty()` and `np.empty_like()`

> [!WARNING]
> `np.empty()` allocates raw RAM without initializing its bits to zero. Whatever garbage bytes were previously left in that memory space will appear in your array!

```python
import numpy as np

# Ultra-fast allocation: avoids the overhead of setting millions of bytes to zero
raw_buffer = np.empty((3, 3), dtype=np.float64)
print("Uninitialized memory (garbage values!):\n", raw_buffer)

# SAFE PATTERN: Always overwrite all elements before reading from an empty array
raw_buffer[:] = 42.0
print("After safe overwrite:\n", raw_buffer)
```

#### 4.4 Numerical Sequences (`arange`, `linspace`, `logspace`, `geomspace`)

```python
import numpy as np

# 1. arange(start, stop, step): Half-open interval [start, stop)
seq_step = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 2. linspace(start, stop, num): Closed interval [start, stop] with exact count
seq_linear = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# 3. logspace(start, stop, num, base): Generates base^start to base^stop
# Essential for parameter tuning (e.g. learning rates 10^-4 to 10^-1)
seq_log = np.logspace(-4, -1, 4, base=10)  # [1e-4, 1e-3, 1e-2, 1e-1]

# 4. geomspace(start, stop, num): Geometric progression directly in data values
seq_geom = np.geomspace(1, 1000, 4)  # [1.0, 10.0, 100.0, 1000.0]

print("arange:   ", seq_step)
print("linspace: ", seq_linear)
print("logspace: ", seq_log)
print("geomspace:", seq_geom)

# Output:
# arange:    [0 2 4 6 8]
# linspace:  [0.   0.25 0.5  0.75 1.  ]
# logspace:  [0.0001 0.001  0.01   0.1   ]
# geomspace: [   1.   10.  100. 1000.]
```

#### 4.5 Identity, Diagonals, and Triangular Matrices

```python
import numpy as np

# 1. Identity matrix (square with 1s on main diagonal)
I = np.eye(3)  # 3x3 identity matrix

# 2. Diagonal matrix from vector, or extract diagonal from matrix
diag_matrix = np.diag([10, 20, 30])
extracted_diag = np.diag(diag_matrix)

# Off-diagonal extraction or placement using k parameter (k=1: superdiagonal, k=-1: subdiagonal)
upper_diag = np.diag([5, 5], k=1)

# 3. Triangular matrices (Lower and Upper)
matrix = np.ones((3, 3))
lower_tri = np.tril(matrix)  # Zeros out everything above diagonal
upper_tri = np.triu(matrix)  # Zeros out everything below diagonal

print("Identity Matrix:\n", I)
print("Diagonal Matrix:\n", diag_matrix)
print("Extracted Diagonal:", extracted_diag)
print("Lower Triangular:\n", lower_tri)
print("Upper Triangular:\n", upper_tri)

# Output:
# Identity Matrix:
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
# Diagonal Matrix:
#  [[10  0  0]
#  [ 0 20  0]
#  [ 0  0 30]]
# Extracted Diagonal: [10 20 30]
# Lower Triangular:
#  [[1. 0. 0.]
#  [1. 1. 0.]
#  [1. 1. 1.]]
# Upper Triangular:
#  [[1. 1. 1.]
#  [0. 1. 1.]
#  [0. 0. 1.]]
```

#### 4.6 Coordinate Grids (`np.meshgrid`, `np.mgrid`, `np.ogrid`)

Coordinate grids are vital for computing 2D/3D functions, surfaces, distance fields, and evaluating heatmaps.

```python
import numpy as np

# Define coordinate axes
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)

# np.meshgrid produces full 2D coordinate matrices for X and Y
X, Y = np.meshgrid(x, y)

# Compute 2D Euclidean distance field from origin: Z = sqrt(X^2 + Y^2)
Z = np.sqrt(X**2 + Y**2)

print("X Coordinates:\n", np.round(X, 1))
print("Y Coordinates:\n", np.round(Y, 1))
print("Computed Distance Field Z:\n", np.round(Z, 2))

# Open mesh with np.ogrid (memory efficient broadcasting grids)
grid_y, grid_x = np.ogrid[0:3, 0:4]
print("\nOpen grid Y shape:", grid_y.shape)  # (3, 1)
print("Open grid X shape:", grid_x.shape)  # (1, 4)
# Broadcasting grid_y and grid_x together forms a (3, 4) grid without allocating full 2D arrays!
```

#### 4.7 Creating Arrays from Buffers, Iterables, and Functions

```python
import numpy as np

# 1. From Python iterators / generators without creating an intermediate list
generator = (x * x for x in range(5))
from_gen = np.fromiter(generator, dtype=int)
print("From generator:", from_gen)

# 2. From raw byte buffer (e.g. reading binary network packets or C structs)
raw_bytes = b'\x01\x00\x02\x00\x03\x00\x04\x00'
from_buf = np.frombuffer(raw_bytes, dtype=np.int16)
print("From raw bytes:", from_buf)

# 3. From mathematical index function: f(row, col)
def matrix_func(i, j):
    return 10 * i + j

from_fn = np.fromfunction(matrix_func, (3, 4), dtype=int)
print("From function:\n", from_fn)

# Output:
# From generator: [ 0  1  4  9 16]
# From raw bytes: [1 2 3 4]
# From function:
#  [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]]
```

### 🏋️ Practice Question
**Q:** Why would you use `np.empty()` over `np.zeros()` when allocating a 10,000 × 10,000 matrix that you plan to immediately populate with sensor measurements?
**A:** `np.zeros()` writes zeros to all 100,000,000 elements (800 MB of memory for float64), which forces the operating system and CPU to touch every single memory page. `np.empty()` merely asks the OS for the memory address reservation without zero-filling it. Because your code will immediately overwrite every cell with sensor measurements, zero-initialization is wasted computation.

---

## 5. Array Properties, Data Types, and Memory Anatomy

### 🤔 WHY

Knowing your array's shape, data type, and memory strides lets you prevent silent integer overflows, minimize memory usage, write cache-optimized algorithms, and debug multidimensional operations effortlessly.

### ⏰ WHEN

Always inspect properties when receiving data from an external file, library, API, or neural network layer.

### 🔧 HOW

#### 5.1 The Four Core Properties

```python
import numpy as np

matrix = np.array([[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [9.0, 10.0, 11.0, 12.0]], dtype=np.float64)

print("Shape (dimensions):          ", matrix.shape)     # (3, 4) -> 3 rows, 4 cols
print("Number of dimensions (ndim): ", matrix.ndim)      # 2
print("Total elements (size):        ", matrix.size)      # 12 (3 * 4)
print("Data type (dtype):            ", matrix.dtype)     # float64
print("Bytes per element (itemsize): ", matrix.itemsize) # 8 bytes
print("Total memory buffer (nbytes): ", matrix.nbytes)   # 96 bytes (12 * 8)
```

#### 5.2 The Complete NumPy DataType Hierarchy

NumPy supports precise bit-width types:

| Category | Type Name | Bit Width | Value Range / Description | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Unsigned Int** | `uint8` | 8 bits | `0` to `255` | Image pixels (RGB), binary masks |
| **Unsigned Int** | `uint16`, `uint32`, `uint64`| 16, 32, 64 bits | Positive integers | Audio sample IDs, hash values |
| **Signed Int** | `int8` | 8 bits | `-128` to `127` | Neural net quantization, low-range sensors |
| **Signed Int** | `int16`, `int32` | 16, 32 bits | Up to $\pm 2.1 \times 10^9$ | Audio signals, general counts |
| **Signed Int** | `int64` | 64 bits | Up to $\pm 9.2 \times 10^{18}$ | Standard integer default |
| **Floating Point**| `float16` | 16 bits | Half precision | Modern AI / Deep Learning GPU inference |
| **Floating Point**| `float32` | 32 bits | Single precision (~7 decimal digits) | Deep learning training, graphics, 3D |
| **Floating Point**| `float64` | 64 bits | Double precision (~16 decimal digits)| Scientific computing default |
| **Complex** | `complex64`, `complex128` | 64, 128 bits | Pairs of floats (real + imaginary) | Fourier transforms, signal processing |
| **Boolean** | `bool_` | 8 bits | `True` or `False` (1 byte per bool) | Filtering masks, logical conditions |
| **String** | `<U<N>` | Variable | Fixed-length Unicode string | Categorical labels |

**Working with Complex Numbers:**
```python
import numpy as np

c_arr = np.array([1 + 2j, 3 - 4j, 5 + 0j], dtype=np.complex128)
print("Complex array:    ", c_arr)
print("Real parts:       ", c_arr.real)
print("Imaginary parts:  ", c_arr.imag)
print("Complex conjugate:", np.conj(c_arr))

# Output:
# Complex array:     [1.+2.j 3.-4.j 5.+0.j]
# Real parts:        [1. 3. 5.]
# Imaginary parts:   [ 2. -4.  0.]
# Complex conjugate: [1.-2.j 3.+4.j 5.-0.j]
```

#### 5.3 Type Casting and the Silent Overflow Hazard

> [!CAUTION]
> In C and NumPy, fixed-width integers overflow silently without throwing an exception! Adding 1 to `np.int8(127)` wraps around to `-128`!

```python
import numpy as np

# Demonstration of Integer Overflow:
safe_val = np.array([126, 127], dtype=np.int8)
overflowed = safe_val + np.int8(1)
print("Original int8:   ", safe_val)
print("After adding 1:  ", overflowed)  # 127 + 1 wrapped to -128!

# Output:
# Original int8:    [126 127]
# After adding 1:   [ 127 -128]

# Safe Type Casting with .astype()
floats = np.array([1.2, 3.8, -2.7])
# Note: casting float to int truncates toward zero (does not round!)
ints = floats.astype(np.int32)
print("Truncated ints:  ", ints)  # [ 1  3 -2]

# Checking safe castability before executing:
print("Can cast int32 to float64 safely?", np.can_cast(np.int32, np.float64))  # True
print("Can cast float64 to int32 safely?", np.can_cast(np.float64, np.int32))  # False (loss of precision!)
```

#### 5.4 Internal Memory Anatomy: Pointer, Strides, and Flags

Every NumPy array consists of a lightweight Python wrapper around a raw memory buffer. The wrapper contains:
1. **`data`**: Pointer to the first byte in memory.
2. **`shape`**: Tuple defining array dimensions (e.g. `(3, 4)`).
3. **`dtype`**: Description of element format (e.g. `float64`, 8 bytes).
4. **`strides`**: Tuple of bytes to step along each dimension!

**Understanding Strides with an ASCII Visual:**
Consider a 2D array of `int64` (8 bytes per number) with shape `(3, 4)`:
```
           Col 0    Col 1    Col 2    Col 3
Row 0:   [   10,      20,      30,      40  ]
Row 1:   [   50,      60,      70,      80  ]
Row 2:   [   90,     100,     110,     120  ]

Physical Layout in RAM:
[ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 ]
  ^               ^
  |-- 32 bytes ---| (Jump to next row: stride[0] = 4 cols * 8 bytes = 32)
  |-- 8 bytes -> (Jump to next col: stride[1] = 1 col * 8 bytes = 8)

Strides Tuple: (32, 8)
To reach element (row=i, col=j):
Memory Address = data_pointer + (i * 32) + (j * 8)
```

```python
import numpy as np

arr = np.array([[10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120]], dtype=np.int64)

print("Array Shape:   ", arr.shape)    # (3, 4)
print("Array Strides: ", arr.strides)  # (32, 8)

# Inspect Array Flags
print("\nArray Memory Flags:")
print("  C_CONTIGUOUS (Row-major):", arr.flags.c_contiguous)
print("  F_CONTIGUOUS (Col-major):", arr.flags.f_contiguous)
print("  OWNDATA (Owns buffer):   ", arr.flags.owndata)
print("  WRITEABLE (Can modify):  ", arr.flags.writeable)
```

### 🏋️ Practice Question
**Q:** If a 3D array of `float32` (4 bytes per element) has shape `(10, 20, 30)`, what will its `strides` tuple be in standard C-contiguous order?
**A:**
- Dimension 2 (innermost, step of 1 element): $1 \times 4\text{ bytes} = 4\text{ bytes}$.
- Dimension 1 (step of 1 row of 30 elements): $30 \times 4\text{ bytes} = 120\text{ bytes}$.
- Dimension 0 (step of 1 matrix of $20 \times 30$ elements): $20 \times 30 \times 4\text{ bytes} = 2400\text{ bytes}$.
- The strides tuple is `(2400, 120, 4)`.

---

## 6. Indexing, Slicing, and Advanced Fancy Indexing

### 🤔 WHY

Data rarely arrives in the exact slice or subset you need. Indexing and slicing allow you to inspect individual measurements, extract subregions (like bounding boxes in images or time slices in sensor logs), and select non-contiguous patterns.

### ⏰ WHEN

- Use **Basic Slicing** when extracting contiguous subarrays, subgrids, or regular intervals (fast, zero memory overhead).
- Use **Advanced / Fancy Indexing** when you have an arbitrary list of coordinate positions, sorted indices from an algorithm, or non-uniform subsets.

### 🔧 HOW

#### 6.1 Basic Slicing in 1D, 2D, and 3D

Basic slicing syntax follows `array[start:stop:step]` for each axis separated by commas.

```python
import numpy as np

# 1D Slicing
a = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print("First 4 elements:      ", a[:4])       # [10 20 30 40]
print("From index 3 to end:   ", a[3:])       # [40 50 60 70 80]
print("Every second element:  ", a[::2])      # [10 30 50 70]
print("Reversed array:        ", a[::-1])     # [80 70 60 50 40 30 20 10]

# 2D Slicing: [row_slice, col_slice]
m = np.array([[ 1,  2,  3,  4],
              [ 5,  6,  7,  8],
              [ 9, 10, 11, 12]])

print("Row 0:                ", m[0, :])      # [1 2 3 4]
print("Column 2:             ", m[:, 2])      # [3 7 11]
print("Subgrid (rows 0-1, cols 1-2):\n", m[0:2, 1:3])
# [[2 3]
#  [6 7]]

# 3D Slicing: [matrix_slice, row_slice, col_slice]
tensor = np.arange(24).reshape(2, 3, 4)
# Extract all rows and cols from the second matrix
print("Tensor 2nd matrix:\n", tensor[1, :, :])
```

#### 6.2 The Ellipsis (`...`) Syntax

When working with high-dimensional tensors (e.g. in deep learning with shapes like `(batch, channels, height, width)`), typing `:` for every intermediate axis is tedious and error-prone. The **Ellipsis (`...`)** represents as many full `:` slices as necessary to match the remaining dimensions.

```python
import numpy as np

# A 4D Tensor: (batch=16, channels=3, height=64, width=64)
images = np.zeros((16, 3, 64, 64))

# Suppose we want the red channel (channel 0) across all batches, heights, and widths:
# Standard way:
red_ch1 = images[:, 0, :, :]

# Using Ellipsis:
red_ch2 = images[:, 0, ...]
print("Matches standard slice?", np.array_equal(red_ch1, red_ch2))  # True

# Extract the first pixel [0, 0] across all batches and channels:
first_pixels = images[..., 0, 0]
print("Shape with trailing ellipsis:", first_pixels.shape)  # (16, 3)
```

#### 6.3 Adding Dimensions (`np.newaxis` and `None`)

You can insert an axis of length 1 into an array using `np.newaxis` (or Python's built-in `None`). This is essential for broadcasting row vectors into column vectors.

```python
import numpy as np

v = np.array([1, 2, 3])  # Shape: (3,)

# Convert 1D vector to a 2D column vector: shape (3, 1)
col_vec = v[:, np.newaxis]
print("Column vector shape:\n", col_vec.shape)
print("Column vector values:\n", col_vec)

# Convert 1D vector to a 2D row vector: shape (1, 3)
row_vec = v[None, :]
print("Row vector shape:\n", row_vec.shape)

# Output:
# Column vector shape: (3, 1)
# Column vector values:
#  [[1]
#  [2]
#  [3]]
# Row vector shape: (1, 3)
```

#### 6.4 Advanced (Fancy) Indexing

> [!IMPORTANT]
> **The Golden Rule of Indexing:**
> - **Basic Slicing** (`m[0:2, 1:3]`) always creates a **VIEW** (shares underlying memory; zero memory copied).
> - **Fancy Indexing** (indexing with lists or integer arrays like `m[[0, 2], :]`) always creates a **COPY** (allocates new memory)!

**Example 1: Single Axis Fancy Indexing**
```python
import numpy as np

arr = np.array([100, 200, 300, 400, 500])
indices = [0, 3, 4]

# Select specific non-consecutive elements:
subset = arr[indices]
print("Selected elements:", subset)  # [100, 400, 500]

# Modifying subset does NOT affect original because it is a COPY!
subset[0] = 9999
print("Original unchanged:", arr[0])  # 100
```

**Example 2: Multidimensional Coordinate Pairing vs Broadcasting**
When you pass two integer arrays `m[rows, cols]`, NumPy pairs them up coordinate-by-coordinate:

```python
import numpy as np

grid = np.array([[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]])

# Selecting specific individual cells: (row 0, col 1) and (row 2, col 3)
row_coords = [0, 2]
col_coords = [1, 3]
selected_points = grid[row_coords, col_coords]
print("Paired coordinates (0,1) and (2,3):", selected_points)  # [1, 11]

# Selecting the RECTANGULAR SUBGRID of row 0 and 2 with col 1 and 3:
# Use np.ix_() to create an open mesh of indices
subgrid = grid[np.ix_([0, 2], [1, 3])]
print("Rectangular subgrid:\n", subgrid)
# [[ 1  3]
#  [ 9 11]]
```

#### 6.5 The In-Place Modification Trap with Repeated Indices

If an index array contains duplicate indices, standard Python in-place addition (`+=`) executes only once per duplicate position! To perform true unbuffered in-place updates, use `np.add.at()`.

```python
import numpy as np

# Trap demonstration:
counts = np.zeros(5, dtype=int)
indices = [1, 1, 1, 3]

# BUG: The Python statement `counts[indices] += 1` reads counts[1], adds 1, and writes back once!
counts[indices] += 1
print("Incorrect repeated update:", counts)  # [0, 1, 0, 1, 0] -> Only incremented once!

# SOLUTION: Use ufunc.at() for unbuffered accumulative updates
counts = np.zeros(5, dtype=int)
np.add.at(counts, indices, 1)
print("Correct accumulative update:", counts)  # [0, 3, 0, 1, 0] -> Correctly incremented 3 times!
```

#### 6.6 Utility Indexing Functions (`take`, `put`, `take_along_axis`)

```python
import numpy as np

# 1. np.take() extracts elements along an axis
data = np.array([10, 20, 30, 40, 50])
print("np.take:", np.take(data, [1, 3]))  # [20, 40]

# 2. np.take_along_axis() is essential for extracting top-k values based on argsort!
matrix = np.array([[90, 10, 40],
                   [20, 80, 50]])
# Find the column index of the maximum value in each row
max_indices = np.argmax(matrix, axis=1, keepdims=True)
# Extract the actual maximum values using those indices
max_values = np.take_along_axis(matrix, max_indices, axis=1)
print("Extracted max values per row:\n", max_values)
# [[90]
#  [80]]
```

### 🏋️ Practice Question
**Q:** You have a matrix `M = np.arange(16).reshape(4, 4)`. Write code to extract:
1. The diagonal elements using fancy indexing.
2. The four corner elements as a 2×2 matrix.
**A:**
```python
import numpy as np
M = np.arange(16).reshape(4, 4)

# 1. Diagonal: coordinate pairs (0,0), (1,1), (2,2), (3,3)
diag_vals = M[[0, 1, 2, 3], [0, 1, 2, 3]]  # or M[np.arange(4), np.arange(4)]

# 2. Corners: subgrid of rows [0, -1] and columns [0, -1]
corners = M[np.ix_([0, -1], [0, -1])]
print("Diagonal:", diag_vals)  # [ 0  5 10 15]
print("Corners:\n", corners)
# [[ 0  3]
#  [12 15]]
```

---

## 7. Array Operations & Universal Functions (ufuncs)

### 🤔 WHY

In traditional programming, performing arithmetic on an array of 1,000,000 numbers requires writing a `for` loop. Universal Functions (**ufuncs**) push that loop down into optimized, compiled C code that executes at bare-metal hardware speeds.

### ⏰ WHEN

Always use vectorized operators (`+`, `-`, `*`, `/`, `@`) and NumPy ufuncs instead of Python `for` loops.

### 🔧 HOW

#### 7.1 Arithmetic, Comparison, and Bitwise Operators

All standard Python operators are mapped to vectorized ufuncs:

| Operator | NumPy Function | Description |
| :--- | :--- | :--- |
| `a + b` | `np.add(a, b)` | Element-wise addition |
| `a - b` | `np.subtract(a, b)` | Element-wise subtraction |
| `a * b` | `np.multiply(a, b)` | Element-wise multiplication |
| `a / b` | `np.divide(a, b)` | Element-wise floating point division |
| `a // b`| `np.floor_divide(a, b)` | Element-wise integer division |
| `a ** b`| `np.power(a, b)` | Element-wise exponentiation |
| `a % b` | `np.remainder(a, b)` | Element-wise modulus (remainder) |
| `a == b`| `np.equal(a, b)` | Element-wise boolean equality |
| `a & b` | `np.bitwise_and(a, b)` | Element-wise bitwise / boolean AND |
| `a \| b` | `np.bitwise_or(a, b)` | Element-wise bitwise / boolean OR |
| `~a` | `np.bitwise_not(a, b)` | Element-wise bitwise / boolean NOT |

#### 7.2 The Memory-Saving `out=` Parameter

When doing math like `c = a + b`, NumPy must allocate brand-new memory for `c`. If you are inside a loop with large arrays, this repeated memory allocation creates garbage collector pressure and slows down your program.
Every ufunc accepts an `out=` parameter to write results into an existing buffer without allocating new memory!

```python
import numpy as np

a = np.ones((1000, 1000))
b = np.ones((1000, 1000))
result = np.empty((1000, 1000))

# Computes a + b directly into the pre-allocated result buffer:
np.add(a, b, out=result)

# In-place shortcut: writes directly back into array a
a += b  # Equivalent to np.add(a, b, out=a)
```

#### 7.3 Hidden Superpowers of ufuncs (`reduce`, `accumulate`, `outer`, `reduceat`, `at`)

Every binary ufunc (like `np.add`, `np.multiply`, `np.maximum`) comes with powerful built-in methods:

**1. `.reduce()`: Reduces an array's dimension by applying the operator repeatedly**
```python
import numpy as np

# Sum of all elements using add.reduce:
total = np.add.reduce([1, 2, 3, 4, 5])
print("np.add.reduce (Sum):", total)  # 15

# Factorial / Product of all elements:
product = np.multiply.reduce([1, 2, 3, 4, 5])
print("np.multiply.reduce (Product):", product)  # 120

# Logical AND reduction:
all_true = np.logical_and.reduce([True, True, False])
print("np.logical_and.reduce:", all_true)  # False
```

**2. `.accumulate()`: Produces running cumulative results**
```python
import numpy as np

# Running sum (same as np.cumsum):
running_sum = np.add.accumulate([1, 2, 3, 4, 5])
print("np.add.accumulate:", running_sum)  # [ 1  3  6 10 15]

# Running maximum:
running_max = np.maximum.accumulate([3, 1, 5, 2, 8, 4])
print("np.maximum.accumulate:", running_max)  # [3 3 5 5 8 8]
```

**3. `.outer()`: Computes the Cartesian outer operation between all pairs of elements**
```python
import numpy as np

# Multiplication table between [1, 2, 3] and [10, 20, 30]
table = np.multiply.outer([1, 2, 3], [10, 20, 30])
print("Outer Multiplication Table:\n", table)
# [[10 20 30]
#  [20 40 60]
#  [30 60 90]]
```

**4. `.reduceat()`: Segmented reduction over slice boundaries**
```python
import numpy as np

# Reduce chunks defined by index start points [0, 3, 5]
# Chunk 1: [0:3] -> 10 + 20 + 30 = 60
# Chunk 2: [3:5] -> 40 + 50 = 90
# Chunk 3: [5:]  -> 60
arr = np.array([10, 20, 30, 40, 50, 60])
chunk_sums = np.add.reduceat(arr, [0, 3, 5])
print("Segmented chunk sums:", chunk_sums)  # [60, 90, 60]
```

#### 7.4 Vectorization Myths: `np.vectorize` vs True Vectorization

> [!WARNING]
> `np.vectorize` is essentially a Python `for` loop wrapped in syntactic sugar! It provides convenience, NOT C-speed performance.

```python
import time
import numpy as np

def custom_logic(x):
    return x**2 + 2*x + 1 if x > 0 else 0

# Vectorize wrapper
vec_func = np.vectorize(custom_logic)

data = np.random.randn(1_000_000)

# 1. np.vectorize (Slow: runs Python function 1,000,000 times)
start = time.perf_counter()
res1 = vec_func(data)
t_vec = time.perf_counter() - start

# 2. Genuine NumPy expression (Fast: compiled C ufuncs)
start = time.perf_counter()
res2 = np.where(data > 0, data**2 + 2*data + 1, 0)
t_native = time.perf_counter() - start

print(f"np.vectorize time: {t_vec:.4f}s")
print(f"Native NumPy time: {t_native:.4f}s")
print(f"Native speedup:    {t_vec / t_native:.1f}x faster!")
# Typical Output: Native speedup is 25x to 50x faster!
```

### 🏋️ Practice Question
**Q:** Using `np.multiply.outer()`, write a one-liner to generate a 12×12 multiplication table (times table from 1 to 12).
**A:**
```python
import numpy as np
times_table = np.multiply.outer(np.arange(1, 13), np.arange(1, 13))
print(times_table)
```

---

## 8. Broadcasting Mechanics Deep Dive

### 🤔 WHY

Broadcasting allows arithmetic operations between arrays of different shapes without copying data. It saves massive amounts of RAM and simplifies code.

### ⏰ WHEN

Broadcasting applies whenever you:
- Add a single scalar to a matrix
- Normalize rows or columns by subtracting their respective means
- Compute pairwise distances or grid functions between coordinates

### 🔧 HOW

#### 8.1 The Formal 2-Rule Algorithm

To determine if two arrays are broadcast-compatible, NumPy compares their shape tuples **from right to left (trailing dimensions first)**:

> **The Two Rules of Broadcasting:**
> 1. If the arrays have a different number of dimensions, prepend `1`s to the shape of the shorter array on the left until shapes have equal length.
> 2. Two dimensions are **compatible** if:
>    - They are equal, OR
>    - One of them is `1`.

If all dimensions are compatible, the resulting array takes the maximum size along each axis. If any dimension fails Rule 2, NumPy raises `ValueError: operands could not be broadcast together`.

#### 8.2 Visual Step-by-Step Alignment Diagrams

**Scenario 1: Adding a 1D Row Vector `(4,)` to a 2D Matrix `(3, 4)`**
```
Step 1: Align shapes right-to-left:
Matrix A:    (3,  4)
Vector B:         (4)  --> Prepends 1: (1, 4)

Step 2: Compare each axis:
Axis 1: 4 and 4  --> Equal (Compatible! Result axis 1 = 4)
Axis 0: 3 and 1  --> One of them is 1 (Compatible! Result axis 0 = 3)

Broadcasting Stretch:
Matrix A (3, 4):          Vector B (1, 4) stretched virtually 3 times:
[[ 1,  2,  3,  4],        [[10, 20, 30, 40],
 [ 5,  6,  7,  8],   +     [10, 20, 30, 40],
 [ 9, 10, 11, 12]]         [10, 20, 30, 40]]
Result Shape: (3, 4)
```

**Scenario 2: Outer Product Grid: `(4, 1)` + `(1, 3)` -> `(4, 3)`**
```
Matrix A:    (4, 1)
Matrix B:    (1, 3)
--------------------
Result:      (4, 3)

A is stretched across 3 columns; B is stretched down across 4 rows!
```

```python
import numpy as np

# Scenario 2 Code Demonstration:
col = np.array([[10], [20], [30], [40]])  # Shape (4, 1)
row = np.array([[1, 2, 3]])               # Shape (1, 3)

grid = col + row
print("Grid shape:", grid.shape)  # (4, 3)
print("Grid values:\n", grid)
# [[11 12 13]
#  [21 22 23]
#  [31 32 33]
#  [41 42 43]]
```

#### 8.3 The Zero-Stride Trick: How Broadcasting Consumes ZERO Extra Memory

How can NumPy "stretch" an array without duplicating memory?
**Answer: By setting the stride of that dimension to 0 bytes!**

When a stride is 0, incrementing the index along that dimension adds 0 bytes to the memory pointer. The CPU reads the exact same memory location repeatedly without allocating a single extra byte of RAM!

```python
import numpy as np

# Create row vector (1, 4)
row = np.array([[10, 20, 30, 40]], dtype=np.int64)
print("Original strides: ", row.strides)  # (32, 8)

# Broadcast virtually to (1000000, 4) using as_strided
broadcasted = np.broadcast_to(row, (1_000_000, 4))
print("Broadcasted shape:   ", broadcasted.shape)    # (1000000, 4)
print("Broadcasted strides: ", broadcasted.strides)  # (0, 8) -> 0 byte stride on rows!
print("Broadcasted nbytes:  ", broadcasted.nbytes)   # Reports 32 MB virtual size...
print("Actual memory used:  ", row.nbytes)           # Only 32 bytes of real RAM!
```

#### 8.4 Diagnosing and Fixing Broadcasting Errors

```python
import numpy as np

matrix = np.ones((3, 4))
vector = np.array([1, 2, 3])  # Shape (3,)

# ATTEMPT: matrix + vector
# Alignment check:
# Matrix: (3, 4)
# Vector: (1, 3)  <-- Axis 1 has 4 vs 3: INCOMPATIBLE!
# Result: ValueError: operands could not be broadcast together with shapes (3,4) (3,)

# SOLUTION: Add new axis to make vector shape (3, 1)
vector_col = vector[:, np.newaxis]  # Shape (3, 1)
# Alignment check:
# Matrix:     (3, 4)
# Vector_col: (3, 1)  <-- Compatible!
result = matrix + vector_col
print("Fixed broadcast shape:", result.shape)  # (3, 4)
```

### 🏋️ Practice Question
**Q:** Are two arrays with shapes `(5, 1, 8)` and `(2, 8)` broadcast-compatible? What will the resulting shape be?
**A:** Yes!
- Pad shorter array: `(2, 8)` becomes `(1, 2, 8)`.
- Compare trailing dimensions:
  - Axis 2: 8 and 8 $\rightarrow$ Equal $\rightarrow$ 8
  - Axis 1: 1 and 2 $\rightarrow$ One is 1 $\rightarrow$ 2
  - Axis 0: 5 and 1 $\rightarrow$ One is 1 $\rightarrow$ 5
- Compatible! Output shape is `(5, 2, 8)`.

---

## 9. Mathematical, Statistical, and Aggregation Functions

### 🤔 WHY

Modern data analysis involves summarization: computing means, finding variances, locating peaks, handling missing sensor data, and maintaining numerical stability.

### ⏰ WHEN

Use aggregation functions to distill multidimensional arrays into summary statistics along specific axes.

### 🔧 HOW

#### 9.1 Aggregations along Axes and `keepdims=True`

In a 2D array:
- `axis=0`: Collapses **ROWS** (computes statistics down each column $\downarrow$)
- `axis=1`: Collapses **COLUMNS** (computes statistics across each row $\rightarrow$)

> [!TIP]
> **Why `keepdims=True` is Essential:**
> When you aggregate with `keepdims=True`, the collapsed axes remain as dimensions of size 1 (e.g. `(3, 1)` instead of `(3,)`). This allows you to immediately broadcast the result back against the original array!

```python
import numpy as np

# Student exam scores: 3 students, 4 exams
scores = np.array([[80, 90, 70, 60],
                   [95, 85, 90, 92],
                   [60, 75, 80, 70]], dtype=float)

# 1. Mean per student (collapse columns across axis 1)
student_means = scores.mean(axis=1, keepdims=True)
print("Student means shape (keepdims=True):", student_means.shape)  # (3, 1)
print("Student means:\n", student_means)

# 2. Mean-center each student's scores (Broadcasting works automatically!)
centered_scores = scores - student_means
print("Centered scores (mean = 0 per student):\n", centered_scores)

# 3. Overall class stats
print("Max score:          ", np.max(scores))
print("Min score:          ", np.min(scores))
print("Standard deviation: ", np.std(scores))
print("Peak-to-peak range: ", np.ptp(scores))  # max - min
```

#### 9.2 Handling Missing Data (`np.nan`) with NaN-Safe Functions

In real-world datasets, missing values are represented by `np.nan` (Not-a-Number). Standard aggregation functions will return `nan` if even a single element is missing. Use the `nan*` family to ignore missing values:

```python
import numpy as np

data = np.array([10.0, 20.0, np.nan, 40.0, 50.0])

# Standard functions fail with NaN:
print("Standard mean:  ", np.mean(data))  # nan

# NaN-safe functions skip NaNs gracefully:
print("NaN-safe mean:  ", np.nanmean(data))     # 30.0
print("NaN-safe sum:   ", np.nansum(data))      # 120.0
print("NaN-safe std:   ", np.nanstd(data))      # 15.81
print("NaN-safe median:", np.nanmedian(data))   # 30.0
print("Count valid:    ", np.count_nonzero(~np.isnan(data)))  # 4
```

#### 9.3 Quantiles and Percentiles

```python
import numpy as np

salaries = np.array([30_000, 45_000, 50_000, 60_000, 75_000, 90_000, 150_000])

# 25th (Q1), 50th (Median), and 75th (Q3) percentiles:
p25, p50, p75 = np.percentile(salaries, [25, 50, 75])
print(f"Q1: ${p25:,.0f}, Median: ${p50:,.0f}, Q3: ${p75:,.0f}")
print(f"Interquartile Range (IQR): ${p75 - p25:,.0f}")

# Quantiles (0.0 to 1.0 scale):
q90 = np.quantile(salaries, 0.90)
print(f"90th percentile salary: ${q90:,.0f}")
```

#### 9.4 Differences, Gradients, and Integration

```python
import numpy as np

# 1. Discrete difference: diff(a) = a[i+1] - a[i]
time_series = np.array([100, 105, 102, 108, 115])
daily_changes = np.diff(time_series)
print("Daily changes:", daily_changes)  # [ 5, -3,  6,  7]

# 2. Numerical gradient (central differences)
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = x**2  # Derivative dy/dx should be 2*x
dydx = np.gradient(y, x)
print("Numerical gradient of x^2:", dydx)  # [0. 2. 4. 6. 8.]

# 3. Numerical Integration via Trapezoidal rule
area = np.trapezoid(y, x)  # Integral of x^2 from 0 to 4 is (4^3)/3 = 21.33
print(f"Integrated area under curve: {area:.2f}")
```

#### 9.5 Rounding, Truncation, and Clamping

```python
import numpy as np

vals = np.array([-2.7, -0.2, 0.5, 1.4, 2.9])

print("Floor (round down):    ", np.floor(vals))  # [-3. -1.  0.  1.  2.]
print("Ceil (round up):       ", np.ceil(vals))   # [-2. -0.  1.  2.  3.]
print("Trunc (round to zero): ", np.trunc(vals))  # [-2. -0.  0.  1.  2.]
print("Round (nearest even):  ", np.round(vals))  # [-3. -0.  0.  1.  3.]
print("Clip (clamp to [0, 2]):", np.clip(vals, 0, 2))  # [0. 0. 0.5 1.4 2.]
```

#### 9.6 Floating-Point Precision and Numerical Stability

> [!CAUTION]
> Never compare floating point numbers with `==`! Due to IEEE 754 floating-point rounding errors, `0.1 + 0.2 == 0.3` evaluates to `False`. Always use `np.isclose()` or `np.allclose()`.

```python
import numpy as np

a = 0.1 + 0.2
b = 0.3

print("Direct equality (a == b):", a == b)  # False!

# np.isclose(a, b, rtol=1e-5, atol=1e-8) handles tiny machine precision differences
print("np.isclose(a, b):        ", np.isclose(a, b))  # True!

# For whole arrays:
arr1 = np.array([0.1 + 0.2, 1.0])
arr2 = np.array([0.3, 1.0])
print("np.allclose(arr1, arr2): ", np.allclose(arr1, arr2))  # True!
```

### 🏋️ Practice Question
**Q:** You have a matrix of sensor readings where missing readings are marked as `-999.0`. Write code to:
1. Replace all `-999.0` with `np.nan`.
2. Compute the mean reading for each sensor (column-wise), ignoring missing values.
**A:**
```python
import numpy as np

readings = np.array([[22.5, -999.0, 101.3],
                     [23.1,   45.2, 102.0],
                     [-999.0, 46.0, -999.0]])

# 1. Replace missing sentinel values with NaN
readings[readings == -999.0] = np.nan

# 2. Compute column-wise mean ignoring NaN
col_means = np.nanmean(readings, axis=0)
print("Sensor means:", col_means)  # [22.8, 45.6, 101.65]
```

---

## 10. Reshaping, Transposing, and Dimension Manipulation

### 🤔 WHY

Multidimensional data constantly needs to be adapted between formats. A flat array of 784 numbers might need to be reshaped into a 28×28 image, or a computer vision tensor in channels-last format `(H, W, C)` may need to be transposed to channels-first `(C, H, W)` for a neural network.

### ⏰ WHEN

- Reshape when transforming flat feature vectors into multidimensional tensors or batching data.
- Transpose / Swap axes when switching between row/column or channels-first/channels-last conventions.
- Squeeze or Expand dimensions when aligning shapes for broadcasting.

### 🔧 HOW

#### 10.1 Reshape and Automatic Dimension Inference (`-1`)

```python
import numpy as np

# 1D array of 12 elements
arr = np.arange(12)

# Reshape into 3 rows, 4 columns
m_3x4 = arr.reshape(3, 4)

# Use -1 to automatically compute an unknown dimension:
m_auto = arr.reshape(2, -1)  # 12 / 2 = 6 columns automatically!
print("Auto-calculated shape:", m_auto.shape)  # (2, 6)

# Batching: 12 elements into 2 batches of 3 rows, 2 cols
tensor = arr.reshape(2, 3, 2)
print("3D Tensor shape:      ", tensor.shape)  # (2, 3, 2)
```

#### 10.2 Flattening: `flatten()` vs `ravel()`

> [!IMPORTANT]
> - `ravel()` returns a **VIEW** whenever possible (ultra-fast, zero memory allocated).
> - `flatten()` always returns a **COPY** in memory (safe, isolated).

```python
import numpy as np

m = np.array([[1, 2], [3, 4]])

# ravel creates a VIEW:
r = m.ravel()
r[0] = 999
print("Original modified via ravel?", m[0, 0] == 999)  # True!

# flatten creates an independent COPY:
m[0, 0] = 1  # Reset
f = m.flatten()
f[0] = 999
print("Original modified via flatten?", m[0, 0] == 999)  # False!
```

#### 10.3 Expanding and Squeezing Dimensions (`expand_dims` & `squeeze`)

```python
import numpy as np

arr = np.array([10, 20, 30])  # Shape: (3,)

# Expand dimensions:
exp_0 = np.expand_dims(arr, axis=0)  # Shape: (1, 3) -> Row vector
exp_1 = np.expand_dims(arr, axis=1)  # Shape: (3, 1) -> Column vector
print("Expanded axis 0 shape:", exp_0.shape)
print("Expanded axis 1 shape:", exp_1.shape)

# Squeeze: removes all singleton dimensions (dimensions of size 1)
redundant = np.zeros((1, 5, 1, 10))
squeezed = np.squeeze(redundant)
print("Squeezed shape:       ", squeezed.shape)  # (5, 10)
```

#### 10.4 Transposing and Axis Permutation (`.T`, `transpose`, `swapaxes`, `moveaxis`)

```python
import numpy as np

# 1. 2D Transpose: Swaps rows and columns
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print("2D Transpose shape:", m.T.shape)  # (3, 2)

# 2. High-dimensional Transpose / Permutation:
# Example: Convert Image tensor from (Batch, Height, Width, Channels) to (Batch, Channels, Height, Width)
batch_images = np.zeros((32, 224, 224, 3))  # (B, H, W, C)

# Permute axes: order = (0: B, 3: C, 1: H, 2: W)
ch_first = np.transpose(batch_images, (0, 3, 1, 2))
print("Channels-first shape:", ch_first.shape)  # (32, 3, 224, 224)

# 3. np.swapaxes() swaps exactly two specific axes
swapped = np.swapaxes(batch_images, 1, 2)  # Swaps H and W
print("Swapped H and W shape:", swapped.shape)

# 4. np.moveaxis() moves a specific axis from source to destination position
moved = np.moveaxis(batch_images, source=-1, destination=1)
print("Moved channels to axis 1:", moved.shape)  # (32, 3, 224, 224)
```

#### 10.5 Replicating Data: `repeat()` vs `tile()`

```python
import numpy as np

a = np.array([1, 2, 3])

# repeat(): Repeats each individual ELEMENT consecutively
rep = np.repeat(a, repeats=3)
print("np.repeat: ", rep)   # [1 1 1 2 2 2 3 3 3]

# tile(): Repeats the entire ARRAY as a tile / block
til = np.tile(a, reps=3)
print("np.tile:   ", til)   # [1 2 3 1 2 3 1 2 3]

# 2D tiling (e.g. creating a checkerboard or repeated pattern):
m_tiled = np.tile(a, reps=(2, 2))
print("2D Tiled:\n", m_tiled)
# [[1 2 3 1 2 3]
#  [1 2 3 1 2 3]]
```

#### 10.6 Padding and Rolling (`np.pad` & `np.roll`)

Padding and circular shifts are foundational for convolutional neural networks, digital signal filtering, and periodic boundary simulations.

```python
import numpy as np

# 1. np.pad(): Adds padding borders around an array
img = np.array([[1, 2],
                [3, 4]])

# Constant padding with 0s: ((top, bottom), (left, right))
pad_zeros = np.pad(img, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)
print("Zero Padded (3x3 border):\n", pad_zeros)

# Edge (replication) padding:
pad_edge = np.pad(img, pad_width=1, mode='edge')
print("Edge Padded:\n", pad_edge)

# 2. np.roll(): Circular shift of elements
seq = np.array([10, 20, 30, 40, 50])
rolled_forward = np.roll(seq, shift=2)   # Shift right by 2
rolled_backward = np.roll(seq, shift=-1) # Shift left by 1
print("Rolled forward by 2: ", rolled_forward)   # [40 50 10 20 30]
print("Rolled backward by 1:", rolled_backward)  # [20 30 40 50 10]
```

### 🏋️ Practice Question
**Q:** You have a batch of 10 grayscale images of shape `(10, 28, 28)`. You need to pass them to a convolutional network that expects shape `(Batch, Channels, Height, Width)` where `Channels = 1`. Write the code to transform the array.
**A:**
```python
import numpy as np
images = np.zeros((10, 28, 28))
# Add the channel dimension at axis 1:
cnn_input = np.expand_dims(images, axis=1)
print(cnn_input.shape)  # (10, 1, 28, 28)
```

---

## 11. Joining, Stacking, and Splitting Arrays

### 🤔 WHY

Machine learning workflows constantly require concatenating feature matrices, combining training and testing splits, assembling batches of tensors, and dividing datasets among parallel workers.

### ⏰ WHEN

- Use `np.concatenate()` when joining arrays along an **existing** axis.
- Use `np.stack()` when joining arrays along a **brand new** axis (e.g. turning ten 2D image slices into a single 3D volume).
- Use `np.array_split()` when dividing an array into chunks that might not divide evenly.

### 🔧 HOW

#### 11.1 `concatenate()` vs `stack()` — The Core Distinction

> [!IMPORTANT]
> - `np.concatenate([a, b], axis=0)` keeps the same number of dimensions.
> - `np.stack([a, b], axis=0)` creates a **NEW** dimension!

```python
import numpy as np

a = np.array([1, 2, 3])  # Shape: (3,)
b = np.array([4, 5, 6])  # Shape: (3,)

# Concatenate: combines along existing axis -> Shape (6,)
concat_res = np.concatenate([a, b], axis=0)
print("np.concatenate shape:", concat_res.shape)  # (6,)
print("np.concatenate value:", concat_res)        # [1 2 3 4 5 6]

# Stack: creates a brand-new axis -> Shape (2, 3)
stack_res = np.stack([a, b], axis=0)
print("np.stack shape:      ", stack_res.shape)   # (2, 3)
print("np.stack value:\n", stack_res)
# [[1 2 3]
#  [4 5 6]]
```

#### 11.2 Specialized Stacking: `vstack`, `hstack`, `dstack`, `column_stack`

```python
import numpy as np

# 1D arrays
r1 = np.array([1, 2, 3])
r2 = np.array([4, 5, 6])

# vstack (vertical stack = row-wise stacking):
v = np.vstack([r1, r2])
print("vstack:\n", v)
# [[1 2 3]
#  [4 5 6]]

# hstack (horizontal stack = column-wise concatenation):
h = np.hstack([r1, r2])
print("hstack:", h)  # [1 2 3 4 5 6]

# column_stack: takes 1D arrays and stacks them as columns of a 2D matrix
cols = np.column_stack([r1, r2])
print("column_stack:\n", cols)
# [[1 4]
#  [2 5]
#  [3 6]]

# dstack: stacks along the 3rd dimension (depth) - ideal for combining R, G, B color channels!
red = np.ones((2, 2))
green = np.zeros((2, 2))
blue = np.full((2, 2), 255)
rgb = np.dstack([red, green, blue])
print("dstack RGB shape:", rgb.shape)  # (2, 2, 3)
```

#### 11.3 Block Matrices with `np.block()`

`np.block()` allows assembling complex block matrices using intuitive nested list syntax:

```python
import numpy as np

# Create 2x2 identity matrix and 2x2 zeros
A = np.eye(2)
B = np.zeros((2, 2))
C = np.ones((2, 2)) * 5
D = np.eye(2) * 9

# Assemble a 4x4 block matrix:
# [[ A, B ],
#  [ C, D ]]
big_matrix = np.block([[A, B],
                       [C, D]])
print("Assembled 4x4 Block Matrix:\n", big_matrix)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [5. 5. 9. 0.]
#  [5. 5. 0. 9.]]
```

#### 11.4 Splitting Arrays: `split()` vs `array_split()`

```python
import numpy as np

# 10 elements
data = np.arange(10)

# np.split() requires an EVEN division; otherwise raises ValueError:
# data has 10 elements; 10 cannot be divided evenly into 3 parts!
# np.split(data, 3) -> ERROR!

# np.array_split() handles uneven divisions gracefully:
chunks = np.array_split(data, 3)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
# Chunk 1: [0 1 2 3] (size 4)
# Chunk 2: [4 5 6]   (size 3)
# Chunk 3: [7 8 9]   (size 3)

# 2D Splitting: vsplit (split rows) and hsplit (split columns)
matrix = np.arange(16).reshape(4, 4)
top_half, bottom_half = np.vsplit(matrix, 2)
left_half, right_half = np.hsplit(matrix, 2)
print("Top half shape:   ", top_half.shape)   # (2, 4)
print("Left half shape:  ", left_half.shape)  # (4, 2)
```

### 🏋️ Practice Question
**Q:** You have two 2D arrays: `X_train` of shape `(800, 20)` and `X_test` of shape `(200, 20)`. Write the code to combine them into `X_total` of shape `(1000, 20)`.
**A:**
```python
import numpy as np
X_total = np.concatenate([X_train, X_test], axis=0)  # or np.vstack([X_train, X_test])
```

---

## 12. Copy vs View and Memory Layout (Strides & Cache Locality)

### 🤔 WHY

Accidentally modifying a view when you meant to modify a copy is one of the most common and dangerous bugs in Python data science. Conversely, making unnecessary copies of 10 GB datasets will exhaust system RAM and crash your program.

### ⏰ WHEN

- Rely on **Views** when reading data, reshaping, transposing, or computing statistics without allocating extra memory.
- Explicitly use `.copy()` when you need to transform or mutate a subset of data without corrupting the original master array.

### 🔧 HOW

#### 12.1 How Strides Create Views Without Copying Memory

A NumPy view does not allocate a new data buffer. It simply creates a new `PyArrayObject` header with:
1. The **same pointer** (or pointer offset) to the original data buffer.
2. A **modified shape** tuple.
3. A **modified strides** tuple.

```
Original Array (4 elements, int64, 8 bytes each):
Physical Memory: [ 10 | 20 | 30 | 40 ]
Base pointer:    0x100
Strides:         (8,)

View with step 2: v = arr[::2]
Physical Memory: [ 10 | 20 | 30 | 40 ]  (Unchanged!)
View pointer:    0x100
Shape:           (2,)
Strides:         (16,)  <-- Strides doubled! To get next item, jump 16 bytes!
```

#### 12.2 C-Order (Row-Major) vs Fortran-Order (Column-Major)

In memory, RAM is a strictly 1-dimensional sequence of physical bytes. How does a 2D matrix sit in 1D RAM?
- **C-Order (Row-Major, default in C and Python):** Rows are laid out one after another. Consecutive elements in the same row are adjacent in memory.
- **Fortran-Order (Column-Major, default in Fortran and MATLAB):** Columns are laid out one after another. Consecutive elements in the same column are adjacent in memory.

```
Matrix (2 rows, 3 cols):
[[ 1, 2, 3 ],
 [ 4, 5, 6 ]]

C-Order (Row-Major) Layout in RAM:
[ 1, 2, 3, 4, 5, 6 ]  --> Moving horizontally along a row steps 1 element in RAM.

Fortran-Order (Column-Major) Layout in RAM:
[ 1, 4, 2, 5, 3, 6 ]  --> Moving vertically down a column steps 1 element in RAM.
```

#### 12.3 Cache Locality Benchmark: Why Traversal Order Matters

When reading contiguous memory, the CPU hardware prefetcher loads future elements into cache ahead of time. Jumping across non-contiguous memory destroys cache performance!

```python
import time
import numpy as np

# Create a large 5000 x 5000 C-contiguous matrix
size = 5000
matrix = np.ones((size, size), order='C')

# 1. Row-Major Traversal (Along rows - Cache Friendly!):
start = time.perf_counter()
row_sum = np.sum(matrix, axis=1)  # Sums across rows (adjacent memory)
t_row = time.perf_counter() - start

# 2. Column-Major Traversal on C-matrix (Cache Unfriendly!):
start = time.perf_counter()
col_sum = np.sum(matrix, axis=0)  # Strides across memory, causing cache misses
t_col = time.perf_counter() - start

print(f"Row-major traversal time:    {t_row:.4f} seconds")
print(f"Column-major traversal time: {t_col:.4f} seconds")
# Row-major traversal is consistently faster because it respects physical memory layout!
```

#### 12.4 Verifying Memory Ownership (`.base`, `shares_memory`)

```python
import numpy as np

master = np.array([1, 2, 3, 4, 5])
view_slice = master[1:4]
explicit_copy = master[1:4].copy()

# 1. The .base attribute: Points to parent array if view; None if copy
print("view_slice.base is master:    ", view_slice.base is master)        # True
print("explicit_copy.base is None:   ", explicit_copy.base is None)       # True

# 2. np.shares_memory(): Exact programmatic check
print("Shares memory (view_slice):   ", np.shares_memory(master, view_slice))     # True
print("Shares memory (explicit_copy):", np.shares_memory(master, explicit_copy)) # False
```

#### 12.5 The Master View vs Copy Reference Table

| Operation | Produces a View? | Produces a Copy? | Explanation / Notes |
| :--- | :---: | :---: | :--- |
| **Basic Slicing** (`a[1:5]`, `m[:, 2]`) | ✅ **YES** | ❌ No | Only changes offset and strides |
| **Reshaping** (`a.reshape(...)`) | ✅ **YES** (usually) | ⚠️ When fragmented | Creates view if array is contiguous |
| **Transposing** (`a.T`, `swapaxes`) | ✅ **YES** | ❌ No | Simply reverses or swaps strides |
| **Ravel** (`a.ravel()`) | ✅ **YES** (usually) | ⚠️ When fragmented | Returns view if memory is contiguous |
| **Flatten** (`a.flatten()`) | ❌ No | ✅ **YES** | Always allocates new buffer |
| **Fancy Indexing** (`a[[0, 2]]`) | ❌ No | ✅ **YES** | Non-regular memory cannot share strides |
| **Boolean Indexing** (`a[a > 5]`) | ❌ No | ✅ **YES** | Ragged matches cannot share strides |
| **Type Casting** (`a.astype(...)`) | ❌ No | ✅ **YES** | New data type requires new byte buffer |
| **Explicit Copy** (`a.copy()`) | ❌ No | ✅ **YES** | Fully independent clone |

### 🏋️ Practice Question
**Q:** Why does `matrix.T.reshape(-1)` sometimes create a copy instead of a view?
**A:** `matrix.T` creates a view with swapped strides (making it Fortran-contiguous instead of C-contiguous). When you call `reshape(-1)`, NumPy attempts to view the elements in standard C-order. Because the physical memory is no longer contiguous in row-major order, NumPy cannot represent the flattened array with a single stride and is forced to allocate a new copy.

---

## 13. Boolean Indexing, Masking, and Conditional Selection

### 🤔 WHY

Real-world datasets contain anomalies, missing sensor spikes, negative values, or records outside target demographics. Boolean indexing lets you query, filter, and conditionally update multidimensional data in pure vectorized code without a single `if` statement or `for` loop.

### ⏰ WHEN

- Filtering records meeting numerical criteria (e.g. `salaries > 75000`).
- Removing outliers or invalid readings.
- Implementing multi-branch conditional business logic (if-elif-else) across arrays.

### 🔧 HOW

#### 13.1 Boolean Masks and Bitwise Operators

> [!CAUTION]
> In Python, you must use bitwise operators `&` (AND), `|` (OR), `~` (NOT), and `^` (XOR) for array conditions, and you **MUST** enclose each condition inside parentheses `()` because bitwise operators have higher operator precedence than comparison operators!

```python
import numpy as np

ages = np.array([16, 21, 25, 17, 34, 45, 19])

# Step 1: Create a boolean mask
adult_mask = ages >= 18
print("Boolean Mask:", adult_mask)
# [False  True  True False  True  True  True]

# Step 2: Apply mask to filter elements
adults = ages[adult_mask]
print("Filtered Adults:", adults)  # [21 25 34 45 19]

# Combining conditions with & and | (Parentheses are mandatory!)
target_group = ages[(ages >= 18) & (ages <= 30)]
print("Ages 18-30:", target_group)  # [21 25 19]

# Negation with ~ (NOT)
minors = ages[~(ages >= 18)]
print("Minors:", minors)  # [16 17]
```

#### 13.2 Vectorized If-Else: `np.where(condition, x, y)`

`np.where()` operates in two fundamentally different modes:
1. **Three arguments `np.where(cond, if_true, if_false)`:** Vectorized ternary operator (like an Excel `IF` or C `? :`).
2. **One argument `np.where(cond)`:** Returns the coordinate indices where condition is `True` (equivalent to `np.nonzero(cond)`).

```python
import numpy as np

scores = np.array([88, 55, 92, 45, 78, 62])

# Mode 1: Vectorized Ternary Assignment
# If score >= 60, "Pass", else "Fail"
grade_status = np.where(scores >= 60, "Pass", "Fail")
print("Grade Status:", grade_status)
# ['Pass' 'Fail' 'Pass' 'Fail' 'Pass' 'Pass']

# Cap scores at 100, floor at 50:
clamped = np.where(scores < 50, 50, scores)
print("Floored at 50:", clamped)

# Mode 2: Coordinate Indices Extraction
failed_indices = np.where(scores < 60)[0]
print("Indices of failing students:", failed_indices)  # [1 3]
```

#### 13.3 Multi-Branch Logic with `np.select()`

When you have multiple conditions (like `if / elif / else`), nesting `np.where()` is messy. `np.select()` provides clean, scalable multi-branch conditional assignment:

```python
import numpy as np

scores = np.array([95, 82, 73, 64, 45])

# Define condition list and choice list
conditions = [
    scores >= 90,
    scores >= 80,
    scores >= 70,
    scores >= 60
]
choices = ['A', 'B', 'C', 'D']

# np.select evaluates conditions in order and falls back to default
letter_grades = np.select(conditions, choices, default='F')
print("Scores:       ", scores)
print("Letter Grades:", letter_grades)
# Scores:        [95 82 73 64 45]
# Letter Grades: ['A' 'B' 'C' 'D' 'F']
```

#### 13.4 Boolean Reductions: `np.any()` and `np.all()`

```python
import numpy as np

data = np.array([[1, 2, -3],
                 [4, 5,  6]])

# Does the array contain ANY negative numbers?
print("Has any negative?", np.any(data < 0))  # True

# Are ALL numbers positive?
print("Are all positive?", np.all(data > 0))  # False

# Axis-wise verification:
# Check if each row contains any negative number
row_has_neg = np.any(data < 0, axis=1)
print("Each row has negative?", row_has_neg)  # [ True False]
```

#### 13.5 Membership Testing with `np.isin()`

```python
import numpy as np

product_ids = np.array([101, 105, 108, 112, 120, 125])
discontinued_items = [105, 120, 999]

# Find mask of products that are discontinued
is_discontinued = np.isin(product_ids, discontinued_items)
print("Discontinued mask:", is_discontinued)
print("Matching IDs:     ", product_ids[is_discontinued])  # [105 120]
```

### 🏋️ Practice Question
**Q:** Why does filtering a 2D matrix with a boolean mask (e.g. `matrix[matrix > 0]`) always return a flat 1D array rather than preserving the 2D shape?
**A:** In a 2D matrix, each row can have a different number of elements satisfying the condition (e.g., row 0 has 2 matches, row 1 has 0 matches, row 2 has 3 matches). A NumPy `ndarray` must be strictly rectangular; it cannot represent a "ragged" array where rows have varying lengths. Therefore, NumPy flattens all matching elements into a 1D copy.

---

## 14. Sorting, Searching, and Counting

### 🤔 WHY

Organizing data and identifying top/bottom performers is central to analytics. NumPy provides ultra-fast sorting algorithms and $O(N)$ partitioning algorithms that avoid the heavy cost of sorting entire datasets when you only need the top $k$ items.

### ⏰ WHEN

- Use `np.sort()` or `.sort()` to order values.
- Use `np.argsort()` when you need to reorder parallel arrays (e.g. sorting student names based on their exam scores).
- Use `np.partition()` or `np.argpartition()` for finding top $k$ items or k-nearest neighbors in $O(N)$ time.
- Use `np.searchsorted()` for binary search in already-sorted data.

### 🔧 HOW

#### 14.1 Sorting Arrays and In-Place vs Copy

```python
import numpy as np

arr = np.array([45, 12, 89, 23, 7])

# np.sort() returns a new sorted COPY:
sorted_copy = np.sort(arr)
print("Sorted copy: ", sorted_copy)  # [ 7 12 45 23 89]
print("Original:    ", arr)          # [45 12 89 23  7] (Unchanged!)

# arr.sort() sorts IN-PLACE:
arr.sort()
print("After in-place sort:", arr)   # [ 7 12 23 45 89]

# Descending sort:
desc = np.sort(arr)[::-1]
print("Descending:", desc)           # [89 45 23 12  7]
```

#### 14.2 Indirect Sorting with `np.argsort()`

`argsort` returns the integer indices that would sort the array. It is the primary tool for synchronizing multiple related arrays.

```python
import numpy as np

athletes = np.array(['Alice', 'Bob', 'Charlie', 'Diana', 'Evan'])
race_times = np.array([12.4, 11.8, 13.1, 11.2, 12.0])  # Lower is better!

# Get sort indices (ascending race times)
sorted_order = np.argsort(race_times)
print("Rank indices: ", sorted_order)  # [3 1 4 0 2] (Diana was fastest!)

# Sort both athletes and times simultaneously:
print("Podium Order:")
for rank, idx in enumerate(sorted_order, start=1):
    print(f"  {rank}. {athletes[idx]:<8} - {race_times[idx]:.1f}s")
```

#### 14.3 Multi-Key Sorting with `np.lexsort()`

`lexsort` performs lexicographical sorting (like `ORDER BY col1, col2` in SQL).

> [!NOTE]
> `np.lexsort((secondary_key, primary_key))` accepts keys in reverse order: the LAST key passed is the PRIMARY sort key!

```python
import numpy as np

last_names = np.array(['Smith', 'Johnson', 'Smith', 'Williams'])
first_names = np.array(['John', 'Alice', 'Adam', 'Bob'])

# Sort primarily by last_name, secondarily by first_name:
sort_order = np.lexsort((first_names, last_names))

print("Sorted Full Names:")
for idx in sort_order:
    print(f"  {last_names[idx]}, {first_names[idx]}")
# Johnson, Alice
# Smith, Adam
# Smith, John
# Williams, Bob
```

#### 14.4 Fast $O(N)$ Top-$K$ Selection with `np.partition()` and `np.argpartition()`

Sorting an array of $N$ elements takes $O(N \log N)$ time. If you only need the top 5 largest elements from 10,000,000 items, sorting the entire array wastes massive compute.
`np.partition()` uses the Introselect algorithm to partition elements in linear $O(N)$ time!

```python
import numpy as np

scores = np.array([45, 12, 98, 76, 23, 89, 54, 100, 31, 67])

# Partition so that the 3 largest elements are placed at the end (index -3):
# All elements before index -3 will be smaller than partition[-3]
# All elements after index -3 will be larger than partition[-3]
part = np.partition(scores, -3)
top_3 = part[-3:]
print("Top 3 scores (unsorted order):", top_3)  # [89, 98, 100]

# argpartition gives the indices of the top 3:
top_3_indices = np.argpartition(scores, -3)[-3:]
print("Top 3 indices:", top_3_indices)
```

#### 14.5 Binary Search with `np.searchsorted()`

For an array that is **already sorted**, `np.searchsorted()` finds the insertion indices in $O(\log N)$ time:

```python
import numpy as np

# Bins / thresholds for tax brackets or letter grades:
grade_cutoffs = np.array([60, 70, 80, 90])
student_scores = np.array([55, 62, 79, 85, 98])

# Find where each score falls in the cutoffs:
bracket_indices = np.searchsorted(grade_cutoffs, student_scores)
print("Bracket indices:", bracket_indices)  # [0 1 2 3 4]
```

#### 14.6 Unique Elements and Frequency Counts (`np.unique`)

```python
import numpy as np

votes = np.array(['Cat', 'Dog', 'Dog', 'Bird', 'Cat', 'Dog', 'Cat', 'Dog'])

# np.unique with all return flags:
uniques, first_indices, inverse_indices, counts = np.unique(
    votes,
    return_index=True,
    return_inverse=True,
    return_counts=True
)

print("Unique labels:       ", uniques)          # ['Bird' 'Cat' 'Dog']
print("First seen at index: ", first_indices)     # [3 0 1]
print("Counts:              ", counts)            # [1 3 4]

# Reconstruct the original array from unique labels and inverse indices:
reconstructed = uniques[inverse_indices]
print("Reconstructed matches original?", np.array_equal(votes, reconstructed))  # True
```

#### 14.7 Fast Non-Zero Counting (`np.count_nonzero`)

`np.count_nonzero(arr)` is significantly faster and uses less memory than `np.sum(arr != 0)` because it avoids allocating an intermediate boolean array in RAM.

```python
import numpy as np

arr = np.array([0, 5, 0, 12, 0, 0, 99])
count = np.count_nonzero(arr)
print("Number of non-zero elements:", count)  # 3
```

### 🏋️ Practice Question
**Q:** You have 1,000,000 sensor readings and need to extract the 10 lowest readings. Which function should you use and why?
**A:** Use `np.partition(readings, 10)[:10]` (or `np.argpartition`). It runs in linear $O(N)$ time instead of $O(N \log N)$ full sort, making it dramatically faster for large arrays.

---

## 15. Random Number Generation (Modern Generator API vs Legacy)

### 🤔 WHY

Random number generation is the heartbeat of modern computation: it powers Monte Carlo simulations, statistical bootstrap sampling, game mechanics, stochastic gradient descent, and deep learning neural network weight initialization.

### ⏰ WHEN

- Always use the **Modern `Generator` API (`np.random.default_rng`)** for new code.
- Recognize the **Legacy API (`np.random.seed`, `np.random.rand`)** when reading older tutorials or legacy codebases.

### 🔧 HOW

#### 15.1 The Legacy API vs The Modern Generator API

In NumPy 1.17+, the legacy random module was overhauled.
**Why the Legacy API is deprecated for modern projects:**
1. **Global Mutable State:** Calling `np.random.seed(42)` sets a single global seed across the entire Python process. If two libraries or threads call `np.random.seed()`, they overwrite each other's seeds.
2. **Poor Statistical Properties:** The legacy generator used the 1990s Mersenne Twister (MT19937) algorithm, which fails modern statistical randomness tests (BigCrush) and has poor high-dimensional uniformity.
3. **No Independent Streams:** Creating separate, un-correlated random streams for parallel CPU processes is difficult with the legacy API.

**The Modern Solution: `default_rng()` (PCG64)**
The modern API uses the **PCG-64 (Permuted Congruential Generator)** bit generator: it is faster, statistically superior, and thread-safe.

```python
import numpy as np

# OLD LEGACY WAY (Avoid in new code):
# np.random.seed(42)
# old_vals = np.random.rand(5)

# MODERN RECOMMENDED WAY:
rng = np.random.default_rng(seed=42)
new_vals = rng.random(5)
print("Modern random floats [0, 1):", new_vals)
```

#### 15.2 Core Methods of `np.random.default_rng`

```python
import numpy as np

rng = np.random.default_rng(seed=123)

# 1. Random Integers: rng.integers(low, high, size, endpoint=False)
dice_rolls = rng.integers(1, 7, size=5)
print("5 Dice Rolls (1 to 6):", dice_rolls)

# 2. Continuous Uniform Floats [0.0, 1.0):
floats = rng.random(size=(2, 3))
print("2x3 Uniform Floats [0, 1):\n", floats)

# 3. Continuous Uniform in Arbitrary Range [low, high):
temps = rng.uniform(low=-10.0, high=35.0, size=5)
print("Temperatures [-10, 35):", temps)

# 4. Standard Normal (Gaussian, mean=0, std=1):
std_normal = rng.standard_normal(size=5)
print("Standard Normal:", std_normal)

# 5. Gaussian with Custom Mean and Standard Deviation:
# Exam scores centered at mean 75 with std 10:
exam_scores = rng.normal(loc=75.0, scale=10.0, size=5)
print("Custom Normal (mean=75, std=10):", exam_scores)
```

#### 15.3 Random Sampling, Shuffling, and Permutation

```python
import numpy as np

rng = np.random.default_rng(seed=42)
cards = np.array(['Ace', 'King', 'Queen', 'Jack', '10', '9'])

# 1. Random Choice with replacement:
draw_with_replacement = rng.choice(cards, size=4, replace=True)
print("Draw with replacement:   ", draw_with_replacement)

# 2. Random Choice WITHOUT replacement (unique items):
hand = rng.choice(cards, size=4, replace=False)
print("Hand without replacement:", hand)

# 3. Weighted Random Choice:
outcomes = np.array(['Win', 'Lose', 'Tie'])
probabilities = [0.6, 0.3, 0.1]  # 60% win, 30% lose, 10% tie
simulated_game = rng.choice(outcomes, size=10, p=probabilities)
print("Weighted game outcomes:  ", simulated_game)

# 4. In-Place Shuffle (mutates original array):
deck = np.array([1, 2, 3, 4, 5])
rng.shuffle(deck)
print("In-place shuffled deck:  ", deck)

# 5. Permutation (returns a NEW shuffled copy without altering original):
original = np.array([10, 20, 30, 40, 50])
permuted = rng.permutation(original)
print("Permuted copy:           ", permuted)
print("Original unchanged:      ", original)
```

#### 15.4 Common Statistical Distributions

```python
import numpy as np

rng = np.random.default_rng(seed=99)

# 1. Binomial Distribution: Coin tosses (n trials, p probability of heads)
# Flip 10 coins, probability 0.5, repeat experiment 5 times:
heads_count = rng.binomial(n=10, p=0.5, size=5)
print("Binomial (heads out of 10):", heads_count)

# 2. Poisson Distribution: Call center arrivals (average rate lambda = 5 calls/hour)
hourly_calls = rng.poisson(lam=5, size=5)
print("Poisson hourly calls:      ", hourly_calls)

# 3. Exponential Distribution: Waiting time between events (scale = 1/lambda)
waiting_times = rng.exponential(scale=2.0, size=5)
print("Exponential wait times:    ", np.round(waiting_times, 2))

# 4. Gamma Distribution: Modeling insurance claim amounts or rainfall
claims = rng.gamma(shape=2.0, scale=100.0, size=5)
print("Gamma distributed claims:  ", np.round(claims, 2))
```

### 🏋️ Practice Question
**Q:** Write a simulation using `default_rng()` to simulate 100,000 flips of a fair coin. Estimate the probability of getting between 49,000 and 51,000 heads.
**A:**
```python
import numpy as np
rng = np.random.default_rng(seed=42)

# Simulate 1 experiment of 100,000 flips directly using binomial:
# Or repeat the experiment 1,000 times to get probability:
experiments = rng.binomial(n=100_000, p=0.5, size=10_000)
in_range = (experiments >= 49_000) & (experiments <= 51_000)
probability = np.mean(in_range)
print(f"Probability: {probability * 100:.2f}%")  # Typically ~100.00% (within 6.3 std deviations!)
```

---

## 16. Linear Algebra and Tensor Operations (np.linalg & np.einsum)

### 🤔 WHY

Linear algebra is the foundational mathematical language of machine learning, deep learning, 3D graphics, physics simulations, and robotics. NumPy's `np.linalg` module interfaces directly with world-class optimized Fortran/C libraries like **OpenBLAS**, **Intel MKL**, and **LAPACK**.

### ⏰ WHEN

- Calculating dot products, matrix multiplications, projections, and embeddings.
- Solving systems of linear equations ($Ax = b$).
- Finding eigenvalues for principal component analysis (PCA) and stability analysis.
- Compressing data or computing pseudoinverses via Singular Value Decomposition (SVD).
- Expressing complex multidimensional tensor contractions cleanly with `np.einsum`.

### 🔧 HOW

#### 16.1 Vector and Matrix Multiplication (`@`, `matmul`, `dot`, `vdot`, `outer`)

> [!NOTE]
> In Python 3.5+, the `@` operator was added specifically for matrix multiplication!
> `A @ B` is identical to `np.matmul(A, B)`. Never use `*` for matrix multiplication; `*` performs element-wise multiplication!

```python
import numpy as np

# Vectors (1D)
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# 1. Vector Dot Product: u . v = (1*4 + 2*5 + 3*6) = 32
print("Dot product (u @ v):      ", u @ v)       # 32
print("np.dot(u, v):             ", np.dot(u, v))# 32

# 2. Vector Outer Product: u (x) v -> Shape (3, 3)
outer_prod = np.outer(u, v)
print("Outer Product:\n", outer_prod)
# [[ 4  5  6]
#  [ 8 10 12]
#  [12 15 18]]

# 3. Matrix Multiplication: (2x3) @ (3x2) -> (2x2)
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape (2, 3)

B = np.array([[7,  8],
              [9, 10],
              [11, 12]])   # Shape (3, 2)

C = A @ B  # Matrix multiplication
print("Matrix Product (A @ B):\n", C)
# [[ 58  64]
#  [139 154]]
```

#### 16.2 Matrix Properties: Trace, Determinant, and Rank

```python
import numpy as np

M = np.array([[2.0, 1.0],
              [1.0, 3.0]])

# 1. Trace: Sum of diagonal elements (2 + 3 = 5)
print("Trace:       ", np.trace(M))  # 5.0

# 2. Determinant: det(M) = (2*3 - 1*1) = 5
print("Determinant: ", np.linalg.det(M))  # 5.0

# 3. Matrix Rank: Number of linearly independent rows/columns
print("Matrix Rank: ", np.linalg.matrix_rank(M))  # 2

# 4. Condition Number: Measures sensitivity to numerical errors
print("Condition No:", np.linalg.cond(M))
```

#### 16.3 Matrix Inversion and Pseudoinverse (`inv` & `pinv`)

> [!WARNING]
> Only square, non-singular matrices (determinant $\neq 0$) have an exact inverse. For singular or non-square rectangular matrices, use the Moore-Penrose **Pseudoinverse (`np.linalg.pinv`)**!

```python
import numpy as np

A = np.array([[4.0, 7.0],
              [2.0, 6.0]])

# 1. Exact Inverse: A^(-1)
A_inv = np.linalg.inv(A)
print("Inverse Matrix A^(-1):\n", A_inv)

# Verify A @ A^(-1) == Identity:
identity_check = A @ A_inv
print("A @ A_inv (Identity):\n", np.round(identity_check, 4))
# [[1. 0.]
#  [0. 1.]]

# 2. Moore-Penrose Pseudoinverse (Works for ANY matrix, even non-square!):
rect_matrix = np.array([[1, 2, 3],
                        [4, 5, 6]], dtype=float)
pinv_res = np.linalg.pinv(rect_matrix)
print("Pseudoinverse shape:", pinv_res.shape)  # (3, 2)
```

#### 16.4 Solving Linear Systems ($Ax = b$)

Suppose you have a system of linear equations:
$$2x + y = 8$$
$$x + 3y = 13$$

In matrix form:
$$\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ 13 \end{bmatrix}$$

> [!IMPORTANT]
> **Never solve systems by computing `inv(A) @ b`!**
> Computing the explicit inverse is slow ($O(N^3)$) and numerically unstable. Always use `np.linalg.solve(A, b)`, which uses optimized LU or Cholesky decomposition!

```python
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 3.0]])
b = np.array([8.0, 13.0])

# Solve Ax = b
solution = np.linalg.solve(A, b)
print("Solution [x, y]:", solution)  # [x = 2.2, y = 3.6]

# Verify solution
print("Ax equals b?", np.allclose(A @ solution, b))  # True
```

#### 16.5 Least Squares Regression (`np.linalg.lstsq`)

When you have more equations than unknowns ($Ax \approx b$), an exact solution does not exist. The least squares solver finds the vector $x$ that minimizes $\|Ax - b\|^2$.

```python
import numpy as np

# Sample data points (x_pts, y_pts)
x_pts = np.array([0, 1, 2, 3, 4, 5])
y_pts = np.array([1.1, 2.9, 5.2, 7.1, 8.8, 11.2])  # Line with slope ~2, intercept ~1

# Design matrix A = [x, 1] for y = m*x + c
A = np.vstack([x_pts, np.ones(len(x_pts))]).T

# Solve least squares fit:
slope, intercept = np.linalg.lstsq(A, y_pts, rcond=None)[0]
print(f"Fitted Line: y = {slope:.2f}x + {intercept:.2f}")
# Output: Fitted Line: y = 2.01x + 1.04
```

#### 16.6 Eigenvalues and Eigenvectors (`eig` & `eigh`)

Eigenvalues $\lambda$ and eigenvectors $v$ satisfy $A v = \lambda v$.
- Use `np.linalg.eig()` for general square matrices.
- Use `np.linalg.eigh()` for symmetric (Hermitian) matrices (faster and guaranteed real eigenvalues!).

```python
import numpy as np

# Symmetric matrix (covariance-like)
cov_matrix = np.array([[4.0, 2.0],
                       [2.0, 3.0]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print("Eigenvalues: \n", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# Verify A @ v = lambda * v for the first eigenvector:
v0 = eigenvectors[:, 0]
lambda0 = eigenvalues[0]
print("A @ v0:     ", cov_matrix @ v0)
print("lambda0 * v0:", lambda0 * v0)
print("Equal?", np.allclose(cov_matrix @ v0, lambda0 * v0))  # True
```

#### 16.7 Singular Value Decomposition (SVD)

SVD factors any $M \times N$ matrix $A$ into:
$$A = U \Sigma V^T$$
Where:
- $U$: Left singular vectors (orthonormal)
- $\Sigma$ (singular values): Strength of each principal dimension
- $V^T$: Right singular vectors (orthonormal)

SVD is the mathematical engine beneath **Principal Component Analysis (PCA)**, **recommender systems**, and **image compression**.

```python
import numpy as np

# 4x3 matrix
data_matrix = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0]])

U, s, Vt = np.linalg.svd(data_matrix)

print("U shape:       ", U.shape)   # (4, 4)
print("Singular values:", s)         # 3 singular values
print("Vt shape:      ", Vt.shape)  # (3, 3)

# Low-rank approximation using only the TOP 1 singular value (rank-1 reconstruction):
rank1_approx = s[0] * np.outer(U[:, 0], Vt[0, :])
print("Rank-1 Compression Approximation:\n", np.round(rank1_approx, 1))
```

#### 16.8 Vector and Matrix Norms (`np.linalg.norm`)

```python
import numpy as np

v = np.array([3.0, -4.0])

# 1. L2 Norm (Euclidean distance): sqrt(3^2 + (-4)^2) = 5.0
print("L2 Norm (Euclidean):", np.linalg.norm(v, ord=2))  # 5.0

# 2. L1 Norm (Manhattan distance): |3| + |-4| = 7.0
print("L1 Norm (Manhattan):", np.linalg.norm(v, ord=1))  # 7.0

# 3. Infinity Norm (Max absolute value): 4.0
print("Infinity Norm:      ", np.linalg.norm(v, ord=np.inf))  # 4.0

# 4. Matrix Frobenius Norm (sqrt of sum of all squared entries):
matrix = np.array([[1, 2], [3, 4]])
print("Frobenius Norm:     ", np.linalg.norm(matrix, ord='fro'))  # 5.477
```

#### 16.9 Einstein Summation Convention (`np.einsum`) — The Master Tensor Tool

`np.einsum` is the Swiss Army knife of tensor computing in NumPy and PyTorch. It allows expressing almost any linear algebra operation with a compact, standardized string convention:

> **The Mental Model of Einsum:**
> 1. Each letter in the subscript string labels a dimension (axis) of an operand.
> 2. Any index that appears on the input side but is **OMITTED** on the output side is **SUMMED OVER**!
> 3. If output indices are omitted entirely (e.g. `i,i->`), all remaining dimensions are summed to a scalar.

**Mastering Einsum by Example:**

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
u = np.array([1, 2])
v = np.array([3, 4])

# 1. Transpose: 'ij->ji' (swap axis 0 and 1)
print("1. Transpose:\n", np.einsum('ij->ji', A))

# 2. Matrix Trace: 'ii->' (sum diagonal elements where row == col)
print("2. Trace:", np.einsum('ii->', A))  # 1 + 4 = 5

# 3. Extract Diagonal: 'ii->i' (keep diagonal index i in output)
print("3. Diagonal:", np.einsum('ii->i', A))  # [1, 4]

# 4. Vector Dot Product: 'i,i->' (multiply and sum over axis i)
print("4. Dot Product:", np.einsum('i,i->', u, v))  # 1*3 + 2*4 = 11

# 5. Matrix-Vector Multiply: 'ij,j->i' (sum over columns j, keep rows i)
print("5. Matrix-Vector Multiply:", np.einsum('ij,j->i', A, u))  # [5, 11]

# 6. Standard Matrix Multiplication: 'ij,jk->ik' (sum over common axis j)
print("6. Matrix Multiplication (A @ B):\n", np.einsum('ij,jk->ik', A, B))
# [[19 22]
#  [43 50]]

# 7. Batched Matrix Multiplication (Common in Deep Learning!):
# Batch of 10 matrices of size 2x3 multiplied with batch of 10 matrices of size 3x4:
batch_A = np.ones((10, 2, 3))
batch_B = np.ones((10, 3, 4))
batch_C = np.einsum('bij,bjk->bik', batch_A, batch_B)
print("7. Batched Matrix Multiplication Shape:", batch_C.shape)  # (10, 2, 4)
```

### 🏋️ Practice Question
**Q:** Write a one-line `np.einsum` call to compute the sum of squares of all elements in a 2D matrix `M`.
**A:**
```python
import numpy as np
M = np.array([[1, 2], [3, 4]])
sum_of_squares = np.einsum('ij,ij->', M, M)
print(sum_of_squares)  # 1^2 + 2^2 + 3^2 + 4^2 = 30
```

---

## 17. Structured Arrays and Record Arrays

### 🤔 WHY

Often in real-world problems, data is tabular and heterogeneous: each row has a string name, an integer age, and a floating-point salary.
- Standard Python lists of dictionaries are slow and consume massive memory.
- Standard NumPy arrays coerce everything into strings if you try to mix data types.
**Structured Arrays** solve this by implementing C-style structs directly in contiguous NumPy memory: you get mixed-type tabular columns running at compiled C speed!

### ⏰ WHEN

- Interfacing with low-level binary files, network telemetry packets, or C code that outputs struct arrays.
- Memory-constrained embedded environments where Pandas is too heavy.
- When you need fixed-schema tabular records without Python object overhead.

### 🔧 HOW

#### 17.1 Defining Structured Data Types (`np.dtype`)

You define a structured dtype as a list of tuples: `('field_name', 'data_type')`.

```python
import numpy as np

# Define schema: name (20-char Unicode), age (32-bit int), salary (64-bit float)
employee_dtype = np.dtype([
    ('name', 'U20'),
    ('age', 'i4'),
    ('salary', 'f8')
])

# Create array of structured records:
employees = np.array([
    ('Alice', 34, 92000.50),
    ('Bob', 28, 67500.00),
    ('Charlie', 45, 115000.00)
], dtype=employee_dtype)

print("Structured Array:\n", employees)
print("Data buffer bytes: ", employees.nbytes)
```

#### 17.2 Querying and Manipulating Fields

```python
import numpy as np

# 1. Accessing a Column (Field) across all rows:
names = employees['name']
print("Employee names:", names)  # ['Alice' 'Bob' 'Charlie']

# 2. Accessing a specific row (Record):
first_person = employees[0]
print("First employee record:", first_person)  # ('Alice', 34, 92000.5)

# 3. Filtering by field condition:
high_earners = employees[employees['salary'] > 70000]
print("High earners:", high_earners['name'])  # ['Alice' 'Charlie']

# 4. Modifying fields in-place:
employees['salary'] *= 1.05  # 5% raise to everyone!
print("Updated Salaries:\n", employees['salary'])
```

#### 17.3 Nested Structured dtypes

Fields can themselves be structured types or fixed-size sub-arrays!

```python
import numpy as np

# Schema with a 3D coordinate sub-array:
particle_dtype = np.dtype([
    ('id', 'i4'),
    ('position', 'f4', (3,)),  # Fixed 3-element float32 vector!
    ('mass', 'f4')
])

particles = np.zeros(2, dtype=particle_dtype)
particles[0] = (1, [0.0, 1.5, 3.2], 9.11)
particles[1] = (2, [1.0, -2.0, 0.5], 1.67)

print("Particle 0 Position:", particles[0]['position'])
print("All Positions (Matrix):\n", particles['position'])
```

#### 17.4 Record Arrays (`np.recarray`)

A `recarray` allows field access via standard Python dot attribute notation (e.g. `employees.salary` instead of `employees['salary']`).

```python
import numpy as np

# Convert structured array to record array:
rec_emp = employees.view(np.recarray)

# Access fields with dot notation:
print("Alice's age via dot notation:", rec_emp.age[0])  # 34
print("Charlie's salary:             ", rec_emp.salary[2])
```

#### 17.5 Structured Arrays vs Pandas DataFrames

| Feature | NumPy Structured Array | Pandas DataFrame |
| :--- | :--- | :--- |
| **Primary Goal** | C-struct compatibility & minimal memory | Exploratory data analysis & rich ETL |
| **Memory Footprint** | Extremely low (pure contiguous C-bytes) | Moderate (contains column indices & metadata) |
| **Missing Data Handling** | Manual (requires sentinel values) | Native (`NaN`, `NA`, `None`) |
| **Indexing & GroupBy** | Basic array indexing | Rich SQL-like groupby, merge, pivot |
| **Recommendation** | Use for binary I/O, C-interop, or low memory | Use for general tabular data science |

### 🏋️ Practice Question
**Q:** Define a structured array dtype for astronomical stars with: `catalog_id` (10-char string), `magnitude` (float32), and `spectral_class` (1-char string).
**A:**
```python
import numpy as np
star_dtype = np.dtype([
    ('catalog_id', 'U10'),
    ('magnitude', 'f4'),
    ('spectral_class', 'U1')
])
stars = np.array([('HD 140283', 7.21, 'F'), ('Betelgeuse', 0.50, 'M')], dtype=star_dtype)
print(stars['catalog_id'], stars['magnitude'])
```

---

## 18. File I/O and Out-of-Core Big Data (np.memmap)

### 🤔 WHY

Data generated in simulations, training runs, or data pipelines must be saved to disk and loaded quickly. When datasets grow to tens or hundreds of gigabytes—larger than available system RAM—standard loading crashes with an `OutOfMemoryError`. NumPy provides both high-speed binary serialization and **Memory Mapping (`memmap`)** for out-of-core computing.

### ⏰ WHEN

- Use `.npy` for single array persistence (fastest binary format).
- Use `.npz` for saving multiple arrays together into a single compressed archive.
- Use `np.memmap` when analyzing or processing datasets too large to fit in physical RAM.

### 🔧 HOW

#### 18.1 Fast Native Binary Storage (`.npy` and `.npz`)

NumPy's native binary format `.npy` stores the raw array buffer along with its dtype and shape header. It is orders of magnitude faster than CSV, JSON, or Pickle.

```python
import numpy as np

arr1 = np.arange(1000).reshape(10, 100)
arr2 = np.ones((5, 5))

# 1. Save single array (.npy)
np.save('my_array.npy', arr1)

# Load single array (.npy)
loaded_arr1 = np.load('my_array.npy')
print("Loaded .npy array shape:", loaded_arr1.shape)

# 2. Save multiple arrays into a single compressed archive (.npz)
np.savez_compressed('archive.npz', weights=arr1, biases=arr2)

# Load compressed archive (.npz): loads lazily on demand
with np.load('archive.npz') as data:
    print("Archive keys:    ", data.files)  # ['weights', 'biases']
    loaded_w = data['weights']
    loaded_b = data['biases']
    print("Weights shape:   ", loaded_w.shape)
    print("Biases shape:    ", loaded_b.shape)
```

#### 18.2 Text & CSV Storage (`loadtxt`, `savetxt`, `genfromtxt`)

```python
import numpy as np

# Export data to CSV
data = np.array([[1.5, 2.3, 3.1],
                 [4.2, 5.0, 6.8]])
np.savetxt('data.csv', data, delimiter=',', fmt='%.2f', header='c1,c2,c3')

# Read data from CSV
loaded_csv = np.loadtxt('data.csv', delimiter=',', skiprows=1)
print("Loaded CSV:\n", loaded_csv)

# For messy text files with missing values, use np.genfromtxt(..., filling_values=...)
```

#### 18.3 Out-of-Core Computing with Memory-Mapped Arrays (`np.memmap`)

> **How Memory Mapping Works:**
> `np.memmap` creates an array whose data buffer points directly to a file stored on your SSD/hard drive using OS virtual memory page tables.
> When you read or slice `memmap_arr[100:200]`, the operating system loads **only those specific disk pages into RAM on demand**. You can seamlessly read, slice, and process a **100 GB array on a machine with only 8 GB of RAM**!

```python
import numpy as np

# Step 1: Create a large memory-mapped array file on disk
# Let's create a 1,000 x 1,000 float64 matrix (8 MB on disk)
filename = 'large_dataset.dat'
shape = (1000, 1000)
dtype = np.float64

# Mode 'w+' creates or overwrites the file for read/write
mmap_arr = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

# Write data into the memory-mapped array just like a regular array:
mmap_arr[:] = np.arange(1000)[:, None] + np.arange(1000)

# Flush changes to physical disk:
mmap_arr.flush()
del mmap_arr  # Close mapping

# Step 2: Open existing disk file in read-only mode ('r')
# No memory is loaded upfront!
readonly_mmap = np.memmap(filename, dtype=dtype, mode='r', shape=shape)

# Slicing only loads the requested 2 rows from disk into RAM:
sub_slice = readonly_mmap[500:502, :5]
print("Read slice from disk via memmap:\n", sub_slice)
```

### 🏋️ Practice Question
**Q:** Why is saving arrays with `np.savez_compressed()` preferred over Python's built-in `pickle` module for numerical checkpoints?
**A:** `np.savez_compressed()` directly writes the raw unboxed C memory buffer to a zip archive with zero serialization overhead and platform-independent endianness headers. Python's `pickle` serializes Python objects individually, resulting in much larger file sizes, slower write/read speeds, and severe security risks when loading untrusted data.

---

## 19. Performance Optimization, Vectorization, and Cache Friendly Code

### 🤔 WHY

Writing fast numerical Python is not about writing clever `for` loops; it is about writing code that aligns with the physical architecture of modern CPUs: cache lines, branch prediction, vector registers, and contiguous memory strides.

### ⏰ WHEN

Optimize performance whenever data analysis pipelines run in production, during high-throughput real-time processing, or when scaling to large datasets.

### 🔧 HOW

#### 19.1 The Vectorization Mindset: Eliminating Loops

Whenever you feel the urge to write a `for` loop over array indices, ask:
1. *Can this be expressed as an element-wise ufunc?*
2. *Can this be broadcasted across a new axis?*
3. *Can this be masked with a boolean condition?*
4. *Can this be accumulated with `.reduce()` or `cumsum()`?*

```python
import time
import numpy as np

# PROBLEM: Compute the Euclidean distance between all pairs of 1000 points
# Points: shape (1000, 2)
points = np.random.rand(1000, 2)

# SLOW: Nested Python for-loop
start = time.perf_counter()
N = len(points)
dist_loop = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        diff = points[i] - points[j]
        dist_loop[i, j] = np.sqrt(diff[0]**2 + diff[1]**2)
t_loop = time.perf_counter() - start

# FAST: Fully Vectorized Broadcasting (Zero Loops!)
# Shape: (1000, 1, 2) - (1, 1000, 2) -> (1000, 1000, 2)
start = time.perf_counter()
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
dist_vec = np.sqrt(np.sum(diff**2, axis=-1))
t_vec = time.perf_counter() - start

print(f"For-loop time:    {t_loop:.4f} seconds")
print(f"Vectorized time:  {t_vec:.4f} seconds")
print(f"Speedup factor:   {t_loop / t_vec:.1f}x faster!")
# Typical Output: Vectorized code is 80x to 150x faster!
```

#### 19.2 Cache Friendliness: Row-Major vs Column-Major

Hardware CPU caches fetch 64-byte chunks.
- When iterating along rows in C-contiguous memory (`arr[i, :]`), every element is adjacent in RAM $\rightarrow$ Cache hits!
- When iterating along columns in C-contiguous memory (`arr[:, j]`), every step jumps across rows in RAM $\rightarrow$ Cache misses!

```python
import numpy as np

# Ensure your memory order matches your access pattern!
# If your algorithm predominantly accesses columns, store the matrix in Fortran order:
col_friendly = np.asfortranarray(np.ones((5000, 5000)))
print("F_CONTIGUOUS flag:", col_friendly.flags.f_contiguous)  # True
```

#### 19.3 Pre-allocating Buffers and In-Place Operations

Avoid allocating temporary arrays inside loops:

```python
import numpy as np

# BAD PATTERN: Allocates a new temporary array on every iteration!
# for step in range(1000):
#     x = x * 0.99 + bias

# GOOD PATTERN: Zero memory allocation using in-place operations
# for step in range(1000):
#     x *= 0.99
#     x += bias
```

#### 19.4 Beyond NumPy: JIT Compilation with Numba

For complex algorithms that cannot be easily vectorized with pure array expressions (like custom simulation state machines or recursive dynamic programming), **Numba** compiles Python functions to machine code using LLVM:

```python
# Conceptual Numba Example:
# from numba import njit
#
# @njit(fastmath=True)
# def fast_custom_simulation(arr):
#     total = 0.0
#     for i in range(arr.shape[0]):
#         if arr[i] > 0:
#             total += np.sin(arr[i])
#     return total
# Runs at identical speed to hand-tuned C/C++!
```

### 🏋️ Practice Question
**Q:** Why does replacing `a = a + b` with `a += b` often result in significant speedups for large arrays?
**A:** `a = a + b` evaluates the right-hand side by allocating a brand-new array in memory to hold the sum, and then re-assigns the name `a` to that new array. This incurs memory allocation, memory copying, and later garbage collection overhead. `a += b` uses the in-place addition ufunc (`np.add(a, b, out=a)`), directly overwriting the bytes in `a`'s existing buffer without allocating any new RAM.

---

## 20. NumPy with Real-World End-to-End Case Studies

### 🎯 Case Study 1: Image Processing from Scratch (No External Image Libraries!)

Images are simply 3D NumPy arrays of shape `(Height, Width, 3)` with `dtype=np.uint8`. Let us build an end-to-end image processing pipeline using pure NumPy:

```python
import numpy as np

# 1. Synthesize a 6x6 RGB Image (Values 0 to 255)
np.random.seed(42)
rgb_image = np.random.randint(0, 256, size=(6, 6, 3), dtype=np.uint8)

print("Original RGB Image Shape:", rgb_image.shape)

# 2. Convert to Grayscale using Standard ITU-R BT.601 Luminance Weights:
# Grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
luminance_weights = np.array([0.2989, 0.5870, 0.1140])
grayscale = np.tensordot(rgb_image, luminance_weights, axes=([2], [0])).astype(np.float32)
print("\nGrayscale Image (6x6):\n", np.round(grayscale, 1))

# 3. Contrast Stretching (Normalization to [0.0, 1.0]):
# Formula: (pixel - min) / (max - min)
min_val = grayscale.min()
max_val = grayscale.max()
normalized = (grayscale - min_val) / (max_val - min_val)
print("\nContrast Normalized [0, 1]:\n", np.round(normalized, 2))

# 4. 2D Spatial Convolution (Gaussian / Box Blur Filter):
# We define a 3x3 averaging blur kernel:
kernel = np.ones((3, 3), dtype=np.float32) / 9.0

# Pad the grayscale image with reflective borders to maintain size:
padded = np.pad(grayscale, pad_width=1, mode='reflect')

# Compute 2D convolution by sliding a 3x3 window:
H, W = grayscale.shape
blurred = np.zeros((H, W), dtype=np.float32)
for i in range(H):
    for j in range(W):
        window = padded[i:i+3, j:j+3]
        blurred[i, j] = np.sum(window * kernel)

print("\nBlurred Image (after 3x3 Convolution):\n", np.round(blurred, 1))

# 5. Thresholding / Binary Mask (Edge or Feature Segmentation):
binary_mask = blurred > 128.0
print("\nBinary Segmentation Mask:\n", binary_mask)
```

---

### 🎯 Case Study 2: Machine Learning from Scratch (K-Means & Linear Regression)

#### A. Vectorized Pairwise Euclidean Distance Matrix (No Loops!)
Given $N$ samples in matrix $A$ of shape $(N, D)$ and $M$ centroids in matrix $B$ of shape $(M, D)$, we compute the Euclidean distance between every sample and every centroid using broadcasting:

```python
import numpy as np

# 100 data points (N=100, D=3), 4 cluster centroids (M=4, D=3)
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 3))
centroids = rng.standard_normal((4, 3))

# Vectorized distance: Shape (100, 1, 3) - Shape (1, 4, 3) -> Shape (100, 4, 3)
diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
print("Distance Matrix Shape (N x M):", dist_matrix.shape)  # (100, 4)

# Assign each sample to the closest centroid:
cluster_assignments = np.argmin(dist_matrix, axis=1)
print("First 10 Cluster Assignments:", cluster_assignments[:10])
```

#### B. Linear Regression via the Normal Equation
The optimal weights $\theta$ that minimize the sum of squared errors in linear regression are given by the analytical Normal Equation:
$$\theta = (X^T X)^{-1} X^T y$$

```python
import numpy as np

rng = np.random.default_rng(42)

# Generate synthetic dataset: y = 3*x1 - 2*x2 + 5 + noise
N = 200
X_raw = rng.uniform(0, 10, size=(N, 2))
true_weights = np.array([3.0, -2.0])
true_intercept = 5.0
noise = rng.normal(0, 0.5, size=N)
y = X_raw @ true_weights + true_intercept + noise

# 1. Add bias / intercept column of 1s to design matrix X:
X_design = np.column_stack([np.ones(N), X_raw])

# 2. Solve Normal Equation: theta = solve(X.T @ X, X.T @ y)
theta = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)

print("Fitted Intercept (True = 5.0): ", np.round(theta[0], 2))
print("Fitted Slopes    (True = 3, -2):", np.round(theta[1:], 2))
```

---

### 🎯 Case Study 3: Quantitative Finance & Signal Processing

```python
import numpy as np

# Simulated 100-day asset price trajectory
rng = np.random.default_rng(42)
base_price = 150.0
daily_returns = rng.normal(loc=0.0005, scale=0.015, size=100)
price_path = base_price * np.cumprod(1 + daily_returns)

# 1. Moving Average Filter using 1D Discrete Convolution:
# A 20-day simple moving average (SMA)
window_size = 20
weights = np.ones(window_size) / window_size
sma = np.convolve(price_path, weights, mode='valid')

# 2. Bollinger Bands: Moving Average +/- 2 Moving Standard Deviations
rolling_std = np.array([
    np.std(price_path[i:i+window_size]) 
    for i in range(len(sma))
])
upper_band = sma + 2 * rolling_std
lower_band = sma - 2 * rolling_std

# 3. Z-Score Anomaly Detection on Price Volatility:
# Identify days where price broke outside Bollinger Bands:
prices_aligned = price_path[window_size - 1:]
buy_signals = np.where(prices_aligned < lower_band)[0]
sell_signals = np.where(prices_aligned > upper_band)[0]

print(f"Asset Starting Price: ${price_path[0]:.2f}")
print(f"Asset Final Price:    ${price_path[-1]:.2f}")
print(f"Detected {len(buy_signals)} Buy Signals and {len(sell_signals)} Sell Signals.")
```

---

## 21. Common Mistakes and Pitfalls (Beginner to Advanced)

### ❌ Mistake 1: Accidentally Modifying Original Arrays Through Slices (View Bug)

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# WRONG: Slicing creates a view!
sub = arr[1:4]
sub[0] = 9999
print("Original corrupted:", arr)  # [10, 9999, 30, 40, 50]

# CORRECT: Use .copy() when you need isolation!
arr = np.array([10, 20, 30, 40, 50])
sub = arr[1:4].copy()
sub[0] = 9999
print("Original safe:", arr)       # [10, 20, 30, 40, 50]
```

### ❌ Mistake 2: Confusing `axis=0` and `axis=1`

```python
import numpy as np

m = np.array([[1, 2, 3],
              [4, 5, 6]])

# Axis 0 collapses ROWS (down each column):
print("axis=0 (Column sums):", m.sum(axis=0))  # [5, 7, 9]

# Axis 1 collapses COLUMNS (across each row):
print("axis=1 (Row sums):   ", m.sum(axis=1))  # [6, 15]
```

### ❌ Mistake 3: Comparing Floating Point Arrays with `==`

```python
import numpy as np

a = np.array([0.1 + 0.2])
b = np.array([0.3])

# WRONG: IEEE 754 precision causes false negatives!
print("a == b:          ", (a == b).all())   # False!

# CORRECT: Use np.isclose or np.allclose!
print("np.allclose(a, b):", np.allclose(a, b)) # True!
```

### ❌ Mistake 4: Forgetting Parentheses in Boolean Indexing

```python
import numpy as np

arr = np.array([1, 5, 10, 15])

# WRONG: Bitwise & has higher precedence than > and <!
# mask = arr > 2 & arr < 12  # Raises ValueError!

# CORRECT: Enclose each condition in parentheses:
mask = (arr > 2) & (arr < 12)
print("Filtered:", arr[mask])  # [5, 10]
```

### ❌ Mistake 5: Repeated Updates in Fancy Indexing

```python
import numpy as np

counts = np.zeros(3, dtype=int)
idx = [0, 0, 0]

# WRONG: Executes once per unique index!
counts[idx] += 1
print("Incorrect in-place update:", counts[0])  # 1

# CORRECT: Use ufunc.at() for unbuffered accumulation!
counts = np.zeros(3, dtype=int)
np.add.at(counts, idx, 1)
print("Correct in-place update:  ", counts[0])  # 3
```

### ❌ Mistake 6: Silent Integer Overflow in Small Types

```python
import numpy as np

# WRONG: uint8 max is 255; 250 + 20 wraps around to 14!
pixels = np.array([250], dtype=np.uint8)
print("Silent overflow:", pixels + np.uint8(20))  # [14]

# CORRECT: Cast to larger type before arithmetic:
safe_pixels = pixels.astype(np.int32) + 20
print("Safe calculation:", safe_pixels)           # [270]
```

### ❌ Mistake 7: Accidental Type Truncation When Writing Floats to Int Arrays

```python
import numpy as np

int_array = np.array([1, 2, 3])

# WRONG: Assigning float 4.9 truncates silently to 4 without warning!
int_array[0] = 4.9
print("Truncated assignment:", int_array)  # [4, 2, 3]

# CORRECT: Ensure array is float if decimals are required:
float_array = np.array([1, 2, 3], dtype=float)
float_array[0] = 4.9
print("Correct assignment:  ", float_array) # [4.9, 2. , 3. ]
```

### ❌ Mistake 8: Writing Procedural Python Loops Instead of Vectorizing

```python
import numpy as np

data = np.arange(100_000)

# WRONG: Terribly slow Python loop
# result = np.empty_like(data)
# for i in range(len(data)):
#     result[i] = data[i] * 2

# CORRECT: Native vectorized expression
result = data * 2
```

### ❌ Mistake 9: Using Deprecated `np.matrix`

> [!CAUTION]
> `np.matrix` is officially deprecated in NumPy and should never be used in modern code. Always use standard 2D `ndarray`s with the `@` operator for matrix multiplication.

---

## 22. When to Use NumPy and When Not To (Ecosystem Decision Guide)

### 🎯 The Python Numerical Ecosystem Decision Matrix

| Tool | Best Used For | Not Recommended For | Key Advantage |
| :--- | :--- | :--- | :--- |
| **Python Standard Library** | Small data (< 100 items), dynamic configs, arbitrary text | Large matrices, heavy math, big arrays | Built-in, no dependencies |
| **NumPy** | Numerical arrays, linear algebra, custom algorithms, image arrays | Heterogeneous tabular data, missing strings | Raw C speed, minimal memory overhead |
| **Pandas** | Tabular DataFrames, CSV/SQL ETL, missing values, group-by, time series | High-dimensional tensors (3D+), raw SIMD loops | Rich relational operations |
| **Polars** | Multi-threaded big tabular data, faster alternative to Pandas | Non-tabular tensor math, linear algebra | Rust-based, lazy evaluation |
| **SciPy** | Advanced scientific routines: ODE solvers, signal processing, optimization | Basic array storage (built on NumPy) | Specialized scientific algorithms |
| **PyTorch / JAX** | Deep learning, neural networks, automatic differentiation, GPU acceleration | Simple offline scripts not requiring GPU | Autograd and GPU/TPU parallelism |

---

## 23. Quick Reference Cheat Sheet & Final Summary

### 📚 The Complete NumPy Master Cheat Sheet

```python
import numpy as np

# --- CREATION ---
np.array([1, 2, 3])                     # From list
np.zeros((3, 4), dtype=float)           # Zeros matrix
np.ones((2, 3))                         # Ones matrix
np.full((2, 2), 7)                      # Constant value
np.empty((3, 3))                        # Uninitialized memory (fastest)
np.arange(0, 10, 2)                     # Range with step [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                    # Evenly spaced [0.0, 0.25, 0.5, 0.75, 1.0]
np.logspace(-3, 3, 7)                   # Logarithmic scale 10^-3 to 10^3
np.eye(3)                               # 3x3 Identity matrix
np.diag([1, 2, 3])                      # Diagonal matrix
np.meshgrid(x, y)                       # 2D coordinate grids

# --- PROPERTIES ---
arr.shape                               # Dimensions tuple (e.g. (3, 4))
arr.ndim                                # Number of dimensions (e.g. 2)
arr.size                                # Total element count
arr.dtype                               # Data type (e.g. float64, int32)
arr.strides                             # Memory stride byte steps
arr.nbytes                              # Total bytes in memory buffer
arr.flags                               # Memory layout flags (C_CONTIGUOUS, etc.)

# --- INDEXING & SLICING ---
arr[0, 1]                               # Single element (row 0, col 1)
arr[:, 2]                               # Entire column 2
arr[1:4, ::2]                           # Subgrid slice (VIEW)
arr[..., 0]                             # Ellipsis syntax (arbitrary dimensions)
arr[:, np.newaxis]                      # Add new dimension
arr[[0, 2], [1, 3]]                     # Fancy indexing (paired coordinates) (COPY)
arr[np.ix_([0, 2], [1, 3])]             # Open mesh rectangular subgrid (COPY)
arr[arr > 5]                            # Boolean mask filtering (COPY)

# --- OPERATIONS & UFUNCS ---
a + b, a - b, a * b, a / b              # Element-wise arithmetic
a @ b                                   # Matrix multiplication (np.matmul)
np.add(a, b, out=a)                     # In-place addition with zero memory allocation
np.add.reduce(arr)                      # Sum reduction
np.multiply.outer(a, b)                 # Cartesian outer product
np.add.at(arr, indices, 1)              # Unbuffered in-place accumulation

# --- AGGREGATIONS ---
np.sum(arr, axis=0, keepdims=True)      # Sum down columns preserving dimension
np.mean(arr), np.median(arr)            # Mean, Median
np.std(arr), np.var(arr)                # Standard deviation, Variance
np.min(arr), np.max(arr)                # Minimum, Maximum
np.argmin(arr), np.argmax(arr)          # Index of Min, Max
np.nanmean(arr), np.nansum(arr)         # NaN-safe aggregations
np.percentile(arr, 75)                  # 75th percentile
np.clip(arr, 0, 100)                    # Clamp values between 0 and 100
np.isclose(a, b), np.allclose(a, b)     # Floating point comparison

# --- SHAPE MANIPULATIONS ---
arr.reshape(3, -1)                      # Reshape with auto-dimension
arr.ravel()                             # Flatten to 1D (VIEW when possible)
arr.flatten()                           # Flatten to 1D (Always COPY)
arr.T, np.transpose(arr, axes)          # Transpose axes
np.swapaxes(arr, 0, 1)                  # Swap two axes
np.expand_dims(arr, axis=0)             # Insert singleton axis
np.squeeze(arr)                         # Remove all singleton axes
np.repeat(arr, 3)                       # Repeat individual elements
np.tile(arr, 3)                         # Repeat whole array as tile
np.pad(arr, pad_width=1, mode='edge')   # Pad array borders

# --- JOINING & SPLITTING ---
np.concatenate([a, b], axis=0)          # Combine along existing axis
np.stack([a, b], axis=0)                # Combine along NEW axis
np.vstack([a, b]), np.hstack([a, b])    # Vertical / Horizontal stack
np.column_stack([a, b])                 # Stack 1D vectors as columns
np.block([[A, B], [C, D]])              # Assemble block matrix
np.array_split(arr, 3)                  # Split into 3 chunks (safe with uneven sizes)

# --- LINEAR ALGEBRA (np.linalg) ---
np.linalg.inv(A), np.linalg.pinv(A)     # Inverse, Moore-Penrose Pseudoinverse
np.linalg.det(A), np.trace(A)           # Determinant, Trace
np.linalg.matrix_rank(A)                # Matrix Rank
np.linalg.solve(A, b)                   # Solve Ax = b (fast & stable)
np.linalg.lstsq(A, y, rcond=None)       # Least squares regression
np.linalg.eig(A), np.linalg.eigh(A)     # Eigenvalues and Eigenvectors
np.linalg.svd(A)                        # Singular Value Decomposition
np.linalg.norm(A, ord=2)                # Vector / Matrix norm
np.einsum('ij,jk->ik', A, B)            # Einstein summation matrix multiplication

# --- RANDOM (Modern Generator API) ---
rng = np.random.default_rng(seed=42)    # Initialize PCG64 Generator
rng.integers(1, 10, size=5)             # Random integers [1, 10)
rng.random(size=(3, 3))                 # Uniform floats [0, 1)
rng.normal(loc=0, scale=1, size=10)     # Gaussian normal distribution
rng.choice(arr, size=3, replace=False)  # Sampling without replacement
rng.shuffle(arr)                        # In-place array shuffle
rng.permutation(arr)                    # Shuffled copy

# --- FILE I/O ---
np.save('file.npy', arr)                # Save single binary array
arr = np.load('file.npy')               # Load single binary array
np.savez_compressed('arch.npz', a=a)    # Save compressed archive
np.memmap('large.dat', dtype=float,     # Memory map big data directly on disk
          mode='r', shape=(1000, 1000))
```

---

## 🎉 Final Words of Encouragement

You now have a complete, master-level understanding of NumPy: from raw hardware memory buffers, strides, and SIMD registers, all the way to tensor contractions, out-of-core big data, and vectorized machine learning algorithms.

**Next Steps to Continue Your Journey:**
1. **Explore Pandas** for tabular data manipulation and SQL-like analytics.
2. **Explore Matplotlib / Seaborn** for visual data storytelling.
3. **Explore Scikit-Learn** for production machine learning algorithms.
4. **Explore PyTorch or JAX** for deep learning neural networks.

*"In God we trust; all others bring data."* - W. Edwards Deming

Happy coding, and go build something incredible! 🚀💻✨
