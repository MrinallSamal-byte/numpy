# 🎓 NumPy Complete Tutorial: From Absolute Basics to Advanced

**Welcome, Future NumPy Master!** 🚀

This tutorial will take you from zero to hero in NumPy. Think of me as your friendly mentor sitting next to you, explaining everything step-by-step.

---

## 📚 Table of Contents

1. [What is NumPy and Why It Exists](#1-what-is-numpy-and-why-it-exists)
2. [Installing and Importing NumPy](#2-installing-and-importing-numpy)
3. [NumPy Arrays vs Python Lists](#3-numpy-arrays-vs-python-lists)
4. [Creating Arrays](#4-creating-arrays)
5. [Array Properties](#5-array-properties)
6. [Indexing and Slicing](#6-indexing-and-slicing)
7. [Array Operations](#7-array-operations)
8. [Broadcasting](#8-broadcasting)
9. [Mathematical Functions](#9-mathematical-functions)
10. [Reshaping Arrays](#10-reshaping-arrays)
11. [Joining and Splitting Arrays](#11-joining-and-splitting-arrays)
12. [Copy vs View](#12-copy-vs-view)
13. [Boolean Indexing and Filtering](#13-boolean-indexing-and-filtering)
14. [Sorting and Searching](#14-sorting-and-searching)
15. [Random Module](#15-random-module)
16. [NumPy with Real-World Examples](#16-numpy-with-real-world-examples)
17. [Common Mistakes Beginners Make](#17-common-mistakes-beginners-make)
18. [When to Use NumPy and When Not To](#18-when-to-use-numpy-and-when-not-to)

---

## 1. What is NumPy and Why It Exists

### 🤔 WHY

**Real-Life Analogy:**
Imagine you're a cashier at a store. You need to calculate the total price of 1000 items.
- **Python lists:** Like using a basic calculator - you add each item one by one. Slow! 🐢
- **NumPy:** Like using a supercomputer - it processes all items at once. Super fast! ⚡

**Problems NumPy Solves:**
- Python lists are SLOW when working with lots of numbers
- Lists can't do math operations directly on all elements
- Lists waste memory by storing extra information about each element

**Why NumPy is Better:**
- **Speed:** 10-100x faster than Python lists
- **Memory:** Uses 5-10x less memory
- **Convenience:** Write less code to do more work


### ⏰ WHEN

**Use NumPy when:**
- Working with large datasets (100s to millions of numbers)
- Doing mathematical/scientific calculations
- Processing images, audio, or video
- Building machine learning models
- Analyzing data

**Avoid NumPy when:**
- Working with small amounts of data (< 100 items)
- Need to store mixed data types (strings + numbers + objects)
- Simple tasks where Python lists are enough

### 🔧 HOW

**How it works internally:**
NumPy stores data in a continuous block of memory (like a row of lockers all next to each other). This makes accessing and processing data super fast. Python lists are like lockers scattered all over the building - takes longer to access each one.

**Example 1: Speed Comparison**
```python
import time
import numpy as np

# Python list way
python_list = list(range(1000000))
start = time.time()
result_list = [x * 2 for x in python_list]
print(f"Python list time: {time.time() - start:.4f} seconds")

# NumPy way
numpy_array = np.array(range(1000000))
start = time.time()
result_numpy = numpy_array * 2
print(f"NumPy time: {time.time() - start:.4f} seconds")

# Output:
# Python list time: 0.0523 seconds
# NumPy time: 0.0018 seconds  # 29x faster!
```

**Example 2: Memory Usage**
```python
import sys
import numpy as np

# Python list
python_list = list(range(1000))
print(f"Python list size: {sys.getsizeof(python_list)} bytes")

# NumPy array
numpy_array = np.array(range(1000))
print(f"NumPy array size: {numpy_array.nbytes} bytes")

# Output:
# Python list size: 9016 bytes
# NumPy array size: 8000 bytes  # Uses less memory!
```

### 🏋️ Practice Question
**Q:** Why would you use NumPy instead of Python lists when calculating grades for 1000 students?
**A:** NumPy is much faster and uses less memory when working with large numbers. Calculating averages, sorting, or finding highest/lowest grades would be 10-100x faster with NumPy.

---

## 2. Installing and Importing NumPy

### 🤔 WHY

**Why we need to install it:**
NumPy doesn't come pre-installed with Python because:
- Not everyone needs it (keeps Python lightweight)
- It's a specialized library for numerical computing
- You can choose which version to install

**What problem does installation solve:**
Gets the NumPy library onto your computer so you can use it!

### ⏰ WHEN

**Install NumPy when:**
- Starting a new data science project
- Working with numerical data
- Following tutorials that use NumPy
- Building ML models

**Version matters when:**
- Working on a team project (everyone should use same version)
- Using specific features from newer versions
- Following along with tutorials (match their version)

### �� HOW

**How it works:**
`pip` is like an app store for Python. It downloads NumPy from the internet and installs it on your computer.

**Installation Steps:**

**Step 1: Install NumPy**
```bash
# Open your terminal/command prompt and type:
pip install numpy

# For specific version:
pip install numpy==1.24.0

# To upgrade:
pip install --upgrade numpy
```

**Step 2: Verify Installation**
```python
import numpy as np

# Check version
print(np.__version__)
# Output: 1.24.3 (or your version)

# Test if it works
test_array = np.array([1, 2, 3])
print(test_array)
# Output: [1 2 3]
```

**Example 1: Import with Alias**
```python
# Standard way - always use 'np' as alias
import numpy as np

# Now you can use 'np' instead of typing 'numpy' every time
arr = np.array([1, 2, 3, 4, 5])
print(arr)
# Output: [1 2 3 4 5]
```

**Example 2: Check Available Functions**
```python
import numpy as np

# See all functions starting with 'a'
print([x for x in dir(np) if x.startswith('a')][:5])
# Output: ['abs', 'absolute', 'add', 'arange', 'arccos']

# NumPy has 500+ functions!
print(f"NumPy has {len(dir(np))} functions and objects!")
# Output: NumPy has 600+ functions and objects!
```

### 🏋️ Practice Question
**Q:** What does `import numpy as np` do? Why do we use `as np`?
**A:** It imports the NumPy library and gives it a shorter nickname "np" so we don't have to type "numpy" every time. It's like calling your friend "Alex" instead of "Alexander" - saves time!

---

## 3. NumPy Arrays vs Python Lists

### 🤔 WHY

**Real-Life Analogy:**
- **Python List:** Like a shopping bag - can hold anything (fruits, books, toys)
- **NumPy Array:** Like an egg carton - designed for one specific thing, but does it perfectly

**Problems with Python Lists:**
```python
# Python lists - slow math operations
my_list = [1, 2, 3, 4, 5]
# To multiply each by 2, you need a loop:
result = [x * 2 for x in my_list]  # Extra work!

# NumPy arrays - fast and easy
my_array = np.array([1, 2, 3, 4, 5])
result = my_array * 2  # Done! So simple!
```

**Why NumPy is Better for Numbers:**
- Direct math operations on entire array
- Much faster (written in C language)
- Uses less memory
- Has built-in functions for complex operations

### ⏰ WHEN

**Use Python Lists when:**
- Storing mixed data types: `[1, "hello", 3.14, True]`
- Needing to add/remove items frequently
- Small amount of data (< 100 items)
- Don't need mathematical operations

**Use NumPy Arrays when:**
- All data is same type (all numbers)
- Doing mathematical calculations
- Large datasets
- Need speed and efficiency

### 🔧 HOW

**How it works:**
- **Python List:** Each element is stored separately in memory with type information
- **NumPy Array:** All elements stored together in one block, all same type

**Key Differences Table:**

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Data Types | Mixed types ✅ | Single type only |
| Speed | Slower 🐢 | Much faster ⚡ |
| Memory | More memory | Less memory |
| Math Operations | Need loops | Direct operations |
| Size | Can change easily | Fixed size (usually) |

**Example 1: Data Type Difference**
```python
import numpy as np

# Python list - mixed types OK
python_list = [1, "hello", 3.14, True]
print(python_list)
# Output: [1, 'hello', 3.14, True]

# NumPy array - converts everything to same type
numpy_array = np.array([1, "hello", 3.14, True])
print(numpy_array)
# Output: ['1' 'hello' '3.14' 'True']  # All converted to strings!

# NumPy with all numbers - stays as numbers
numpy_numbers = np.array([1, 2, 3, 4, 5])
print(numpy_numbers)
print(f"Data type: {numpy_numbers.dtype}")
# Output: [1 2 3 4 5]
# Data type: int64
```

**Example 2: Mathematical Operations**
```python
import numpy as np

# Python list - need loop
python_list = [1, 2, 3, 4, 5]
doubled_list = [x * 2 for x in python_list]
print(f"Python list doubled: {doubled_list}")
# Output: Python list doubled: [10, 20, 30, 40, 50]

# NumPy array - direct operation
numpy_array = np.array([1, 2, 3, 4, 5])
doubled_array = numpy_array * 2
print(f"NumPy array doubled: {doubled_array}")
# Output: NumPy array doubled: [ 2  4  6  8 10]

# More complex - works element-by-element
result = numpy_array ** 2 + numpy_array * 3
print(f"Complex operation: {result}")
# Output: Complex operation: [ 4 10 18 28 40]
# Calculation: [1² + 1×3, 2² + 2×3, 3² + 3×3, 4² + 4×3, 5² + 5×3]
```

**Example 3: Memory and Speed**
```python
import numpy as np
import sys

# Memory comparison
python_list = [1, 2, 3, 4, 5]
numpy_array = np.array([1, 2, 3, 4, 5])

print(f"List item size: {sys.getsizeof(python_list[0])} bytes")
print(f"Array item size: {numpy_array.itemsize} bytes")
# Output:
# List item size: 28 bytes
# Array item size: 8 bytes  # Much smaller!

# Addition comparison
list1 = [1, 2, 3]
list2 = [4, 5, 6]
# Python lists concatenate
print(list1 + list2)
# Output: [1, 2, 3, 4, 5, 6]

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
# NumPy arrays add element-wise
print(array1 + array2)
# Output: [5 7 9]  # [1+4, 2+5, 3+6]
```

### 🏋️ Practice Question
**Q:** You have 10,000 prices and want to add 10% tax to each. Should you use a Python list or NumPy array? Why?
**A:** Use NumPy array! You can do `prices * 1.10` directly on the entire array (one line, super fast). With a Python list, you'd need a loop to process each price one by one (slower and more code).

---

## 4. Creating Arrays

### 🤔 WHY

**Why we need different ways to create arrays:**
Just like you need different tools to build different things (hammer, screwdriver, saw), you need different functions to create different types of arrays.

**Problems each function solves:**
- `array()`: Convert existing data (list) to NumPy array
- `zeros()`: Need array full of zeros (common starting point)
- `ones()`: Need array full of ones (useful for calculations)
- `arange()`: Need sequence of numbers (like range() but better)
- `linspace()`: Need evenly spaced numbers between two values

### ⏰ WHEN

**When to use each function:**
- **array()**: When you already have data in a list
- **zeros()**: Initializing arrays before filling with data, placeholders
- **ones()**: Creating multipliers, masks, or starting values
- **arange()**: Creating sequences, loops, indices
- **linspace()**: Creating smooth curves, graphs, scientific plots

### 🔧 HOW

### 4.1 Creating Arrays with `array()`

**How it works:** Converts Python list (or tuple) into NumPy array.

**Example 1: 1D Array (One Dimension - Like a Line)**
```python
import numpy as np

# From a list
simple_array = np.array([1, 2, 3, 4, 5])
print(simple_array)
print(f"Type: {type(simple_array)}")
# Output:
# [1 2 3 4 5]
# Type: <class 'numpy.ndarray'>

# From a tuple
tuple_array = np.array((10, 20, 30))
print(tuple_array)
# Output: [10 20 30]

# With floating point numbers
float_array = np.array([1.5, 2.7, 3.9])
print(float_array)
print(f"Data type: {float_array.dtype}")
# Output:
# [1.5 2.7 3.9]
# Data type: float64
```

**Example 2: 2D Array (Two Dimensions - Like a Table/Matrix)**
```python
import numpy as np

# Think of this like a table with rows and columns
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(matrix)
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Real-world: Student grades
# Rows = students, Columns = subjects (Math, Science, English)
grades = np.array([[85, 90, 78],    # Student 1
                   [92, 88, 95],    # Student 2
                   [78, 85, 82]])   # Student 3
print("Student grades:")
print(grades)
print(f"Student 1 grades: {grades[0]}")
print(f"Math grades for all: {grades[:, 0]}")
# Output:
# Student grades:
# [[85 90 78]
#  [92 88 95]
#  [78 85 82]]
# Student 1 grades: [85 90 78]
# Math grades for all: [85 92 78]
```

**Example 3: 3D Array (Three Dimensions - Like a Cube/Stack of Tables)**
```python
import numpy as np

# Think of this as multiple 2D arrays stacked together
# Like multiple pages in a book
cube = np.array([[[1, 2], [3, 4]],
                 [[5, 6], [7, 8]]])
print(cube)
# Output:
# [[[1 2]
#   [3 4]]
#
#  [[5 6]
#   [7 8]]]

print(f"Shape: {cube.shape}")  # (2, 2, 2) = 2 matrices, each 2x2
# Output: Shape: (2, 2, 2)

# Real-world: RGB image (height, width, color channels)
# Let's create a tiny 2x2 pixel image with 3 color channels (R, G, B)
image = np.array([[[255, 0, 0], [0, 255, 0]],      # Row 1: Red, Green
                  [[0, 0, 255], [255, 255, 0]]])    # Row 2: Blue, Yellow
print("Tiny image shape:", image.shape)  # (2, 2, 3)
# Output: Tiny image shape: (2, 2, 3)
```

### 4.2 Creating Arrays with `zeros()`

**How it works:** Creates array filled with zeros. Useful for initializing arrays.

**Example 1: Basic zeros()**
```python
import numpy as np

# 1D array of zeros
zeros_1d = np.zeros(5)
print(zeros_1d)
# Output: [0. 0. 0. 0. 0.]

# 2D array of zeros (3 rows, 4 columns)
zeros_2d = np.zeros((3, 4))
print(zeros_2d)
# Output:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Specify data type
zeros_int = np.zeros(5, dtype=int)
print(zeros_int)
# Output: [0 0 0 0 0]
```

**Example 2: Real-world Use**
```python
import numpy as np

# Initialize score tracker for 10 players
player_scores = np.zeros(10)
print("Initial scores:", player_scores)
# Output: Initial scores: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# Now you can update scores as game progresses
player_scores[0] = 10
player_scores[1] = 15
print("Updated scores:", player_scores)
# Output: Updated scores: [10. 15.  0.  0.  0.  0.  0.  0.  0.  0.]

# Create attendance sheet for 5 days, 30 students
attendance = np.zeros((30, 5))
print(f"Attendance sheet shape: {attendance.shape}")
# Output: Attendance sheet shape: (30, 5)
# Each row = student, Each column = day
# 0 = absent, 1 = present (you'll fill this in)
```

### 4.3 Creating Arrays with `ones()`

**How it works:** Creates array filled with ones.

**Example 1: Basic ones()**
```python
import numpy as np

# 1D array of ones
ones_1d = np.ones(4)
print(ones_1d)
# Output: [1. 1. 1. 1.]

# 2D array of ones
ones_2d = np.ones((2, 3))
print(ones_2d)
# Output:
# [[1. 1. 1.]
#  [1. 1. 1.]]

# With integer type
ones_int = np.ones(6, dtype=int)
print(ones_int)
# Output: [1 1 1 1 1 1]
```

**Example 2: Real-world Use**
```python
import numpy as np

# Create prices array
prices = np.array([100, 200, 150, 300])
print("Original prices:", prices)

# Apply 10% discount to all items
# Create multiplier array (0.9 = 90% = 10% off)
discount = np.ones(4) * 0.9
discounted_prices = prices * discount
print("Discounted prices:", discounted_prices)
# Output:
# Original prices: [100 200 150 300]
# Discounted prices: [ 90. 180. 135. 270.]

# Or even simpler:
all_items_available = np.ones(50, dtype=bool)
print(f"All 50 items available: {all_items_available.sum()} items")
# Output: All 50 items available: 50 items
```

### 4.4 Creating Arrays with `arange()`

**How it works:** Like Python's `range()` but returns NumPy array and works with floats!

**Syntax:** `np.arange(start, stop, step)`

**Example 1: Basic arange()**
```python
import numpy as np

# Just stop value (starts from 0)
arr1 = np.arange(10)
print(arr1)
# Output: [0 1 2 3 4 5 6 7 8 9]

# Start and stop
arr2 = np.arange(5, 15)
print(arr2)
# Output: [ 5  6  7  8  9 10 11 12 13 14]

# Start, stop, and step
arr3 = np.arange(0, 20, 3)
print(arr3)
# Output: [ 0  3  6  9 12 15 18]

# Works with floats!
arr4 = np.arange(0, 1, 0.2)
print(arr4)
# Output: [0.  0.2 0.4 0.6 0.8]

# Countdown
arr5 = np.arange(10, 0, -1)
print(arr5)
# Output: [10  9  8  7  6  5  4  3  2  1]
```

**Example 2: Real-world Use**
```python
import numpy as np

# Create indices for 100 items
indices = np.arange(100)
print(f"First 10 indices: {indices[:10]}")
# Output: First 10 indices: [0 1 2 3 4 5 6 7 8 9]

# Create time series (every 0.5 seconds for 10 seconds)
time_seconds = np.arange(0, 10, 0.5)
print(f"Time points: {time_seconds}")
# Output: Time points: [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5 7.  7.5 8.  8.5 9.  9.5]

# Create even numbers from 0 to 100
even_numbers = np.arange(0, 101, 2)
print(f"Even numbers: {even_numbers}")
# Output: Even numbers: [  0   2   4   6 ... 96  98 100]
```

### 4.5 Creating Arrays with `linspace()`

**How it works:** Creates array with evenly spaced numbers between two values. Unlike `arange()`, you specify how many numbers you want, not the step size.

**Syntax:** `np.linspace(start, stop, num)`

**Example 1: Basic linspace()**
```python
import numpy as np

# 5 numbers between 0 and 10 (includes both ends!)
arr1 = np.linspace(0, 10, 5)
print(arr1)
# Output: [ 0.   2.5  5.   7.5 10. ]

# 10 numbers between 0 and 1
arr2 = np.linspace(0, 1, 10)
print(arr2)
# Output: [0.    0.111 0.222 0.333 0.444 0.556 0.667 0.778 0.889 1.   ]

# Exclude the endpoint
arr3 = np.linspace(0, 10, 5, endpoint=False)
print(arr3)
# Output: [0. 2. 4. 6. 8.]  # 10 is not included
```

**Example 2: Real-world Use**
```python
import numpy as np

# Creating smooth curve points for plotting
# Want to plot a graph from x=0 to x=10 with smooth line
x_values = np.linspace(0, 10, 50)  # 50 points for smooth curve
print(f"First 5 x-values: {x_values[:5]}")
print(f"Last 5 x-values: {x_values[-5:]}")
# Output:
# First 5 x-values: [0.     0.204  0.408  0.612  0.816]
# Last 5 x-values: [9.184 9.388 9.592 9.796 10.   ]

# Creating percentage points
percentages = np.linspace(0, 100, 11)  # 0%, 10%, 20%, ..., 100%
print(f"Percentages: {percentages}")
# Output: Percentages: [  0.  10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]
```

### 🎯 Comparison: arange() vs linspace()

```python
import numpy as np

# arange: You specify the STEP
arr1 = np.arange(0, 10, 2)
print(f"arange with step=2: {arr1}")
# Output: arange with step=2: [0 2 4 6 8]
# You don't know how many numbers you'll get

# linspace: You specify HOW MANY numbers
arr2 = np.linspace(0, 10, 5)
print(f"linspace with 5 numbers: {arr2}")
# Output: linspace with 5 numbers: [ 0.   2.5  5.   7.5 10. ]
# You get exactly 5 numbers
```

### 🏋️ Practice Question
**Q:** You need to create 100 evenly spaced points between 0 and 2π (for plotting a sine wave). Should you use `arange()` or `linspace()`? Write the code.
**A:** Use `linspace()` because you know exactly how many points you want (100):
```python
import numpy as np
x = np.linspace(0, 2*np.pi, 100)
```
With `arange()`, you'd have to calculate the step size yourself: `np.arange(0, 2*np.pi, 2*np.pi/100)` - more complicated!

---

## 5. Array Properties

### 🤔 WHY

**Why we need to know array properties:**
Before working with data, you need to know:
- What shape is it? (dimensions)
- How many elements?
- What type of data?

**Real-Life Analogy:**
It's like checking a box before opening it:
- How big is the box? (shape)
- Is it flat, cube, or stack? (dimensions)  
- What's inside? (data type)

### ⏰ WHEN

**Use these properties when:**
- Debugging code (check if array is correct shape)
- Before performing operations (ensure compatibility)
- Reading data from files (verify what you loaded)
- Reshaping or transforming data

### 🔧 HOW

**Key Properties:**
1. **shape**: The dimensions (rows, columns, etc.)
2. **ndim**: Number of dimensions
3. **size**: Total number of elements
4. **dtype**: Data type of elements

**Example 1: Understanding All Properties**
```python
import numpy as np

# Create different arrays
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 1D Array Properties
print("=== 1D Array ===")
print(f"Array: {arr_1d}")
print(f"Shape: {arr_1d.shape}")      # (5,) means 1D with 5 elements
print(f"Dimensions: {arr_1d.ndim}")  # 1
print(f"Size: {arr_1d.size}")        # 5 elements total
print(f"Data type: {arr_1d.dtype}")  # int64
print(f"Item size: {arr_1d.itemsize} bytes")  # 8 bytes per element

# Output:
# === 1D Array ===
# Array: [1 2 3 4 5]
# Shape: (5,)
# Dimensions: 1
# Size: 5
# Data type: int64
# Item size: 8 bytes

# 2D Array Properties
print("\n=== 2D Array ===")
print(f"Array:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")      # (2, 3) = 2 rows, 3 columns
print(f"Dimensions: {arr_2d.ndim}")  # 2
print(f"Size: {arr_2d.size}")        # 6 elements total (2 × 3)
print(f"Data type: {arr_2d.dtype}")  # int64

# Output:
# === 2D Array ===
# Array:
# [[1 2 3]
#  [4 5 6]]
# Shape: (2, 3)
# Dimensions: 2
# Size: 6
# Data type: int64

# 3D Array Properties
print("\n=== 3D Array ===")
print(f"Shape: {arr_3d.shape}")      # (2, 2, 2) = 2 matrices of 2×2
print(f"Dimensions: {arr_3d.ndim}")  # 3
print(f"Size: {arr_3d.size}")        # 8 elements total (2 × 2 × 2)

# Output:
# === 3D Array ===
# Shape: (2, 2, 2)
# Dimensions: 3
# Size: 8
```

**Example 2: Data Types**
```python
import numpy as np

# Different data types
int_array = np.array([1, 2, 3])
float_array = np.array([1.5, 2.7, 3.9])
string_array = np.array(['a', 'b', 'c'])
bool_array = np.array([True, False, True])

print(f"Integer array dtype: {int_array.dtype}")    # int64
print(f"Float array dtype: {float_array.dtype}")    # float64
print(f"String array dtype: {string_array.dtype}")  # <U1 (Unicode string)
print(f"Boolean array dtype: {bool_array.dtype}")   # bool

# You can specify dtype when creating
custom_array = np.array([1, 2, 3], dtype=np.float32)
print(f"Custom dtype: {custom_array}")
print(f"Type: {custom_array.dtype}")
# Output:
# Custom dtype: [1. 2. 3.]
# Type: float32

# Memory calculation
print(f"\nMemory usage: {int_array.nbytes} bytes")  # size × itemsize
# Output: Memory usage: 24 bytes (3 elements × 8 bytes each)
```

**Example 3: Real-World Use**
```python
import numpy as np

# Load student grades (imagine this came from a file)
grades = np.array([[85, 90, 78, 92],  # Student 1: Math, Sci, Eng, Hist
                   [92, 88, 95, 89],  # Student 2
                   [78, 85, 82, 91],  # Student 3
                   [95, 92, 88, 94]]) # Student 4

print("Student Grades System")
print(f"Number of students: {grades.shape[0]}")     # 4
print(f"Number of subjects: {grades.shape[1]}")     # 4
print(f"Total grades recorded: {grades.size}")      # 16
print(f"Array dimensions: {grades.ndim}D array")    # 2D

# Output:
# Student Grades System
# Number of students: 4
# Number of subjects: 4
# Total grades recorded: 16
# Array dimensions: 2D array

# Verify data before processing
if grades.shape[1] == 4:
    print("✓ All students have 4 subject grades")
else:
    print("✗ Error: Missing grades!")
```

### 🏋️ Practice Question
**Q:** You have an array with shape (10, 5, 3). What does this represent? How many total elements?
**A:** It's a 3D array with:
- 10 "pages" (first dimension)
- 5 rows per page (second dimension)
- 3 columns per row (third dimension)
- Total elements: 10 × 5 × 3 = 150 elements

---

## 6. Indexing and Slicing

### 🤔 WHY

**Why we need indexing and slicing:**
- Access specific elements (like finding a book on a specific shelf)
- Extract portions of data (like copying specific pages)
- Modify specific elements (like editing specific cells in Excel)

**Real-Life Analogy:**
Think of an array like apartment building:
- **Indexing**: Finding a specific apartment (e.g., Floor 3, Room 5)
- **Slicing**: Getting all apartments on Floor 2

### ⏰ WHEN

**Use indexing when:**
- Need one specific element
- Updating a single value
- Checking a specific data point

**Use slicing when:**
- Need multiple elements
- Extracting a portion of data
- Getting rows or columns from a table

### 🔧 HOW

**How it works:**
- **Indexing**: Uses `[row, column]` notation
- **Slicing**: Uses `[start:stop:step]` notation
- **Python starts counting from 0!**

### 6.1 1D Array Indexing

**Example 1: Basic 1D Indexing**
```python
import numpy as np

numbers = np.array([10, 20, 30, 40, 50])

# Access elements (0-based indexing)
print(f"First element: {numbers[0]}")     # 10
print(f"Third element: {numbers[2]}")     # 30
print(f"Last element: {numbers[-1]}")     # 50 (negative index from end)
print(f"Second-last: {numbers[-2]}")      # 40

# Modify elements
numbers[0] = 100
print(f"After modification: {numbers}")
# Output: After modification: [100  20  30  40  50]
```

**Example 2: 1D Array Slicing**
```python
import numpy as np

numbers = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

# Slicing syntax: array[start:stop:step]
print(f"First 3 elements: {numbers[:3]}")        # [10 20 30]
print(f"Elements 2-5: {numbers[2:6]}")          # [30 40 50 60]
print(f"Last 3 elements: {numbers[-3:]}")       # [70 80 90]
print(f"Every 2nd element: {numbers[::2]}")     # [10 30 50 70 90]
print(f"Reverse array: {numbers[::-1]}")        # [90 80 70 60 50 40 30 20 10]
print(f"Middle elements: {numbers[3:7]}")       # [40 50 60 70]
```

### 6.2 2D Array Indexing

**Example 1: Basic 2D Indexing**
```python
import numpy as np

# Create a 2D array (like a spreadsheet)
matrix = np.array([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])

print("Matrix:")
print(matrix)

# Access single element [row, column]
print(f"\nElement at row 0, col 0: {matrix[0, 0]}")    # 10
print(f"Element at row 1, col 2: {matrix[1, 2]}")      # 60
print(f"Element at row 2, col 1: {matrix[2, 1]}")      # 80

# Access entire row
print(f"\nFirst row: {matrix[0]}")        # [10 20 30]
print(f"Last row: {matrix[-1]}")          # [70 80 90]

# Access entire column (need : for rows, then column number)
print(f"\nFirst column: {matrix[:, 0]}")   # [10 40 70]
print(f"Second column: {matrix[:, 1]}")    # [20 50 80]
print(f"Last column: {matrix[:, -1]}")     # [30 60 90]

# Modify elements
matrix[0, 0] = 100
print(f"\nAfter modifying [0,0]:\n{matrix}")
```

**Example 2: 2D Array Slicing**
```python
import numpy as np

# Student grades: 5 students × 4 subjects
grades = np.array([[85, 90, 78, 92],  # Student 0
                   [92, 88, 95, 89],  # Student 1
                   [78, 85, 82, 91],  # Student 2
                   [95, 92, 88, 94],  # Student 3
                   [88, 91, 86, 90]]) # Student 4

print("All grades:")
print(grades)

# Get first 3 students (rows 0-2)
print(f"\nFirst 3 students:\n{grades[:3]}")

# Get last 2 subjects (columns 2-3)
print(f"\nLast 2 subjects for all students:\n{grades[:, 2:]}")

# Get specific subset: first 2 students, first 3 subjects
print(f"\nFirst 2 students, first 3 subjects:\n{grades[:2, :3]}")

# Get every other student
print(f"\nEvery other student:\n{grades[::2]}")

# Real example: Get math grades (column 0) for all students
math_grades = grades[:, 0]
print(f"\nAll Math grades: {math_grades}")
# Output: All Math grades: [85 92 78 95 88]
```

**Example 3: Advanced Slicing**
```python
import numpy as np

# Image-like data: 5×5 matrix
image = np.array([[1,  2,  3,  4,  5],
                  [6,  7,  8,  9,  10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25]])

print("Original image:")
print(image)

# Extract center 3×3 portion
center = image[1:4, 1:4]
print(f"\nCenter 3×3:\n{center}")
# Output:
# [[ 7  8  9]
#  [12 13 14]
#  [17 18 19]]

# Extract corners
top_left = image[:2, :2]
bottom_right = image[-2:, -2:]
print(f"\nTop-left corner:\n{top_left}")
print(f"\nBottom-right corner:\n{bottom_right}")

# Extract border elements (first/last rows and columns)
border_top = image[0, :]
border_bottom = image[-1, :]
border_left = image[:, 0]
border_right = image[:, -1]
print(f"\nBorder elements:")
print(f"Top: {border_top}")
print(f"Bottom: {border_bottom}")
print(f"Left: {border_left}")
print(f"Right: {border_right}")
```

### 🏋️ Practice Question
**Q:** Given array `arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])`, write code to:
1. Get element at row 1, column 2
2. Get the entire second row
3. Get the last column

**A:**
```python
# 1. Element at row 1, column 2
print(arr[1, 2])  # Output: 7

# 2. Entire second row
print(arr[1])  # Output: [5 6 7 8]

# 3. Last column
print(arr[:, -1])  # Output: [ 4  8 12]
```

---

## 7. Array Operations

### 🤔 WHY

**Why array operations are powerful:**
In regular Python, to add two lists element-by-element, you need a loop:
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
result = [a + b for a, b in zip(list1, list2)]  # Complex!
```

With NumPy, it's super simple:
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = arr1 + arr2  # Done! That's it!
```

**Problems it solves:**
- No need for loops (faster and cleaner code)
- Operations work element-by-element automatically
- Can mix arrays with single numbers (broadcasting)

### ⏰ WHEN

**Use array operations when:**
- Doing calculations on entire datasets
- Applying same operation to all elements
- Combining multiple arrays
- Scientific/financial calculations

**Examples:**
- Calculating total sales with tax
- Converting temperatures
- Normalizing data
- Applying discounts to prices

### 🔧 HOW

**How it works:**
Operations are applied element-by-element (element-wise). Arrays must have compatible shapes.

### 7.1 Arithmetic Operations

**Example 1: Basic Arithmetic**
```python
import numpy as np

arr1 = np.array([10, 20, 30, 40])
arr2 = np.array([1, 2, 3, 4])

# Addition
result_add = arr1 + arr2
print(f"Addition: {result_add}")
# Output: Addition: [11 22 33 44]  # [10+1, 20+2, 30+3, 40+4]

# Subtraction
result_sub = arr1 - arr2
print(f"Subtraction: {result_sub}")
# Output: Subtraction: [9 18 27 36]

# Multiplication (element-wise!)
result_mul = arr1 * arr2
print(f"Multiplication: {result_mul}")
# Output: Multiplication: [ 10  40  90 160]  # [10×1, 20×2, 30×3, 40×4]

# Division
result_div = arr1 / arr2
print(f"Division: {result_div}")
# Output: Division: [10. 10. 10. 10.]

# Integer division
result_floor = arr1 // arr2
print(f"Floor division: {result_floor}")
# Output: Floor division: [10 10 10 10]

# Modulus (remainder)
result_mod = arr1 % arr2
print(f"Modulus: {result_mod}")
# Output: Modulus: [0 0 0 0]

# Power
result_pow = arr2 ** 2
print(f"Power: {result_pow}")
# Output: Power: [ 1  4  9 16]  # [1², 2², 3², 4²]
```

**Example 2: Operations with Scalars**
```python
import numpy as np

prices = np.array([100, 200, 150, 300])

# Add fixed amount to all prices
increased = prices + 50
print(f"Prices + 50: {increased}")
# Output: Prices + 50: [150 250 200 350]

# Apply 10% discount (multiply by 0.9)
discounted = prices * 0.9
print(f"10% off: {discounted}")
# Output: 10% off: [ 90. 180. 135. 270.]

# Add 5% tax (multiply by 1.05)
with_tax = prices * 1.05
print(f"With 5% tax: {with_tax}")
# Output: With 5% tax: [105. 210. 157.5 315. ]

# Divide all by 10
divided = prices / 10
print(f"Divided by 10: {divided}")
# Output: Divided by 10: [10. 20. 15. 30.]
```

**Example 3: Combined Operations**
```python
import numpy as np

# Temperature conversion: Celsius to Fahrenheit
# Formula: F = (C × 9/5) + 32
celsius = np.array([0, 10, 20, 30, 40])
fahrenheit = (celsius * 9/5) + 32
print(f"Celsius: {celsius}")
print(f"Fahrenheit: {fahrenheit}")
# Output:
# Celsius: [ 0 10 20 30 40]
# Fahrenheit: [ 32.  50.  68.  86. 104.]

# Calculate final price: base + tax - discount
base_prices = np.array([100, 200, 300])
tax_rate = 0.08  # 8% tax
discount = np.array([10, 20, 30])  # Different discount per item

final_prices = (base_prices * (1 + tax_rate)) - discount
print(f"Final prices: {final_prices}")
# Output: Final prices: [ 98. 196. 294.]  # [100×1.08-10, 200×1.08-20, 300×1.08-30]
```

### 7.2 Comparison Operations

**Example 1: Comparison Operators**
```python
import numpy as np

grades = np.array([85, 92, 78, 95, 88])

# Compare with scalar
passed = grades >= 80
print(f"Grades >= 80: {passed}")
# Output: Grades >= 80: [ True  True False  True  True]

excellent = grades >= 90
print(f"Excellent (>=90): {excellent}")
# Output: Excellent (>=90): [False  True False  True False]

# Count how many passed
print(f"Number of students passed: {passed.sum()}")
# Output: Number of students passed: 4

# Compare arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([1, 4, 3, 2, 5])

print(f"arr1 == arr2: {arr1 == arr2}")
# Output: arr1 == arr2: [ True False  True False  True]

print(f"arr1 > arr2: {arr1 > arr2}")
# Output: arr1 > arr2: [False False False  True False]
```

**Example 2: Logical Operations**
```python
import numpy as np

scores = np.array([85, 92, 78, 95, 88, 65, 91])

# Multiple conditions using & (and), | (or)
good_students = (scores >= 80) & (scores < 90)
print(f"Scores between 80-90: {good_students}")
# Output: Scores between 80-90: [ True False False False  True False False]

# Extract actual values
print(f"Students with 80-90 scores: {scores[good_students]}")
# Output: Students with 80-90 scores: [85 88]

# Either very low or very high
extreme = (scores < 70) | (scores >= 95)
print(f"Extreme scores (<70 or >=95): {scores[extreme]}")
# Output: Extreme scores (<70 or >=95): [95 65]
```

### 🏋️ Practice Question
**Q:** You have prices `np.array([50, 100, 75, 120])`. Write code to:
1. Add 10% tax
2. Find which prices are over 80 (after tax)
3. Apply extra 5% discount to those over 80

**A:**
```python
import numpy as np

prices = np.array([50, 100, 75, 120])

# 1. Add 10% tax
prices_with_tax = prices * 1.10
print(f"With tax: {prices_with_tax}")  # [55. 110. 82.5 132.]

# 2. Find which are over 80
over_80 = prices_with_tax > 80
print(f"Over 80: {over_80}")  # [False True True True]

# 3. Apply 5% discount to those over 80
prices_with_tax[over_80] *= 0.95
print(f"Final prices: {prices_with_tax}")  # [55. 104.5 78.375 125.4]
```

---

## 8. Broadcasting

### 🤔 WHY

**Why broadcasting is magical:**
Broadcasting lets you do operations on arrays of different shapes automatically!

**Real-Life Analogy:**
Imagine you have 10 students and want to add 5 bonus points to everyone:
- **Without broadcasting:** Make a list `[5, 5, 5, 5, 5, 5, 5, 5, 5, 5]`, then add
- **With broadcasting:** Just do `grades + 5` - NumPy duplicates the 5 automatically!

**Problem it solves:**
You don't need to manually resize arrays to match shapes. NumPy does it for you!

### ⏰ WHEN

**Broadcasting happens when:**
- Operating on arrays of different shapes
- Adding scalar to array
- Operating array with smaller array

**Real examples:**
- Adding same tax rate to all prices
- Normalizing entire dataset by one number
- Applying same transformation to all rows

### 🔧 HOW

**How broadcasting works:**
NumPy automatically "stretches" the smaller array to match the larger one (virtually, not in memory).

**Broadcasting Rules:**
1. If arrays have different dimensions, pad the smaller one with 1s on the left
2. Arrays are compatible if dimensions are equal OR one of them is 1
3. After broadcasting, each dimension is the maximum of the two

**Example 1: Scalar Broadcasting (Most Common)**
```python
import numpy as np

# Array + Scalar
arr = np.array([1, 2, 3, 4, 5])
result = arr + 10
print(f"Array: {arr}")
print(f"Array + 10: {result}")
# Output:
# Array: [1 2 3 4 5]
# Array + 10: [11 12 13 14 15]

# What NumPy does internally (conceptually):
# 10 becomes [10, 10, 10, 10, 10]
# Then adds: [1, 2, 3, 4, 5] + [10, 10, 10, 10, 10]

# Works with all operations
arr = np.array([10, 20, 30])
print(f"Multiply by 2: {arr * 2}")     # [20 40 60]
print(f"Divide by 5: {arr / 5}")       # [2. 4. 6.]
print(f"Power of 2: {arr ** 2}")       # [100 400 900]
```

**Example 2: 1D Array Broadcasting**
```python
import numpy as np

# Adding 1D array to each row of 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

row_to_add = np.array([10, 20, 30])

result = matrix + row_to_add
print("Original matrix:")
print(matrix)
print(f"\nRow to add: {row_to_add}")
print(f"\nResult:\n{result}")
# Output:
# Original matrix:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
#
# Row to add: [10 20 30]
#
# Result:
# [[11 22 33]  # [1, 2, 3] + [10, 20, 30]
#  [14 25 36]  # [4, 5, 6] + [10, 20, 30]
#  [17 28 39]] # [7, 8, 9] + [10, 20, 30]

# The row [10, 20, 30] is added to EACH row of the matrix!
```

**Example 3: Real-World Example - Price Calculations**
```python
import numpy as np

# Product prices (3 products)
prices = np.array([[100],   # Product A
                   [200],   # Product B  
                   [150]])  # Product C

# Discount rates for different customer types (3 types)
discount_rates = np.array([0.9, 0.85, 0.8])  # 10%, 15%, 20% off

# Broadcasting creates a 3×3 matrix of final prices
final_prices = prices * discount_rates

print("Prices (3 products):")
print(prices)
print(f"\nDiscount rates (3 customer types): {discount_rates}")
print("\nFinal prices (products × customer types):")
print(final_prices)
# Output:
# Prices (3 products):
# [[100]
#  [200]
#  [150]]
#
# Discount rates (3 customer types): [0.9  0.85 0.8 ]
#
# Final prices (products × customer types):
# [[ 90.  85.  80.]   # Product A for each customer type
#  [180. 170. 160.]   # Product B for each customer type
#  [135. 127.5 120.]] # Product C for each customer type
```

**Example 4: Standardizing Data (Common in ML)**
```python
import numpy as np

# Student test scores (4 students × 3 tests)
scores = np.array([[85, 90, 78],
                   [92, 88, 95],
                   [78, 85, 82],
                   [95, 92, 88]])

print("Original scores:")
print(scores)

# Calculate mean for each test (column-wise)
mean_scores = scores.mean(axis=0)  # Average across students
print(f"\nMean scores per test: {mean_scores}")
# Output: Mean scores per test: [87.5  88.75 85.75]

# Calculate standard deviation for each test
std_scores = scores.std(axis=0)
print(f"Std deviation per test: {std_scores}")
# Output: Std deviation per test: [6.85  2.59  5.91]

# Standardize: (score - mean) / std
# Broadcasting automatically applies to each column
standardized = (scores - mean_scores) / std_scores
print(f"\nStandardized scores:\n{standardized}")
# Each column now has mean=0, std=1
```

**Example 5: Broadcasting Visualization**
```python
import numpy as np

# Example: Shape (3, 1) + Shape (4,)
a = np.array([[1],
              [2],
              [3]])  # Shape: (3, 1)

b = np.array([10, 20, 30, 40])  # Shape: (4,)

result = a + b
print(f"Shape of a: {a.shape}")      # (3, 1)
print(f"Shape of b: {b.shape}")      # (4,)
print(f"Shape of result: {result.shape}")  # (3, 4)
print(f"\nResult:\n{result}")
# Output:
# Shape of a: (3, 1)
# Shape of b: (4,)
# Shape of result: (3, 4)
#
# Result:
# [[11 21 31 41]  # 1 + [10, 20, 30, 40]
#  [12 22 32 42]  # 2 + [10, 20, 30, 40]
#  [13 23 33 43]] # 3 + [10, 20, 30, 40]
```

### 🏋️ Practice Question
**Q:** You have monthly sales for 12 months: `sales = np.array([100, 120, 110, ..., 150])` and want to calculate quarterly totals (Q1=Jan+Feb+Mar, etc.). How would broadcasting help?

**A:** You can reshape sales to (4, 3) for 4 quarters × 3 months, then sum along axis 1:
```python
sales = np.array([100, 120, 110, 130, 140, 135, 145, 150, 155, 160, 165, 150])
quarterly = sales.reshape(4, 3)
quarterly_totals = quarterly.sum(axis=1)
print(quarterly_totals)  # [330, 405, 450, 475] for Q1, Q2, Q3, Q4
```

---

## 9. Mathematical Functions

### 🤔 WHY

**Why built-in math functions:**
NumPy provides optimized functions for common mathematical operations. They're:
- **Faster** than writing your own loops
- **More accurate** (better algorithms)
- **More convenient** (less code)

**Real-Life Analogy:**
It's like using a calculator app vs doing math by hand - faster, easier, less error-prone!

### ⏰ WHEN

**Use these functions for:**
- Statistical analysis (mean, median, std)
- Data summarization (sum, min, max)
- Data science and ML (normalization, standardization)
- Financial calculations (total, average, variance)

### 🔧 HOW

**Categories of functions:**
1. **Aggregation**: sum, mean, median
2. **Statistics**: std, var, percentile
3. **Extremes**: min, max, argmin, argmax
4. **Other**: round, abs, sqrt, exp, log

### 9.1 Sum, Mean, and Basic Stats

**Example 1: Basic Aggregation**
```python
import numpy as np

scores = np.array([85, 92, 78, 95, 88, 91, 83])

# Sum - total of all elements
total = np.sum(scores)
print(f"Total score: {total}")
# Output: Total score: 612

# Mean - average
average = np.mean(scores)
print(f"Average score: {average:.2f}")
# Output: Average score: 87.43

# Median - middle value (when sorted)
median = np.median(scores)
print(f"Median score: {median}")
# Output: Median score: 88.0

# Alternative: use methods
print(f"Sum (method): {scores.sum()}")
print(f"Mean (method): {scores.mean():.2f}")
```

**Example 2: Min, Max, and Indices**
```python
import numpy as np

temperatures = np.array([72, 68, 75, 80, 65, 70, 78])

# Minimum and maximum values
min_temp = np.min(temperatures)
max_temp = np.max(temperatures)
print(f"Min temperature: {min_temp}°F")
print(f"Max temperature: {max_temp}°F")
# Output:
# Min temperature: 65°F
# Max temperature: 80°F

# Find INDEX of min/max (useful for finding which day)
min_day = np.argmin(temperatures)
max_day = np.argmax(temperatures)
print(f"Coldest day: Day {min_day} with {temperatures[min_day]}°F")
print(f"Hottest day: Day {max_day} with {temperatures[max_day]}°F")
# Output:
# Coldest day: Day 4 with 65°F
# Hottest day: Day 3 with 80°F

# Range (difference between max and min)
temp_range = np.ptp(temperatures)  # peak-to-peak
print(f"Temperature range: {temp_range}°F")
# Output: Temperature range: 15°F
```

**Example 3: Standard Deviation and Variance**
```python
import numpy as np

# Test scores from two classes
class_a = np.array([85, 87, 86, 85, 87, 86, 85])  # Consistent
class_b = np.array([70, 95, 75, 90, 80, 85, 95])  # Varied

# Standard deviation - measures "spread" of data
# Low std = data points close to mean
# High std = data points spread out
std_a = np.std(class_a)
std_b = np.std(class_b)

print(f"Class A - Mean: {np.mean(class_a):.2f}, Std Dev: {std_a:.2f}")
print(f"Class B - Mean: {np.mean(class_b):.2f}, Std Dev: {std_b:.2f}")
# Output:
# Class A - Mean: 85.86, Std Dev: 0.83  # Very consistent!
# Class B - Mean: 84.29, Std Dev: 9.05  # More variation!

# Variance - square of standard deviation
var_a = np.var(class_a)
var_b = np.var(class_b)
print(f"\nClass A variance: {var_a:.2f}")
print(f"Class B variance: {var_b:.2f}")
# Output:
# Class A variance: 0.69
# Class B variance: 81.92
```

**Example 4: Axis-wise Operations (2D Arrays)**
```python
import numpy as np

# Sales data: 3 products × 4 quarters
sales = np.array([[100, 120, 110, 130],  # Product A
                  [200, 210, 205, 215],  # Product B
                  [150, 155, 160, 165]]) # Product C

print("Sales data (products × quarters):")
print(sales)

# Total sales per product (sum across columns)
product_totals = np.sum(sales, axis=1)
print(f"\nTotal sales per product: {product_totals}")
# Output: Total sales per product: [460 830 630]

# Total sales per quarter (sum across rows)
quarter_totals = np.sum(sales, axis=0)
print(f"Total sales per quarter: {quarter_totals}")
# Output: Total sales per quarter: [450 485 475 510]

# Average sales per quarter for each product
product_averages = np.mean(sales, axis=1)
print(f"Average quarterly sales per product: {product_averages}")
# Output: Average quarterly sales per product: [115.  207.5 157.5]

# Best performing quarter (max sales per quarter)
best_quarter = np.argmax(quarter_totals)
print(f"Best quarter: Q{best_quarter + 1} with {quarter_totals[best_quarter]} sales")
# Output: Best quarter: Q4 with 510 sales
```

**Example 5: Percentiles**
```python
import numpy as np

# Student ages in a class
ages = np.array([18, 19, 18, 20, 19, 18, 21, 19, 20, 18, 22, 19, 20, 18, 19])

# Percentiles - value below which a percentage of data falls
p25 = np.percentile(ages, 25)  # 25th percentile (Q1)
p50 = np.percentile(ages, 50)  # 50th percentile (median)
p75 = np.percentile(ages, 75)  # 75th percentile (Q3)

print(f"25th percentile: {p25}")
print(f"50th percentile (median): {p50}")
print(f"75th percentile: {p75}")
# Output:
# 25th percentile: 18.0
# 50th percentile (median): 19.0
# 75th percentile: 20.0

# This means: 25% of students are 18 or younger
#             50% of students are 19 or younger
#             75% of students are 20 or younger
```

**Example 6: Cumulative Operations**
```python
import numpy as np

# Monthly savings
monthly_savings = np.array([100, 150, 120, 180, 200])

# Cumulative sum - running total
cumulative_total = np.cumsum(monthly_savings)
print(f"Monthly savings: {monthly_savings}")
print(f"Cumulative total: {cumulative_total}")
# Output:
# Monthly savings: [100 150 120 180 200]
# Cumulative total: [100 250 370 550 750]

# Cumulative product
investments = np.array([1.05, 1.03, 1.07, 1.02])  # Growth factors
cumulative_growth = np.cumprod(investments)
print(f"\nGrowth factors: {investments}")
print(f"Cumulative growth: {cumulative_growth}")
# Output:
# Growth factors: [1.05 1.03 1.07 1.02]
# Cumulative growth: [1.05   1.0815 1.157205 1.180349]
# Meaning: After 4 periods, you have 1.18x your original investment
```

### 🏋️ Practice Question
**Q:** Given daily temperatures `temps = np.array([72, 68, 75, 80, 65, 70, 78, 82, 69, 73])`:
1. Find average temperature
2. Find the day with highest temperature
3. Find how many days were above average

**A:**
```python
import numpy as np

temps = np.array([72, 68, 75, 80, 65, 70, 78, 82, 69, 73])

# 1. Average
avg_temp = np.mean(temps)
print(f"Average: {avg_temp}°F")  # 73.2°F

# 2. Day with highest temp
hottest_day = np.argmax(temps)
print(f"Hottest: Day {hottest_day} ({temps[hottest_day]}°F)")  # Day 7 (82°F)

# 3. Days above average
above_avg = temps > avg_temp
print(f"Days above average: {np.sum(above_avg)}")  # 5 days
```

---


## 10. Reshaping Arrays

### 🤔 WHY

**Why reshape arrays:**
Sometimes data comes in one shape but you need it in another:
- Load image as 1D array, need 2D for display
- Flatten 2D matrix for machine learning models  
- Reorganize data for calculations

**Real-Life Analogy:**
Like rearranging books:
- All books in a single pile → Organize into shelves (rows) with books per shelf (columns)
- Books on shelves → Stack into one pile
Same books, different arrangement!

### ⏰ WHEN

**Reshape when:**
- Preparing data for ML models (need specific shapes)
- Converting between 1D and 2D representations
- Organizing data for calculations (matrix operations)
- Image processing (converting between flat and 2D)

### 🔧 HOW

**Key Functions:**
- `reshape()`: Change shape (creates view if possible)
- `flatten()`: Convert to 1D (creates copy)
- `ravel()`: Convert to 1D (creates view)
- `resize()`: Change shape with size modification

### 10.1 Reshape

**Example 1: Basic Reshaping**
```python
import numpy as np

# 1D to 2D
arr_1d = np.array([1, 2, 3, 4, 5, 6])
arr_2d = arr_1d.reshape(2, 3)  # 2 rows, 3 columns

print(f"Original 1D: {arr_1d}")
print(f"Reshaped to 2×3:\n{arr_2d}")
# Output:
# Original 1D: [1 2 3 4 5 6]
# Reshaped to 2×3:
# [[1 2 3]
#  [4 5 6]]

# Different shapes (total elements must match!)
arr_3x2 = arr_1d.reshape(3, 2)  # 3 rows, 2 columns
print(f"Reshaped to 3×2:\n{arr_3x2}")
# Output:
# [[1 2]
#  [3 4]
#  [5 6]]

# Can use -1 to auto-calculate one dimension
arr_auto = arr_1d.reshape(2, -1)  # -1 means "figure it out"
print(f"Reshape with -1:\n{arr_auto}")
# Output: (2, 3) automatically calculated
```

**Example 2: Real-World - Image Data**
```python
import numpy as np

# Simulated grayscale image data (9 pixels)
image_flat = np.array([255, 200, 150, 100, 50, 0, 128, 192, 64])
print(f"Flat image (9 pixels): {image_flat}")

# Reshape to 3×3 image
image_2d = image_flat.reshape(3, 3)
print(f"\n3×3 Image:\n{image_2d}")
# Output:
# 3×3 Image:
# [[255 200 150]
#  [100  50   0]
#  [128 192  64]]

# Now we can process it like an image (e.g., get center pixel)
center_pixel = image_2d[1, 1]
print(f"Center pixel value: {center_pixel}")
# Output: Center pixel value: 50
```

**Example 3: 2D to 3D (Batching)**
```python
import numpy as np

# 6 data points, each with 2 features
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
print(f"Original shape: {data.shape}")  # (6, 2)

# Reshape to 3 batches of 2 samples with 2 features
batched = data.reshape(3, 2, 2)
print(f"Batched shape: {batched.shape}")  # (3, 2, 2)
print(f"Batched data:\n{batched}")
# Output:
# Batched data:
# [[[ 1  2]
#   [ 3  4]]
#
#  [[ 5  6]
#   [ 7  8]]
#
#  [[ 9 10]
#   [11 12]]]
```

### 10.2 Flatten and Ravel

**Example 1: Flatten vs Ravel**
```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# flatten() - always returns a COPY
flat_copy = matrix.flatten()
print(f"Flattened: {flat_copy}")
# Output: Flattened: [1 2 3 4 5 6]

# Modify flattened - doesn't affect original
flat_copy[0] = 999
print(f"Original after modifying flatten: {matrix[0, 0]}")
# Output: 1 (unchanged)

# ravel() - returns VIEW when possible (no copy)
flat_view = matrix.ravel()
print(f"Raveled: {flat_view}")
# Output: Raveled: [1 2 3 4 5 6]

# Modify raveled - DOES affect original!
flat_view[0] = 999
print(f"Original after modifying ravel: {matrix[0, 0]}")
# Output: 999 (changed!)
```

**Example 2: When to Use Each**
```python
import numpy as np

# When you need independent copy - use flatten()
original = np.array([[1, 2], [3, 4]])
backup = original.flatten()  # Safe copy
original[0, 0] = 100
print(f"Original changed: {original}")
print(f"Backup unchanged: {backup}")
# Backup is safe!

# When you want memory efficiency - use ravel()
large_matrix = np.arange(1000000).reshape(1000, 1000)
flat = large_matrix.ravel()  # No copy, very fast!
# Use flat for calculations without duplicating memory
```

### 10.3 Transpose

**Example 1: Basic Transpose**
```python
import numpy as np

# Transpose swaps rows and columns
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"Original (2×3):\n{matrix}")

transposed = matrix.T  # or matrix.transpose()
print(f"\nTransposed (3×2):\n{transposed}")
# Output:
# Original (2×3):
# [[1 2 3]
#  [4 5 6]]
#
# Transposed (3×2):
# [[1 4]
#  [2 5]
#  [3 6]]
```

**Example 2: Real Use - Student Grades**
```python
import numpy as np

# Grades: rows=students, columns=subjects
grades = np.array([[85, 90, 78],  # Student 1
                   [92, 88, 95],  # Student 2
                   [78, 85, 82]]) # Student 3

print("By Students (rows):")
print(grades)
print(f"Student 1 grades: {grades[0]}")

# Transpose to get by subjects
by_subjects = grades.T
print("\nBy Subjects (rows):")
print(by_subjects)
print(f"Math grades (all students): {by_subjects[0]}")
# Output: Math grades (all students): [85 92 78]
```

### 🏋️ Practice Question
**Q:** You have 12 monthly sales figures in 1D array. Reshape them to show quarters (4 rows) with 3 months each (3 columns).
**A:**
```python
sales = np.array([100, 120, 110, 130, 140, 135, 145, 150, 155, 160, 165, 150])
quarterly = sales.reshape(4, 3)
print(quarterly)
# [[100 120 110]  # Q1: Jan, Feb, Mar
#  [130 140 135]  # Q2: Apr, May, Jun
#  [145 150 155]  # Q3: Jul, Aug, Sep
#  [160 165 150]] # Q4: Oct, Nov, Dec
```

---

## 11. Joining and Splitting Arrays

### 🤔 WHY

**Why join/split arrays:**
- Combine data from different sources
- Split data for training/testing in ML
- Merge results from parallel processing
- Separate data into chunks

**Real-Life Analogy:**
- **Joining**: Combining multiple smaller boxes into one big box
- **Splitting**: Cutting a pizza into equal slices

### ⏰ WHEN

**Join when:**
- Merging data from multiple files
- Combining train and test datasets
- Stacking results from multiple operations

**Split when:**
- Creating train/validation/test sets
- Parallel processing (split work, merge results)
- Batch processing

### 🔧 HOW

**Joining Functions:**
- `concatenate()`: Join along existing axis
- `vstack()`: Vertical stack (row-wise)
- `hstack()`: Horizontal stack (column-wise)
- `stack()`: Join along new axis

**Splitting Functions:**
- `split()`: Split into equal parts
- `vsplit()`: Vertical split
- `hsplit()`: Horizontal split

### 11.1 Concatenate

**Example 1: Concatenate 1D Arrays**
```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Concatenate multiple arrays
result = np.concatenate([arr1, arr2, arr3])
print(f"Concatenated: {result}")
# Output: Concatenated: [1 2 3 4 5 6 7 8 9]
```

**Example 2: Concatenate 2D Arrays**
```python
import numpy as np

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Concatenate along rows (axis=0) - stack vertically
vertical = np.concatenate([matrix1, matrix2], axis=0)
print(f"Vertical (axis=0):\n{vertical}")
# Output:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Concatenate along columns (axis=1) - stack horizontally
horizontal = np.concatenate([matrix1, matrix2], axis=1)
print(f"\nHorizontal (axis=1):\n{horizontal}")
# Output:
# [[1 2 5 6]
#  [3 4 7 8]]
```

### 11.2 VStack and HStack

**Example 1: VStack (Vertical)**
```python
import numpy as np

row1 = np.array([1, 2, 3])
row2 = np.array([4, 5, 6])
row3 = np.array([7, 8, 9])

# Stack rows vertically
stacked = np.vstack([row1, row2, row3])
print(f"VStack:\n{stacked}")
# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Real example: Adding new row of data
existing_data = np.array([[10, 20], [30, 40]])
new_row = np.array([50, 60])
updated_data = np.vstack([existing_data, new_row])
print(f"\nAdded new row:\n{updated_data}")
# Output:
# [[10 20]
#  [30 40]
#  [50 60]]
```

**Example 2: HStack (Horizontal)**
```python
import numpy as np

col1 = np.array([[1], [2], [3]])
col2 = np.array([[4], [5], [6]])
col3 = np.array([[7], [8], [9]])

# Stack columns horizontally
stacked = np.hstack([col1, col2, col3])
print(f"HStack:\n{stacked}")
# Output:
# [[1 4 7]
#  [2 5 8]
#  [3 6 9]]

# Real example: Adding new feature column
features = np.array([[1, 2], [3, 4], [5, 6]])
new_feature = np.array([[10], [20], [30]])
extended_features = np.hstack([features, new_feature])
print(f"\nAdded new feature:\n{extended_features}")
# Output:
# [[ 1  2 10]
#  [ 3  4 20]
#  [ 5  6 30]]
```

### 11.3 Splitting Arrays

**Example 1: Split 1D Array**
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Split into 3 equal parts
parts = np.split(arr, 3)
print(f"Split into 3 parts:")
for i, part in enumerate(parts):
    print(f"  Part {i+1}: {part}")
# Output:
# Split into 3 parts:
#   Part 1: [1 2 3]
#   Part 2: [4 5 6]
#   Part 3: [7 8 9]

# Split at specific indices
parts2 = np.split(arr, [2, 5])  # Split at index 2 and 5
print(f"\nSplit at [2, 5]:")
for i, part in enumerate(parts2):
    print(f"  Part {i+1}: {part}")
# Output:
#   Part 1: [1 2]
#   Part 2: [3 4 5]
#   Part 3: [6 7 8 9]
```

**Example 2: VSplit and HSplit**
```python
import numpy as np

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Vertical split (split rows)
v_parts = np.vsplit(matrix, 3)
print("Vertical split (3 parts):")
for i, part in enumerate(v_parts):
    print(f"Part {i+1}:\n{part}")
# Output:
# Part 1: [[1 2 3 4]]
# Part 2: [[5 6 7 8]]
# Part 3: [[9 10 11 12]]

# Horizontal split (split columns)
h_parts = np.hsplit(matrix, 2)
print("\nHorizontal split (2 parts):")
for i, part in enumerate(h_parts):
    print(f"Part {i+1}:\n{part}")
# Output:
# Part 1:
# [[1 2]
#  [5 6]
#  [9 10]]
# Part 2:
# [[3 4]
#  [7 8]
#  [11 12]]
```

**Example 3: Train/Test Split**
```python
import numpy as np

# Dataset: 10 samples
data = np.arange(100).reshape(10, 10)
print(f"Dataset shape: {data.shape}")  # (10, 10)

# Split: 80% train, 20% test
split_point = int(0.8 * len(data))
train_data = data[:split_point]
test_data = data[split_point:]

print(f"Train set: {train_data.shape}")  # (8, 10)
print(f"Test set: {test_data.shape}")    # (2, 10)
```

### 🏋️ Practice Question
**Q:** You have two score arrays for tests: `test1 = np.array([80, 85, 90])` and `test2 = np.array([75, 88, 92])`. Combine them so each row represents one student with both test scores.

**A:**
```python
test1 = np.array([80, 85, 90])
test2 = np.array([75, 88, 92])

# Reshape to column vectors
test1_col = test1.reshape(-1, 1)
test2_col = test2.reshape(-1, 1)

# Horizontally stack
combined = np.hstack([test1_col, test2_col])
print(combined)
# Output:
# [[80 75]
#  [85 88]
#  [90 92]]
```

---
