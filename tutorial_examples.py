#!/usr/bin/env python3
"""
NumPy Tutorial Examples - Test Script
This script demonstrates key concepts from the tutorial

Requirements:
    pip install numpy

Usage:
    python tutorial_examples.py
"""

try:
    import numpy as np
except ImportError:
    print("❌ NumPy is not installed!")
    print("📦 Please install it using: pip install numpy")
    print("📚 See Section 2 of numpy_tutorial_complete.md for installation instructions")
    exit(1)

def separator(title):
    """Print a section separator"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def main():
    print("🎓 NumPy Tutorial - Example Demonstrations")
    
    # 1. Basic Array Creation
    separator("1. Creating Arrays")
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    zeros = np.zeros((2, 3))
    ones = np.ones(5)
    range_arr = np.arange(0, 10, 2)
    linspace_arr = np.linspace(0, 1, 5)
    
    print(f"1D Array: {arr_1d}")
    print(f"2D Array:\n{arr_2d}")
    print(f"Zeros:\n{zeros}")
    print(f"Ones: {ones}")
    print(f"Range: {range_arr}")
    print(f"Linspace: {linspace_arr}")
    
    # 2. Array Properties
    separator("2. Array Properties")
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"Matrix:\n{matrix}")
    print(f"Shape: {matrix.shape}")
    print(f"Dimensions: {matrix.ndim}")
    print(f"Size: {matrix.size}")
    print(f"Data Type: {matrix.dtype}")
    
    # 3. Indexing and Slicing
    separator("3. Indexing and Slicing")
    arr = np.array([10, 20, 30, 40, 50])
    print(f"Original: {arr}")
    print(f"First element: {arr[0]}")
    print(f"Last element: {arr[-1]}")
    print(f"Slice [1:4]: {arr[1:4]}")
    print(f"Matrix row 1: {matrix[1]}")
    print(f"Matrix column 0: {matrix[:, 0]}")
    
    # 4. Array Operations
    separator("4. Array Operations")
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * 2 = {a * 2}")
    print(f"a ** 2 = {a ** 2}")
    
    # 5. Broadcasting
    separator("5. Broadcasting")
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    scalar = 10
    print(f"Matrix:\n{matrix}")
    print(f"Matrix + {scalar}:\n{matrix + scalar}")
    
    # 6. Mathematical Functions
    separator("6. Mathematical Functions")
    scores = np.array([85, 92, 78, 95, 88, 91, 83])
    print(f"Scores: {scores}")
    print(f"Mean: {np.mean(scores):.2f}")
    print(f"Median: {np.median(scores)}")
    print(f"Std Dev: {np.std(scores):.2f}")
    print(f"Min: {np.min(scores)}")
    print(f"Max: {np.max(scores)}")
    print(f"Sum: {np.sum(scores)}")
    
    # 7. Reshaping
    separator("7. Reshaping")
    arr = np.arange(12)
    print(f"Original (1D): {arr}")
    reshaped_3x4 = arr.reshape(3, 4)
    print(f"Reshaped (3×4):\n{reshaped_3x4}")
    flattened = reshaped_3x4.flatten()
    print(f"Flattened back: {flattened}")
    
    # 8. Boolean Indexing
    separator("8. Boolean Indexing")
    ages = np.array([25, 17, 32, 19, 45, 16, 28])
    print(f"Ages: {ages}")
    adults = ages >= 18
    print(f"Adults (>=18): {adults}")
    print(f"Adult ages: {ages[adults]}")
    
    # 9. Sorting
    separator("9. Sorting")
    unsorted = np.array([64, 34, 25, 12, 22, 11, 90])
    print(f"Unsorted: {unsorted}")
    sorted_arr = np.sort(unsorted)
    print(f"Sorted: {sorted_arr}")
    sorted_indices = np.argsort(unsorted)
    print(f"Sort indices: {sorted_indices}")
    
    # 10. Random Numbers
    separator("10. Random Numbers")
    np.random.seed(42)  # For reproducibility
    random_floats = np.random.rand(5)
    random_ints = np.random.randint(1, 100, 5)
    random_normal = np.random.randn(5)
    print(f"Random floats [0,1): {random_floats}")
    print(f"Random integers [1,100): {random_ints}")
    print(f"Random normal (mean=0, std=1): {random_normal}")
    
    # 11. Real Example: Grade Analysis
    separator("11. Real Example - Grade Analysis")
    # Students × Subjects (Math, Science, English)
    grades = np.array([[85, 90, 78],
                       [92, 88, 95],
                       [78, 85, 82]])
    
    students = ['Alice', 'Bob', 'Charlie']
    print("Grade Matrix:")
    print(grades)
    print("\nStudent Averages:")
    for i, student in enumerate(students):
        avg = grades[i].mean()
        print(f"  {student}: {avg:.2f}")
    
    print("\nSubject Averages:")
    subjects = ['Math', 'Science', 'English']
    for j, subject in enumerate(subjects):
        avg = grades[:, j].mean()
        print(f"  {subject}: {avg:.2f}")
    
    top_student_idx = grades.mean(axis=1).argmax()
    print(f"\nTop Student: {students[top_student_idx]} (avg: {grades[top_student_idx].mean():.2f})")
    
    separator("Tutorial Examples Complete!")
    print("✅ All examples ran successfully!")
    print("📚 Learn more in numpy_tutorial_complete.md")
    print("🚀 You're ready to use NumPy!")

if __name__ == "__main__":
    main()
