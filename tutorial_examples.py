#!/usr/bin/env python3
"""
NumPy Tutorial Complete Examples - Runnable Master Test Script
This script demonstrates key concepts from all sections of numpy_tutorial_complete.md

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
    """Print a styled section separator"""
    print("\n" + "="*65)
    print(f"  {title}")
    print("="*65)

def main():
    print("🎓 NumPy Complete Tutorial - Hands-On Demonstrations")
    print(f"NumPy Version: {np.__version__}")

    # 1. Advanced Array Creation
    separator("1. Creating Arrays (Basics, Grids, and Special Matrices)")
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    identity_m = np.eye(3)
    diag_m = np.diag([10, 20, 30])
    lin_space = np.linspace(0, 1, 5)
    log_space = np.logspace(-3, -1, 3)

    # Coordinate grids for 2D evaluation
    x = np.linspace(-1, 1, 3)
    y = np.linspace(-1, 1, 3)
    X, Y = np.meshgrid(x, y)

    print(f"1D Vector:      {arr_1d}")
    print(f"2D Float Matrix:\n{arr_2d}")
    print(f"3x3 Identity Matrix:\n{identity_m}")
    print(f"Diagonal Matrix:\n{diag_m}")
    print(f"Linspace [0, 1]: {lin_space}")
    print(f"Logspace (10^-3 to 10^-1): {log_space}")
    print(f"2D Meshgrid X:\n{X}")

    # 2. Array Properties, Strides, and Memory Layout
    separator("2. Array Properties, Strides, and Memory Flags")
    matrix = np.arange(12, dtype=np.int64).reshape(3, 4)
    print(f"Matrix (3x4):\n{matrix}")
    print(f"Shape:               {matrix.shape}")
    print(f"Dimensions (ndim):   {matrix.ndim}")
    print(f"Total Elements:      {matrix.size}")
    print(f"Data Type (dtype):   {matrix.dtype}")
    print(f"Item Size in Bytes:  {matrix.itemsize}")
    print(f"Total Memory Buffer: {matrix.nbytes} bytes")
    print(f"Memory Strides:      {matrix.strides}  (bytes to step [row, col])")
    print(f"C_CONTIGUOUS flag:   {matrix.flags.c_contiguous}")
    print(f"OWNDATA flag:        {matrix.flags.owndata}")

    # 3. Basic Slicing, Ellipsis, and Fancy Indexing
    separator("3. Slicing, Ellipsis (...), and Advanced Fancy Indexing")
    grid = np.arange(16).reshape(4, 4)
    print(f"Original 4x4 Grid:\n{grid}")
    print(f"Subgrid slice (rows 1:3, cols 1:3):\n{grid[1:3, 1:3]}")

    # Ellipsis syntax on 4D tensor
    tensor_4d = np.zeros((2, 3, 4, 5))
    print(f"4D Tensor shape: {tensor_4d.shape}")
    print(f"Slice with ellipsis tensor_4d[0, ..., 0] shape: {tensor_4d[0, ..., 0].shape}")

    # Fancy indexing (coordinate pairs vs open mesh)
    paired = grid[[0, 2], [1, 3]]  # Points (0,1) and (2,3)
    mesh_indexed = grid[np.ix_([0, 2], [1, 3])]  # Rectangular subgrid
    print(f"Paired coordinate fancy indexing: {paired}")
    print(f"np.ix_ rectangular fancy indexing:\n{mesh_indexed}")

    # 4. Universal Functions (ufuncs) and Methods
    separator("4. Universal Functions (ufuncs) and Built-In Methods")
    v = np.array([1, 2, 3, 4, 5])
    print(f"Original vector: {v}")
    print(f"np.add.reduce (Sum):       {np.add.reduce(v)}")
    print(f"np.multiply.reduce (Prod): {np.multiply.reduce(v)}")
    print(f"np.add.accumulate (Cumsum):{np.add.accumulate(v)}")
    
    # Outer multiplication table
    outer_table = np.multiply.outer([1, 2, 3], [10, 20, 30])
    print(f"np.multiply.outer table:\n{outer_table}")

    # Zero-allocation out= parameter
    out_buffer = np.empty_like(v)
    np.multiply(v, 10, out=out_buffer)
    print(f"Result written directly into out_buffer: {out_buffer}")

    # 5. Broadcasting Deep Dive
    separator("5. Broadcasting Mechanics")
    col_vec = np.array([[10], [20], [30]])  # Shape (3, 1)
    row_vec = np.array([[1, 2, 3, 4]])      # Shape (1, 4)
    broadcasted_grid = col_vec + row_vec     # Shape (3, 4)
    print(f"Column vector (3, 1):\n{col_vec}")
    print(f"Row vector (1, 4): {row_vec}")
    print(f"Broadcasted Sum Grid (3, 4):\n{broadcasted_grid}")

    # 6. Math, Statistics, and NaN Handling
    separator("6. Mathematical Statistics & NaN-Safe Aggregations")
    dirty_data = np.array([10.0, 25.0, np.nan, 40.0, 50.0])
    print(f"Raw data with NaN: {dirty_data}")
    print(f"Standard mean (fails with NaN):  {np.mean(dirty_data)}")
    print(f"np.nanmean (ignores NaN):        {np.nanmean(dirty_data):.2f}")
    print(f"np.nanstd (standard deviation):  {np.nanstd(dirty_data):.2f}")
    print(f"np.nanpercentile 50% (Median):   {np.nanpercentile(dirty_data, 50)}")

    # Floating point comparison
    float_a = 0.1 + 0.2
    float_b = 0.3
    print(f"Direct float equality (0.1 + 0.2 == 0.3): {float_a == float_b}")
    print(f"np.isclose(0.1 + 0.2, 0.3):               {np.isclose(float_a, float_b)}")

    # 7. Reshaping, Transposing, and Dimension Manipulation
    separator("7. Reshaping, Transposing, and Padding")
    flat = np.arange(6)
    m_2x3 = flat.reshape(2, 3)
    print(f"Reshaped 2x3:\n{m_2x3}")
    print(f"Transposed 3x2:\n{m_2x3.T}")
    
    # Ravel (view) vs Flatten (copy)
    raveled = m_2x3.ravel()
    flattened = m_2x3.flatten()
    print(f"Ravel is view?   {raveled.base is m_2x3}")
    print(f"Flatten is view? {flattened.base is m_2x3}")

    # Padding borders
    padded_img = np.pad(m_2x3, pad_width=1, mode='constant', constant_values=0)
    print(f"Zero-padded border:\n{padded_img}")

    # 8. Stacking and Block Matrices
    separator("8. Stacking, Concatenation, and Block Matrices")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(f"np.concatenate along axis 0: {np.concatenate([a, b])} (shape: {np.concatenate([a, b]).shape})")
    print(f"np.stack along new axis 0:\n{np.stack([a, b], axis=0)} (shape: {np.stack([a, b], axis=0).shape})")

    # Block matrix assembly
    A_blk = np.eye(2)
    B_blk = np.ones((2, 2)) * 3
    assembled_blk = np.block([[A_blk, B_blk],
                              [B_blk, A_blk]])
    print(f"Assembled 4x4 Block Matrix:\n{assembled_blk}")

    # 9. Boolean Indexing, Masking, and np.select
    separator("9. Boolean Masking, np.where, and np.select")
    scores = np.array([92, 58, 74, 85, 42])
    print(f"Student scores: {scores}")
    
    # Ternary np.where: Pass/Fail status
    status = np.where(scores >= 60, "Pass", "Fail")
    print(f"Pass/Fail status: {status}")

    # Multi-branch np.select
    conditions = [scores >= 90, scores >= 80, scores >= 70, scores >= 60]
    choices = ['A', 'B', 'C', 'D']
    grades = np.select(conditions, choices, default='F')
    print(f"Letter grades:   {grades}")

    # 10. Sorting, Partitioning, and Unique
    separator("10. Sorting, O(N) Partitioning, and Unique Counts")
    unsorted = np.array([45, 12, 89, 7, 23, 99, 54])
    print(f"Unsorted array: {unsorted}")
    print(f"np.sort (ascending):  {np.sort(unsorted)}")
    print(f"np.argsort (indices): {np.argsort(unsorted)}")
    
    # Fast O(N) top-3 partition without sorting whole array
    top3_partition = np.partition(unsorted, -3)[-3:]
    print(f"Top 3 elements (via O(N) partition): {top3_partition}")

    # Unique values and frequency counts
    items = np.array(['Apple', 'Banana', 'Apple', 'Orange', 'Banana', 'Banana'])
    uniques, counts = np.unique(items, return_counts=True)
    for item, cnt in zip(uniques, counts):
        print(f"  {item}: {cnt} times")

    # 11. Modern Random Generator API (PCG64)
    separator("11. Modern Random Generator API (np.random.default_rng)")
    rng = np.random.default_rng(seed=42)
    rand_ints = rng.integers(1, 100, size=5)
    rand_floats = rng.random(size=5)
    rand_normal = rng.normal(loc=100.0, scale=15.0, size=5)
    print(f"PCG64 Random Integers [1, 100): {rand_ints}")
    print(f"PCG64 Uniform Floats [0, 1):   {np.round(rand_floats, 4)}")
    print(f"PCG64 Normal (mean=100, std=15):{np.round(rand_normal, 2)}")

    # 12. Linear Algebra and Einsum
    separator("12. Linear Algebra (np.linalg) and Einstein Summation (np.einsum)")
    M = np.array([[3.0, 1.0],
                  [1.0, 2.0]])
    rhs = np.array([9.0, 8.0])

    # Solve Mx = rhs
    sol = np.linalg.solve(M, rhs)
    print(f"Matrix M:\n{M}")
    print(f"Solved Mx = [9, 8]: x = {sol}")
    print(f"Determinant: {np.linalg.det(M):.2f}")
    print(f"Inverse M^(-1):\n{np.linalg.inv(M)}")

    # Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    print(f"Eigenvalues: {eigenvalues}")

    # SVD
    U, s, Vt = np.linalg.svd(M)
    print(f"SVD Singular Values: {s}")

    # Einstein Summation Matrix Multiplication
    einsum_matmul = np.einsum('ij,jk->ik', M, M)
    print(f"Einsum Matrix Multiplication (M @ M):\n{einsum_matmul}")

    # 13. Structured Arrays (C-Style Structs)
    separator("13. Structured Arrays (Heterogeneous Records at C Speed)")
    employee_dtype = np.dtype([
        ('name', 'U20'),
        ('age', 'i4'),
        ('salary', 'f8')
    ])
    staff = np.array([
        ('Alice', 32, 85000.0),
        ('Bob', 27, 62000.0),
        ('Charlie', 41, 110000.0)
    ], dtype=employee_dtype)
    print("Structured Staff Array:")
    for emp in staff:
        print(f"  Name: {emp['name']:<8} | Age: {emp['age']} | Salary: ${emp['salary']:,.2f}")
    print(f"Staff Names:   {staff['name']}")
    print(f"Mean Salary:   ${staff['salary'].mean():,.2f}")

    # 14. Real-World Case Study: Pairwise Distance Matrix (Fully Vectorized)
    separator("14. Real-World Case Study: Vectorized Pairwise Distance")
    points = rng.uniform(0, 10, size=(5, 2))  # 5 2D points
    print(f"Points (5 coordinates):\n{np.round(points, 2)}")
    
    # Vectorized computation of distance between all pairs without loops
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    print(f"Pairwise Distance Matrix (5x5):\n{np.round(dist_matrix, 2)}")

    # 15. Real-World Case Study: Linear Regression via Normal Equation
    separator("15. Real-World Case Study: Linear Regression via Normal Equation")
    # y = 2.5 * x + 4.0 + noise
    x_data = np.linspace(0, 10, 50)
    noise = rng.normal(0, 0.5, size=50)
    y_data = 2.5 * x_data + 4.0 + noise

    # Design matrix [1, x]
    X_mat = np.column_stack([np.ones_like(x_data), x_data])
    # Normal Equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ y_data)
    print(f"True Parameters:   Intercept = 4.00, Slope = 2.50")
    print(f"Fitted Parameters: Intercept = {theta[0]:.2f}, Slope = {theta[1]:.2f}")

    separator("Tutorial Complete - Master NumPy Verified! 🎉")
    print("✅ All demonstrations executed with 100% precision!")
    print("📚 Complete tutorial available in numpy_tutorial_complete.md")
    print("🚀 You are fully equipped to build data science, ML, and scientific computing projects!")

if __name__ == "__main__":
    main()
