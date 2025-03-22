## Assignment 1: The Finite Difference Method, Boundary Value Problems and Iterative Solution of Linear Systems

**Duration**: 6 weeks  
**Deadline**: Sunday, March 23, 2025, 23:59

This assignment consists of three parts, covering fundamental concepts in numerical analysis related to the Finite Difference Method (FDM), solving Boundary Value Problems (BVPs), and using iterative methods for linear systems. The exercises align with the course lectures and are meant to build a strong theoretical and practical understanding.

---

### 📌 Exercise 1: Basics of Finite Difference Methods

#### a) Stencil Derivation
- Derive two FDM stencils for approximating \( u''(x) \):
  - Off-centered stencil: (α, β) = (4, 0)
  - Centered stencil: α = β = 2
- Show full derivation of stencil weights.

#### b) Accuracy Analysis
- Use order conditions to determine and compare the accuracy of both stencils.

#### c) MATLAB Implementation
- Write code to approximate \( u''(x) \) for \( u(x) = \exp(\cos(x)) \) at \( x = 0 \).
- Use `fdcoeffF.m` from LeVeque’s book.

#### d) Convergence Test
- Plot the error vs. \( h \), using \( h = 1/2^s \), for \( s = 2,3,4,... \)
- Analyze truncation error behavior for large and small \( h \).

#### e) Interpolation Stencil
- Derive an interpolation stencil at \( x = 0 \) of order 3 accuracy.
- Use symmetric points \( x = \pm h/2, \pm 3h/2, ... \)
- Perform and plot a convergence test.

---

### 📌 Exercise 2a: Solving a 2-Point Nonlinear BVP

We consider the BVP:

\[
\epsilon u'' + u(u' - 1) = 0 \quad \text{on } [0, 1], \quad u(0) = -1, \quad u(1) = 1.5, \quad \epsilon = 0.1
\]

#### a) Discretization
- Derive a second-order accurate finite difference scheme.
- Compute and analyze the local truncation error \( \tau_j \).

#### b) Newton’s Method
- Implement Newton’s method to solve the BVP.
- Use LeVeque’s suggested initial guess and Jacobian.
- Estimate global error and check convergence order.

---

### 📌 Exercise 2b: 5- and 9-Point Laplacians in 2D

On the unit square domain \([0,1] \times [0,1]\), using a uniform grid of \( m \times m \) interior points:

#### c) 5-Point Laplacian
- Implement the 5-point Laplacian scheme.
- Validate convergence using the exact solution:
  \[
  u(x, y) = \sin(4\pi(x + y)) + \cos(4\pi x y)
  \]

#### d) 9-Point Laplacian
- Implement the 9-point Laplacian with a corrected right-hand side (per LeVeque’s Section 3.5).
- Demonstrate \( O(h^4) \) convergence.

---

### 📌 Exercise 3: Iterative Solvers in 2D – Multigrid

Using the same domain as in 2b:

#### a) Matrix-Free 5-Point Laplacian
- Implement `function AU=Amult(U,m)` to compute \( -A_h U \) without forming \( A_h \).

#### b) Conjugate Gradient Method
- Use MATLAB’s `pcg` to solve \( -A_h U = -F \)
- Report convergence history and rate.

---

### 🔄 Jacobi Smoothing

- Analyze under-/over-relaxed Jacobi iteration.
- Plot \( \max_{m/2 \le p,q \le m} |\gamma_{p,q}| \) vs. \( \omega \) for various \( m \).
- Implement a relaxed Jacobi smoother:
  ```matlab
  function Unew = smooth(U, omega, m, F)
  ```

---

### 🔀 Multigrid V-Cycle

- Implement coarsening and interpolation:
  ```matlab
  function Rc = coarsen(R, m)
  function R = interpolate(Rc, m)
  ```
- Integrate everything into a `VCycle.m` implementation.

#### b) Multigrid Performance
- For varying \( m = 2^k - 1 \), analyze how the number of outer V-cycle iterations changes (fixed tolerance).

---

### 📄 Submission Requirements

- Submit:
  - A **written report (PDF)** explaining your approach, results, and conclusions.
  - A **ZIP file** with all your MATLAB code.

Make sure your report is clear, reproducible, and explains any deviations from theoretical expectations.

---

Let me know if you’d like this saved as a `.md` file or need a LaTeX or HTML version too!