# Backend coupling: plan

Status: draft (2026-11-24). Scope: connecting the symbolic `equations` package to the
FEM backends (`qewton/backends/fem`) and to the computation graph (`qewton/graphs`),
including time discretization and repeated solves. Continues milestones M9 and M10
of [problem_graph_plan.md](problem_graph_plan.md).

---

## 1. Starting point

- `weak_form(task)` returns a **semi-discrete** `WeakProblem`: time derivatives stay
  inside the integrands, the test functions depend on space only, and the initial
  conditions are kept next to the form. For the heat fixture:

  ```
  F[v_w] = ∫ ∂ₜw·v_w dx + ∫ ∇w·∇v_w dx − ∫ f·v_w dx − ∫ (∇w·n)|_∂Ω·v_w ds = 0
  also: w|_{Ω×{t=0}} − w0 = 0
  ```

- The FEM backend (`backends/fem`) has four sub-backends: `FormBackend` (UFL-like
  operations, assembly, Dirichlet conditions), `FunctionSpaceBackend`, `MeshBackend`
  and `SolveBackend`. `SolveBackend` already works with reusable solver handles.
  `FormBackend.time_derivative` is a TODO.
- The computation graph is a strict DAG of `Node`s. A node's ports come from the
  signature of `forward`, or are built at runtime (as in `GraphNode`). There is no
  loop construct.
- Nothing yet turns an equations `Expr` into backend objects.

## 2. Target pipeline

```
Problem + Task ──AnalysisContext──► weak_form ──► WeakProblem        (semi-discrete, ∂t kept)
                                                    │ discretize_time(scheme)      equations/compile
                                                    ▼
                                               StepProblem           (stationary per step + history)
                                                    │ lower(target)                 equations/compile
                                                    ▼
                                    backend objects: V, a/L or F, bcs, input slots
                                                    │
                       FEMSolveNode (one solve) ──► ScanNode (repeats it, carries state)    graphs
```

The equations package stays independent of any solver: the lowering only talks to
the backend protocols in `qewton/backends`.

## 3. Time discretization in the equations package

A **symbolic compiler pass** turns the semi-discrete form into a stationary problem
per step. Backward Euler for heat:

```
∫ (w − w_n)/Δt · v dx + ∫ ∇w·∇v dx − ∫ f(·, t_{n+1})·v dx = 0
unknown: w        data: w_n (history), Δt, t_{n+1}
```

- **Why here and not in the backend.** A θ-scheme changes every term, not only
  `∂ₜw`; with `transform` the pass is short and testable on symbols without FEniCS.
  Every backend then only solves stationary problems, so
  `FormBackend.time_derivative` can be removed. The same pass serves PINN-style time
  marching.
- **Schemes are data.** The θ-family (θ = 1 backward Euler, ½ Crank–Nicolson,
  0 explicit), BDF-k as coefficients over the history states, and for `∂ₜₜ` a
  second-difference stencil or Newmark later.
- **What the pass returns** (`StepProblem`): the stationary per-step `WeakProblem`;
  the history symbols (`w → (w_n, w_{n-1}, ...)`), which are plain fixed
  `FunctionSymbol`s so that linearity and the operator/source split treat them as
  data; the `Δt` and `t` symbols; and how to start. The initial conditions in
  `WeakProblem.constraints` give `w_0`. BDF2 starts with one backward-Euler step;
  second order in time needs the initial velocity, which classification already
  counts per order.
- **Data depending on time** (`f(x, t)`, boundary data `g(x, t)`) is evaluated at
  the step's time, which is a scalar input of the step. A θ-scheme also needs the
  previous time; `Shift(f, t, −Δt)` expresses that without a new node type.
- **Kept open.** A backend with its own integrator (PETSc TS, adaptive steps) can
  take the semi-discrete form `F(t, u, u̇)` directly later.

## 4. Repeated solves as graph nodes

### 4.1 The Task is the node's signature

`FEMSolveNode(step_problem, backend, discretization)` builds its ports from the task:

| Role | Port |
|---|---|
| UNKNOWN | output |
| VARYING, history (`w_n`), `t`, `Δt` | input |
| FIXED | built in during `setup()`, no port |

`setup()` builds function spaces and forms and creates the solver handle;
`forward()` writes the inputs into the backend's Constants/Functions and solves.

### 4.2 What can be reused comes from the analysis

`split` and `batch_placement` already say whether a symbol enters the operator or
only the source. History, `t` and `Δt` are more symbols to place:

| Changes between solves | Enters | Reuse |
|---|---|---|
| `w_n`, `f(t)`, `g(t)`, a varying source | source only | assemble and factor `A` once, reassemble the RHS |
| `Δt` (adaptive), a time-dependent coefficient | operator | reassemble and refactor `A` |
| varying `ν`, separable (`OPERATOR_AFFINE`) | operator | assemble each `A_q` once; `A(ν) = Σ θ_q(ν) A_q` |
| nonlinear | everything | Newton each solve |

The first row is what the `SingleFactorization` suggestion recommends; the node makes
it happen.

### 4.3 Looping: a generic `ScanNode`

A control node in `graphs/control_nodes` that runs an inner graph `n` times, feeds
chosen outputs back to inputs (`w_n ← w`) and collects the trajectory, like
`jax.lax.scan`. Time stepping is `ScanNode(step_graph)`; the same node covers
fixed-point coupling and staggered multiphysics. Unrolling a time loop in the DAG is
not viable for long runs.

### 4.4 Port payload

Proposal: dof arrays at node boundaries (backend-neutral, easy to stack into a
trajectory, can be handed to torch); the backend `Function`s stay inside the node.

## 5. Lowering: equations → backends

One visitor `lower(expr, target)` over `Expr` maps each `OPS` name to a method of the
target, plus `Coord`, `Normal`, `Restrict` and `Measure`. Two targets:

- **Weak form, via `FormBackend`.** Unknown → TrialFunction (linear) or Function
  (nonlinear); `v_w` → TestFunction; `Coord` → spatial coordinate; `Measure(Γ_N)` →
  `ds(tag)`; `Normal` → facet normal; essential conditions → `build_dirichlet`, with
  the value read from the condition's source terms via `split_constraint`.
- **Pointwise residual, via a DL backend with autograd** (PINNs, M10). The
  hand-written residual functions in the current PINN notebooks are what this target
  would generate.

Choices the equations cannot make go into a `Discretization` object:

- the mesh, from `Task.values[Omega]` or the domain's geometry
  (`MeshBackend.create_mesh`);
- the element per unknown, defaulting from `WeakProblem.spaces` (H1 → Lagrange 1,
  L2 → DG 0, H(div) → Raviart–Thomas 1), overridable (Taylor–Hood);
- the mapping from `Part`s to mesh tags (the open TODO in the main plan; today
  matched by name).

## 6. Backend issues fixed on the way

All fixed (2026-11-24); `tests/test_backends/test_fenicsx.py` covers them and runs
where dolfinx is installed.

- `FenicsXFormBackend.derivative` used `ufl.derivative`, a Gateaux derivative with
  respect to a coefficient; it is now the partial derivative `u.dx(i)`, and the
  abstract signature takes the direction.
- `FormBackend` gained `sin`, `cos`, `exp`, which the equations use.
- dolfinx's `LinearProblem.solve()` reassembles and refactors `A` on every call. The
  FEniCSx handle is now a `ReusableLinearProblem`, and
  `SolveBackend.solve(handle, reassemble_operator=False)` reuses the operator and
  its factorization, reassembling only the right-hand side (table in 4.2, first row).
- `FenicsXMeshBackend.get_subdomain` now returns the submesh and its entity map.
- A linear `SolveResult` reports the KSP's convergence reason, iterations and
  residual norm instead of a fixed `converged=True, iterations=1`.
- Also: `LinearSolverOptions.backend_options` now take precedence for linear solvers
  too (they were only applied to nonlinear ones), and every handle gets its own PETSc
  options prefix.

Still open: `FormBackend.time_derivative` goes once `discretize_time` exists (B3).

## 7. Order of work

| # | Content | Done when |
|---|---|---|
| B1 | Lowering visitor and the FEniCSx weak-form target | stationary Poisson solves and matches a manufactured solution |
| B2 | `FEMSolveNode` with ports from the task; operator assembled once | repeated solves with a varying source reuse one factorization |
| B3 | `discretize_time` (θ-family) and `ScanNode` | heat converges in time at the expected order |
| B4 | Batching: many right-hand sides, then the affine operator | a varying-source sweep runs as one node |
| B5 | Pointwise lowering for PINN residuals | a Poisson PINN trains from the compiled residual |

## 8. Open decisions

- Time discretization in the equations package (proposed) or in the backend?
- Port payload: dof arrays (proposed) or backend `Function` objects?
- Loop: generic `ScanNode` (proposed) or a FEM-specific time-loop node first?
