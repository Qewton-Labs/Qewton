# Problem graph: architecture plan

Status: draft. Scope: representing PDE problems as a typed graph and analysing their structure, as the basis for suggesting numerical methods.

---

## 1. Goals

- Represent PDE problems (PDEs, boundary conditions, initial conditions, inequalities) as a typed, immutable graph.
- Infer what users don't declare: function signatures (input axes, output shape) and the domains of data functions.
- Analyse the structure: derivative orders, equation type, boundary-condition kinds, coupling between unknowns, linearity, and where batched parameters enter.
- Suggest discretisation and solver strategies, each with a traceable rationale.
- Support many tasks on one model: forward solve, parametric batches, inverse problems, operator learning.
- Compile the strong form automatically into a weak formulation, and make it exportable (e.g., via a conversion function to UFL (FEniCS / Firedrake) or other software).
- Expose problems to neural networks in two ways: as a graph input (tensorisation) and as a training objective (PINN, variational and energy losses).

Non-goals for now: writing our own assembler or mesher (weak forms go to UFL instead), and symbolic simplification beyond what the analysis needs.

---

## 2. Design principles

1. **The model and the task are separate.** `Problem` contains only mathematics and no roles. `Task` assigns roles (unknown, fixed, batched, observed) to symbols and sets.
2. **A signature is not a role.** A function's inputs are always declared on the symbol, and its output shape is declared or inferred. Whether it is unknown belongs to the task. "Constant" is a function with `inputs=None`, not a role. Batch axes are added by the task (`batch_axes`), so the signature itself is geometric; reports show both.
3. **Three layers with typed edges.** The layers are geometry (axes, sets), expressions (symbols, operators) and constraints. The edge types are: *uses*, *defined on*, *holds on*, *subset of*, *boundary of*.
4. **Nodes are immutable; analysis results live in side tables.** Nodes are frozen dataclasses, so they are hashable and can be shared. Passes write their results into an `AnalysisContext` keyed by node. That way the same model can be analysed under different tasks without being mutated.
5. **Operators are data.** There is one `Apply` node plus an operator registry, not one class per operator. Operator semantics (shape rule, derivative order, linearity) live in an `OpSpec`.
6. **Explicit over clever.** Evaluation on a lower-dimensional set is an explicit `Restrict` node (spatial trace or time slice). Normalisation rewrites notational conventions (e.g. "in Ω" for a transient problem) explicitly and reports what it changed.

---

## 3. Package layout

M1–M6 import nothing numeric except `ContinuousGeometry`, used by `Domain` to couple a domain to its geometry. Meshes and everything else numeric stay out until M10. Reuse, if any, starts at M10.

```
qewton/equations/
    ├── geometry/
    │   ├── coordinates.py       # check_coordinate: a coordinate is a config Variable with an integer dim
    │   └── sets.py              # SetNode hierarchy, Partition, subset queries
    ├── expr/
    │   ├── nodes.py             # Expr, FunctionSymbol, Coord, Const, Normal, Apply, Restrict
    │   ├── ops.py               # OpSpec and the operator registry (grad, div, dt, ...)
    │   ├── forms.py             # TestFunction, Measure, Integral
    │   └── api.py               # user-facing helpers: grad, div, curl, lap, d, dt, dot, inner, sin, cos, exp (Eq goes in constraints.py)
    ├── constraints.py           # Relation, Constraint, PendingConstraint
    ├── model.py                 # Problem
    ├── task.py                  # Task, Role
    ├── analysis/
    │   ├── context.py           # AnalysisContext, Fact, Suggestion, Diagnostic
    │   ├── pipeline.py          # run_pipeline(task)
    │   ├── validate.py
    │   ├── normalize.py
    │   ├── infer.py             # signatures and shapes
    │   ├── structure.py         # orders, operator/source split, linearity, coupling
    │   ├── classify.py          # PDE type, BC kinds, function spaces
    │   └── suggest.py           # method suggestions with rationale
    ├── compile/
    │   ├── weak_form.py         # strong form → WeakProblem
    │   ├── ufl_export.py        # WeakProblem → UFL (FEniCS / Firedrake)
    │   ├── samplers.py          # SetNode → point sampler (interior, boundary parts, slices)
    │   ├── losses.py            # constraints / weak forms → PINN, VPINN, Deep Ritz losses
    │   └── tensorize.py         # Problem + Task → heterogeneous graph tensors
    ├── report.py                # human-readable summary (M6)
    ├── pretty.py                # readable text for expressions, sets, constraints, problems (M1)
    ├── printable.py             # mixin: str() and notebook display through pretty
    ├── fixtures.py              # canonical problems: Poisson, heat, wave, advection, Stokes, obstacle
src/qewton/equations/tests/      # all tests live under the equations folder
src/qewton/equations/notebooks/  # notebooks; import everything as `import qewton.equations as eq`
    └── test_<pass>.py           # golden tests per pass
```

---

## 4. Mapping from the current code

| Current | Planned | Reason |
|---|---|---|
| `EquationNode` (common base) | Separate hierarchies `SetNode`, `Expr`, `Constraint`. Optionally a minimal `Node` base with only `children()` for traversal. | These are different kinds of object; a shared base invites mixing them. |
| `AxisNode(variable)` | `Variable` (unchanged) | The kind (space, time, parameter) is not stored on the variable. The problem declares its time variable, and the Task creates parameter variables for batched symbols. A coordinate can be an argument of a derivative operator. |
| (none) | `Coord(axis, component)` | The coordinate function x ↦ xᵢ, usable inside expressions, e.g. `sin(x[0])`. |
| `OperationNode` | `Apply(op, args)` plus the `OpSpec` registry | Operators as data. |

`Variable` is the coordinate type. It stays the config object and is used directly, with no wrapper. The kind is not stored on it: `Problem(time=t)` declares the time variable, the Task creates parameter variables for batched symbols, and every other coordinate is spatial. One change lives outside the equations folder: `Variable.__mul__` (and `+`) return `NotImplemented` for non-`Variable` operands, so `t * u` works. Multi-component coordinates like `x` need a component: `Coord(x, 0) * u`. If `Variable` is also used for unknowns like `u`, split that use off into `FunctionSymbol`, since `dim` means input dimension for a coordinate but output size for a function.

---

## 5. Core classes (sketch)

### 5.1 Geometry

```python
# A coordinate is a config Variable with an integer dim. There is no wrapper class.
# The kind (space, time, parameter) is not stored on it: Problem(time=...) declares
# the time coordinate, and the Task creates parameter coordinates for batched symbols.
x = Variable("x", dim=2)        # space: components x_1, x_2
t = Variable("t", dim=1)        # time: declared as Problem(time=t)
Coord(x, 0)                     # the component x_1
```

```python
@dataclass(frozen=True)
class SetNode:
    @property
    def axes(self) -> frozenset[Variable]: ...
    @property
    def codim(self) -> int: ...       # 0 for open sets, 1 for boundaries, ...

@dataclass(frozen=True)
class Domain(SetNode):          # named open set, e.g. Ω ⊂ ℝ³
    name: str
    ambient: Variable | None = None
    geometry: ContinuousGeometry | None = None   # optional; its coordinate becomes ambient; not part of equality; meshes are bound in the Task

@dataclass(frozen=True)
class Boundary(SetNode):        # ∂S; codim increases by 1
    of: SetNode

@dataclass(frozen=True)
class Part(SetNode):            # named piece of a set, e.g. Γ_D ⊂ ∂Ω
    name: str
    of: SetNode

@dataclass(frozen=True)
class Product(SetNode):         # e.g. Ω × I
    factors: tuple[SetNode, ...]

@dataclass(frozen=True)
class Slice(SetNode):           # {axis = value}, e.g. {t = 0}
    axis: Variable
    value: float

@dataclass(frozen=True)
class Partition:                # declaration: the parts cover `whole` and are disjoint
    whole: SetNode
    parts: tuple[Part, ...]
```

All subset reasoning lives in one function, `is_subset(a, b)` in `sets.py`, which walks the `Part`, `Boundary` and `Product` edges. A time slice is always a factor of a `Product`: Ω × {0} is `Product((Omega, Slice(t, 0.0)))`, and the time interval is `Domain("I", ambient=t)`. A parametrized (batched) domain, `ParametrizedDomain(reference, map)`, comes in milestone M7.

### 5.2 Expressions

```python
class Expr:   # plain class, not a dataclass: a generated __eq__ would be inherited by FunctionSymbol
    def __add__(self, other): return Apply(OPS["add"], (self, as_expr(other)))
    # The arithmetic operators build Apply nodes, reflected forms included. Division is mul with pow(-1). A one-dimensional coordinate works as an operand, e.g. t * u.
    # Do NOT overload __eq__: structural equality is needed for hashing.

@dataclass(frozen=True, eq=False)       # compared by identity: two "u"s are different symbols
class FunctionSymbol(Expr):
    variable: Variable                 # name and output: dim 1 scalar, n vector, tuple tensor, None inferred
    inputs: Variable | None             # always given: product of coordinates, e.g. x * t (order = input layout); None = constant

@dataclass(frozen=True)
class Coord(Expr):              # coordinate function x ↦ x_i
    variable: Variable
    component: int                  # 0-based; must be below variable.dim

@dataclass(frozen=True)
class Const(Expr):              # number, vector or matrix (nested tuples); not typing.Literal, which the annotations below use
    value: float | tuple

@dataclass(frozen=True)
class Normal(Expr):             # outward unit normal; only on a Boundary or a Part of one
    of: SetNode

@dataclass(frozen=True)
class Apply(Expr):
    op: "OpSpec"
    args: tuple[Expr | Variable, ...]   # expressions, then one coordinate product if the operator takes axes, e.g. grad(u, x)

@dataclass(frozen=True)
class Restrict(Expr):           # expr evaluated on a set (trace, time slice); lower-dimensionality is checked in M2
    expr: Expr
    on: SetNode
```

```python
Shape = tuple[int, ...]         # () scalar, (3,) vector, (3, 3) tensor

@dataclass(frozen=True)
class Signature:
    axes: frozenset[Variable] | None = None     # input coordinates, from the declared inputs; frozenset() means constant
    shape: Shape | None = None
    domain: SetNode | None = None           # inferred: smallest set on which it is evaluated
```

### 5.3 Operator registry

```python
@dataclass(frozen=True)
class OpSpec:
    name: str
    arity: int                          # number of expression arguments
    shape_rule: Callable[[tuple[Shape, ...], Variable | None], Shape]  # argument shapes and derivative axes in, shape out; raises ShapeError
    takes_axes: bool = False            # a coordinate product follows the expression arguments
    order: int = 0                      # derivative order along each leaf of the axes argument
    linearity: Literal["linear", "multilinear", "nonlinear"] = "linear"

OPS: dict[str, OpSpec] = {}
def register(spec: OpSpec) -> OpSpec: ...
```

The initial registry:

| op | expression args | shape rule (derivative axes are passed separately) | order | linearity |
|---|---|---|---|---|
| add, sub | 2 | equal shapes (scalars broadcast) | 0 | linear |
| neg | 1 | same shape | 0 | linear |
| mul | 2 | scalar · any | 0 | multilinear |
| dot | 2 | contracts the last axis of the first with the first axis of the second: (k, k)·(k,) → (k,), (d,)·(d,) → () | 0 | multilinear |
| pow, exp, sin, cos | 1 or 2 | scalars only | 0 | nonlinear |
| d(u, a) (partial derivative along one coordinate) | 1 | same shape; a must be a single coordinate | 1 per leaf | linear |
| dt(u, t) | alias for d(u, t) | | | |
| grad(u, axes) | 1 | shape + (d,), d the total dim of the axes | 1 per leaf | linear |
| div(v, axes) | 1 | last axis must be d, which is removed | 1 per leaf | linear |
| curl(v, axes) | 1 | (3,) → (3,), with three coordinates | 1 per leaf | linear |
| inner | 2 | equal shapes → () (full contraction) | 0 | multilinear |
| lap(u, axes) | 1 | same shape | 2 per leaf | linear |

"Multilinear" means the operator is linear in each argument separately. An expression is linear in the unknowns if, at every multilinear node, at most one argument depends on the unknowns.

### 5.4 Constraints and model

```python
class Relation(Enum):
    EQ = "="
    LE = "<="
    GE = ">="
    COMPLEMENTARITY = "⊥"

@dataclass(frozen=True)
class Constraint:
    lhs: Expr
    rel: Relation
    rhs: Expr
    on: SetNode                 # the "for all points in" set
    name: str | None = None

@dataclass(frozen=True)
class PendingConstraint:          # not a Constraint yet: Problem refuses it
    lhs: Expr
    rel: Relation
    rhs: Expr
    def on(self, on, name=None) -> Constraint: ...

def Eq(lhs, rhs) -> PendingConstraint: ...   # also Le, Ge and Complementary

@dataclass(frozen=True)
class Problem:
    constraints: tuple[Constraint, ...]
    partitions: tuple[Partition, ...] = ()
    time: Variable | None = None    # the time coordinate; drives time lifting and time-derivative checks
    # derived: symbols(), sets(); coordinates() comes in M2 with the leaf rules
```

`Problem` has no roles.

### 5.5 Task

```python
class Role(Enum):
    UNKNOWN = auto()    # solve for this
    FIXED = auto()      # one known value; can be precomputed or constant-folded
    BATCHED = auto()    # known per instance; introduces a parameter axis
    OBSERVED = auto()   # known on part of the domain (inverse problems)

@dataclass(frozen=True)
class Task:
    problem: Problem
    roles: Mapping[FunctionSymbol | SetNode, Role]
    values: Mapping[FunctionSymbol | SetNode, object]   # Expr, array, mesh, samples
    batch_axes: Mapping[FunctionSymbol | SetNode, Variable] # created automatically for BATCHED entries
```

User-facing API:

```python
task = problem.task(unknown=[u],
                    fixed={f: sin(x[0]), Omega: mesh},
                    batched={nu: nu_samples})
```

Unknowns must be listed. Any other free symbol defaults to FIXED; if it has no value, that's a warning ("symbolic fixed").

---

## 6. Analysis pipeline

```python
ctx = run_pipeline(task)   # validate → axis pre-pass → normalize → infer → structure → classify → suggest
print(report(ctx))
```

Each pass has the form `run(task, ctx) -> None`. It reads the tables of earlier passes, writes its own, and appends diagnostics. No pass mutates nodes; normalisation builds a new `Problem` and stores it in `ctx.problem`.

```python
@dataclass
class AnalysisContext:
    task: Task
    problem: Problem                                  # the normalised version
    signatures: dict[FunctionSymbol, Signature]
    shapes: dict[Expr, Shape]
    orders: dict[tuple[Constraint, FunctionSymbol], dict[Variable, int]]
    coupling: "BipartiteGraph"
    facts: list["Fact"]             # e.g. Fact("pde_type", c1, "elliptic", evidence=[...])
    suggestions: list["Suggestion"]
    diagnostics: list["Diagnostic"] # severity: error | warning | info
```

### 6.0 axis pre-pass

A syntactic scan of the constraints: which unknowns have a time derivative (`d(u, t)`) or appear under a time `Slice`. It runs before normalize, so time lifting does not depend on inference.

### 6.1 validate

- The axes of every constraint's set are known axes, and `Part`/`Boundary` chains are well-formed.
- Every declared `Partition` is consistent.
- Every unknown listed in the task appears in at least one constraint.

### 6.2 normalize

- Bring every constraint into the form `lhs − rhs REL 0`, keeping the original for display.
- **Time lifting.** If the pre-pass found a time derivative or time slice for an unknown, a constraint holding on S without time is rewritten to hold on S × I. Constraints containing a `Slice` (initial conditions) are left alone. Each lift is reported as an info diagnostic.
- **Batch lifting.** Constraints that involve a batched symbol get its parameter axis (× P).
- Choose canonical forms: expand `lap(u, axes)` into `div(grad(u, axes), axes)`, so that −div(ν∇u) and −νΔu are analysed consistently.

### 6.3 infer: signatures and shapes

- **Inputs are declared, not inferred.** Usage cannot tell a constant from a function that is not differentiated along x, so no inputs are inferred from where a symbol is used. An unknown with `inputs=None` is an error.
- **Lower bound check.** If a derivative along axis a is applied directly to s (or to `Restrict(s, …)`), then a must be among s's declared inputs, or the problem is an error. Nested cases such as grad(u·v) give no requirement.
- **Data values.** A supplied value must agree with the declared inputs. A constant (`inputs=None`) cannot get a spatial array, and a numeric value with spatial inputs needs a geometry through `Task.values`. This is checked when the Task is built.
- **Conflicts and unused inputs.** A declared output shape must agree with the inferred one. Declared inputs that no constraint needs are a warning.
- **Shapes.** Compute them bottom-up with `OpSpec.shape_rule`. Unknown shapes are variables, unified across all occurrences until a fixpoint is reached. Anything still unresolved defaults to scalar with a warning.
- The declared inputs and the inferred shapes are printed first in the report, the way a type checker shows types.

### 6.4 structure

- **Derivative orders.** For each (constraint, unknown) pair, the maximum order per axis, accumulated through differential operators while walking the tree.
- **Operator/source split.** Expand top-level sums. Terms that depend on unknowns form the operator part; the rest form the source part.
- **Linearity.** Classify each term as linear, affine or nonlinear in the unknowns, using `OpSpec.linearity`.
- **Batch placement.** Record whether each batched symbol appears in the operator part or only in the source part, and whether it enters affinely (term = θ(μ) · term′, where term′ does not depend on μ). A symbol with no spatial axes can be pulled out of `div`, `grad` and `lap`. That rule is what makes `div(nu * grad(u, x), x)` affine, so the structure pass must state it explicitly.
- **Coupling graph.** A bipartite graph between scalar equations and scalar unknowns (a vector constraint or unknown contributes one node per component). Maximum matching gives a structural well-posedness check; strongly connected components or a Dulmage–Mendelsohn decomposition give the block-triangular order. networkx provides the matching and the SCCs. A Dulmage–Mendelsohn decomposition is built from those two, not from one library call.
- **Decoupled axes.** If no derivative of any unknown along axis a appears anywhere, the problem decouples along a: pointwise in space, quasi-static in time, or parallel over a batch.

### 6.5 classify

- **PDE type** (heuristics, to be extended):
  - order 2 in space, no time derivative: elliptic candidate
  - order 1 in time, order 2 in space: parabolic
  - order 2 in time, order 2 in space: hyperbolic (wave type)
  - order 1 in time, order 1 in space, no second-order terms: hyperbolic (transport)
  - order 0: algebraic / pointwise
  - a coupled system with a zero diagonal block: saddle point
- **BC kinds.** `Restrict(u)` is essential (Dirichlet). `Restrict` of a derivative is natural (Neumann). Both in one condition is Robin.
- **Counts.** For a second-order elliptic problem, every boundary point gets one condition per scalar component of the unknown (Dirichlet on a d-vector is d conditions), so the parts of the `Partition` must be covered per component. For order k in time, k initial conditions are needed. Hyperbolic problems need inflow information, which is flagged.
- **Function spaces** per unknown, from the weak form (§7.1), not from which operators appear. Integration by parts decides: p in ∇p·v only needs L², so Stokes has u ∈ H¹ and p ∈ L². The operator-based rule below is only a first guess: grad gives H¹, div only gives H(div), curl gives H(curl), none gives L².

### 6.6 suggest

A rule table: conditions on facts produce a `Suggestion(method, evidence)`.

| Facts | Suggestion |
|---|---|
| elliptic, linear | continuous Lagrange FEM; CG with multigrid, or a direct solver |
| parabolic | method of lines with implicit time stepping (BDF, Crank–Nicolson) |
| hyperbolic transport | FV or DG with an upwind flux; explicit Runge–Kutta under a CFL limit |
| saddle point | inf-sup stable pairs (Taylor–Hood); block preconditioner |
| nonlinear in an unknown | Newton (or Picard) iteration |
| inequality / complementarity | active-set or semismooth Newton |
| zero-order constraint | pointwise solve; boundary conditions are an error |
| decoupled along t | a sequence of stationary solves |
| batched only in the source part | factor once, solve with many right-hand sides |
| batched affinely in the operator part | reduced basis with an offline/online split |
| batched non-affinely | empirical interpolation (EIM) or a learned surrogate |
| task asks to generalise to unseen parameters | DeepONet, FNO, or a parametric PINN |
| batched domain with fixed topology | pull back to a reference domain |

Every `Fact` and `Suggestion` carries its evidence (the nodes involved and the rule id), so the report can explain why each suggestion was made.

---

## 7. Compilation targets

The strong-form `Problem` stays the single source of truth. Everything in this section is a compiler from (`Problem`, `Task`, `AnalysisContext`) to another representation. None of them modifies the model, and all of them reuse results of the analysis passes.

### 7.1 Weak formulation

New node types in `expr/forms.py`:

```python
@dataclass(frozen=True, eq=False)
class TestFunction(Expr):
    of: FunctionSymbol              # the unknown whose space it shares

@dataclass(frozen=True)
class Measure:
    on: SetNode                     # Ω → dx, Γ_N → ds(Γ_N)

@dataclass(frozen=True)
class Integral(Expr):
    integrand: Expr
    measure: Measure

@dataclass(frozen=True)
class WeakProblem:
    unknowns: tuple[FunctionSymbol, ...]
    spaces: dict[FunctionSymbol, "SpaceSpec"]   # H1, H(div), ...; from classify (6.5)
    essential: tuple[Constraint, ...]          # imposed on the space, not in the form
    residuals: tuple[Expr, ...]                             # one per test function: F_i(u; v_i), a sum of Integrals; = 0 for all v_i
    kind: Literal["equation", "variational_inequality"]
    bilinear: Expr | None = None               # a(u, v), if linear
    linear: Expr | None = None                 # ℓ(v), if linear
    energy: Expr | None = None                 # J(u) = ½ a(u, u) − ℓ(u), if symmetric and coercive
```

Algorithm (`compile/weak_form.py`):

1. **Pair constraints with test functions.** The coupling graph (6.4) gives a matching, but it only fixes the pairing for square, structurally nonsingular systems. Stokes: continuity div u = 0 does not contain p, so the matching is momentum ↔ p and continuity ↔ u. The test-space pairing (momentum tested with V, continuity tested with Q) is the inf-sup pairing, a modelling choice recorded explicitly, not derived from the matching.
2. **Test and integrate.** For each interior constraint `lhs − rhs = 0` on S, add `Integral(inner(lhs − rhs, v), Measure(S))`. For constraints on S × I, the measure is over S only (method of lines; see step 5).
3. **Integrate by parts** using rewrite rules on the expression tree:
   - ∫_Ω −div(σ)·v dx → ∫_Ω σ : ∇v dx − ∫_∂Ω (σ·n)·v ds
   - ∫_Ω ∇p·v dx → −∫_Ω p div(v) dx + ∫_∂Ω p (v·n) ds
   - ∫_Ω curl(E)·v dx → ∫_Ω E·curl(v) dx + ∫_∂Ω (n × E)·v ds

   Policy: integrate by parts until the derivative order on the trial function is no more than its space allows (H¹: order 1). This is the default "primal" formulation. First-order transport terms are not integrated by parts by default (DG comes later), and a "mixed" option, with the flux σ as an extra unknown in H(div), comes later.
4. **Handle the boundary terms** using the `Partition` of ∂Ω:
   - **Essential BC on a part.** The test function vanishes there (v ∈ V₀), so the boundary term is dropped. The BC goes into `essential`, and the trial space becomes g + V₀.
   - **Natural BC on a part.** The BC's expression must match the boundary flux σ·n produced in step 3 (the conormal derivative of the operator). If it matches, substitute the given value, which gives −∫_{Γ_N} h·v ds. If it doesn't, for example when the user wrote ∂ₙu = h but the flux is ν∂ₙu, emit a warning and require an explicit choice.
   - **Robin BC** (σ·n + αu = h). Substitute σ·n = h − αu, which gives ∫ αu·v ds − ∫ h·v ds.
   - **A part with no BC.** This is an error, unless classification says none is needed there (e.g. outflow for transport).
5. **Time.** The weak form is built in space only. ∂ₜu contributes the mass term ∫ ∂ₜu·v dx, and initial conditions stay as constraints on Ω × {0}. Time stepping is chosen by the suggestion pass.
6. **Inequalities.** A constraint with LE, GE or ⊥ on an unknown makes `kind = "variational_inequality"`, with a convex set K, and the problem becomes: find u ∈ K with F(u; v − u) ≥ 0 for all v ∈ K.
7. **Split and symmetry.** If the problem is linear, split the residual into a(u, v) and ℓ(v), reusing the operator/source split (6.4). If a(u, v) is structurally equal to a(v, u) after swapping trial and test, and a is coercive (positive definite; see §10), also emit the energy J(u). Symmetry alone is not enough: Helmholtz and wave forms are symmetric but indefinite, so J is unbounded below. The structural check is conservative and may miss symmetric forms that need algebra. Otherwise keep F(u; v) for Newton.

Export (`compile/ufl_export.py`): `WeakProblem` maps almost one-to-one onto UFL (`TrialFunction`, `TestFunction`, `inner`, `grad`, `dx`, `ds(tag)`, `DirichletBC`), with boundary `Part`s becoming mesh tags. FEniCS or Firedrake then does the assembly and solving.

### 7.2 Neural network interfaces

"Feeding the problem into a network" covers two different things. Both are supported, as separate compilers.

**(a) The problem as input: `tensorize.py`.** Turn the normalised `Problem` plus `Task` into a heterogeneous graph (PyTorch Geometric `HeteroData`, or DGL) for a GNN encoder. Uses include predicting a solver, or conditioning a neural operator on the equation itself.

| Graph element | Encoding |
|---|---|
| node types | axis, set, symbol, literal, coord, normal, apply, restrict, constraint |
| apply node features | op id from the registry (embedding), derivative order, linearity |
| symbol node features | role (from `Task`), inferred shape, axis kinds (multi-hot), function space |
| set node features | set kind (domain, boundary, part, product, slice), codim, axis kinds |
| literal node features | scaled value plus a literal flag |
| constraint node features | relation, inferred PDE type, BC kind |
| edge types | `arg` (with the argument index as edge feature, needed for sub, div and other non-commutative ops), `defined_on`, `holds_on`, `restrict_to`, `subset_of`, `boundary_of`, `part_of` |

Rules:

- **Tensorise after normalisation.** Two formulations must give the same graph if they differ only in argument order of commutative ops, in sign placement (`-a = b` and `a + b = 0` both become `lhs − rhs = 0`), or in `lap` versus `div ∘ grad`. Algebraic identities are out of scope (non-goal), so M11 tests only such pairs. Otherwise the network has to learn invariances it shouldn't need to.
- **Keep sharing.** Hash-consing makes shared subexpressions shared nodes, which suits message passing.
- **Version the vocabulary.** Op and node-type ids are derived from the registry and versioned (`VOCAB_VERSION`), because a trained model depends on them.
- **Numeric data are not graph nodes.** Data functions (f, samples of ν) go through a function encoder (e.g. values at sensor points, like the DeepONet branch net) and are attached to their symbol node as features. Geometry (meshes, point clouds, signed distance functions) goes through a geometry encoder attached to set nodes. That gives three channels: symbolic structure (GNN), data (function encoders) and geometry (geometry encoders).
- **Batches.** The batch dimension of the tensors is the parameter axis P.

**(b) The problem as objective: `samplers.py` and `losses.py`.** Compile the problem into training losses for a network u_θ.

- **Samplers.** Every `SetNode` gets a sampler: the interior of a `Domain` (from a mesh, or by rejection sampling), `Boundary` and `Part` (from tagged boundary facets), `Product` (by product sampling), `Slice` (by fixing the coordinate). Parameter axes are sampled or enumerated.
- **Network inputs.** The input layer is the declared inputs of the unknown, x and t here, plus the parameter axes of batched symbols. The declared inputs therefore determine the input layer directly.
- **PINN losses.** Each constraint becomes a mean squared residual over samples from its `on` set, with one weight per constraint (configurable, possibly adaptive). Derivatives come from autodiff (`torch.func` or JAX).
- **Inequalities.** Use a penalty, or a complementarity residual such as min(u − ψ, −Δu − f), or a Fischer–Burmeister function.
- **Variational losses from the `WeakProblem`.** Either a variational PINN (test the residual against a finite set of test functions), or Deep Ritz (minimise J(u) when `energy` is present).
- **Essential BCs.** Either as a loss term, or as a hard constraint through the ansatz u_θ = g̃ + φ·N_θ, where φ vanishes on Γ_D (e.g. a distance function). Use the hard constraint whenever a distance function for the part is available.

---

## 8. Worked example (target for milestones M1–M6)

```python
x = Variable("x", dim=2)
Omega = Domain("Omega", ambient=x)
dO = Boundary(Omega)
GD, GN = Part("Gamma_D", dO), Part("Gamma_N", dO)
u  = FunctionSymbol(Variable("u"), x)        # unknown on Omega
f  = FunctionSymbol(Variable("f"), None)     # constant source
g  = FunctionSymbol(Variable("g"), x)        # Dirichlet data on Gamma_D
h  = FunctionSymbol(Variable("h"), x)        # Neumann data on Gamma_N
nu = FunctionSymbol(Variable("nu"), None)    # batched coefficient, constant in x
n = Normal(dO)

problem = Problem(
    constraints=(
        Eq(-div(nu * grad(u, x), x), f).on(Omega),
        Eq(Restrict(u, GD), g).on(GD),
        Eq(dot(Restrict(nu * grad(u, x), GN), n), h).on(GN),
    ),
    partitions=(Partition(dO, (GD, GN)),),
)

task = problem.task(unknown=[u],
                    fixed={f: 1.0, g: 0.0, h: 0.0},
                    batched={nu: nu_samples})
```

Expected report:

```
Signatures
  u  : x × P → ℝ   declared inputs x; coupled to batched nu
  nu : P → ℝ                    batched; declared constant in x (inputs None)
  f  : ℝ                        fixed constant
Structure
  C1: order 2 in x on u; linear; nu in operator part, affine (nu · A0)
  C2: essential BC on Gamma_D
  C3: natural BC on Gamma_N
  partition of ∂Omega fully covered
  decoupled along P → parallel over batch
Classification
  C1: elliptic (order 2, no time derivative); u ∈ H¹
Suggestions
  continuous Lagrange FEM            ← elliptic, u ∈ H¹
  factor A0 once, scale per instance ← nu batched, enters as a scalar factor of the operator
```

---

## 9. Milestones

| # | Content | Done when |
|---|---|---|
| M1 | Geometry, expressions, constraints, pretty printer | Poisson and heat fixtures build and print |
| M2 | Task (roles, batch axes), validate, normalize, infer; signature report | golden tests for signatures and shapes pass |
| M3 | structure: orders, operator/source split, linearity, coupling graph | Stokes coupling graph shows the saddle-point block |
| M4 | classify: PDE type, BC kinds and counts, function spaces | wave, advection, Stokes and obstacle fixtures classified correctly |
| M5 | Batch placement and batched reporting (parameter axes) | worked example (Section 8) reports the batch correctly |
| M6 | suggest and report with rationale | every suggestion lists its evidence |
| M7 | parametrized domains (reference maps), OBSERVED role | batched-domain fixture uses the pull-back |
| M8 | weak_form: pairing, testing, integration by parts, boundary terms | Poisson, heat and Stokes weak forms match hand-derived ones |
| M9 | UFL export | exported Poisson and Stokes problems solve in FEniCS or Firedrake and converge under refinement |
| M10 | samplers and PINN / Deep Ritz losses | a Poisson PINN trains from the compiled loss alone |
| M11 | tensorize | all fixtures convert to `HeteroData`; two equivalent formulations give identical graphs |

**Scope for the first implementation: M1–M6.** M7–M11 are deferred until the analysis core works.

---

## 10. Open questions

**Decided after review:**
- Time lifting uses a syntactic pre-pass (§6.0), not inferred signatures.
- No set union is needed for the axis checks. Signatures use unions of axis sets. Closure is not needed in M1–M6.
- Coordinates are plain `Variable`s. Their equality is `Variable`'s name-based rule, so inside one problem `x` means the same x everywhere. The kind comes from `Problem(time=...)` and the Task.
- The symbolic package is parallel to the numeric code. M1–M6 reuse nothing.
- Derivative operators take their axes as arguments: `grad(u, x)`, `lap(u, x)`, `div(v, x)`, `dt(u, t)`.
- `Domain` holds an optional continuous geometry, and its coordinate becomes the ambient. The geometry is excluded from equality. Meshes are bound in `Task.values`.
- `set_from_geometry(geometry, name)` turns a continuous geometry into a `Domain`, and a `ProductGeometry` into a `Product` whose factors are named after their coordinates. Discrete geometries are rejected.
- The `Literal` node is renamed `Const`.

The questions below are still open.

- Several batched symbols: zip them onto one shared P, or form a product P₁ × P₂? Proposal: zip by default, with an explicit `grid=` option for products.
- Checking sign and definiteness for "elliptic": do it symbolically (e.g. with sympy), or assume it and only flag?
- Closure: an explicit `Closure` set node, or implicit in the subset queries?
- Lateral boundary: ∂(Ω × I) is not ∂Ω × I. Normalise boundaries of products into products of factors.
- Interoperability: export to sympy for simplification, or to UFL for FEM backends?
- Resolved: `Variable` gets no `kind`. The problem declares its time variable (see §5.1).
- Resolved: function inputs are always declared. `inputs=None` means constant, and nothing about inputs is inferred from usage.
- Time is declared on the problem (`Problem(time=t)`), not inferred and not a role. Mathematically t is just another independent variable. The analysis treats it specially because it is the evolution direction: classification (time versus space derivative order), initial conditions (slices in time), time lifting, and time stepping all depend on it. Steady problems have `time=None`.
- The package exports its user-facing names, so notebooks use `import qewton.equations as eq`. Notebooks live in `equations/notebooks/`.
- Integration-by-parts policy: primal by default, but when should the compiler offer a mixed formulation (flux as an extra unknown in H(div)) or DG (for transport)?
- Natural BCs written as ∂ₙu although the operator's flux is ν∂ₙu: fix automatically by multiplying with the coefficient, or always ask?
- Trained models depend on the vocabulary ids. How should new operators be handled without retraining: a reserved "unknown op" id, or op embeddings computed from features such as order and linearity?
- Which function encoder for data functions: values at fixed sensor points, Fourier coefficients, or a small neural field?

---

## 11. Conventions

- All nodes are frozen dataclasses. `FunctionSymbol` is compared by identity (`eq=False`); everything else is compared structurally.
- Never overload `==` on `Expr` to build equations; use `Eq()`.
- Passes never mutate nodes; they write into `AnalysisContext`.
- Every `Fact` and `Suggestion` carries its evidence.
- Every pass gets golden tests against the canonical fixtures.
