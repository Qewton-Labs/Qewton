# Code Documentation Guidelines

Documentation conventions for the Qewton backend (`src/qewton`): how
docstrings, comments, and per-package docs are written. Design rules for
the code itself are in [CODE_DESIGN.md](CODE_DESIGN.md). All
documentation it covers — docstrings, comments, `README.md`,
`DECISIONS.md` — is written in English.

## Writing

Keep all documentation, this file included, compact: state rules and
behavior, not side details. Don't record each change together with its
cause. A reason is written down only if it stays important long-term,
beyond the current piece of work — usually in `DECISIONS.md`, in a
docstring when it is needed to use the code correctly.

## Docstrings

- Google style, wrapped at the code's line length. A single Sphinx
  cross-reference that can't be wrapped may exceed it.
- Required on every public module, class, function, and method,
  including dunder methods (`__add__`, `__getitem__`, ...). Not required
  on `_private` members; add a short comment there instead.
- Omit a parameter's type in the docstring when the signature already
  carries a type hint: `name: description`. Otherwise give it as
  `name (type): description`.
- Describe current behavior only. Rationale, alternatives, and history
  go in `DECISIONS.md`.
- No `Examples` section or extra background beyond what was asked.
  Raise anything beyond that for discussion first.
- Section order: summary → `Args` → `Returns` / `Yields` → `Raises` →
  `Attributes` (classes) → `Notes` → `See Also`.
- `__init__` stays undocumented. Its parameters go under `Args` in the
  class docstring.
- Properties are documented once, under `Attributes` in the class
  docstring.
- Validate inputs by raising exceptions such as `ValueError`, not with
  `assert`. Existing `assert`s are replaced over time; until then they
  are documented under `Raises` as `AssertionError`.
- Abstract base classes get a full docstring plus `Notes` on what
  subclasses must or may override.
- Link to other Qewton symbols with Sphinx roles, e.g.
  `` :class:`~qewton.visualization.renderers.base.Renderer` ``.

## Comments

May explain non-obvious implementation detail. Never recount decision
history or alternatives — see `DECISIONS.md`.

## Per-package docs

Each package directory (has `__init__.py`) worth documenting gets:

- **`README.md`** — purpose, public API, usage. Becomes the package's
  module docstring automatically via `docs/source/conf.py`; `__init__.py`
  then carries no module docstring of its own. Explains how the
  package's pieces relate.
- **`DECISIONS.md`** — design rationale, rejected alternatives. Linked from the bottom of `README.md`.
- **`REQUIREMENTS.md`** (optional, for packages with an architecture of
  their own) — Part 1: scenarios (`S1`, ...) and general requirements
  (`G1`, ...), independent of the solution. Part 2: the architecture, one
  section per component with its requirements (IDs prefixed per
  component, e.g. `P1`), the general requirements they follow from, and
  their status.

Skip each where there's nothing to capture. Where a `REQUIREMENTS.md`
exists, `README.md` and `DECISIONS.md` follow its components in the same
order: the README explains how to use them, without requirements;
`DECISIONS.md` groups decisions under the component they concern, each
heading naming the requirement IDs it serves. Subpackage docs stay short
and add only what is specific to the subpackage.

## Scope

Applies to new and modified code. Existing code is brought up to
standard in its own dedicated pass.

## Example

```python
class VectorScale(Scale):
    """Maps vectors to arrows: trained on the vector magnitudes, it scales
    the longest vector to `length`.

    Args:
        length: Length of the longest arrow in data units; None for the
            typical point spacing of the drawn positions.
        normalize: Draw every arrow with the same length.

    Attributes:
        length (float or None): Length of the longest arrow.
        normalize (bool): Whether every arrow has the same length.
    """

    def arrows(self, vectors: np.ndarray, spacing: float) -> np.ndarray:
        """The arrows of `vectors`, in data units.

        Args:
            vectors: Shape ``(n, dim)``.
            spacing: Typical point spacing of the drawn positions.

        Returns:
            numpy.ndarray: Shape ``(n, dim)``.
        """
```
