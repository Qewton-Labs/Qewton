# Code Documentation Guidelines

Documentation conventions for the Qewton backend (`src/qewton`): how
docstrings, comments, and per-package docs are written. All
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

Skip both where there's nothing to capture.

## Scope

Applies to new and modified code. Existing code is brought up to
standard in its own dedicated pass.

## Example

```python
class Renderer:
    """Backend that turns a Plot into a concrete figure.

    Args:
        figure (Figure): The figure this renderer builds and keeps in
            sync.

    Attributes:
        name (str): Identifier of the backend (e.g. ``"plotly"``).

    Notes:
        Subclasses must implement :meth:`setup`; others may be left as
        no-ops.
    """

    @property
    def name(self):
        return self._name

    @staticmethod
    def setup(figure: "Figure") -> "BackendFigure":
        """Build a new, empty backend figure for `figure`.

        Args:
            figure: The figure to build the backend figure for.

        Returns:
            BackendFigure: An empty figure sized to `figure`'s facet grid.
        """
        raise NotImplementedError
```
