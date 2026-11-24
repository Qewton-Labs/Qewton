# Code Design Guidelines

Architecture and design rules for the Qewton backend (`src/qewton`): how
concepts are mapped onto code. How code is documented is described in
[CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md). Package-specific
rationale and trade-offs belong in that package's `DECISIONS.md`, not
here.

## Concepts first

- Every class, attribute and function implements a concept that can be
  named and explained without referring to the implementation. If it
  cannot, the concept is missing or the code is in the wrong place.
- One concept, one name. A term means the same everywhere (code, docs,
  errors); two names for one concept or one name for two concepts are
  bugs.
- Before adding a class, check whether an existing concept already covers
  it (e.g. a product of discrete geometries is `ProductDiscreteGeometry`)
  and extend that instead. Prefer generalizing an existing class over
  adding a parallel one.

## Knowledge in one place

- Knowledge lives with the class it is about: what a geometry's points and
  cells are belongs to the geometry, how a mark is drawn belongs to the
  mark. Other code asks; it does not re-derive.
- No concept is implemented twice, e.g. two result types or two code paths
  for the same thing. Where a second way exists for historical reasons,
  it is removed when the first one can take over.
- Declarations are the single source: what a class declares (e.g. its
  channels) is read from the declaration, never repeated as a special case
  elsewhere.
- A class that can be saved implements `Serializable` (`config/saving`)
  once; nothing re-derives its own save/load logic for the same data.

## Polymorphism over type switches

- No `if`/`elif` chains over own classes or over strings standing for
  them. The class answers the question itself (a method or class
  attribute), so adding a variant means adding a class, not editing a
  switch.
- Mappings between independent hierarchies (e.g. result type to artist)
  are explicit tables, not chains.
- Concepts are types, not strings: classes or enums instead of
  `"grid"`/`"mesh"` and the like.
- Several related boolean flags on one constructor are usually a missing
  hierarchy, not a parameter set; model the variants as types instead.

## Interfaces

- Code uses other classes only through their public API: no access to
  `_private` members across modules, no `getattr`/`hasattr` duck typing
  on own classes. What is needed becomes public API, documented.
- Public signatures are explicit (no `**kwargs` passed through blindly
  where the accepted arguments are known).
- A package's public surface is exactly what its `__init__.py`
  re-exports; everything else is internal, even where importable.
- Layers depend in one direction only (e.g. plots never import
  renderers).
- Framework-specific code — torch, tensorflow, the numpy fallback, or a
  visualization backend such as Plotly — lives only inside its own
  backend/renderer module, behind the shared protocols (e.g.
  `backends/base.py`). Core logic never imports a specific framework
  directly.
- An optional dependency (e.g. a rendering export library) is imported
  lazily, where it is used, never as a hard import of the package.

## Size and structure

- A function does one thing and is short enough to read at once; a long
  function is split along its steps.
- A module holds one concept or one closely related group. Helpers used by
  one class live with it; helpers shared by several live in a module of
  their own with a name for what they share.
- Private helper classes are a warning sign: either they are a real
  concept (then name and document them, and give them a proper place) or
  the knowledge belongs to an existing class.

## Errors and behavior

- Invalid input fails as early as possible (at construction, not at
  drawing) with a `ValueError` that names the problem and, where there is
  one, the fix.
- Nothing requested is dropped silently: where something cannot be done,
  a defined fallback is used and a warning says so.
- Progress and diagnostic output goes through `logging`, never `print`.
- While the framework is in development, a cleaner design takes
  precedence over backward compatibility: breaking changes are made as a
  hard cut, without aliases.

## Verification

- Behavior is verified by tests; combinations of features are verified by
  test matrices over all of them, with the implausible combinations listed
  explicitly with their reason.
- Behavior shared across backends is tested once and run parametrized
  over every backend; results are compared with an explicit numeric
  tolerance, not exact equality.
- A `Serializable` class's save/load round-trip is covered by a test.
- Known defects are marked in the tests (strict xfail with a reason), not
  left untested.