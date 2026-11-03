from qewton.equations import Domain, Eq, FunctionSymbol, Problem, Variable
from qewton.equations.analysis.coupling import coupling_graph
from qewton.equations.fixtures import poisson, stokes


def _named(problem, name):
    return next(s for s in problem.symbols() if s.variable.name == name)


def test_stokes_has_a_full_matching_and_one_saddle_point_block():
    problem = stokes()
    task = problem.task(unknown=[_named(problem, "u"), _named(problem, "p")])
    graph = coupling_graph(task)

    assert len(graph.equations) == 3
    assert len(graph.unknowns) == 3
    assert len(graph.edges) == 8
    assert len(graph.matching) == 3
    assert graph.has_full_matching
    assert len(graph.blocks) == 1
    assert len(graph.blocks[0].equations) == 3
    assert {node.symbol.variable.name for node in graph.blocks[0].unknowns} == {"u", "p"}


def test_vector_equations_connect_conservatively_to_unknown_components():
    problem = stokes()
    task = problem.task(unknown=[_named(problem, "u"), _named(problem, "p")])
    graph = coupling_graph(task)
    momentum_edges = [edge for edge in graph.edges if edge[0].constraint == 0]

    assert len(momentum_edges) == 6
    assert {equation.component for equation, _ in momentum_edges} == {0, 1}
    assert {unknown.symbol.variable.name for _, unknown in momentum_edges} == {"u", "p"}


def test_underdetermined_poisson_reports_unmatched_equations():
    problem = poisson()
    task = problem.task(unknown=[_named(problem, "u")])
    graph = coupling_graph(task)

    assert len(graph.equations) == 3
    assert len(graph.unknowns) == 1
    assert len(graph.matching) == 1
    assert not graph.has_full_matching
    assert len(graph.unmatched_equations) == 2
    assert not graph.unmatched_unknowns


def test_triangular_system_reports_blocks_in_dependency_order():
    x = Variable("x", dim=1)
    domain = Domain("Omega", ambient=x)
    u = FunctionSymbol(Variable("u", dim=1), x)
    v = FunctionSymbol(Variable("v", dim=1), x)
    f = FunctionSymbol(Variable("f", dim=1), x)
    g = FunctionSymbol(Variable("g", dim=1), x)
    problem = Problem(constraints=(Eq(u, f).on(domain), Eq(u + v, g).on(domain)))
    graph = coupling_graph(problem.task(unknown=[u, v]))

    assert graph.has_full_matching
    assert [
        tuple(node.symbol.variable.name for node in block.unknowns)
        for block in graph.blocks
    ] == [("u",), ("v",)]


def test_a_problem_without_unknowns_leaves_all_equations_unmatched():
    problem = stokes()
    task = problem.task(unknown=[])
    graph = coupling_graph(task)

    assert not graph.matching
    assert len(graph.unmatched_equations) == 3
    assert not graph.unmatched_unknowns
    assert not graph.has_full_matching
