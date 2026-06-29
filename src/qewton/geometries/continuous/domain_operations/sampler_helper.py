"""This file contains some sample functions for the domain operations.
Since Union/Cut/Intersection follow the same idea for sampling for a given
number of points.
"""

from qewton.config.devices import Device

# TODO: Update docstring


def _inside_random_with_n(domain_a, domain_b, n, invert, device: Device):
    """Creates a random uniform points inside of a cut or intersection domain."""
    number_valid = 0
    scaled_n = n
    while number_valid < n:
        # first create in a
        new_points = domain_a.sample_random_uniform(n_points=int(scaled_n), device=device)
        # check how many are in the other domain
        index_valid = _check_in_b(domain_b, invert, new_points)
        number_valid = len(index_valid)
        # scale up the number of point and try again
        scaled_n = 5 * scaled_n if number_valid == 0 else scaled_n**2 / number_valid + 1
    return new_points[index_valid[:n],]  # type: ignore


def _inside_grid_with_n(domain_a, domain_b, n, invert, device: Device):
    """Creates a point grid inside of a cut or intersection domain."""
    # first sample grid inside the domain_a
    grid_a = domain_a.sample_grid(n_points=n, device=device)
    index_valid = _check_in_b(domain_b, invert, grid_a)
    number_inside = max(len(index_valid), 1)
    if number_inside == n:
        return grid_a
    # if the grid does not fit, scale the number of points
    scaled_n = int(n**2 / number_inside)
    grid_a = domain_a.sample_grid(n_points=scaled_n, device=device)
    index_valid = _check_in_b(domain_b, invert, grid_a)
    grid_a = grid_a[index_valid,]
    if len(grid_a) >= n:
        return grid_a[:n,]
    # add some random ones if still some missing
    rand_points = _inside_random_with_n(
        domain_a, domain_b, n - len(grid_a), invert, device=device
    )
    return domain_a.backend.math.concatenate([grid_a, rand_points], axis=0)


def _check_in_b(domain_b, invert, grid_a):
    # check what points are correct
    inside_b = domain_b.contains(grid_a)
    backend = domain_b.backend
    if invert:
        inside_b = backend.math.logical_not(inside_b)
    index = backend.math.where(inside_b)[0]
    return index


def _boundary_random_with_n(main_domain, domain_a, domain_b, n, device: Device):
    """Creates a point grid on the boundary of a domain operation.

    Parameters
    ----------
    main_domain : Domain
        The domain that represents the new created domain.
    domain_a, domain_b : Domain
        The two domains that define the main domain.
    n : int
        The number of points.
    """
    backend = main_domain.backend
    random_points = backend.math.empty((0, main_domain.dim), device=device)
    domains = [domain_a, domain_b]
    # scale n such that the number of points corresponds to the size
    # of the boundary
    scaled_n = _compute_boundary_ratio(main_domain, domain_a, domain_b, n)
    use_b = False  # to switch between sampling on a and b
    while len(random_points) < n:
        new_points = domains[use_b].boundary.sample_random_uniform(
            n_points=scaled_n[use_b], device=device
        )
        index_valid = main_domain.contains(new_points)
        index_valid = backend.math.where(index_valid)[0]
        ith_points = new_points[index_valid]
        use_b = not use_b  # switch to other domain
        random_points = backend.math.concatenate([random_points, ith_points], axis=0)
    return random_points[:n]


def _compute_boundary_ratio(main_domain, domain_a, domain_b, n):
    main_volume = main_domain.volume()
    a_volume = domain_a.boundary.volume()
    b_volume = domain_b.boundary.volume()
    return [int(n * a_volume / main_volume) + 1, int(n * b_volume / main_volume) + 1]


def _boundary_grid_with_n(main_domain, domain_a, domain_b, n, device: Device):
    """Creates a point grid on the boundary of a domain operation."""
    # first sample a grid on both boundaries
    backend = main_domain.backend
    grid_a = domain_a.boundary.sample_grid(n_points=n, device=device)
    grid_b = domain_b.boundary.sample_grid(n_points=n, device=device)
    # check how many points are on the boundary of the operation domain
    on_bound_a, on_bound_b, a_correct, b_correct = _check_points_on_main_boundary(
        main_domain, grid_a, grid_b
    )
    sum_of_correct = a_correct + b_correct
    if sum_of_correct == n:
        return backend.math.concatenate(
            [grid_a[on_bound_a,], grid_b[on_bound_b,]], axis=0
        )
    # scale the n so that more or fewer points are sampled and try again
    # to get a better grid. For the scaling we approximate the volume of the
    # the main domain.
    a_surface = domain_a.boundary.volume()
    b_surface = domain_b.boundary.volume()
    approx_surface = a_surface * a_correct / n + b_surface * b_correct / n
    scaled_a = int(n * a_surface / approx_surface) + 1  # round up
    scaled_b = max(int(n * b_surface / approx_surface), 1)  # round to floor, but not 0
    grid_a = domain_a.boundary.sample_grid(n_points=scaled_a, device=device)
    grid_b = domain_b.boundary.sample_grid(n_points=scaled_b, device=device)
    # check again how what points are correct and now just stay with this grid
    # if still some points are missing add random ones.
    on_bound_a, on_bound_b, a_correct, b_correct = _check_points_on_main_boundary(
        main_domain, grid_a, grid_b
    )
    final_grid = backend.math.concatenate(
        [grid_a[on_bound_a,], grid_b[on_bound_b,]], axis=0
    )
    if len(final_grid) >= n:
        return final_grid[:n,]
    rand_points = _boundary_random_with_n(
        main_domain, domain_a, domain_b, n - len(final_grid), device=device
    )
    return backend.math.concatenate([final_grid, rand_points], axis=0)


def _check_points_on_main_boundary(main_domain, grid_a, grid_b):
    backend = main_domain.backend
    on_bound_a = backend.math.where(main_domain.contains(grid_a))[0]
    on_bound_b = backend.math.where(main_domain.contains(grid_b))[0]
    a_correct = len(on_bound_a)
    b_correct = len(on_bound_b)
    return on_bound_a, on_bound_b, a_correct, b_correct
