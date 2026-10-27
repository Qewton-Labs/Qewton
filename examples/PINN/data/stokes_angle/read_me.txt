Solution to the Stokes equation:

\begin{align*}
    -\frac{\mu}{2} \text{div}(\nabla u_\alpha + \nabla u_\alpha^T) + \nabla p_\alpha &= 0 &&\text{ in } [0, 1]^2  \quad \text{(Conservation of momentum)}, \\
    \text{div}(u_\alpha) &= 0 &&\text{ in } [0, 1]^2 \quad  \text{(Conservation of mass)}, \\
    u_\alpha &= u_{in}(\alpha, x) , &&\text{ on } \{0 \} \times [0, 1] \quad \text{(Inflow condition)},\\
    u_\alpha &= 0 , &&\text{ on } [0, 1] \times \{0, 1\} \quad \text{(Noslip condition)}, \\
    -\frac{\mu}{2}(\nabla u_\alpha + \nabla u_\alpha^T) \cdot n + p_\alpha n &= 0 , &&\text{ on }  \{1 \} \times [0, 1] \quad \text{(Outflow condition)}.
\end{align*}

The domain is a unit square, on the left side we have variable input given by an 
inflow angle $\alpha$.
The data is:
    - coords.pt, shape=(3, 33, 33, 2), the coordinates of the underlying mesh. Is a 
    33x33 grid on the unit square.
    - angles.pt, shape=(3, 33, 33, 1), three different angles for the inflow function.
    - velocity.pt, shape=(3, 33, 33, 2), the FEM solution for each angle and on each grid point.
All data has the same shape, angles.pt only varies along the first axis, coords.pt doesnt
dependt on the first axis.