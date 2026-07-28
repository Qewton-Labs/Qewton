The data is for the ODE:
u_t = f in (0, 1)
u(0) = 0

The shape of the data is:
x_data = (N_x, 1)
f_data = (N_B, N_x, 1)
u_data = (N_B, N_x, 1)

N_B = 5000, N_x = 100
where N_B is the amount of pairs (f, u) and N_x the discretization 
of the spatial domain.