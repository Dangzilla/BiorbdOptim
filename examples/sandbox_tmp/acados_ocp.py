from acados_template import AcadosOcp, AcadosOcpSolver
from export_eocar_ode_model import export_eocar_ode_model
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
# from utils import plot_pendulum
import os
import time
# import BiorbdViz as brbdv

os.environ["ACADOS_SOURCE_DIR"] = "/home/dangzilla/Documents/Programmation/acados"
# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_eocar_ode_model()
ocp.model = model



Tf = 2
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = 31
x0 = np.array([0]*nx)
xT = np.array([1]*nu+[0]*nu)

# set dimensions
ocp.dims.nx    = nx
ocp.dims.ny    = ny
ocp.dims.ny_e  = ny_e
ocp.dims.nu    = nu
ocp.dims.N     = N

# set cost module
ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

Q = 0.00*np.eye(nx)
R = 5*np.eye(nu)

ocp.cost.W = scipy.linalg.block_diag(Q, R)

ocp.cost.W_e = Q

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:] = np.eye(nx)

Vu = np.zeros((ny, nu))
Vu[nx:,:] = np.eye(nu)
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.zeros((nx,nx))

ocp.cost.yref  = np.zeros((ny, ))
ocp.cost.yref_e = np.ones((ny_e, ))

# set constraints
ocp.constraints.x0 = x0
ocp.dims.nbx_0 = nx
Fmax = 1000000
ocp.constraints.constr_type = 'BGH'
ocp.constraints.lbu = -Fmax*np.ones(nu,)
ocp.constraints.ubu = Fmax*np.ones(nu,)
ocp.constraints.idxbu = np.array(range(nu))
ocp.dims.nbu   = nu

# terminal constraints
ocp.constraints.Jbx_e  = np.eye(nx)
ocp.constraints.ubx_e  = xT
ocp.constraints.lbx_e  = xT
ocp.constraints.idxbx_e = np.array(range(nx))
ocp.dims.nbx_e = nx


#path constraints
# ocp.constraints.Jbx   = np.eye(nx)
# ocp.constraints.ubx   = 100*np.ones(nx,)
# ocp.constraints.lbx   = -100*np.ones(nx,)
# ocp.constraints.idxbx = np.array(range(nx))
# ocp.dims.nbx   = nx



# ocp.solver_options.nlp_solver_max_iter = 200
# ocp.solver_options.nlp_solver_step_length = 1.0


# set prediction horizon
ocp.solver_options.tf = Tf

ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

# initial guess
# t_traj = np.linspace(0, Tf, N+1)
# x_traj = np.linspace(x0,xT,N+1)
# u_traj = np.ones((N, nu))+np.random.rand(N, nu)*0
# for n in range(N+1):
#   ocp_solver.set(n, 'x', x_traj[n,:])
# for n in range(N):
#   ocp_solver.set(n, 'u', u_traj[n])

simX = np.ndarray((N+1, nx))
simU = np.ndarray((N, nu))

t = time.time()
status = ocp_solver.solve()
t_solve = time.time()-t
print(f"Time to solve regular problem {t_solve}")

if status != 0:
    raise Exception('acados returned status {}. Exiting.'.format(status))

# get solution
stat_fields = ['time_tot', 'time_lin', 'time_qp', 'time_qp_solver_call', 'time_reg', 'sqp_iter']
for field in stat_fields:
  print(f"{field} : {ocp_solver.get_stats(field)}")
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

print(simX)
tgrid = [Tf/N*k for k in range(N+1)]
fig,(ax1,ax2,ax3) = plt.subplots(3,1)

ax1.plot(tgrid, simX[:,:nu], '-',label='Positions')
ax1.legend()
ax2.plot(tgrid, simX[:,nu:], '-',label='Velocities')
ax2.legend()
plt.step(tgrid, np.vstack([np.nan*np.ones(nu),simU]), '-.',label='Controls')
ax3.legend()
plt.suptitle(f"EOCAR {nu}D ACADOS\nTime to solve {t_solve} s\n {N} nodes")
plt.show(block=True)


# plt.plot(simX[:,:nu],'o',label='opt_sol')
# # plt.plot(x_traj[:,:nu],'x',label='init_sol')
# plt.legend()
# plt.title('position')
# plt.figure()
# plt.plot(simX[:,nu:],'o',label='opt_sol')
# # plt.plot(x_traj[:,nu:],'x',label='init_sol')
# plt.legend()
# plt.title('velocity')
# plt.figure()
# plt.plot(simU,'o',label='opt_sol')
# # plt.plot(u_traj,'x',label='init_traj')
# plt.legend()
# plt.title('control')
# plt.show(block=True)
print()