import os
import biorbd
from acados_template import AcadosModel, AcadosOcp
from casadi import MX, Function, SX, vertcat
import numpy as np
import scipy.linalg


def acados_export_model(self):
    # Declare model variables
    x = self.nlp[0]['x']
    u = self.nlp[0]['u']
    p = self.nlp[0]['p']
    mod = self.nlp[0]['model']
    x_dot = MX.sym("x_dot", mod.nbQdot() * 2, 1)

    f_expl = self.nlp[0]['dynamics_func'](x, u, p)
    f_impl = x_dot - f_expl

    acados_model = AcadosModel()
    acados_model.f_impl_expr = f_impl
    acados_model.f_expl_expr = f_expl
    acados_model.x = x
    acados_model.xdot = x_dot
    acados_model.u = u
    acados_model.p = []
    acados_model.name = "model_name"

    return acados_model


def prepare_acados(self):
    # create ocp object to formulate the OCP
    acados_ocp = AcadosOcp()

    # # set model
    acados_model = acados_export_model(self)
    acados_ocp.model = acados_model

    for i in range(self.nb_phases):
        # set time
        acados_ocp.solver_options.tf = self.nlp[i]["tf"]
        # set dimensions
        acados_ocp.dims.nx = self.nlp[i]["nx"]
        acados_ocp.dims.nu = self.nlp[i]["nu"]
        acados_ocp.dims.ny = acados_ocp.dims.nx + acados_ocp.dims.nu
        acados_ocp.dims.ny_e = self.nlp[i]["nx"]
        acados_ocp.dims.N = self.nlp[i]["ns"]

    # set cost module
    # TODO: test external cost_type
    acados_ocp.cost.cost_type = 'EXTERNAL'
    acados_ocp.cost.cost_type_e = 'EXTERNAL'

    # Cost for states and controls (default: 1.00)
    Q = 1.00 * np.eye(acados_ocp.dims.nx) #TODO Get the weights from J matrix
    R = 1.00 * np.eye(acados_ocp.dims.nu)

    acados_ocp.cost.W = scipy.linalg.block_diag(Q, R)

    acados_ocp.cost.W_e = Q

    # set Lagrange term
    if acados_ocp.cost.cost_type == 'LINEAR_LS':
        acados_ocp.cost.Vx = np.zeros((acados_ocp.dims.ny, acados_ocp.dims.nx))
        acados_ocp.cost.Vx[:acados_ocp.dims.nx, :] = np.eye(acados_ocp.dims.nx)

        Vu = np.zeros((acados_ocp.dims.ny, acados_ocp.dims.nu))
        Vu[acados_ocp.dims.nx:, :] = np.eye(acados_ocp.dims.nu)
        acados_ocp.cost.Vu = Vu

    elif acados_ocp.cost.cost_type == 'EXTERNAL':
        # set Lagrange term
        acados_ocp.model.cost_expr_ext_cost = self.nlp[0]['J'][0][2]
        # acados_ocp.model.cost_expr_ext_cost = vertcat(acados_model.x, acados_model.u).T @ acados_ocp.cost.W @ vertcat(acados_model.x, acados_model.u)

    else:
        raise RuntimeError("Available acados cost type: 'LINEAR_LS' and 'EXTERNAL'.")

    # set Mayer term
    if acados_ocp.cost.cost_type_e == 'LINEAR_LS':
        acados_ocp.cost.Vx_e = np.zeros((acados_ocp.dims.nx, acados_ocp.dims.nx))

    elif acados_ocp.cost.cost_type_e == 'EXTERNAL':
        acados_ocp.model.cost_expr_ext_cost_e = acados_model.x.T @ Q @ acados_model.x #Not working for now

    else:
        raise RuntimeError("Available acados cost type: 'LINEAR_LS' and 'EXTERNAL'.")

    #TODO: link with nlp
    acados_ocp.cost.yref = np.zeros((acados_ocp.dims.ny,))
    acados_ocp.cost.yref_e = np.ones((acados_ocp.dims.ny_e,))

    for i in range(self.nb_phases):
        # set constraints
        for j in range(-1,0):
            for k in range(self.nlp[i]['nx']):
                if self.nlp[i]["X_bounds"].min[k, j] != self.nlp[i]["X_bounds"].max[k,j]:
                    raise RuntimeError("The initial values must be set and fixed.")

        acados_ocp.constraints.x0 = np.array(self.nlp[i]["X_bounds"].min[:, 0])
        acados_ocp.dims.nbx_0 = acados_ocp.dims.nx
        acados_ocp.constraints.constr_type = 'BGH'  # TODO: put as an option in ocp?
        acados_ocp.constraints.lbu = np.array(self.nlp[i]["U_bounds"].min[:, 0])
        acados_ocp.constraints.ubu = np.array(self.nlp[i]["U_bounds"].max[:, 0])
        acados_ocp.constraints.idxbu = np.array(range(acados_ocp.dims.nu))
        acados_ocp.dims.nbu = acados_ocp.dims.nu

        # set control constraints
        acados_ocp.constraints.Jbx_e = np.eye(acados_ocp.dims.nx)
        acados_ocp.constraints.ubx_e = np.array(self.nlp[i]["X_bounds"].max[:, -1])
        acados_ocp.constraints.lbx_e = np.array(self.nlp[i]["X_bounds"].min[:, -1])
        acados_ocp.constraints.idxbx_e = np.array(range(acados_ocp.dims.nx))
        acados_ocp.dims.nbx_e = acados_ocp.dims.nx

    return acados_ocp
