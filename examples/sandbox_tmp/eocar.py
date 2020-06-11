import biorbd
import numpy as np
import matplotlib.pyplot as plt

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
    Data,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, nb_threads):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -1000000, 1000000, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()


    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100},
    {"type": Objective.Lagrange.MINIMIZE_STATE, "weight": 50},


    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    X_bounds.min[:, 0] = 0
    X_bounds.max[:, 0] = 0
    X_bounds.min[:n_q, -1] = 1
    X_bounds.max[:n_q, -1] = 1
    X_bounds.min[n_q:, -1] = 1
    X_bounds.max[n_q:, -1] = 1

    # Initial guess
    X_init = InitialConditions([0] * (n_q + n_qdot))

    # Define control path constraint
    U_bounds = Bounds(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)

    # Control initial guess
    U_init = InitialConditions([torque_init] * n_tau)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        nb_threads=nb_threads,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="eocar-6D.bioMod", final_time=2, number_shooting_points=31, nb_threads=1)

    # --- Solve the program --- #
    data_sol_acados = ocp.solve(solver='acados', acados_dir={"/home/dangzilla/Documents/Programmation/acados"}, options_acados={'print_level' :1, 'nlp_solver_tol_comp' : 1e-06})

    sol = ocp.solve()
    data_sol_ipopt = Data.get_data(ocp, sol, concatenate=False)

    state_diff = np.linalg.norm(data_sol_acados["qqdot"]-np.vstack([data_sol_ipopt[0]["q"],data_sol_ipopt[0]["q_dot"]]))
    control_diff = np.linalg.norm(data_sol_acados["u"]-data_sol_ipopt[1]["tau"][:,:-1])


    print(f'Total time ACADOS : {data_sol_acados["time_tot"]}')
    print(f'Total time IPOPT : {sol["time_tot"]}')
    print(f'Difference on state : {state_diff}')
    print(f'Difference on control : {control_diff}')

    # --- Show results --- #
    result = ShowResult(ocp, sol)

    result_acados = ShowResult(ocp,data_sol_acados)
    result_acados.graphs()
    result_acados.animate()

