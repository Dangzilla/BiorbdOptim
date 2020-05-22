import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,

)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, nb_threads):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -1000000, 1000000, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()


    # Add objective functions
    objective_functions = {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 100}

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
    X_bounds.min[n_q:, -1] = 0
    X_bounds.max[n_q:, -1] = 0

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
        biorbd_model_path="../eocar/eocar-6D.bioMod", final_time=2, number_shooting_points=31,nb_threads=4)

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
