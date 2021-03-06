from math import inf
from enum import Enum

from casadi import vertcat, sum1, horzcat

from .enums import Instant, InterpolationType
from .penalty import PenaltyType, PenaltyFunctionAbstract
from .path_conditions import Bounds

# TODO: Convert the constraint in CasADi function?


class ConstraintFunction(PenaltyFunctionAbstract):
    """
    Different conditions between biorbd geometric structures.
    """

    class Functions:
        @staticmethod
        def contact_force_inequality(
            constraint_type, ocp, nlp, t, x, u, direction, contact_force_idx, boundary, **parameters
        ):
            """
            To be completed when this function will be fully developed, in particular the fact that policy is either a
            tuple/list or a tuple of tuples/list of lists,
            with in the 1st index the number of the contact force and in the 2nd index the associated bound.
            """
            # To be modified later so that it can handle something other than lower bounds for greater than
            for i in range(len(u)):
                if direction == "GREATER_THAN":
                    min_bound = boundary
                    max_bound = inf
                elif direction == "LESSER_THAN":
                    min_bound = -inf
                    max_bound = boundary
                else:
                    raise RuntimeError(
                        "direction parameter of contact_force_inequality must either be GREATER_THAN or LESSER_THAN"
                    )
                ConstraintFunction._add_to_penalty(
                    ocp,
                    nlp,
                    nlp["contact_forces_func"](x[i], u[i])[contact_force_idx, 0],
                    min_bound=min_bound,
                    max_bound=max_bound,
                    **parameters
                )

        @staticmethod
        def non_slipping(
            constraint_type,
            ocp,
            nlp,
            t,
            x,
            u,
            tangential_component_idx,
            normal_component_idx,
            static_friction_coefficient,
            **parameters
        ):
            """
            :param coeff: It is the coefficient of static friction.
            """
            if not isinstance(tangential_component_idx, int):
                raise RuntimeError("tangential_component_idx must be a unique integer")

            if isinstance(normal_component_idx, int):
                normal_component_idx = [normal_component_idx]

            mu = static_friction_coefficient
            for i in range(len(u)):
                contact = nlp["contact_forces_func"](x[i], u[i])
                normal_contact_force = sum1(contact[normal_component_idx, 0])
                tangential_contact_force = contact[tangential_component_idx, 0]

                # Since it is non-slipping normal forces are supposed to be greater than zero
                ConstraintFunction._add_to_penalty(
                    ocp,
                    nlp,
                    mu * normal_contact_force - tangential_contact_force,
                    min_bound=0,
                    max_bound=inf,
                    **parameters
                )
                ConstraintFunction._add_to_penalty(
                    ocp,
                    nlp,
                    mu * normal_contact_force + tangential_contact_force,
                    min_bound=0,
                    max_bound=inf,
                    **parameters
                )

    @staticmethod
    def continuity_constraint(ocp):
        """
        Adds continuity constraints between each nodes and its neighbours. It is possible to add a continuity
        constraint between first and last nodes to have a loop (nlp.is_cyclic_constraint).
        :param ocp: An OptimalControlProgram class.
        """
        # Dynamics must be sound within phases
        for i, nlp in enumerate(ocp.nlp):
            penalty_idx = ConstraintFunction._reset_penalty(ocp, None, -1)
            # Loop over shooting nodes or use parallelization
            if ocp.nb_threads > 1:
                end_nodes = nlp["par_dynamics"](horzcat(*nlp["X"][:-1]), horzcat(*nlp["U"]))
                vals = horzcat(*nlp["X"][1:]) - end_nodes
                ConstraintFunction._add_to_penalty(ocp, None, vals.reshape((nlp["nx"] * nlp["ns"], 1)), penalty_idx)
            else:
                for k in range(nlp["ns"]):
                    # Create an evaluation node
                    end_node = nlp["dynamics"][k](x0=nlp["X"][k], p=nlp["U"][k])["xf"]

                    # Save continuity constraints
                    val = end_node - nlp["X"][k + 1]
                    ConstraintFunction._add_to_penalty(ocp, None, val, penalty_idx)

        # Dynamics must be continuous between phases
        for i in range(len(ocp.nlp) - 1):
            penalty_idx = ConstraintFunction._reset_penalty(ocp, None, -1)
            if ocp.nlp[i]["nx"] != ocp.nlp[i + 1]["nx"]:
                raise RuntimeError("Phase constraints without same nx is not supported yet")

            val = ocp.nlp[i]["X"][-1] - ocp.nlp[i + 1]["X"][0]
            ConstraintFunction._add_to_penalty(ocp, None, val, penalty_idx)

        if ocp.is_cyclic_constraint:
            # Save continuity constraints between final integration and first node
            if ocp.nlp[0]["nx"] != ocp.nlp[-1]["nx"]:
                raise RuntimeError("Cyclic constraint without same nx is not supported yet")

            val = ocp.nlp[-1]["X"][-1][1:] - ocp.nlp[0]["X"][0][1:]
            ConstraintFunction._add_to_penalty(ocp, None, val)

    @staticmethod
    def _add_to_penalty(ocp, nlp, g, penalty_idx, min_bound=0, max_bound=0, **extra_param):
        g_bounds = Bounds(interpolation_type=InterpolationType.CONSTANT)
        for _ in range(g.rows()):
            g_bounds.concatenate(Bounds(min_bound, max_bound, interpolation_type=InterpolationType.CONSTANT))

        if nlp:
            nlp["g"][penalty_idx].append(g)
            nlp["g_bounds"][penalty_idx].append(g_bounds)
        else:
            ocp.g[penalty_idx].append(g)
            ocp.g_bounds[penalty_idx].append(g_bounds)

    @staticmethod
    def _reset_penalty(ocp, nlp, penalty_idx):
        if nlp:
            g_to_add_to = nlp["g"]
            g_bounds_to_add_to = nlp["g_bounds"]
        else:
            g_to_add_to = ocp.g
            g_bounds_to_add_to = ocp.g_bounds

        if penalty_idx < 0:
            g_to_add_to.append([])
            g_bounds_to_add_to.append([])
            return len(g_to_add_to) - 1
        else:
            g_to_add_to[penalty_idx] = []
            g_bounds_to_add_to[penalty_idx] = []
            return penalty_idx

    @staticmethod
    def _parameter_modifier(constraint_function, parameters):
        # Everything that should change the entry parameters depending on the penalty can be added here
        super(ConstraintFunction, ConstraintFunction)._parameter_modifier(constraint_function, parameters)

    @staticmethod
    def _span_checker(constraint_function, instant, nlp):
        # Everything that is suspicious in terms of the span of the penalty function ca be checked here
        super(ConstraintFunction, ConstraintFunction)._span_checker(constraint_function, instant, nlp)
        if (
            constraint_function == Constraint.CONTACT_FORCE_INEQUALITY.value[0]
            or constraint_function == Constraint.NON_SLIPPING.value[0]
        ):
            if instant == Instant.END or instant == nlp["ns"]:
                raise RuntimeError("No control u at last node")


class Constraint(Enum):
    """
    Different conditions between biorbd geometric structures.
    """

    TRACK_STATE = (PenaltyType.TRACK_STATE,)
    TRACK_MARKERS = (PenaltyType.TRACK_MARKERS,)
    TRACK_MARKERS_VELOCITY = (PenaltyType.TRACK_MARKERS_VELOCITY,)
    ALIGN_MARKERS = (PenaltyType.ALIGN_MARKERS,)
    PROPORTIONAL_STATE = (PenaltyType.PROPORTIONAL_STATE,)
    PROPORTIONAL_CONTROL = (PenaltyType.PROPORTIONAL_CONTROL,)
    TRACK_TORQUE = (PenaltyType.TRACK_TORQUE,)
    TRACK_MUSCLES_CONTROL = (PenaltyType.TRACK_MUSCLES_CONTROL,)
    TRACK_ALL_CONTROLS = (PenaltyType.TRACK_ALL_CONTROLS,)
    TRACK_CONTACT_FORCES = (PenaltyType.TRACK_CONTACT_FORCES,)
    ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
    ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
    CUSTOM = (PenaltyType.CUSTOM,)
    CONTACT_FORCE_INEQUALITY = (ConstraintFunction.Functions.contact_force_inequality,)
    NON_SLIPPING = (ConstraintFunction.Functions.non_slipping,)

    @staticmethod
    def _get_type():
        return ConstraintFunction
