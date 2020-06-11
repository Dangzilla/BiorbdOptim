
from acados_template import AcadosModel
from casadi import MX,Function, external,SX, vertcat, sin, cos
import biorbd as brbd
from ctypes import *

def export_eocar_ode_model(ocp):

    m = brbd.Model("eocar-6D.bioMod")
    model_name = 'eocar_ode'

    # Declare model variables
    model = AcadosModel()
    x = ocp.nlp[0]['X'][0]
    u = ocp.nlp[0]['U'][0]
    model.x = x
    model.u = u
    xdot = MX.sym('dx', m.nbQ() * 2)
    f_expl = ocp.nlp[0]['dynamics'][0](x,u)
    f_impl = xdot - f_expl


    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl

    model.xdot = xdot

    # model.z = z
    model.p = []
    model.name = model_name

    return model 

