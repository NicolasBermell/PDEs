#
# PDE SOLVER DE NICOLÁS - EL DE PABLO ES DISTINTO
#

from dataclasses import dataclass
from typing import Callable, Any, Dict
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np

@dataclass
class PDE_Model:
    method_to_use: str
    Lmin: float
    Lmax: float
    m: int
    tmin: float
    tmax: float
    n: int
    param: Any
                    #   x     t    param
    coef_a: Callable[[float, float, Any], float]
    coef_b: Callable[[float, float, Any], float]
    coef_c: Callable[[float, float, Any], float]
    payoff_T: Callable[[float, float, Any], float]
                    
                    #   ti    tmax    xj  param
    bc_left: Callable[[float, float, float, Any], float]
    bc_right: Callable[[float, float, float, Any], float]

                            # t-index U-grid param
    neumann_bc_left: Callable[[float, NDArray, Any], float] = None
    neumann_bc_right: Callable[[float, NDArray, Any], float] = None

    rannacher_steps: int = 0  # Fully-implicit steps near payoff (0 = pure CN)

    @property
    def has_neumann(self):
        return self.neumann_bc_left is not None or self.neumann_bc_right is not None

    @property
    def dx(self):
        return (self.Lmax - self.Lmin) / (self.m - 1)
    @property
    def dt(self):
        return (self.tmax - self.tmin) / (self.n - 1)

    def validate(self) -> None:
        allowed = {"expl", "impl", "cn"}
        if self.method_to_use not in allowed:
            raise ValueError(f"method_to_use must be one of {allowed}")
        if self.m < 3 or self.n < 2:
            raise ValueError("m must be >= 3 and n must be >= 2")
        if self.Lmax <= self.Lmin:
            raise ValueError("Lmax must be > Lmin")
        if self.tmax <= self.tmin:
            raise ValueError("tmax must be > tmin")
        

class PDE_Solver:
    def solve(self, model: PDE_Model) -> Dict[str, Any]:
        model.validate()
        x_mesh, t_mesh, dt, dx, alpha, rho = self._make_mesh(model)
        U = self._init_solution_grid(model, x_mesh, t_mesh)

        if model.method_to_use == "expl":
            self._solve_explicit(model, U, x_mesh, t_mesh, dt, dx, alpha, rho)
        elif model.method_to_use == "impl":
            self._solve_implicit(model, U, x_mesh, t_mesh, dt, dx, alpha, rho)
        elif model.method_to_use == "cn" and not model.has_neumann:
            self._solve_cn_dir(model, U, x_mesh, t_mesh, dt, dx, alpha, rho)
        elif model.method_to_use == "cn" and model.has_neumann:
            self._solve_cn(model, U, x_mesh, t_mesh, dt, dx, alpha, rho)

        return {
            "U": U,
            "x_mesh": x_mesh,
            "t_mesh": t_mesh,
            "meta": {
                "dt": dt,
                "dx": dx,
                "alpha": alpha,
                "rho": rho,
                "method": model.method_to_use
            }
        }

    def _make_mesh(self, model: PDE_Model):
        x_mesh = np.linspace(model.Lmin, model.Lmax, model.m)
        t_mesh = np.linspace(model.tmin, model.tmax, model.n)
        dt = t_mesh[1] - t_mesh[0]
        dx = x_mesh[1] - x_mesh[0]
        alpha = dt / (dx * dx)
        rho = dt / dx
        return x_mesh, t_mesh, dt, dx, alpha, rho

    def _init_solution_grid(self, model: PDE_Model, x_mesh, t_mesh):
        n, m = model.n, model.m
        U = np.zeros((n, m))
        t_max = t_mesh[-1]
        dx = x_mesh[1] - x_mesh[0]
        
        # Initialize payoff at t=T
        for j in range(m):
            U[n - 1, j] = model.payoff_T(t_max, x_mesh[j], model.param)
        
        # Initialize Dirichlet boundaries if provided
        if model.bc_left is not None:
            for i in range(n):
                U[i, 0] = model.bc_left(t_mesh[i], t_max, x_mesh[0], model.param)
        elif model.neumann_bc_left is not None:
            # Initialize Neumann boundary using ghost point formula
            g_left = model.neumann_bc_left(n - 1, U[n - 1, :], model.param)
            U[n - 1, 0] = (4 * U[n - 1, 1] - U[n - 1, 2] - 2 * dx * g_left) / 3.0
        
        if model.bc_right is not None:
            for i in range(n):
                U[i, -1] = model.bc_right(t_mesh[i], t_max, x_mesh[-1], model.param)
        elif model.neumann_bc_right is not None:
            # Initialize Neumann boundary using ghost point formula
            g_right = model.neumann_bc_right(n - 1, U[n - 1, :], model.param)
            U[n - 1, -1] = (4 * U[n - 1, -2] - U[n - 1, -3] + 2 * dx * g_right) / 3.0
        
        return U

    def _eval_coeffs_at_time(self, model: PDE_Model, t: float, x_mesh):
        m = model.m
        atx = np.zeros(m - 2)
        btx = np.zeros(m - 2)
        ctx = np.zeros(m - 2)
        for j in range(m - 2):
            xj = x_mesh[j + 1]
            atx[j] = model.coef_a(t, xj, model.param)
            btx[j] = model.coef_b(t, xj, model.param)
            ctx[j] = model.coef_c(t, xj, model.param)
        return atx, btx, ctx

    def _solve_explicit(self, model: PDE_Model, U, x_mesh, t_mesh, dt, dx, alpha, rho):
        n, m = model.n, model.m
        for i in range(n - 2, -1, -1):
            atx, btx, ctx = self._eval_coeffs_at_time(model, t_mesh[i], x_mesh)
            for j in range(1, m - 1):
                k = j - 1
                U[i, j] = (
                    U[i + 1, j + 1] * (alpha * atx[k] + 0.5 * rho * btx[k])
                    + U[i + 1, j] * (1 - 2 * alpha * atx[k] + dt * ctx[k])
                    + U[i + 1, j - 1] * (alpha * atx[k] - 0.5 * rho * btx[k])
                )

    def _solve_implicit(self, model: PDE_Model, U, x_mesh, t_mesh, dt, dx, alpha, rho):
        n, m = model.n, model.m
        for i in range(n - 2, -1, -1):
            atx, btx, ctx = self._eval_coeffs_at_time(model, t_mesh[i], x_mesh)
            dl = -alpha * atx[1:] + 0.5 * rho * btx[1:]
            d = 1 + 2 * alpha * atx - dt * ctx
            du = -alpha * atx[:-1] - 0.5 * rho * btx[:-1]
            M = diags([dl, d, du], [-1, 0, 1], shape=(m - 2, m - 2)).tocsc()
            rhs = U[i + 1, 1:-1].copy()
            rhs[0] += (alpha * atx[0] - 0.5 * rho * btx[0]) * U[i, 0]
            rhs[-1] += (alpha * atx[-1] + 0.5 * rho * btx[-1]) * U[i, -1]
            U[i, 1:-1] = spsolve(M, rhs)

    def _solve_cn(self, model: PDE_Model, U, x_mesh, t_mesh, dt, dx, alpha, rho):
        n, m = model.n, model.m
        for i in range(n - 2, -1, -1):
            # Rannacher: fully implicit near payoff, CN otherwise
            theta = 1.0 if (i >= n - 2 - model.rannacher_steps) else 0.5
            om = 1.0 - theta

            atx, btx, ctx = self._eval_coeffs_at_time(model, t_mesh[i], x_mesh)

            # LHS matrix (implicit side, scaled by theta)
            dlM = -theta * alpha * atx[1:] + 0.5 * theta * rho * btx[1:]
            dM = 1 + 2 * theta * alpha * atx - theta * dt * ctx
            duM = -theta * alpha * atx[:-1] - 0.5 * theta * rho * btx[:-1]

            # Boundary coefficients (implicit side)
            coef_U0 = -theta * alpha * atx[0] + 0.5 * theta * rho * btx[0]
            coef_Un = -theta * alpha * atx[-1] - 0.5 * theta * rho * btx[-1]

            # Inject Neumann BC into matrix M via second-order one-sided formula:
            #   U[i,0] = (4*U[i,1] - U[i,2] - 2*dx*g_left) / 3
            if model.neumann_bc_left is not None:
                dM[0] += (4.0 / 3.0) * coef_U0
                duM[0] -= (1.0 / 3.0) * coef_U0

            #   U[i,-1] = (4*U[i,-2] - U[i,-3] + 2*dx*g_right) / 3
            if model.neumann_bc_right is not None:
                dM[-1] += (4.0 / 3.0) * coef_Un
                dlM[-1] -= (1.0 / 3.0) * coef_Un

            M = diags([dlM, dM, duM], [-1, 0, 1], shape=(m - 2, m - 2)).tocsc()

            # RHS matrix (explicit side, scaled by 1-theta)
            dlN = om * alpha * atx[1:] - 0.5 * om * rho * btx[1:]
            dN = 1 - 2 * om * alpha * atx + om * dt * ctx
            duN = om * alpha * atx[:-1] + 0.5 * om * rho * btx[:-1]
            N = diags([dlN, dN, duN], [-1, 0, 1], shape=(m - 2, m - 2)).tocsc()

            rhs = N @ U[i + 1, 1:-1]

            # Explicit side boundary contributions (known time level i+1)
            coef_exp_left = om * alpha * atx[0] - 0.5 * om * rho * btx[0]
            coef_exp_right = om * alpha * atx[-1] + 0.5 * om * rho * btx[-1]

            if model.neumann_bc_left is not None:
                g_left_next = model.neumann_bc_left(i + 1, U[i + 1, :], model.param)
                U_left_next = (4 * U[i + 1, 1] - U[i + 1, 2] - 2 * dx * g_left_next) / 3.0
                rhs[0] += coef_exp_left * U_left_next
            else:
                rhs[0] += coef_exp_left * U[i + 1, 0]

            if model.neumann_bc_right is not None:
                g_right_next = model.neumann_bc_right(i + 1, U[i + 1, :], model.param)
                U_right_next = (4 * U[i + 1, -2] - U[i + 1, -3] + 2 * dx * g_right_next) / 3.0
                rhs[-1] += coef_exp_right * U_right_next
            else:
                rhs[-1] += coef_exp_right * U[i + 1, -1]

            # Implicit side: Neumann derivative constant term moved to RHS
            if model.neumann_bc_left is not None:
                g_left_now = model.neumann_bc_left(i, U[i + 1, :], model.param)
                rhs[0] += (2.0 * dx / 3.0) * coef_U0 * g_left_now

            if model.neumann_bc_right is not None:
                g_right_now = model.neumann_bc_right(i, U[i + 1, :], model.param)
                rhs[-1] -= (2.0 * dx / 3.0) * coef_Un * g_right_now

            # Solve
            U[i, 1:-1] = spsolve(M, rhs)

            # Update boundary values from interior using Neumann condition
            if model.neumann_bc_left is not None:
                g_left_update = model.neumann_bc_left(i, U[i, :], model.param)
                U[i, 0] = (4 * U[i, 1] - U[i, 2] - 2 * dx * g_left_update) / 3.0
            if model.neumann_bc_right is not None:
                g_right_update = model.neumann_bc_right(i, U[i, :], model.param)
                U[i, -1] = (4 * U[i, -2] - U[i, -3] + 2 * dx * g_right_update) / 3.0

    def _solve_cn_dir(self, model: PDE_Model, U, x_mesh, t_mesh, dt, dx, alpha, rho):
        n, m = model.n, model.m
        for i in range(n - 2, -1, -1):
            # Rannacher: fully implicit near payoff, CN otherwise
            theta = 1.0 if (i >= n - 2 - model.rannacher_steps) else 0.5
            om = 1.0 - theta

            atx, btx, ctx = self._eval_coeffs_at_time(model, t_mesh[i], x_mesh)

            # LHS matrix (implicit side)
            dlM = -theta * alpha * atx[1:] + 0.5 * theta * rho * btx[1:]
            dM = 1 + 2 * theta * alpha * atx - theta * dt * ctx
            duM = -theta * alpha * atx[:-1] - 0.5 * theta * rho * btx[:-1]
            M = diags([dlM, dM, duM], [-1, 0, 1], shape=(m - 2, m - 2)).tocsc()

            # RHS matrix (explicit side)
            dlN = om * alpha * atx[1:] - 0.5 * om * rho * btx[1:]
            dN = 1 - 2 * om * alpha * atx + om * dt * ctx
            duN = om * alpha * atx[:-1] + 0.5 * om * rho * btx[:-1]
            N = diags([dlN, dN, duN], [-1, 0, 1], shape=(m - 2, m - 2)).tocsc()

            rhs = N @ U[i + 1, 1:-1]
            # Boundary contributions
            rhs[0] += (om * alpha * atx[0] - 0.5 * om * rho * btx[0]) * U[i + 1, 0]
            rhs[0] -= (-theta * alpha * atx[0] + 0.5 * theta * rho * btx[0]) * U[i, 0]
            rhs[-1] += (om * alpha * atx[-1] + 0.5 * om * rho * btx[-1]) * U[i + 1, -1]
            rhs[-1] -= (-theta * alpha * atx[-1] - 0.5 * theta * rho * btx[-1]) * U[i, -1]
            U[i, 1:-1] = spsolve(M, rhs)