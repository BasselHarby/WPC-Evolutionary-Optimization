import sys
FREECADPATH = 'C:\\Program Files\\FreeCAD 1.0\\bin'
sys.path.append(FREECADPATH)

import FreeCAD
from Panels import*
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

import multiprocessing
from pymoo.core.problem import StarmapParallelization

class Optimize(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=1,
                         xl=[6 ,20 ,50 , 20],  # Lower bounds x = [thickness , amp , p, phi]
                         xu=[50,300,200, 85], # Upper bounds
                         **kwargs)
    def _evaluate(self, x, out, *args, **kwargs):
        thickness= int(x[0])
        amp = int(x[1])
        p = int(x[2])
        phi = int(x[3])
        doc = FreeCAD.newDocument()
        width = 3000
        panel = TrPanel(doc, 400, width, thickness, amp, p, phi, 20)
        out["G"] = panel.max_disp - (width/360) #Displacement constraint
        out["F"] = panel.volume #Volume Objective


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best_F"] = []  # Best objective value
        self.data["best_X"] = []  # Best decision variables
        self.data["best_G"] = []  # Best constraint value

    def notify(self, algorithm):
        F = algorithm.pop.get("F")  # Objective values
        X = algorithm.pop.get("X")  # Decision variables
        G = algorithm.pop.get("G")  # Constraint values

        best_idx = F.argmin()  # Index of best solution
        self.data["best_F"].append(F[best_idx])  # Store best objective
        self.data["best_X"].append(X[best_idx])  # Store best variables
        self.data["best_G"].append(G[best_idx])  # Store best constraint

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    runner = StarmapParallelization(pool.starmap)

    #algorithm = GA(pop_size=47, eliminate_duplicates=True)
    algorithm = PSO(pop_size=47)
    problem = Optimize(elementwise_runner=runner)
    termination = get_termination("n_gen", 30)
    res = minimize(problem, algorithm, termination,callback=MyCallback(), seed=2, save_history=True, verbose=True)
    X = res.X
    F = res.F
    G = res.G
    print(X)
    print(F)
    print(G)

    # Retrieve data from callback
    best_F = res.algorithm.callback.data["best_F"]  # Best objective values
    best_X = res.algorithm.callback.data["best_X"]  # Corresponding variables
    best_G = res.algorithm.callback.data["best_G"]  # Corresponding constraint values

    # Convert to a NumPy array for structured saving
    data = np.column_stack((best_F, best_X, best_G))

    # Save to CSV
    np.savetxt("TrPanelHistory_PSO.csv", data, delimiter=", ", fmt='%s', header="F, T, Amp, P, Phi, G", comments="")

    pool.close()
