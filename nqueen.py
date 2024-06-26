#!/usr/bin/env python3

import argparse
import sympy as sp
import gurobipy as gp


class sp_Binary(sp.Symbol):
    def _eval_power(self, _):
        return self


class NQUEEN:
    def __init__(self, n):
        self.n = n
        self.X = [
            [sp_Binary(f"x_{i}_{j}") for j in range(self.n)] for i in range(self.n)
        ]

    def generate(self):
        sumH = [sum([self.X[i][j] for j in range(self.n)]) for i in range(self.n)]
        sumV = [sum([self.X[i][j] for i in range(self.n)]) for j in range(self.n)]
        A = [
            [self.X[i + k][k] for k in range(self.n) if 0 <= i + k and i + k < self.n]
            for i in range(-self.n, 2 * self.n)
        ]
        B = [
            [self.X[i - k][k] for k in range(self.n) if 0 <= i - k and i - k < self.n]
            for i in range(-self.n, 2 * self.n)
        ]
        sumA = [sum(x) for x in A if len(x) >= 2]
        sumB = [sum(x) for x in B if len(x) >= 2]
        formula = (
            sum([(x - 1) ** 2 for x in sumH])
            + sum([(x - 1) ** 2 for x in sumV])
            + sum([x * (x - 1) for x in sumA])
            + sum([x * (x - 1) for x in sumB])
        )
        self.qubo = formula.expand().as_coefficients_dict()

    def gurobi(self, timelimit):
        self.model = gp.Model()
        self.model.setParam("TimeLimit", timelimit)
        var = {}
        for i in range(self.n):
            for j in range(self.n):
                var[self.X[i][j]] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name=f"x_{i}_{j}"
                )
        obj = []
        self.optimal = -self.qubo.pop(1, 0)
        for key, val in self.qubo.items():
            if isinstance(key, sp.Mul):
                quad = key.expand().args
                obj.append(val * var[quad[0]] * var[quad[1]])
            else:
                obj.append(val * var[key] * var[key])
        self.model.setObjective(gp.quicksum(obj), sense=gp.GRB.MINIMIZE)
        self.model.optimize()

    def print(self):
        for i in range(self.n):
            for j in range(self.n):
                print(int(self.model.getVarByName(f"x_{i}_{j}").X), end="")
                print(" ", end="")
            print()
        print(f"optimal={self.optimal} obtained={int(self.model.ObjVal)}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve N-Queens puzzle through SymPy using Gurobi"
    )
    parser.add_argument(
        "-n", default=8, type=int, help="Value of N of N-QUEENS (default:8)"
    )
    parser.add_argument(
        "-t", default=10, type=int, help="Time limit of Gurobi optimizer"
    )
    args = parser.parse_args()

    nqueen = NQUEEN(args.n)
    nqueen.generate()
    nqueen.gurobi(args.t)
    nqueen.print()


if __name__ == "__main__":
    main()
