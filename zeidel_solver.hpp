#pragma once

#include "solver.hpp"

class ZeidelSolver : public Solver {
public:
    ZeidelSolver(const std::vector<std::vector<double>>& a, 
                 const std::vector<double>& b, 
                 double eps = 1e-6, 
                 int maxIterations = 10000)
        : Solver(a, b, eps, maxIterations, "Метод Зейделя") {}
    
    std::vector<double> solve() override;
};