#pragma once

#include "solver.hpp"

class MSGSolver : public Solver {
private:
    // Вспомогательная функция для вычисления скалярного произведения
    double dot(const std::vector<double>& v1, const std::vector<double>& v2) const;
    
    // Вспомогательная функция для умножения матрицы на вектор A*v
    std::vector<double> multiply(const std::vector<std::vector<double>>& A, 
                                const std::vector<double>& v) const;
    
    // Вычисление нормы вектора
    double norm(const std::vector<double>& v) const;

public:
    MSGSolver(const std::vector<std::vector<double>>& a, 
              const std::vector<double>& b, 
              double eps = 1e-6, 
              int maxIterations = 10000)
        : Solver(a, b, eps, maxIterations, "Метод серединных градиентов") {}
    
    std::vector<double> solve() override;
};