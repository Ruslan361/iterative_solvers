#pragma once

#include "solver.hpp"

class MSGSolver : public Solver {
private:
    // Вспомогательная функция для вычисления скалярного произведения
    double dot(const KokkosVector& v1, const KokkosVector& v2) const;
    
    // Вспомогательная функция для умножения матрицы на вектор A*v
    KokkosVector multiply(const KokkosCrsMatrix& A, const KokkosVector& v) const;
    
    // Вычисление нормы вектора
    double norm(const KokkosVector& v) const;
    
    // Вычисление максимальной нормы вектора (максимальный модуль элемента)
    double max_norm(const KokkosVector& v) const;

public:
    MSGSolver(const KokkosCrsMatrix& a, 
              const KokkosVector& b, 
              double eps = 1e-6, 
              int maxIterations = 10000)
        : Solver(a, b, eps, maxIterations, "Метод серединных градиентов") {}
    
    KokkosVector solve(const KokkosVector& true_solution) override;
};