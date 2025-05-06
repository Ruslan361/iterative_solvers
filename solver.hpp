#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

// Определяем псевдонимы для типов Kokkos
using execution_space = Kokkos::DefaultExecutionSpace;
using memory_space = Kokkos::HostSpace;
using KokkosVector = Kokkos::View<double*, memory_space>;
using KokkosCrsMatrix = KokkosSparse::CrsMatrix<double, int, execution_space, void, int>;

class Solver {
protected:
    const KokkosCrsMatrix& a;        // Матрица коэффициентов
    const KokkosVector& b;           // Вектор правой части
    KokkosVector x;                  // Решение
    double eps;                       // Точность
    int maxIterations;                // Максимальное число итераций
    int iterationsDone;               // Выполнено итераций
    double achievedPrecision;         // Достигнутая точность
    bool converged;                   // Сошелся ли метод
    std::string solverName;           // Название метода решателя

public:
    Solver(const KokkosCrsMatrix& a, 
           const KokkosVector& b, 
           double eps = 1e-6, 
           int maxIterations = 10000,
           const std::string& name = "Базовый решатель")
        : a(a), b(b), eps(eps), maxIterations(maxIterations), 
          iterationsDone(0), achievedPrecision(0.0), converged(false),
          solverName(name) {
        // Инициализация решения
        x = KokkosVector("x", b.extent(0));
        Kokkos::deep_copy(x, 0.0);
    }
    
    virtual ~Solver() = default;
    
    // Абстрактный метод решения системы
    virtual KokkosVector solve(const KokkosVector& true_solution)  = 0;
    
    // Получить решение
    KokkosVector getSolution() const {
        return x;
    }
    
    // Общие методы для всех решателей
    int getIterations() const {
        return iterationsDone;
    }
    
    double getPrecision() const {
        return achievedPrecision;
    }
    
    bool hasConverged() const {
        return converged;
    }
    
    std::string getName() const {
        return solverName;
    }
};