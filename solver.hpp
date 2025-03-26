#pragma once

#include <vector>
#include <string>
#include <iostream>

class Solver {
protected:
    std::vector<std::vector<double>> a; // Матрица коэффициентов
    std::vector<double> b;             // Вектор правой части
    std::vector<double> x;             // Решение
    double eps;                         // Точность
    int maxIterations;                  // Максимальное число итераций
    int iterationsDone;                 // Выполнено итераций
    double achievedPrecision;           // Достигнутая точность
    bool converged;                     // Сошелся ли метод
    std::string solverName;             // Название метода решателя

public:
    Solver(const std::vector<std::vector<double>>& a, 
           const std::vector<double>& b, 
           double eps = 1e-6, 
           int maxIterations = 10000,
           const std::string& name = "Базовый решатель")
        : a(a), b(b), eps(eps), maxIterations(maxIterations), 
          iterationsDone(0), achievedPrecision(0.0), converged(false),
          solverName(name) {
        x.resize(b.size(), 0.0); // Инициализация начального приближения
    }
    
    virtual ~Solver() = default;
    
    // Абстрактный метод решения системы
    virtual std::vector<double> solve() = 0;
    
    // Общие методы для всех решателей
    std::vector<double> getSolution() const {
        return x;
    }
    
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