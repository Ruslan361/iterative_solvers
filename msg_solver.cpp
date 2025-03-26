#include "msg_solver.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// MSGSolver::MSGSolver(const std::vector<std::vector<double>>& a, const std::vector<double>& b, 
//                      double eps, int maxIterations)
//     : a(a), b(b), eps(eps), maxIterations(maxIterations), 
//       iterationsDone(0), achievedPrecision(0.0), converged(false) {
//     x.resize(b.size(), 0.0); // Инициализация начального приближения
// }

double MSGSolver::dot(const std::vector<double>& v1, const std::vector<double>& v2) const {
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

std::vector<double> MSGSolver::multiply(const std::vector<std::vector<double>>& A, 
                                        const std::vector<double>& v) const {
    std::vector<double> result(v.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

double MSGSolver::norm(const std::vector<double>& v) const {
    double sum = 0.0;
    for (const auto& val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

std::vector<double> MSGSolver::solve() {
    int n = b.size();
    
    // Начальное приближение
    x = std::vector<double>(n, 0.0);
    
    // Вычисляем начальную невязку r0 = b - A*x0
    std::vector<double> r = b; // так как x0 = 0, r0 = b
    
    // Начальное направление p0 = r0
    std::vector<double> p = r;
    
    // Норма невязки
    double r_norm = norm(r);
    achievedPrecision = r_norm;
    
    iterationsDone = 0;
    
    while (r_norm > eps && iterationsDone < maxIterations) {
        // Вычисляем A*p
        std::vector<double> Ap = multiply(a, p);
        
        // Вычисляем шаг alpha = (r, p) / (p, A*p)
        double alpha = dot(r, p) / dot(p, Ap);
        
        // Обновляем решение x = x + alpha*p
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
        }
        
        // Сохраняем предыдущую невязку
        std::vector<double> r_prev = r;
        
        // Обновляем невязку r = r - alpha*A*p
        for (int i = 0; i < n; ++i) {
            r[i] -= alpha * Ap[i];
        }
        
        // Вычисляем A*r
        std::vector<double> Ar = multiply(a, r);
        
        // Вычисляем A*(r - r_prev)
        std::vector<double> Ar_diff(n);
        for (int i = 0; i < n; ++i) {
            Ar_diff[i] = Ar[i] - multiply(a, r_prev)[i];
        }
        
        // Вычисляем beta = (r, A*(r - r_prev)) / (r_prev, A*r_prev)
        double beta = dot(r, Ar_diff) / dot(r_prev, multiply(a, r_prev));
        
        // Обновляем направление p = r + beta*p
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        
        // Обновляем норму невязки
        r_norm = norm(r);
        achievedPrecision = r_norm;
        
        iterationsDone++;
    }
    
    converged = (r_norm <= eps);
    
    std::cout << solverName << " " << (converged ? "сошёлся" : "не сошёлся") 
              << " за " << iterationsDone << " итераций, достигнув точности " 
              << std::uppercase << std::scientific << achievedPrecision << std::endl;
    
    return x;
}

// std::vector<double> MSGSolver::getSolution() const {
//     return x;
// }

// int MSGSolver::getIterations() const {
//     return iterationsDone;
// }

// double MSGSolver::getPrecision() const {
//     return achievedPrecision;
// }

// bool MSGSolver::hasConverged() const {
//     return converged;
// }