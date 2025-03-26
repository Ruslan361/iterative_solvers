#include "zeidel_solver.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// ZeidelSolver::ZeidelSolver(const std::vector<std::vector<double>>& a, const std::vector<double>& b, 
//                            double eps, int maxIterations)
//     : a(a), b(b), eps(eps), maxIterations(maxIterations), 
//       iterationsDone(0), achievedPrecision(0.0), converged(false) {
//     x.resize(b.size(), 0.0); // Инициализация начального приближения
// }

std::vector<double> ZeidelSolver::solve() {
    int n = b.size();
    std::vector<double> x_prev(n, 0.0);
    
    do {
        x_prev = x;
        
        // Основной цикл метода Зейделя
        for (int i = 0; i < n; ++i) {
            double sum1 = 0.0;  // Сумма с уже обновленными x_j
            double sum2 = 0.0;  // Сумма с еще не обновленными x_j
            
            for (int j = 0; j < i; ++j) {
                sum1 += a[i][j] * x[j];
            }
            
            for (int j = i + 1; j < n; ++j) {
                sum2 += a[i][j] * x_prev[j];
            }
            
            // Проверка на нулевой диагональный элемент
            if (fabs(a[i][i]) < 1e-10) {
                std::cerr << "Ошибка: деление на ноль (диагональный элемент близок к нулю): " << a[i][i] << std::endl;
                return x_prev;
            }
            
            x[i] = (b[i] - sum1 - sum2) / a[i][i];
        }
        
        // Проверка сходимости
        achievedPrecision = 0.0;
        for (int i = 0; i < n; ++i) {
            achievedPrecision = std::max(achievedPrecision, fabs(x[i] - x_prev[i]));
        }
        
        iterationsDone++;
        
    } while (achievedPrecision > eps && iterationsDone < maxIterations);
    
    converged = (achievedPrecision <= eps);
    
    std::cout << solverName << " " << (converged ? "сошёлся" : "не сошёлся") 
              << " за " << iterationsDone << " итераций, достигнув точности " 
              << std::uppercase << std::scientific << achievedPrecision << std::endl;
    
    return x;
}

// std::vector<double> ZeidelSolver::getSolution() const {
//     return x;
// }

// int ZeidelSolver::getIterations() const {
//     return iterationsDone;
// }

// double ZeidelSolver::getPrecision() const {
//     return achievedPrecision;
// }

// bool ZeidelSolver::hasConverged() const {
//     return converged;
// }