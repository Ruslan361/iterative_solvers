#include "dirichlet_solver.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

// Функция правой части уравнения Пуассона
double function2(double x, double y) {
    return 4 * (x * x + y * y) * std::exp(x * x - y * y);
}

// Точное решение уравнения
double solution2(double x, double y) {
    return std::exp(x*x-y*y);
}

// Граничные условия, определенные на основе точного решения
double boundary_left(double x, double y) {
    return solution2(x, y);
}

double boundary_right(double x, double y) {
    return solution2(x, y);
}

double boundary_bottom(double x, double y) {
    return solution2(x, y);
}

double boundary_top(double x, double y) {
    return solution2(x, y);
}

// Класс для тестов с известным решением
class DirichletSolverSquareTest {
private:
    DirichletSolver solver;
    int n, m;
    double a, b, c, d;

public:
    DirichletSolverSquareTest(int grid_n = 50, int grid_m = 50,
                             double x_min = -1.0, double x_max = 1.0,
                             double y_min = -1.0, double y_max = 1.0) 
        : n(grid_n), m(grid_m), a(x_min), b(x_max), c(y_min), d(y_max) {
        
        // Создаем решатель для квадратной области с известной точной функцией решения
        solver = DirichletSolver(DomainType::SQUARE, n, m, a, b, c, d, function2, solution2);
        
        // Настраиваем параметры солвера
        solver.setSolverParameters(1e-10, 1e-10, 1e-10, 10000);
        solver.enableErrorStopping(true);
    }
    
    void run() {
        // Запускаем таймер
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Решаем задачу
        SolverResults results = solver.solve();
        
        // Останавливаем таймер
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        
        // Выводим информацию о решении
        std::cout << "===== Решение уравнения Пуассона на квадратной области =====" << std::endl;
        std::cout << "Размер сетки: " << n << "x" << m << std::endl;
        std::cout << "Область: [" << a << ", " << b << "] x [" << c << ", " << d << "]" << std::endl;
        std::cout << "Метод решения: " << solver.getMethodName() << std::endl;
        std::cout << "Число итераций: " << results.iterations << std::endl;
        std::cout << "Сходимость: " << (results.converged ? "Да" : "Нет") << std::endl;
        std::cout << "Причина останова: " << results.stop_reason << std::endl;
        std::cout << "Норма невязки: " << std::scientific << results.residual_norm << std::endl;
        std::cout << "Норма ошибки: " << std::scientific << results.error_norm << std::endl;
        std::cout << "Время решения: " << elapsed_seconds.count() << " с" << std::endl;
        
        // Сохраняем результаты в файл
        std::string result_filename = "square_test_results.txt";
        if (solver.saveResultsToFile(result_filename)) {
            std::cout << "Результаты сохранены в файл: " << result_filename << std::endl;
        } else {
            std::cerr << "Ошибка при сохранении результатов в файл" << std::endl;
        }
        
        // Сохраняем данные для 3D визуализации
        std::string viz_filename = "square_test_solution.dat";
        if (ResultsIO::saveSolutionFor3D(viz_filename, solver.solutionToMatrix(), a, b, c, d)) {
            std::cout << "Данные для визуализации сохранены в файл: " << viz_filename << std::endl;
        } else {
            std::cerr << "Ошибка при сохранении данных для визуализации" << std::endl;
        }
    }
};

// Класс для решения задачи без известного решения, только с граничными условиями
class DirichletSolverSquareMain {
private:
    DirichletSolver solver;
    int n, m;
    double a, b, c, d;

public:
    DirichletSolverSquareMain(int grid_n = 50, int grid_m = 50,
                             double x_min = -1.0, double x_max = 1.0,
                             double y_min = -1.0, double y_max = 1.0) 
        : n(grid_n), m(grid_m), a(x_min), b(x_max), c(y_min), d(y_max) {
        
        // Создаем решатель для квадратной области с граничными условиями
        solver = DirichletSolver(DomainType::SQUARE, n, m, a, b, c, d, 
                               function2, 
                               boundary_left, boundary_right, boundary_bottom, boundary_top);
        
        // Настраиваем параметры солвера (без критерия точности)
        solver.setSolverParameters(1e-10, 1e-10, 1e-10, 10000);
        solver.enableErrorStopping(false);  // Нет точного решения для сравнения
    }
    
    void run() {
        // Запускаем таймер
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Решаем задачу
        SolverResults results = solver.solve();
        
        // Останавливаем таймер
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        
        // Выводим информацию о решении
        std::cout << "===== Решение уравнения Пуассона на квадратной области (только граничные условия) =====" << std::endl;
        std::cout << "Размер сетки: " << n << "x" << m << std::endl;
        std::cout << "Область: [" << a << ", " << b << "] x [" << c << ", " << d << "]" << std::endl;
        std::cout << "Метод решения: " << solver.getMethodName() << std::endl;
        std::cout << "Число итераций: " << results.iterations << std::endl;
        std::cout << "Сходимость: " << (results.converged ? "Да" : "Нет") << std::endl;
        std::cout << "Причина останова: " << results.stop_reason << std::endl;
        std::cout << "Норма невязки: " << std::scientific << results.residual_norm << std::endl;
        std::cout << "Время решения: " << elapsed_seconds.count() << " с" << std::endl;
        
        // Сохраняем результаты в файл
        std::string result_filename = "square_main_results.txt";
        if (solver.saveResultsToFile(result_filename)) {
            std::cout << "Результаты сохранены в файл: " << result_filename << std::endl;
        } else {
            std::cerr << "Ошибка при сохранении результатов в файл" << std::endl;
        }
        
        // Сохраняем данные для 3D визуализации
        std::string viz_filename = "square_main_solution.dat";
        if (ResultsIO::saveSolutionFor3D(viz_filename, solver.solutionToMatrix(), a, b, c, d)) {
            std::cout << "Данные для визуализации сохранены в файл: " << viz_filename << std::endl;
        } else {
            std::cerr << "Ошибка при сохранении данных для визуализации" << std::endl;
        }
    }
};

int main(int argc, char *argv[]) {
    try {
        int grid_size = 100;  // Размер сетки по умолчанию
        if (argc > 1) {
            grid_size = std::atoi(argv[1]);
        }
        
        // Инициализируем Kokkos
        Kokkos::initialize(argc, argv);
        
        // Запускаем тесты
        std::cout << "=== Запуск DirichletSolverSquareTest ===" << std::endl;
        DirichletSolverSquareTest test(grid_size, grid_size);
        test.run();
        
        std::cout << "\n=== Запуск DirichletSolverSquareMain ===" << std::endl;
        DirichletSolverSquareMain main_solver(grid_size, grid_size);
        main_solver.run();
        
        // Завершаем работу с Kokkos
        Kokkos::finalize();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
}