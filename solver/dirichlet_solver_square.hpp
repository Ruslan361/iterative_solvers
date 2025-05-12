#ifndef DIRICHLET_SOLVER_SQUARE_HPP
#define DIRICHLET_SOLVER_SQUARE_HPP

#include "grid_system_square.h"
#include "msg_solver.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>

// Структура для хранения результатов решения для квадратной области
struct SquareSolverResults {
    std::vector<double> solution;       // Численное решение
    std::vector<double> true_solution;  // Точное решение (если доступно)
    std::vector<double> residual;       // Вектор невязки (Ax - b)
    std::vector<double> error;          // Вектор ошибки (solution - true_solution)
    std::vector<double> x_coords;       // Координаты X узлов сетки (для основного решения)
    std::vector<double> y_coords;       // Координаты Y узлов сетки (для основного решения)
    int iterations;                     // Количество выполненных итераций
    bool converged;                     // Флаг сходимости
    std::string stop_reason;            // Причина останова
    double residual_norm;               // Норма невязки ||Ax - b||
    double error_norm;                  // Норма ошибки ||x - x_true||
    double precision;                   // Достигнутая точность (например, ||x_k - x_{k-1}||)
    double initial_residual_norm;       // Начальная норма невязки (например, ||b|| или ||Ax_0 - b||)

    // Поля для результатов на подробной сетке
    double refined_grid_error;                        // Норма разности решений (текущее - подробное)
    int refined_grid_iterations;                    // Итерации на подробной сетке
    double refined_grid_precision;                  // Точность на подробной сетке
    double refined_grid_residual_norm;              // Норма невязки на подробной сетке
    double refined_grid_initial_residual_norm;      // Начальная норма невязки на подробной сетке
    std::vector<double> refined_grid_solution;      // Решение на подробной сетке
    std::vector<double> solution_refined_diff;      // Разность между решением на основной и подробной сетке
    std::vector<double> refined_grid_x_coords;      // Координаты X узлов подробной сетки
    std::vector<double> refined_grid_y_coords;      // Координаты Y узлов подробной сетки


    // Конструктор по умолчанию для инициализации
    SquareSolverResults() : iterations(0), converged(false), residual_norm(0.0), 
                          error_norm(0.0), precision(0.0), initial_residual_norm(0.0),
                          refined_grid_error(-1.0), refined_grid_iterations(0),
                          refined_grid_precision(0.0), refined_grid_residual_norm(0.0),
                          refined_grid_initial_residual_norm(0.0) {}
};

// Класс для работы с решателем уравнения Дирихле в квадратной области
class DirichletSolverSquare {
public:
    // Конструкторы
    DirichletSolverSquare(int n, int m, double a, double b, double c, double d);
    DirichletSolverSquare(int n, int m, double a, double b, double c, double d,
                        double (*f)(double, double), double (*sol)(double, double) = nullptr);
    DirichletSolverSquare(int n, int m, double a, double b, double c, double d,
                        double (*f)(double, double),
                        double (*mu1_func)(double, double),
                        double (*mu2_func)(double, double),
                        double (*mu3_func)(double, double),
                        double (*mu4_func)(double, double));
    
    // Деструктор
    ~DirichletSolverSquare();

    // Методы для установки параметров
    void setGridParameters(int n, int m, double a, double b, double c, double d);
    void setSolverParameters(double eps_p, double eps_r, double eps_e, int max_iter);
    
    // Включение/выключение различных критериев остановки
    void setUsePrecisionStopping(bool value) { use_precision_stopping = value; }
    void setUseResidualStopping(bool value) { use_residual_stopping = value; }
    void setUseErrorStopping(bool value) { use_error_stopping = value; }
    void setUseMaxIterationsStopping(bool value) { use_max_iterations_stopping = value; }
    
    // Решение задачи
    SquareSolverResults solve();
    
    // Получение результатов
    std::vector<double> getSolution() const;
    std::vector<double> getTrueSolution() const;
    std::vector<std::vector<double>> solutionToMatrix() const;
    
    // Методы для работы с решением на более мелкой сетке
    const SquareSolverResults* getRefinedGridResults() const {
        return refined_grid_results.get();
    }
    
    SquareSolverResults* getRefinedGridResultsPtr() {
        if (!refined_grid_results) {
            refined_grid_results = std::make_unique<SquareSolverResults>();
        }
        return refined_grid_results.get();
    }
    
    // Вспомогательные методы
    std::string generateReport() const;
    bool saveResultsToFile(const std::string& filename) const;
    bool saveMatrixAndRhsToFile(const std::string& filename) const;
    
    // Установка функций обратного вызова
    using IterationCallbackType = std::function<void(int, double, double, double)>;
    using CompletionCallbackType = std::function<void(const SquareSolverResults&)>;
    
    void setIterationCallback(IterationCallbackType callback) {
        iteration_callback = callback;
    }
    
    void setCompletionCallback(CompletionCallbackType callback) {
        completion_callback = callback;
    }
    
    // Проверка доступности точного решения
    bool hasTrueSolution() const {
        return (exact_solution != nullptr);
    }
    
    // Получение имени метода решения
    std::string getMethodName() const {
        return (solver ? solver->getName() : "Unknown");
    }

    void requestStop() {
        if (solver) {
            solver->requestStop();
        }
    }

    // Метод для получения доступа к GridSystemSquare
    const GridSystemSquare* getGridSystem() const { return grid.get(); }

    // Вычисление ошибки с использованием решения на более мелкой сетке
    double computeRefinedGridError();
    
    // Флаг использования сравнения с решением на более мелкой сетке
    void setUseRefinedGridComparison(bool value) { use_refined_grid_comparison = value; }
    bool getUseRefinedGridComparison() const { return use_refined_grid_comparison; }

private:
    // Параметры сетки
    int n_internal, m_internal;
    double a_bound, b_bound, c_bound, d_bound;
    
    // Параметры точности
    double eps_precision;
    double eps_residual;
    double eps_exact_error;
    int max_iterations;
    
    // Флаги критериев остановки
    bool use_precision_stopping;
    bool use_residual_stopping;
    bool use_error_stopping;
    bool use_max_iterations_stopping;
    bool use_refined_grid_comparison = false;  // Флаг использования сравнения с решением на более мелкой сетке
    
    // Функции для уравнения и граничных условий
    double (*func)(double, double) = nullptr;  // Функция правой части уравнения
    double (*exact_solution)(double, double) = nullptr;  // Функция точного решения
    
    // Функции граничных условий
    double (*mu1)(double, double) = nullptr;  // Левая граница (x = a_bound)
    double (*mu2)(double, double) = nullptr;  // Правая граница (x = b_bound)
    double (*mu3)(double, double) = nullptr;  // Нижняя граница (y = c_bound)
    double (*mu4)(double, double) = nullptr;  // Верхняя граница (y = d_bound)
    
    // Объекты для сетки и решателя
    std::unique_ptr<GridSystemSquare> grid;
    std::unique_ptr<MSGSolver> solver;
    
    // Векторы решений
    KokkosVector solution;
    KokkosVector true_solution;
    
    // Результаты решения на более мелкой сетке
    std::unique_ptr<SquareSolverResults> refined_grid_results; // Для хранения полных результатов с мелкой сетки
    
    // Функции обратного вызова
    IterationCallbackType iteration_callback;
    CompletionCallbackType completion_callback;
    
    // Вспомогательные методы
    std::vector<double> kokkosToStdVector(const KokkosVector& kv) const;
    KokkosVector computeResidual(const KokkosCrsMatrix& A, const KokkosVector& x, const KokkosVector& b) const;
    std::vector<double> computeError(const KokkosVector& v) const;
    double calculateSolutionDifferenceNorm(const std::vector<double>& sol1, const GridSystemSquare& grid1,
                                          const std::vector<double>& sol2, const GridSystemSquare& grid2);
};

// Класс для ввода/вывода результатов
class SquareResultsIO {
public:
    static bool saveResults(const std::string& filename, const SquareSolverResults& results,
                          int n, int m, double a, double b, double c, double d,
                          const std::string& solver_name);
    
    static bool loadResults(const std::string& filename, SquareSolverResults& results,
                          int& n, int& m, double& a, double& b, double& c, double& d,
                          std::string& solver_name);
    
    static bool saveMatrixAndRhs(const std::string& filename, const KokkosCrsMatrix& A,
                               const KokkosVector& b, int n, int m);
};

#endif // DIRICHLET_SOLVER_SQUARE_HPP