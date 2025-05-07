#pragma once

#include "solver.hpp"
#include "grid_system.h"
#include "msg_solver.hpp"
#include <memory>
#include <functional>
#include <fstream>

// Структура для хранения результатов решения
struct SolverResults {
    std::vector<double> solution;         // Решение
    std::vector<double> true_solution;    // Истинное решение
    std::vector<double> residual;         // Невязка (Ax - b)
    std::vector<double> error;            // Ошибка (u - x)
    std::vector<double> x_coords;         // Координаты X точек решения
    std::vector<double> y_coords;         // Координаты Y точек решения
    double residual_norm;                 // Норма невязки
    double error_norm;                    // Норма ошибки
    int iterations;                       // Число итераций
    double precision;                     // Достигнутая точность
    bool converged;                       // Сошелся ли метод
    std::string stop_reason;              // Причина остановки
};

// Класс для работы с файлами результатов
class ResultsIO {
public:
    // Сохранение результатов в файл
    static bool saveResults(const std::string& filename, const SolverResults& results,
                          int n, int m, double a, double b, double c, double d,
                          const std::string& solver_name);
    
    // Загрузка результатов из файла
    static bool loadResults(const std::string& filename, SolverResults& results,
                          int& n, int& m, double& a, double& b, double& c, double& d,
                          std::string& solver_name);
    
    // Сохранение матрицы и вектора правой части в файл
    static bool saveMatrixAndRhs(const std::string& filename, const KokkosCrsMatrix& A,
                               const KokkosVector& b, int n, int m);
    
    // Сохранение решения в формате для 3D визуализации
    static bool saveSolutionFor3D(const std::string& filename, 
                                 const std::vector<std::vector<double>>& solution,
                                 double a_bound, double b_bound,
                                 double c_bound, double d_bound) {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            return false;
        }

        int m = solution.size();
        if (m <= 0) return false;
        
        int n = solution[0].size();
        if (n <= 0) return false;
        
        // Расчет шагов сетки
        double h_x = (b_bound - a_bound) / (n + 1);
        double h_y = (d_bound - c_bound) / (m + 1);
        
        // Запись данных в формате для gnuplot: x y z
        for (int i = 0; i < m; ++i) {
            double y = c_bound + (i + 1) * h_y;
            
            for (int j = 0; j < n; ++j) {
                double x = a_bound + (j + 1) * h_x;
                outFile << x << " " << y << " " << solution[i][j] << std::endl;
            }
            outFile << std::endl; // Пустая строка для разделения строк сетки (для gnuplot)
        }
        
        outFile.close();
        return true;
    }
};

class DirichletSolver {
private:
    // Параметры сетки
    int n_internal;    // Число внутренних узлов по x
    int m_internal;    // Число внутренних узлов по y
    double a_bound;    // Левая граница по x
    double b_bound;    // Правая граница по x
    double c_bound;    // Нижняя граница по y
    double d_bound;    // Верхняя граница по y
    
    // Границы локальной матрицы
    int local_i_begin, local_i_end;
    int local_j_begin, local_j_end;

    // Параметры для критериев остановки
    double eps_precision;   // Точность по разнице между xn и xn-1
    double eps_residual;    // Точность по норме невязки
    double eps_exact_error; // Точность по сравнению с точным решением
    int max_iterations;     // Максимальное число итераций
    
    // Флаги активации критериев останова
    bool use_precision_stopping;     // Использовать критерий точности
    bool use_residual_stopping;      // Использовать критерий невязки
    bool use_error_stopping;         // Использовать критерий ошибки
    bool use_max_iterations_stopping; // Использовать критерий максимального числа итераций
    
    // Флаг запроса на остановку решателя
    bool stop_requested = false;

    // Колбэки для отслеживания итераций и завершения
    std::function<void(int, double, double, double)> iteration_callback;
    std::function<void(const SolverResults&)> completion_callback;
    
    // Система сетки для дискретизации
    std::unique_ptr<GridSystem> grid;
    
    // Решатель для системы уравнений
    std::unique_ptr<MSGSolver> solver;
    
    // Вектор решения и истинного решения
    KokkosVector solution;
    KokkosVector true_solution;
    
    // Вспомогательные методы
    std::vector<double> kokkosToStdVector(const KokkosVector& kv) const;
    KokkosVector computeResidual(const KokkosCrsMatrix& A, const KokkosVector& x, const KokkosVector& b) const;
    std::vector<double> computeError(const KokkosVector& v) const;
    
    // Методы решения
    SolverResults solveWithMSG();
    void finalizeResults(SolverResults& results);

public:
    // Конструктор
    DirichletSolver(int n = 10, int m = 10, 
                   double a = 0.0, double b = 1.0, 
                   double c = 0.0, double d = 1.0);
    
    // Деструктор
    ~DirichletSolver();
    
    // Настройка параметров
    void setGridParameters(int n, int m, double a, double b, double c, double d);
    void setSolverParameters(double eps_p, double eps_r, double eps_e, int max_iter);
    
    // Управление критериями останова
    void enablePrecisionStopping(bool enable) { use_precision_stopping = enable; }
    void enableResidualStopping(bool enable) { use_residual_stopping = enable; }
    void enableErrorStopping(bool enable) { use_error_stopping = enable; }
    void enableMaxIterationsStopping(bool enable) { use_max_iterations_stopping = enable; }
    
    // Прерывание решения по запросу пользователя
    void requestStop() {
        stop_requested = true;
        if (solver) {
            solver->requestStop();
        }
    }
    
    // Получение названия метода
    std::string getMethodName() const {
        return solver ? solver->getName() : "МСГ";
    }
    
    // Установка колбэков для отслеживания прогресса
    void setIterationCallback(std::function<void(int, double, double, double)> callback);
    void setCompletionCallback(std::function<void(const SolverResults&)> callback) {
        completion_callback = callback;
    }
    
    // Решение задачи
    SolverResults solve();
    
    // Получение результатов
    std::vector<double> getSolution() const;
    std::vector<double> getTrueSolution() const;
    std::vector<std::vector<double>> solutionToMatrix() const;
    
    // Работа с отчетами
    std::string generateReport() const;
    bool saveResultsToFile(const std::string& filename) const;
    bool saveMatrixAndRhsToFile(const std::string& filename) const;

    // Метод для получения доступа к GridSystem
    const GridSystem* getGridSystem() const { return grid.get(); }
};