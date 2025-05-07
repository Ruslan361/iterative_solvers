#pragma once

#include "solver.hpp"
#include <string>
#include <functional>
#include <atomic>

// Перечисление критериев остановки
enum class StopCriterion {
    ITERATIONS,        // По числу итераций
    PRECISION,         // По точности (норма разности xn и xn-1)
    RESIDUAL,          // По малости невязки
    EXACT_ERROR,       // По норме разности между численным и истинным решением
    INTERRUPTED        // Прервано пользователем
};

class MSGSolver : public Solver {
private:
    // Параметры для критериев остановки
    double eps_precision;       // Точность по разнице между xn и xn-1
    double eps_residual;        // Точность по норме невязки
    double eps_exact_error;     // Точность по сравнению с точным решением
    
    // Флаги и информация о сходимости
    bool converged;             // Флаг успешной сходимости
    StopCriterion stop_reason;  // Причина остановки
    double final_residual_norm; // Итоговая норма невязки
    double final_error_norm;    // Итоговая норма ошибки (сравнение с точным решением)
    double final_precision;     // Итоговая точность (разница между xn и xn-1)

    // Колбэк для отслеживания промежуточных итераций
    std::function<void(int, double, double, double)> iteration_callback;
    
    // Флаг для прерывания решения извне
    std::atomic<bool> stop_requested;
    
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
        : Solver(a, b, eps, maxIterations, "Метод серединных градиентов"),
          eps_precision(eps), eps_residual(eps), eps_exact_error(eps),
          converged(false), stop_reason(StopCriterion::ITERATIONS),
          final_residual_norm(0.0), final_error_norm(0.0), final_precision(0.0),
          stop_requested(false) {}
    
    // Устанавливает точность для критерия разницы между xn и xn-1
    void setPrecisionEps(double eps) { eps_precision = eps; }
    
    // Устанавливает точность для критерия нормы невязки
    void setResidualEps(double eps) { eps_residual = eps; }
    
    // Устанавливает точность для критерия сравнения с точным решением
    void setExactErrorEps(double eps) { eps_exact_error = eps; }
    
    // Получение информации о сходимости
    bool hasConverged() const { return converged; }
    
    // Получение причины остановки итераций
    StopCriterion getStopReason() const { return stop_reason; }
    
    // Запрос на остановку решения
    void requestStop() { stop_requested = true; }
    
    // Сброс флага остановки
    void resetStop() { stop_requested = false; }
    
    // Проверка, запрошена ли остановка
    bool isStopRequested() const { return stop_requested; }
    
    // Получение текста с причиной остановки
    std::string getStopReasonText() const {
        switch(stop_reason) {
            case StopCriterion::ITERATIONS: 
                return "Достигнуто максимальное число итераций";
            case StopCriterion::PRECISION: 
                return "Достигнута требуемая точность по норме разности xn и xn-1";
            case StopCriterion::RESIDUAL: 
                return "Достигнута требуемая точность по норме невязки";
            case StopCriterion::EXACT_ERROR: 
                return "Достигнута требуемая точность по норме разности с истинным решением";
            case StopCriterion::INTERRUPTED:
                return "Прервано пользователем";
            default: 
                return "Неизвестная причина остановки";
        }
    }
    
    // Получение итоговой нормы невязки
    double getFinalResidualNorm() const { return final_residual_norm; }
    
    // Получение итоговой нормы ошибки
    double getFinalErrorNorm() const { return final_error_norm; }
    
    // Получение итоговой точности
    double getFinalPrecision() const { return final_precision; }
    
    // Установка обратного вызова для отслеживания итераций
    void setIterationCallback(std::function<void(int, double, double, double)> callback) {
        iteration_callback = callback;
    }
    
    // Метод для решения СЛАУ с дополнительными критериями остановки
    KokkosVector solve(const KokkosVector& true_solution) override;
    
    // Генерация отчета о решении
    std::string generateReport(int n, int m, double a, double b, double c, double d) const;
};