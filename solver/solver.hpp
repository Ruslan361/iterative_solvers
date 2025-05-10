#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <functional>
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
    double eps;                       // Точность
    int maxIterations;                // Максимальное число итераций
    int iterations;                   // Выполнено итераций
    std::string name;                 // Название метода решателя

    // Обратный вызов для отслеживания промежуточных результатов
    std::function<void(int, double, double, double)> iteration_callback;
    
    // Обратный вызов для уведомления о завершении решения
    std::function<void(bool, const std::string&)> completion_callback;

public:
    Solver(const KokkosCrsMatrix& a, 
           const KokkosVector& b, 
           double eps = 1e-6, 
           int maxIterations = 10000,
           const std::string& name = "Базовый решатель")
        : a(a), b(b), eps(eps), maxIterations(maxIterations), 
          iterations(0), name(name) {}
    
    virtual ~Solver() = default;
    
    // Абстрактный метод решения системы
    virtual KokkosVector solve() = 0;
    
    // Установка колбэка для отслеживания итераций
    // Параметры: итерация, точность ||xn-x(n-1)||, невязка ||Ax-b||, ошибка ||u-x||
    void setIterationCallback(std::function<void(int, double, double, double)> callback) {
        iteration_callback = callback;
    }
    
    // Установка колбэка для уведомления о завершении
    // Параметры: успех/неудача, причина остановки
    void setCompletionCallback(std::function<void(bool, const std::string&)> callback) {
        completion_callback = callback;
    }
    
    // Общие методы для всех решателей
    int getIterations() const {
        return iterations;
    }
    
    std::string getName() const {
        return name;
    }
};