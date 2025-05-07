#include "msg_solver.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <chrono>

// Метод для решения СЛАУ с дополнительными критериями остановки
KokkosVector MSGSolver::solve(const KokkosVector& true_solution) {
    // Reset stop flag at the beginning of the solve process
    converged = false;
    stop_requested = false;
    
    // Инициализация таймера для измерения времени работы
    auto start_time = std::chrono::high_resolution_clock::now();

    // Размерность задачи
    int n = b.extent(0);
    
    // Инициализируем вектор решения нулями
    KokkosVector x("x", n);
    
    // Инициализируем предыдущее решение и вектор невязки
    KokkosVector x_prev("x_prev", n);
    KokkosVector r("r", n);  // Невязка: r = b - A*x
    
    // Рабочие векторы для градиентного метода
    KokkosVector z("z", n);  // Направление спуска
    KokkosVector A_z("A_z", n);  // A*z
    
    // Заполняем x нулями
    Kokkos::deep_copy(x, 0.0);
    
    // Изначально r = b - A*x = b (т.к. x = 0)
    Kokkos::deep_copy(r, b);  

    // Используем z как направление спуска, изначально z = r
    Kokkos::deep_copy(z, r);
    
    // Норма невязки
    double r_norm = norm(r);
    double r_max_norm = max_norm(r);
    double initial_r_norm = r_norm;
    
    // Счетчик итераций
    int iterationsDone = 0;
    
    // Флаг сходимости
    converged = false;
    
    // Флаг для определения причины остановки
    stop_reason = StopCriterion::ITERATIONS;
    
    // Норма разности между последовательными приближениями
    double precision_norm = std::numeric_limits<double>::max();
    double precision_max_norm = std::numeric_limits<double>::max();
    
    // Норма ошибки (разницы с истинным решением)
    double error_norm = std::numeric_limits<double>::max();
    double error_max_norm = std::numeric_limits<double>::max();
    
    // Рассчитываем начальную ошибку сравнения с истинным решением
    if (true_solution.extent(0) > 0) {
        KokkosVector error("error", n);
        // error = x - u_true
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
            error(i) = x(i) - true_solution(i);
        });
        error_norm = norm(error);
        error_max_norm = max_norm(error);
    }
    
    // Вызываем колбэк для первой итерации
    if (iteration_callback) {
        iteration_callback(0, precision_max_norm, r_max_norm, error_max_norm);
    }
    
    // Основной цикл итераций
    while (iterationsDone < maxIterations) {
        // Проверка, запросил ли пользователь остановку
        if (stop_requested) {
            // Устанавливаем, что процесс остановлен пользователем
            stop_reason = StopCriterion::INTERRUPTED;
            converged = false; // Решение не сошлось, т.к. было прервано
            break;
        }
        
        // Запоминаем текущее решение для проверки сходимости
        Kokkos::deep_copy(x_prev, x);
        
        // Умножаем матрицу A на вектор z
        KokkosSparse::spmv("N", 1.0, a, z, 0.0, A_z);
        
        // Скалярное произведение (r, z)
        double rz = dot(r, z);
        
        // Скалярное произведение (A*z, z)
        double Az_z = dot(A_z, z);
        
        // Вычисляем оптимальную длину шага
        double alpha = rz / Az_z;
        
        // Обновляем решение: x = x + alpha * z
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
            x(i) = x(i) + alpha * z(i);
        });
        
        // Обновляем невязку: r = r - alpha * A*z
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
            r(i) = r(i) - alpha * A_z(i);
        });
        
        // Обновляем счетчик итераций
        iterationsDone++;
        
        // Вычисляем нормы для всех критериев остановки
        
        // 1. Норма невязки
        r_norm = norm(r);
        r_max_norm = max_norm(r);
        
        // 2. Норма разности между последовательными приближениями
        KokkosVector diff("diff", n);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
            diff(i) = x(i) - x_prev(i);
        });
        precision_norm = norm(diff);
        precision_max_norm = max_norm(diff);
        
        // 3. Норма ошибки (сравнение с истинным решением)
        if (true_solution.extent(0) > 0) {
            KokkosVector error("error", n);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
                error(i) = x(i) - true_solution(i);
            });
            error_norm = norm(error);
            error_max_norm = max_norm(error);
        }
        
        // Проверяем все критерии остановки
        
        // Проверка по разности последовательных приближений
        if (eps_precision > 0 && precision_max_norm < eps_precision) {
            converged = true;
            stop_reason = StopCriterion::PRECISION;
            break;
        }
        
        // Проверка по норме невязки
        if (eps_residual > 0 && r_max_norm < eps_residual) {
            converged = true;
            stop_reason = StopCriterion::RESIDUAL;
            break;
        }
        
        // Проверка по сравнению с истинным решением
        if (eps_exact_error > 0 && true_solution.extent(0) > 0 && error_max_norm < eps_exact_error) {
            converged = true;
            stop_reason = StopCriterion::EXACT_ERROR;
            break;
        }
        
        // Обновляем направление спуска по методу сопряженных градиентов
        double beta = (r_norm * r_norm) / (rz);
        
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
            z(i) = r(i) + beta * z(i);
        });
        
        // Вывод промежуточной информации каждые 100 итераций или по запросу
        if (iterationsDone % 100 == 0 || iterationsDone == 1) {
            // Вывод промежуточной информации в консоль
            std::cout << "Итерация: " << iterationsDone << "\n";
            std::cout << "Точность ||x(n)-x(n-1)||: max-норма = " << std::scientific << precision_max_norm << "\n";
            std::cout << "Невязка ||Ax-b||: max-норма = " << std::scientific << r_max_norm << "\n";
            std::cout << "Ошибка ||u-x||: max-норма = " << std::scientific << error_max_norm << "\n\n";
            
            // Вызов колбэка для обновления UI
            if (iteration_callback) {
                iteration_callback(iterationsDone, precision_max_norm, r_max_norm, error_max_norm);
            }
        }
    }
    
    // Сохраняем финальные значения для отчета
    iterations = iterationsDone;
    final_residual_norm = r_max_norm;
    final_precision = precision_max_norm;
    final_error_norm = error_max_norm;
    
    // Вызов колбэка для финальной итерации
    if (iteration_callback) {
        iteration_callback(iterationsDone, precision_max_norm, r_max_norm, error_max_norm);
    }
    
    // Вычисляем время работы
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Вывод информации о решении
    std::cout << "Метод серединных градиентов (MSG)\n"
              << "Итераций: " << iterationsDone << "\n"
              << "Время: " << duration << " мс\n"
              << "Начальная невязка: " << initial_r_norm << "\n"
              << "Конечная невязка: " << r_norm << "\n"
              << "Сходимость: " << (converged ? "Да" : "Нет") << "\n"
              << "Причина остановки: " << getStopReasonText() << std::endl;
    
    // Возвращаем решение
    return x;
}

// Вспомогательная функция для вычисления скалярного произведения
double MSGSolver::dot(const KokkosVector& v1, const KokkosVector& v2) const {
    double result = 0.0;
    
    auto v1_host = Kokkos::create_mirror_view(v1);
    auto v2_host = Kokkos::create_mirror_view(v2);
    
    Kokkos::deep_copy(v1_host, v1);
    Kokkos::deep_copy(v2_host, v2);
    
    for (int i = 0; i < v1.extent(0); ++i) {
        result += v1_host(i) * v2_host(i);
    }
    
    return result;
}

// Вспомогательная функция для умножения матрицы на вектор A*v
KokkosVector MSGSolver::multiply(const KokkosCrsMatrix& A, const KokkosVector& v) const {
    int n = v.extent(0);
    KokkosVector result("A*v", n);
    
    KokkosSparse::spmv("N", 1.0, A, v, 0.0, result);
    
    return result;
}

// Вычисление нормы вектора (Евклидова норма)
double MSGSolver::norm(const KokkosVector& v) const {
    return std::sqrt(dot(v, v));
}

// Вычисление максимальной нормы вектора (максимальный модуль элемента)
double MSGSolver::max_norm(const KokkosVector& v) const {
    double max_val = 0.0;
    
    auto v_host = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(v_host, v);
    
    for (int i = 0; i < v.extent(0); ++i) {
        max_val = std::max(max_val, std::abs(v_host(i)));
    }
    
    return max_val;
}

// Генерация отчета о решении
std::string MSGSolver::generateReport(int n, int m, double a, double b, double c, double d) const {
    std::stringstream ss;
    
    ss << "ОТЧЕТ О РЕШЕНИИ ЗАДАЧИ ДИРИХЛЕ\n";
    ss << "===========================\n\n";
    
    // Параметры задачи
    ss << "ПАРАМЕТРЫ ЗАДАЧИ:\n";
    ss << "----------------\n";
    ss << "Размер сетки: " << n << "x" << m << " внутренних узлов\n";
    ss << "Область: [" << a << ", " << b << "] x [" << c << ", " << d << "]\n";
    ss << "Шаг по x: " << (b - a) / (n + 1) << "\n";
    ss << "Шаг по y: " << (d - c) / (m + 1) << "\n";
    ss << "Общее количество неизвестных: " << n * m << "\n\n";
    
    // Информация о методе решения
    ss << "МЕТОД РЕШЕНИЯ:\n";
    ss << "-------------\n";
    ss << "Название метода: " << name << "\n";
    ss << "Максимальное число итераций: " << maxIterations << "\n";
    ss << "Критерии остановки:\n";
    ss << "  - Точность ||xn-x(n-1)||: " << eps_precision << "\n";
    ss << "  - Норма невязки ||Ax-b||: " << eps_residual << "\n";
    ss << "  - Норма ошибки ||u-x||: " << eps_exact_error << "\n\n";
    
    // Результаты решения
    ss << "РЕЗУЛЬТАТЫ РЕШЕНИЯ:\n";
    ss << "-----------------\n";
    ss << "Выполнено итераций: " << iterations << "\n";
    ss << "Сходимость: " << (converged ? "Да" : "Нет") << "\n";
    ss << "Причина остановки: " << getStopReasonText() << "\n";
    ss << "Достигнутые величины:\n";
    ss << "  - Точность ||xn-x(n-1)||: " << std::scientific << final_precision << "\n";
    ss << "  - Норма невязки ||Ax-b||: " << std::scientific << final_residual_norm << "\n";
    ss << "  - Норма ошибки ||u-x||: " << std::scientific << final_error_norm << "\n\n";
    
    // Примечания
    ss << "ПРИМЕЧАНИЯ:\n";
    ss << "----------\n";
    ss << "- Все нормы вычислены как maximum-norm (максимальный модуль элемента)\n";
    ss << "- Для сравнения с истинным решением используется функция u(x,y) = exp(x^2 - y^2)\n";
    
    return ss.str();
}