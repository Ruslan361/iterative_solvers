#include "msg_solver.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <chrono>

// Функция для проверки условий завершения итераций
bool MSGSolver::checkTerminationConditions(double precision_max_norm, double r_max_norm,
                                          double error_max_norm, int iterationsDone,
                                          StopCriterion& reason) {
    // Проверка прерывания пользователем (имеет наивысший приоритет)
    if (stop_requested) {
        reason = StopCriterion::INTERRUPTED;
        return true;
    }
    
    // Проверка критериев остановки в порядке приоритета
    
    // 1. Проверка по разности последовательных приближений
    if (use_precision && !std::isnan(precision_max_norm) && precision_max_norm < eps_precision) {
        reason = StopCriterion::PRECISION;
        return true;
    }
    
    // 2. Проверка по норме невязки
    if (use_residual && !std::isnan(r_max_norm) && r_max_norm < eps_residual) {
        reason = StopCriterion::RESIDUAL;
        return true;
    }
    
    // 3. Проверка по сравнению с истинным решением
    if (use_exact_error && !std::isnan(error_max_norm) && error_max_norm < eps_exact_error) {
        reason = StopCriterion::EXACT_ERROR;
        return true;
    }
    
    // 4. Проверка по максимальному числу итераций
    if (use_max_iterations && iterationsDone >= maxIterations) {
        reason = StopCriterion::ITERATIONS;
        return true;
    }
    
    // Ни один из критериев не выполнен - продолжаем итерации
    return false;
}

// Метод для решения СЛАУ с дополнительными критериями остановки
KokkosVector MSGSolver::solve() {
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
    
    // Определяем, нужно ли вычислять различные нормы
    bool need_residual_norm_check = use_residual; // Renamed to avoid conflict with local r_norm
    bool need_precision_norm_check = use_precision; // Renamed
    
    // Проверяем, есть ли истинное решение и включено ли использование точного решения
    bool has_true_solution = use_exact_error && exact_solution.extent(0) > 0;
    bool need_error_norm_check = has_true_solution; // Renamed
    
    // Нормы для критериев остановки (max-norms)
    double r_max_norm = std::numeric_limits<double>::quiet_NaN();
    double initial_r_max_norm = std::numeric_limits<double>::quiet_NaN(); // For consistent initial reporting if needed
    double initial_r_L2_norm = 0.0; // For existing console output

    // Значения для callback - используем NaN вместо отрицательных значений
    double cb_precision_max_norm = std::numeric_limits<double>::quiet_NaN();
    double cb_r_max_norm = std::numeric_limits<double>::quiet_NaN();
    double cb_error_max_norm = std::numeric_limits<double>::quiet_NaN();
    
    if (need_residual_norm_check) {
        r_max_norm = max_norm(r);
        initial_r_max_norm = r_max_norm; // Store initial max-norm
        cb_r_max_norm = r_max_norm;
    }
    initial_r_L2_norm = norm(r); // Calculate initial L2 norm for console output
    
    // Счетчик итераций
    int iterationsDone = 0;
    
    // Флаг сходимости
    converged = false;
    
    // Флаг для определения причины остановки
    stop_reason = StopCriterion::ITERATIONS;
    
    // Норма разности между последовательными приближениями
    double precision_max_norm = std::numeric_limits<double>::quiet_NaN();
    
    // Значение для callback (начальное)
    cb_precision_max_norm = need_precision_norm_check ? precision_max_norm : std::numeric_limits<double>::quiet_NaN();
    
    // Норма ошибки (разницы с истинным решением)
    double error_max_norm = std::numeric_limits<double>::quiet_NaN();
    
    // Рассчитываем начальную ошибку сравнения с истинным решением
    if (need_error_norm_check) {
        KokkosVector error_vec("error_vec", n); // Renamed to avoid conflict
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
            error_vec(i) = x(i) - exact_solution(i);
        });
        error_max_norm = max_norm(error_vec);
        cb_error_max_norm = error_max_norm;
    }
    
    // Вызываем колбэк для первой итерации (итерация 0)
    if (iteration_callback) {
        iteration_callback(0, cb_precision_max_norm, cb_r_max_norm, cb_error_max_norm);
    }
    
    // Проверяем условия завершения перед началом итераций
    if (checkTerminationConditions(precision_max_norm, r_max_norm, error_max_norm, iterationsDone, stop_reason)) {
        converged = (stop_reason != StopCriterion::INTERRUPTED && stop_reason != StopCriterion::ITERATIONS);
    } else {
        // Основной цикл итераций
        while (true) {
            Kokkos::deep_copy(x_prev, x);
            
            KokkosSparse::spmv("N", 1.0, a, z, 0.0, A_z);
            
            double r_dot_z = dot(r, z); // This is r_k^T z_k, which equals r_k^T r_k
            double Az_dot_z = dot(A_z, z);
            
            if (std::abs(Az_dot_z) < std::numeric_limits<double>::epsilon()) {
                std::cerr << "Error: Denominator Az_dot_z for alpha is close to zero at iteration " << iterationsDone + 1
                          << ". Stopping." << std::endl;
                stop_reason = StopCriterion::ITERATIONS; // Or a new StopCriterion::NUMERICAL_ISSUE
                converged = false;
                break;
            }
            double alpha = r_dot_z / Az_dot_z;
            
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
                x(i) = x(i) + alpha * z(i);
            });
            
            // Store r_dot_z (which is r_k^T r_k) for beta calculation later
            double r_old_dot_r_old = r_dot_z; 
            
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
                r(i) = r(i) - alpha * A_z(i); // r is now r_{k+1}
            });
            
            iterationsDone++;
            
            // Calculate r_new_dot_r_new (r_{k+1}^T r_{k+1}) for beta
            double r_new_dot_r_new = dot(r, r);

            // Сбрасываем значения callback перед пересчетом
            cb_precision_max_norm = std::numeric_limits<double>::quiet_NaN();
            cb_r_max_norm = std::numeric_limits<double>::quiet_NaN();
            cb_error_max_norm = std::numeric_limits<double>::quiet_NaN();
            
            // Вычисляем нормы только для требуемых критериев остановки
            if (need_residual_norm_check) {
                r_max_norm = max_norm(r);
                cb_r_max_norm = r_max_norm;
            }
            
            if (need_precision_norm_check) {
                KokkosVector diff("diff", n);
                Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
                    diff(i) = x(i) - x_prev(i);
                });
                precision_max_norm = max_norm(diff);
                cb_precision_max_norm = precision_max_norm;
            }
            
            if (need_error_norm_check) {
                KokkosVector error_vec("error_vec", n); // Renamed
                Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
                    error_vec(i) = x(i) - exact_solution(i);
                });
                error_max_norm = max_norm(error_vec);
                cb_error_max_norm = error_max_norm;
            }
            
            if (checkTerminationConditions(precision_max_norm, r_max_norm, error_max_norm, iterationsDone, stop_reason)) {
                converged = (stop_reason != StopCriterion::ITERATIONS && stop_reason != StopCriterion::INTERRUPTED);
                break;
            }
            
            double beta;
            if (std::abs(r_old_dot_r_old) < std::numeric_limits<double>::epsilon() * r_new_dot_r_new && std::abs(r_old_dot_r_old) < std::numeric_limits<double>::epsilon()) {
                 // If r_old_dot_r_old is very small (effectively r_k was zero), restart (beta=0)
                 // This condition means r_k was already very small, should have converged.
                beta = 0.0;
            } else {
                beta = r_new_dot_r_new / r_old_dot_r_old; // Fletcher-Reeves: (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            }
            
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
                z(i) = r(i) + beta * z(i); // z_{k+1} = r_{k+1} + beta * z_k
            });
            
            if (iteration_callback && (iterationsDone % 100 == 0 || iterationsDone == 1)) {
                 if (iterationsDone % 100 == 0 || iterationsDone == 1) { // Console output condition
                    std::cout << "Итерация: " << iterationsDone << "\n";
                    if (need_precision_norm_check) {
                        std::cout << "Точность ||x(n)-x(n-1)||: max-норма = " << std::scientific << precision_max_norm << "\n";
                    }
                    if (need_residual_norm_check) {
                        std::cout << "Невязка ||Ax-b||: max-норма = " << std::scientific << r_max_norm << "\n";
                    }
                    if (need_error_norm_check) {
                        std::cout << "Ошибка ||u-x||: max-норма = " << std::scientific << error_max_norm << "\n";
                    } else {
                        std::cout << "Ошибка ||u-x||: не вычисляется (нет точного решения или критерий отключен)\n";
                    }
                    std::cout << "\n";
                }
                if (iteration_callback) { // Callback may have different frequency
                    iteration_callback(iterationsDone, cb_precision_max_norm, cb_r_max_norm, cb_error_max_norm);
                }
            }
        }
    }
    
    iterations = iterationsDone;
    final_residual_norm = need_residual_norm_check ? r_max_norm : std::numeric_limits<double>::quiet_NaN();
    final_precision = need_precision_norm_check ? precision_max_norm : std::numeric_limits<double>::quiet_NaN();
    final_error_norm = need_error_norm_check ? error_max_norm : std::numeric_limits<double>::quiet_NaN();
    
    if (iteration_callback) {
        iteration_callback(iterationsDone, 
                           final_precision,      // Use final values for the last callback
                           final_residual_norm, 
                           final_error_norm);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Метод сопряженных градиентов (CG)\n" // Changed name for clarity, assuming it's standard CG
              << "Итераций: " << iterationsDone << "\n"
              << "Время: " << duration << " мс\n";
    
    if (need_residual_norm_check) { // Use the renamed flag
        std::cout << "Начальная невязка (L2): " << initial_r_L2_norm << "\n"; // Kept as L2 for now, labeled
        if (!std::isnan(final_residual_norm)) {
             std::cout << "Конечная невязка (max-norm): " << final_residual_norm << "\n";
        } else {
             std::cout << "Конечная невязка (max-norm): не вычислялась\n";
        }
    }
    
    std::cout << "Сходимость: " << (converged ? "Да" : "Нет") << "\n"
              << "Причина остановки: " << getStopReasonText() << std::endl;
    
    if (need_error_norm_check && !std::isnan(final_error_norm)) { // Use renamed flag
        std::cout << "Ошибка сравнения с точным решением (max-norm): " << final_error_norm << std::endl;
    }
    
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
    
    // Выводим информацию только о включенных критериях остановки
    ss << "Критерии остановки:\n";
    
    if (use_max_iterations) {
        ss << "  - Максимальное число итераций: " << maxIterations << "\n";
    }
    
    if (use_precision) {
        ss << "  - Точность ||xn-x(n-1)||: " << eps_precision << "\n";
    }
    
    if (use_residual) {
        ss << "  - Норма невязки ||Ax-b||: " << eps_residual << "\n";
    }
    
    if (use_exact_error) {
        ss << "  - Норма ошибки ||u-x||: " << eps_exact_error << "\n";
    }
    
    ss << "\n";
    
    // Результаты решения
    ss << "РЕЗУЛЬТАТЫ РЕШЕНИЯ:\n";
    ss << "-----------------\n";
    ss << "Выполнено итераций: " << iterations << "\n";
    ss << "Сходимость: " << (converged ? "Да" : "Нет") << "\n";
    ss << "Причина остановки: " << getStopReasonText() << "\n";
    ss << "Достигнутые величины:\n";
    
    // Выводим результаты только для включенных критериев
    if (use_precision) {
        ss << "  - Точность ||xn-x(n-1)||: " << std::scientific << final_precision << "\n";
    }
    
    if (use_residual) {
        ss << "  - Норма невязки ||Ax-b||: " << std::scientific << final_residual_norm << "\n";
    }
    
    if (use_exact_error) {
        ss << "  - Норма ошибки ||u-x||: " << std::scientific << final_error_norm << "\n";
    }
    
    ss << "\n";
    
    // Примечания
    ss << "ПРИМЕЧАНИЯ:\n";
    ss << "----------\n";
    ss << "- Все нормы вычислены как maximum-norm (максимальный модуль элемента)\n";
    
    if (use_exact_error) {
        ss << "- Для сравнения с истинным решением используется аналитическое решение задачи\n";
    }
    
    return ss.str();
}
