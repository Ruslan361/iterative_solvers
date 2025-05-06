#include "msg_solver.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono> // Для замера времени
#include <iostream> // Для вывода в консоль
#include <KokkosBlas1_axpby.hpp> // Для использования KokkosBlas::axpy

//#define deb  // Раскомментировал эту строку для активации отладочного вывода

double MSGSolver::dot(const KokkosVector& v1, const KokkosVector& v2) const {
    double result = 0.0;
    auto v1_host = Kokkos::create_mirror_view(v1);
    auto v2_host = Kokkos::create_mirror_view(v2);
    
    Kokkos::deep_copy(v1_host, v1);
    Kokkos::deep_copy(v2_host, v2);
    
    for (size_t i = 0; i < v1.extent(0); ++i) {
        result += v1_host(i) * v2_host(i);
    }
    return result;
}

KokkosVector MSGSolver::multiply(const KokkosCrsMatrix& A, const KokkosVector& v) const {
    KokkosVector result("result", v.extent(0));
    KokkosSparse::spmv("N", 1.0, A, v, 0.0, result);
    return result;
}

double MSGSolver::norm(const KokkosVector& v) const {
    return std::sqrt(dot(v, v));
}

double MSGSolver::max_norm(const KokkosVector& v) const {
    auto v_host = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(v_host, v);
    
    double max_value = 0.0;
    for (size_t i = 0; i < v.extent(0); ++i) {
        double abs_val = std::abs(v_host(i));
        if (abs_val > max_value) {
            max_value = abs_val;
        }
    }
    return max_value;
}

// Вспомогательная функция для операции x = x + alpha * p
// Заменяем на использование KokkosBlas::axpy
void axpy(double alpha, const KokkosVector& p, KokkosVector& x) {
    KokkosBlas::axpy(alpha, p, x);
}

KokkosVector MSGSolver::solve(const KokkosVector& true_solution) { // New signature
    int n_solve = b.extent(0); // Renamed n to n_solve to avoid conflict with class member n if any
    if (true_solution.extent(0) != n_solve) {
        throw std::runtime_error("True solution vector size does not match system size in MSGSolver.");
    }

#ifdef deb
    std::ofstream debug_file("msg_debug.txt");
    debug_file << std::scientific << std::setprecision(10);
    debug_file << "=== Метод серединных градиентов: отладочная информация ===" << std::endl;
#endif

    // Начальное приближение x = 0
    Kokkos::deep_copy(x, 0.0);
    
    // Сохраняем предыдущее приближение для вычисления точности
    KokkosVector x_prev("x_prev", n_solve);
    Kokkos::deep_copy(x_prev, x); // Начальное x_prev = x = 0
    
    // Инициализация r = b - A*x = b (т.к. x0 = 0)
    KokkosVector r("r", n_solve);
    auto b_mirror_for_r_init = Kokkos::create_mirror_view(b);
    Kokkos::deep_copy(b_mirror_for_r_init, b);
    auto r_mirror_init = Kokkos::create_mirror_view(r);
    for (int i = 0; i < n_solve; ++i) {
        r_mirror_init(i) = b_mirror_for_r_init(i);  // r = b при x=0
    }
    Kokkos::deep_copy(r, r_mirror_init);
    
    // Начальное направление h = r
    KokkosVector h("h", n_solve);
    Kokkos::deep_copy(h, r);  // h0 = r0

    // Для хранения предыдущего h
    KokkosVector h_old("h_old", n_solve);

    // Для хранения предыдущей невязки
    KokkosVector r_old("r_old", n_solve);
    
    auto start_time = std::chrono::steady_clock::now(); // Засекаем время начала решения
    long long last_print_iteration = 0; // Для контроля частоты вывода
    const long long print_interval_iterations = 100; // Выводить каждые N итераций
    
#ifdef deb
    debug_file << "Итерация 0:" << std::endl;
    debug_file << "x0 = [";
    auto x0_host = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(x0_host, x);
    for (int i = 0; i < n_solve; ++i) {
        debug_file << x0_host(i);
        if (i < n_solve-1) debug_file << ", ";
    }
    debug_file << "]" << std::endl;
    
    auto h_host_debug = Kokkos::create_mirror_view(h);
    Kokkos::deep_copy(h_host_debug, h);
    debug_file << "h0 = [";
    for (int i = 0; i < n_solve; ++i) {
        debug_file << h_host_debug(i);
        if (i < n_solve-1) debug_file << ", ";
    }
    debug_file << "]" << std::endl;
#endif

    // Предвычисляем A*h
    KokkosVector Ah = multiply(a, h);
    
    // Норма невязки
    double r_norm = norm(r);
    double r_max_norm = max_norm(r);
    achievedPrecision = r_norm;
    
    // Для расчета ошибки: разница между истинным решением и текущим
    KokkosVector error_vec("error_vec", n_solve);
    auto x_host_for_error = Kokkos::create_mirror_view(x);
    auto true_solution_host = Kokkos::create_mirror_view(true_solution);
    auto error_host = Kokkos::create_mirror_view(error_vec);
    
    Kokkos::deep_copy(x_host_for_error, x);
    Kokkos::deep_copy(true_solution_host, true_solution);
    
    for (int i = 0; i < n_solve; ++i) {
        error_host(i) = true_solution_host(i) - x_host_for_error(i);
    }
    Kokkos::deep_copy(error_vec, error_host);
    
    double error_norm = norm(error_vec);
    double error_max_norm = max_norm(error_vec);
    
    // Для первой итерации разница с предыдущим решением равна начальному x (т.е. нули)
    double precision_norm = norm(x);
    double precision_max_norm = max_norm(x);
    
    iterationsDone = 0;
    
    std::cout << "\n--- Начало решения методом: " << solverName << " ---\n";
    std::cout << "Итерация: " << iterationsDone << "\n";
    std::cout << "Точность ||x(n)-x(n-1)||: евклидова = " << std::scientific << precision_norm 
              << ", max-норма = " << std::scientific << precision_max_norm << "\n";
    std::cout << "Невязка ||Ax-b||: евклидова = " << std::scientific << r_norm 
              << ", max-норма = " << std::scientific << r_max_norm << "\n";
    std::cout << "Ошибка ||u-x||: евклидова = " << std::scientific << error_norm 
              << ", max-норма = " << std::scientific << error_max_norm << "\n";

    double r_dot_r = dot(r, r);  // Сохраняем для формулы beta

    while (r_norm > eps && iterationsDone < maxIterations) {
        // Сохраняем текущее приближение для следующей итерации
        Kokkos::deep_copy(x_prev, x);
        Kokkos::deep_copy(r_old, r);
        Kokkos::deep_copy(h_old, h); // Сохраняем текущее h
        
        // Вычисляем шаг alpha = (r, r) / (A*h, h)
        double h_Ah_dot = dot(h, Ah);
        double alpha = r_dot_r / h_Ah_dot;
        
#ifdef deb
        debug_file << "alpha" << iterationsDone << " = " << alpha << std::endl;
#endif
        
        // Обновляем решение x = x + alpha*h используя KokkosBlas::axpy
        KokkosBlas::axpy(alpha, h, x);
        
#ifdef deb
        auto x_host_debug = Kokkos::create_mirror_view(x);
        Kokkos::deep_copy(x_host_debug, x);
        debug_file << "x" << iterationsDone+1 << " = [";
        for (int i = 0; i < n_solve; ++i) {
            debug_file << x_host_debug(i);
            if (i < n_solve-1) debug_file << ", ";
        }
        debug_file << "]" << std::endl;
#endif
        
        // Обновляем невязку r = r - alpha*A*h используя KokkosBlas::axpy
        KokkosBlas::axpy(-alpha, Ah, r);
        
#ifdef deb
        auto r_host_debug = Kokkos::create_mirror_view(r);
        Kokkos::deep_copy(r_host_debug, r);
        debug_file << "r" << iterationsDone+1 << " = [";
        for (int i = 0; i < n_solve; ++i) {
            debug_file << r_host_debug(i);
            if (i < n_solve-1) debug_file << ", ";
        }
        debug_file << "]" << std::endl;
#endif

        // Вычисляем новое значение (r, r)
        double r_dot_r_new = dot(r, r);
        
        // Вычисляем коэффициент beta по формуле Fletcher-Reeves
        double beta = r_dot_r_new / r_dot_r;
        r_dot_r = r_dot_r_new;  // Обновляем для следующей итерации
        
#ifdef deb
        debug_file << "beta" << iterationsDone << " = " << beta << std::endl;
#endif
        
        // Обновляем направление h = r + beta*h_old
        Kokkos::deep_copy(h, r);         // h = r
        KokkosBlas::axpy(beta, h_old, h); // h = h + beta*h_old = r + beta*h_old
        
#ifdef deb
        debug_file << "h" << iterationsDone+1 << " = [";
        auto h_host = Kokkos::create_mirror_view(h);
        Kokkos::deep_copy(h_host, h);
        for (int i = 0; i < n_solve; ++i) {
            debug_file << h_host(i);
            if (i < n_solve-1) debug_file << ", ";
        }
        debug_file << "]" << std::endl;
        debug_file << "-------------------------------------" << std::endl;
#endif
        
        // Предвычисляем A*h для следующей итерации
        Ah = multiply(a, h);
        
        iterationsDone++;
        
        // Вычисляем точность - разница между текущим и предыдущим решением
        KokkosVector precision_vec("precision_vec", n_solve);
        Kokkos::deep_copy(precision_vec, x);              // precision_vec = x
        KokkosBlas::axpy(-1.0, x_prev, precision_vec);    // precision_vec = x - x_prev
        
        precision_norm = norm(precision_vec);
        precision_max_norm = max_norm(precision_vec);
        
        // Обновляем норму невязки
        r_norm = norm(r);
        r_max_norm = max_norm(r);
        achievedPrecision = r_norm; // Используем норму невязки в качестве достигнутой точности
        
        // Вычисляем ошибку - разница между истинным и текущим решением
        Kokkos::deep_copy(error_vec, true_solution);     // error_vec = true_solution
        KokkosBlas::axpy(-1.0, x, error_vec);            // error_vec = true_solution - x
        
        error_norm = norm(error_vec);
        error_max_norm = max_norm(error_vec);
        
        // Вывод промежуточной информации
        if (iterationsDone % print_interval_iterations == 0 || iterationsDone == 1 || r_norm <= eps) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time).count();
            double iter_per_second = (elapsed_seconds > 0) ? (iterationsDone / elapsed_seconds) : 0;

            std::cout << "Итерация: " << iterationsDone << "\n";
            std::cout << "Точность ||x(n)-x(n-1)||: евклидова = " << std::scientific << precision_norm 
                      << ", max-норма = " << std::scientific << precision_max_norm << "\n";
            std::cout << "Невязка ||Ax-b||: евклидова = " << std::scientific << r_norm 
                      << ", max-норма = " << std::scientific << r_max_norm << "\n";
            std::cout << "Ошибка ||u-x||: евклидова = " << std::scientific << error_norm 
                      << ", max-норма = " << std::scientific << error_max_norm << "\n";
            std::cout << "Скорость: " << std::fixed << std::setprecision(2) << iter_per_second << " итер/сек\n" << std::endl;
            
            last_print_iteration = iterationsDone;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    double final_iter_per_second = (total_elapsed_seconds > 0 && iterationsDone > 0) ? (iterationsDone / total_elapsed_seconds) : 0;

    std::cout << "\n--- Завершение решения методом: " << solverName << " ---\n";
    std::cout << "Итого итераций: " << iterationsDone << "\n";
    std::cout << "Финальная точность ||x(n)-x(n-1)||: евклидова = " << std::scientific << precision_norm 
              << ", max-норма = " << std::scientific << precision_max_norm << "\n";
    std::cout << "Финальная невязка ||Ax-b||: евклидова = " << std::scientific << r_norm 
              << ", max-норма = " << std::scientific << r_max_norm << "\n";
    std::cout << "Финальная ошибка ||u-x||: евклидова = " << std::scientific << error_norm 
              << ", max-норма = " << std::scientific << error_max_norm << "\n";
    std::cout << "Общее время: " << std::fixed << std::setprecision(2) << total_elapsed_seconds << " с\n";
    std::cout << "Средняя скорость: " << std::fixed << std::setprecision(2) << final_iter_per_second << " итер/сек\n";

    converged = (r_norm <= eps);
    
    std::cout << solverName << " " << (converged ? "сошёлся" : "не сошёлся") 
              << " за " << iterationsDone << " итераций, достигнув точности " 
              << std::uppercase << std::scientific << achievedPrecision << std::endl;
    
#ifdef deb
    debug_file << "Итого сделано " << iterationsDone << " итераций" << std::endl;
    debug_file << "Достигнутая точность: " << achievedPrecision << std::endl;
    debug_file << "Итоговое решение x = [";
    auto final_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(final_x, x);
    for (int i = 0; i < n_solve; ++i) {
        debug_file << final_x(i);
        if (i < n_solve-1) debug_file << ", ";
    }
    debug_file << "]" << std::endl;
    debug_file.close();
#endif
    
    return x;
}