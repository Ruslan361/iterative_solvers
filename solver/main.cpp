#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>
#include "grid_system.h"
#include "solver.hpp"
#include "msg_solver.hpp"
#include <locale.h>

/**
 * Вычисляет невязку системы уравнений с использованием Kokkos.
 * Arg: a Матрица коэффициентов системы.
 * Arg: v Вектор решений.
 * Arg: b Вектор свободных членов.
 * Returns: Вектор невязки.
 */
KokkosVector compute_residual(const KokkosCrsMatrix& a, const KokkosVector& v, const KokkosVector& b)
{
    int n = b.extent(0);
    KokkosVector residual("residual", n);
    
    // Создаем вид на хосте для работы с данными
    auto b_host = Kokkos::create_mirror_view(b);
    Kokkos::deep_copy(b_host, b);
    
    // Для матрицы A*v используем Kokkos SpMV
    KokkosVector temp("temp", n);
    KokkosSparse::spmv("N", 1.0, a, v, 0.0, temp);
    
    // Вычисляем r = A*v - b
    auto temp_host = Kokkos::create_mirror_view(temp);
    auto residual_host = Kokkos::create_mirror_view(residual);
    
    Kokkos::deep_copy(temp_host, temp);
    
    for (int i = 0; i < n; ++i) {
        residual_host(i) = temp_host(i) - b_host(i);
    }
    
    Kokkos::deep_copy(residual, residual_host);
    return residual;
}

/**
 * Возвращает std::vector из KokkosVector для вывода и работы с CPU
 */
std::vector<double> kokkos_to_std_vector(const KokkosVector& kv) {
    auto host_view = Kokkos::create_mirror_view(kv);
    Kokkos::deep_copy(host_view, kv);
    std::vector<double> result(kv.extent(0));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = host_view(i);
    }
    return result;
}

/**
 * Вычисляет координату y по индексу сеточной точки.
 */
double y(int j, int m, double c_bound, double d_bound)
{
    double k = (d_bound - c_bound) / (m + 1);
    return c_bound + j * k;
}

/**
 * Вычисляет координату x по индексу сеточной точки.
 */
double x(int i, int n, double a_bound, double b_bound)
{
    double h = (b_bound - a_bound) / (n + 1);
    return a_bound + i * h;
}

/**
 * Истинное решение функции u(x, y).
 */
double u(double x_val, double y_val)
{
    return exp(pow(x_val, 2) - pow(y_val, 2));
}

/**
 * Вычисляет ошибку решения.
 */
std::vector<double> compute_error(const KokkosVector& v, int n_internal, int m_internal, 
                                 double a_bound, double b_bound, double c_bound, double d_bound)
{
    std::vector<double> error(v.extent(0), 0.0);
    auto v_host = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(v_host, v);
    
    double h = (b_bound - a_bound) / (n_internal + 1);
    double k = (d_bound - c_bound) / (m_internal + 1);
    
    for (size_t i = 0; i < error.size(); ++i) {   
        int row = i / n_internal + 1;
        int col = i % n_internal + 1;
        error[i] = fabs(v_host(i) - u(x(col, n_internal, a_bound, b_bound), 
                                     y(row, m_internal, c_bound, d_bound)));
    }
    return error;
}

/**
 *  Выводит вектор невязки и её нормы.
 */
void print_residual(const KokkosVector& kokkos_residual)
{
    std::vector<double> residual = kokkos_to_std_vector(kokkos_residual);
    
    std::cout << "\nНевязка (r):\n";
    // Выводим только первые 5 элементов невязки
    int max_output = std::min(5, static_cast<int>(residual.size()));
    for (int i = 0; i < max_output; ++i) {
        std::cout << std::setw(15) << std::uppercase << std::scientific << residual[i] << "\n";
    }
    if (residual.size() > max_output) {
        std::cout << "... (показаны только первые " << max_output << " элементов из " << residual.size() << ")\n";
    }
    
    double maxResidual = *std::max_element(residual.begin(), residual.end(), 
                                          [](double a, double b) {return fabs(a) < fabs(b);});
    std::cout << "Максимальная невязка: " << std::uppercase << std::scientific << maxResidual << "\n";

    double evk_norm = 0;
    for (double r : residual) {
        evk_norm += r * r;
    }
    evk_norm = sqrt(evk_norm);
    std::cout << "Евклидова норма невязки: " << std::uppercase << std::scientific << evk_norm << "\n";
}

/**
 * Выводит вектор ошибки и её нормы.
 */
void print_error(const std::vector<double>& error)
{
    std::cout << "\nВектор погрешности:\n";
    // Выводим только первые 5 элементов вектора погрешности
    int max_output = std::min(5, static_cast<int>(error.size()));
    for (int i = 0; i < max_output; ++i) {
        std::cout << std::fixed << std::setprecision(6) << error[i] << "\n";
    }
    if (error.size() > max_output) {
        std::cout << "... (показаны только первые " << max_output << " элементов из " << error.size() << ")\n";
    }
    
    double maxError = *std::max_element(error.begin(), error.end(), 
                                       [](double a, double b) {return fabs(a) < fabs(b);});
    std::cout << "Максимальная погрешность: " << std::uppercase << std::scientific << maxError << "\n";
}

/**
 * Сохраняет результаты вычислений в файл.
 */
void save_results_to_file(const std::string& filename,
                         const std::vector<double>& solution,
                         const std::vector<double>& residual,
                         const std::vector<double>& error,
                         const std::vector<double>& u_true_std, // Added u_true_std
                         int n_internal, int m_internal,
                         double a_bound, double b_bound,
                         double c_bound, double d_bound,
                         const std::string& solver_name,
                         int iterations, double precision)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << " для записи.\n";
        return;
    }
    
    // Вычисляем шаги сетки
    double x_step = (b_bound - a_bound) / n_internal;
    double y_step = (d_bound - c_bound) / m_internal;
    
    file << "Результаты решения системы уравнений\n";
    file << "------------------------------------\n\n";
    
    // Параметры сетки
    file << "Параметры сетки:\n";
    file << "  Размерность: " << n_internal << " x " << m_internal << "\n";
    file << "  Область: [" << a_bound << ", " << b_bound << "] x [" << c_bound << ", " << d_bound << "]\n";
    file << "  Шаг по x: " << std::fixed << std::setprecision(6) << x_step << "\n";
    file << "  Шаг по y: " << std::fixed << std::setprecision(6) << y_step << "\n\n";
    
    // Информация о методе
    file << "Метод решения: " << solver_name << "\n";
    file << "  Итераций: " << iterations << "\n";
    file << "  Точность на последней итерации: " << std::uppercase << std::scientific << precision << "\n";
    file << "    (максимальная разница между решениями на последних двух итерациях, max-норма)\n\n";
    
    // Невязка
    double max_residual = *std::max_element(residual.begin(), residual.end(), 
                                          [](double a, double b) {return fabs(a) < fabs(b);});
    double residual_norm = 0;
    for (double r : residual) {
        residual_norm += r * r;
    }
    residual_norm = sqrt(residual_norm);
    
    file << "Невязка (r = A*x - b):\n";
    file << "  Евклидова норма невязки: " << std::uppercase << std::scientific << residual_norm << "\n";
    file << "  Max-норма невязки: " << std::uppercase << std::scientific << max_residual << "\n";
    file << "    (Евклидова норма - корень из суммы квадратов всех элементов вектора невязки)\n";
    file << "    (Max-норма - максимальный по модулю элемент вектора невязки)\n\n";
    
    // Погрешность
    double max_error = *std::max_element(error.begin(), error.end(), 
                                       [](double a, double b) {return fabs(a) < fabs(b);});
    double error_norm = 0;
    for (double e : error) {
        error_norm += e * e;
    }
    error_norm = sqrt(error_norm);
    
    file << "Погрешность (разница между истинным и численным решением):\n";
    file << "  Евклидова норма погрешности: " << std::uppercase << std::scientific << error_norm << "\n";
    file << "  Max-норма погрешности: " << std::uppercase << std::scientific << max_error << "\n";
    file << "    (Евклидова норма - корень из суммы квадратов элементов вектора погрешности)\n";
    file << "    (Max-норма - максимальный по модулю элемент вектора погрешности)\n\n";
    
    // Истинное решение u_true
    file << "Истинное решение (вектор u_true):\n";
    // Ограничиваем вывод для больших размерностей сетки
    int max_output_u_true = std::min(100, static_cast<int>(u_true_std.size()));
    for (int i = 0; i < max_output_u_true; ++i) {
        int row = i / n_internal + 1;
        int col = i % n_internal + 1;
        double xi = x(col, n_internal, a_bound, b_bound);
        double yi = y(row, m_internal, c_bound, d_bound);
        
        file << "u_true[" << i << "] (x=" << xi << ", y=" << yi << ") = " 
             << std::fixed << std::setprecision(6) << u_true_std[i] << "\n";
    }
    if (u_true_std.size() > max_output_u_true) {
        file << "... (показаны только первые " << max_output_u_true << " элементов из " << u_true_std.size() << ")\n";
    }
    file << "\n";

    // Решение
    file << "Решение (вектор v):\n";
    // Ограничиваем вывод для больших размерностей сетки
    int max_output = std::min(100, static_cast<int>(solution.size()));
    for (int i = 0; i < max_output; ++i) {
        int row = i / n_internal + 1;
        int col = i % n_internal + 1;
        double xi = x(col, n_internal, a_bound, b_bound);
        double yi = y(row, m_internal, c_bound, d_bound);
        
        file << "v[" << i << "] (x=" << xi << ", y=" << yi << ") = " 
             << std::fixed << std::setprecision(6) << solution[i] << "\n";
    }
    
    if (solution.size() > max_output) {
        file << "... (показаны только первые " << max_output << " элементов из " << solution.size() << ")\n";
    }
    
    file.close();
    std::cout << "Результаты сохранены в файл: " << filename << std::endl;
}

/**
 * Сохраняет матрицу системы и вектор правой части в файл.
 */
void save_matrix_and_rhs_to_file(const std::string& filename,
                               const KokkosCrsMatrix& A,
                               const KokkosVector& b,
                               int n_internal, int m_internal)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << " для записи.\n";
        return;
    }
    
    int n_rows = A.numRows();
    int n_cols = A.numCols();
    
    // Создаем вид на хосте для работы с данными
    auto row_map = Kokkos::create_mirror_view(A.graph.row_map);
    auto entries = Kokkos::create_mirror_view(A.graph.entries);
    auto values = Kokkos::create_mirror_view(A.values);
    
    Kokkos::deep_copy(row_map, A.graph.row_map);
    Kokkos::deep_copy(entries, A.graph.entries);
    Kokkos::deep_copy(values, A.values);
    
    auto b_host = Kokkos::create_mirror_view(b);
    Kokkos::deep_copy(b_host, b);
    
    // Заголовок файла
    file << "Матрица системы и вектор правой части\n";
    file << "------------------------------------\n\n";
    
    // Информация о матрице
    file << "Информация о матрице:\n";
    file << "  Размер: " << n_rows << " x " << n_cols << "\n";
    file << "  Ненулевых элементов: " << A.nnz() << "\n";
    file << "  Плотность: " << (double)A.nnz() / (n_rows * n_cols) * 100.0 << "%\n\n";
    
    // Формат CSR матрицы
    file << "Матрица в CSR формате:\n";
    file << "Row Map (" << (n_rows + 1) << " элементов):\n";
    for (int i = 0; i <= n_rows; ++i) {
        file << row_map(i) << " ";
        if ((i+1) % 10 == 0) file << "\n";
    }
    file << "\n\n";
    
    file << "Entries и Values (" << A.nnz() << " элементов):\n";
    file << "Индекс | Столбец | Значение\n";
    file << "-------|---------|--------\n";
    for (int i = 0; i < A.nnz(); ++i) {
        file << std::setw(6) << i << " | " 
             << std::setw(7) << entries(i) << " | " 
             << std::fixed << std::setprecision(8) << values(i) << "\n";
        
        // Ограничиваем вывод для очень больших матриц
        if (i >= 1000) {
            file << "... (выведено только первые 1000 из " << A.nnz() << " элементов)\n";
            break;
        }
    }
    file << "\n";
    
    // Матрица в плотном формате (ограниченная часть)
    file << "Матрица в плотном формате (ограниченная часть):\n";
    int max_dense_rows = std::min(50, n_rows);
    int max_dense_cols = std::min(50, n_cols);
    
    // Заголовок столбцов
    file << "    ";
    for (int j = 0; j < max_dense_cols; ++j) {
        file << std::setw(10) << j << " ";
    }
    file << "\n    ";
    for (int j = 0; j < max_dense_cols; ++j) {
        file << "----------- ";
    }
    file << "\n";
    
    // Строки матрицы
    for (int i = 0; i < max_dense_rows; ++i) {
        file << std::setw(3) << i << " | ";
        
        // Создаем временный вектор для полной строки
        std::vector<double> row_values(n_cols, 0.0);
        
        // Заполняем ненулевые элементы строки
        for (int j = row_map(i); j < row_map(i + 1); ++j) {
            int col = entries(j);
            if (col < n_cols) {
                row_values[col] = values(j);
            }
        }
        
        // Выводим только первые max_dense_cols элементов
        for (int j = 0; j < max_dense_cols; ++j) {
            if (std::abs(row_values[j]) < 1e-10) {
                file << std::setw(10) << "0" << " ";
            } else {
                file << std::setw(10) << std::fixed << std::setprecision(4) << row_values[j] << " ";
            }
        }
        file << "\n";
    }
    
    if (n_rows > max_dense_rows || n_cols > max_dense_cols) {
        file << "\n... (выведена только часть матрицы " << max_dense_rows << "x" << max_dense_cols 
             << " из " << n_rows << "x" << n_cols << ")\n";
    }
    file << "\n\n";
    
    // Вектор правой части
    file << "Вектор правой части (b):\n";
    file << "Размер: " << b.extent(0) << " элементов\n\n";
    
    file << "Индекс |    Значение\n";
    file << "-------|-------------\n";
    
    int max_elements = std::min(1000, static_cast<int>(b.extent(0)));
    for (int i = 0; i < max_elements; ++i) {
        file << std::setw(6) << i << " | " 
             << std::setw(12) << std::fixed << std::setprecision(8) << b_host(i) << "\n";
    }
    
    if (b.extent(0) > max_elements) {
        file << "\n... (выведено только " << max_elements << " из " << b.extent(0) << " элементов)\n";
    }
    
    // Вычисляем некоторые статистики
    double min_val = b_host(0), max_val = b_host(0), sum = 0.0;
    for (int i = 0; i < b.extent(0); ++i) {
        min_val = std::min(min_val, b_host(i));
        max_val = std::max(max_val, b_host(i));
        sum += b_host(i);
    }
    
    file << "\nСтатистика вектора правой части:\n";
    file << "  Минимальное значение: " << std::fixed << std::setprecision(8) << min_val << "\n";
    file << "  Максимальное значение: " << std::fixed << std::setprecision(8) << max_val << "\n";
    file << "  Среднее значение: " << std::fixed << std::setprecision(8) << sum / b.extent(0) << "\n";
    file << "  Сумма всех элементов: " << std::fixed << std::setprecision(8) << sum << "\n";
    
    file.close();
    std::cout << "Матрица и вектор правой части сохранены в файл: " << filename << std::endl;
}

void print_separator() {
    std::cout << "\n_________________________________________________\n\n";
}

void print_summary(const std::string& solver_name, int iterations, double precision, 
                   double residual_norm, int n_internal, int m_internal, 
                   double a_bound, double b_bound, double c_bound, double d_bound,
                   const std::vector<double>& error)
{
    // Вычисляем шаги сетки
    double x_step = (b_bound - a_bound) / n_internal;
    double y_step = (d_bound - c_bound) / m_internal;
    
    // Вычисляем нормы погрешности
    double max_error = *std::max_element(error.begin(), error.end(), 
                                      [](double a, double b) {return fabs(a) < fabs(b);});
    double error_norm = 0;
    for (double e : error) {
        error_norm += e * e;
    }
    error_norm = sqrt(error_norm);
    
    std::cout << "\n******* ИТОГИ РЕШЕНИЯ *******\n";
    std::cout << "Параметры сетки:\n";
    std::cout << "  Размерность: " << n_internal << " x " << m_internal << "\n";
    std::cout << "  Область: [" << a_bound << ", " << b_bound << "] x [" 
              << c_bound << ", " << d_bound << "]\n";
    std::cout << "  Шаг по x: " << std::fixed << std::setprecision(6) << x_step << "\n";
    std::cout << "  Шаг по y: " << std::fixed << std::setprecision(6) << y_step << "\n\n";
    
    std::cout << "Метод решения: " << solver_name << "\n";
    std::cout << "  Итераций: " << iterations << "\n";
    std::cout << "  Точность на последней итерации: " << std::uppercase << std::scientific << precision << " (max-норма)\n";
    std::cout << "    (максимальная разница между решениями на последних двух итерациях)\n\n";
    
    std::cout << "Невязка (r = A*x - b):\n";
    std::cout << "  Евклидова норма невязки: " << std::uppercase << std::scientific << residual_norm << "\n";
    std::cout << "    (корень из суммы квадратов всех элементов вектора невязки)\n\n";
    
    std::cout << "Погрешность (разница между истинным и численным решением):\n";
    std::cout << "  Максимальная погрешность в C-норме: " << std::uppercase << std::scientific << max_error << "\n";
    std::cout << "  Евклидова норма погрешности: " << std::uppercase << std::scientific << error_norm << "\n";
    std::cout << "******************************\n";
}

/**
 * Выводит матрицу в человекочитаемом виде.
 * Если матрица большая, выводит только часть.
 * 
 * @param A Матрица в формате KokkosCrsMatrix
 * @param max_rows Максимальное количество строк для вывода
 * @param max_cols Максимальное количество столбцов для вывода 
 */
void print_matrix(const KokkosCrsMatrix& A, int max_rows = 10, int max_cols = 10)
{
    int n_rows = A.numRows();
    int n_cols = A.numCols();
    
    // Ограничиваем количество выводимых строк и столбцов
    max_rows = std::min(max_rows, n_rows);
    max_cols = std::min(max_cols, n_cols);
    
    std::cout << "\nМатрица системы (формат разреженной матрицы):\n";
    std::cout << "Размер: " << n_rows << " x " << n_cols << "\n";
    std::cout << "Ненулевых элементов: " << A.nnz() << "\n";
    std::cout << "Плотность: " << (double)A.nnz() / (n_rows * n_cols) * 100.0 << "%\n\n";
    
    // Создаем вид на хосте для работы с данными
    auto row_map = Kokkos::create_mirror_view(A.graph.row_map);
    auto entries = Kokkos::create_mirror_view(A.graph.entries);
    auto values = Kokkos::create_mirror_view(A.values);
    
    Kokkos::deep_copy(row_map, A.graph.row_map);
    Kokkos::deep_copy(entries, A.graph.entries);
    Kokkos::deep_copy(values, A.values);
    
    // Выводим индексы столбцов
    std::cout << "     ";
    for (int j = 0; j < max_cols; ++j) {
        std::cout << std::setw(10) << j << " ";
    }
    std::cout << "\n     ";
    for (int j = 0; j < max_cols; ++j) {
        std::cout << "----------- ";
    }
    std::cout << "\n";
    
    // Выводим строки матрицы
    for (int i = 0; i < max_rows; ++i) {
        std::cout << std::setw(3) << i << " | ";
        
        // Создаем временный вектор для полной строки
        std::vector<double> row_values(n_cols, 0.0);
        
        // Заполняем ненулевые элементы строки
        for (int j = row_map(i); j < row_map(i + 1); ++j) {
            int col = entries(j);
            if (col < n_cols) {
                row_values[col] = values(j);
            }
        }
        
        // Выводим только первые max_cols элементов
        for (int j = 0; j < max_cols; ++j) {
            if (std::abs(row_values[j]) < 1e-10) {
                std::cout << std::setw(10) << "0" << " ";
            } else {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << row_values[j] << " ";
            }
        }
        std::cout << "\n";
    }
    
    // Если матрица большая, добавляем многоточие
    if (n_rows > max_rows || n_cols > max_cols) {
        std::cout << "\n... (выведена только часть матрицы " << max_rows << "x" << max_cols << " из " 
                  << n_rows << "x" << n_cols << ")\n";
    }
    
    std::cout << "\nСтруктура разреженной матрицы (CSR формат):\n";
    std::cout << "Row Map (первые " << std::min(10, n_rows + 1) << " элементов): ";
    for (int i = 0; i < std::min(10, n_rows + 1); ++i) {
        std::cout << row_map(i) << " ";
    }
    if (n_rows + 1 > 10) std::cout << "...";
    std::cout << "\n";
    
    std::cout << "Entries & Values (первые " << std::min(20, (int)A.nnz()) << " элементов):\n";
    for (int i = 0; i < std::min(20, (int)A.nnz()); ++i) {
        std::cout << "(" << entries(i) << ", " << std::fixed << std::setprecision(4) << values(i) << ") ";
        if ((i+1) % 5 == 0) std::cout << "\n";
    }
    if (A.nnz() > 20) std::cout << "...";
    std::cout << "\n";
}

/**
 * Выводит вектор правой части системы в человекочитаемом виде.
 * 
 * @param b Вектор правой части
 * @param max_elements Максимальное количество элементов для вывода
 */
void print_rhs(const KokkosVector& b, int max_elements = 20)
{
    int n = b.extent(0);
    max_elements = std::min(max_elements, n);
    
    // Создаем вид на хосте для работы с данными
    auto b_host = Kokkos::create_mirror_view(b);
    Kokkos::deep_copy(b_host, b);
    
    std::cout << "\nВектор правой части (b):\n";
    std::cout << "Размер: " << n << " элементов\n\n";
    
    std::cout << "Индекс |    Значение\n";
    std::cout << "-------|-------------\n";
    
    for (int i = 0; i < max_elements; ++i) {
        std::cout << std::setw(6) << i << " | " 
                  << std::setw(12) << std::fixed << std::setprecision(6) << b_host(i) << "\n";
    }
    
    if (n > max_elements) {
        std::cout << "\n... (выведено только " << max_elements << " из " << n << " элементов)\n";
    }
    
    // Вычисляем некоторые статистики
    double min_val = b_host(0), max_val = b_host(0), sum = 0.0;
    for (int i = 0; i < n; ++i) {
        min_val = std::min(min_val, b_host(i));
        max_val = std::max(max_val, b_host(i));
        sum += b_host(i);
    }
    
    std::cout << "\nСтатистика вектора правой части:\n";
    std::cout << "  Минимальное значение: " << std::fixed << std::setprecision(6) << min_val << "\n";
    std::cout << "  Максимальное значение: " << std::fixed << std::setprecision(6) << max_val << "\n";
    std::cout << "  Среднее значение: " << std::fixed << std::setprecision(6) << sum / n << "\n";
    std::cout << "  Сумма всех элементов: " << std::fixed << std::setprecision(6) << sum << "\n";
}

int main(int argc, char* argv[]) {
    // Инициализация Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Define eps_val and max_iter_val here
        double eps_val = 1e-9; // Default precision
        int max_iter_val = 2; // Default max iterations
        
        int m_val = 20; 
        int n_val = 20; 
        
        setlocale(LC_ALL, "RU");

        // Ввод размерности сетки
        int n_internal, m_internal;
        std::cout << "Введите размерность сетки по оси X: ";
        std::cin >> n_internal;
        std::cout << "Введите размерность сетки по оси Y: ";
        std::cin >> m_internal;
        
        // Параметры области
        double a_bound = 1.0; // Левая граница по x
        double b_bound = 2.0; // Правая граница по x
        double c_bound = 1.0; // Нижняя граница по y
        double d_bound = 2.0; // Верхняя граница по y
        
        std::cout << "\nСоздание сетки размером " << n_internal << "x" << m_internal 
                  << " для области [" << a_bound << "," << b_bound << "] x [" 
                  << c_bound << "," << d_bound << "]" << std::endl;
        
        GridSystem grid(m_internal, n_internal, a_bound, b_bound, c_bound, d_bound);
        std::cout << grid << std::endl;

        KokkosVector u_true = grid.get_true_solution_vector(); // Get true solution vector

        // MSG Solver
        // Corrected constructor call: removed "MSG Solver" string, using eps_val and max_iter_val
        MSGSolver msg_solver(grid.get_matrix(), grid.get_rhs(), eps_val, max_iter_val);
        KokkosVector solution_msg = msg_solver.solve(u_true); // Pass true solution to solver


        // Вычисляем невязку
        // Using grid.get_matrix(), solution_msg, and grid.get_rhs()
        KokkosVector kokkos_residual = compute_residual(grid.get_matrix(), solution_msg, grid.get_rhs());
        std::vector<double> residual = kokkos_to_std_vector(kokkos_residual);
        print_residual(kokkos_residual);

        // Вычисляем и выводим ошибку
        // Using solution_msg
        std::vector<double> error = compute_error(solution_msg, n_internal, m_internal, 
                                                 a_bound, b_bound, c_bound, d_bound);
        print_error(error);

        // Преобразуем решение в std::vector для сохранения и вывода
        // Using solution_msg
        std::vector<double> solution_std = kokkos_to_std_vector(solution_msg);
        
        // Рассчитываем метрики для итогового отчета
        double max_residual = *std::max_element(residual.begin(), residual.end(), 
                                               [](double a, double b) {return fabs(a) < fabs(b);});
        double residual_norm = 0;
        for (double r : residual) {
            residual_norm += r * r;
        }
        residual_norm = sqrt(residual_norm);
        
        double max_error = *std::max_element(error.begin(), error.end(), 
                                           [](double a, double b) {return fabs(a) < fabs(b);});
        
        // Выводим краткую сводку результатов
        print_summary(msg_solver.getName(), msg_solver.getIterations(), msg_solver.getPrecision(),
                      residual_norm, n_internal, m_internal, a_bound, b_bound, c_bound, d_bound, error);
        
        // Спрашиваем, нужно ли сохранить результаты в файл
        std::string save_choice;
        std::cout << "\nСохранить результаты в файл? (да/нет): ";
        std::cin >> save_choice;
        
        if (save_choice == "да" || save_choice == "Да" || save_choice == "ДА" || 
            save_choice == "y" || save_choice == "Y" || save_choice == "yes") {
            
            std::string filename;
            std::cout << "Введите имя файла: ";
            std::cin >> filename;
            
            std::vector<double> u_true_std = kokkos_to_std_vector(u_true); // Convert u_true to std::vector

            save_results_to_file(filename, solution_std, residual, error, u_true_std, // Pass u_true_std
                                n_internal, m_internal, a_bound, b_bound, c_bound, d_bound,
                                msg_solver.getName(), msg_solver.getIterations(), msg_solver.getPrecision());
        }
        
        // Спрашиваем, нужно ли сохранить матрицу и правую часть в файл
        std::string save_matrix_choice;
        std::cout << "\nСохранить матрицу и вектор правой части в файл? (да/нет): ";
        std::cin >> save_matrix_choice;
        
        if (save_matrix_choice == "да" || save_matrix_choice == "Да" || save_matrix_choice == "ДА" || 
            save_matrix_choice == "y" || save_matrix_choice == "Y" || save_matrix_choice == "yes") {
            
            std::string filename;
            std::cout << "Введите имя файла для сохранения матрицы и вектора правой части: ";
            std::cin >> filename;
            
            // Сохраняем матрицу и вектор правой части
            save_matrix_and_rhs_to_file(filename, grid.get_matrix(), grid.get_rhs(), n_internal, m_internal);
        }
    }
    // catch (std::exception& e) {
    //     std::cerr << "Ошибка: " << e.what() << std::endl;
    // }
    
    // Финализация Kokkos
    Kokkos::finalize();
    
    return 0;
}