#include "dirichlet_solver.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// Конструктор DirichletSolver
DirichletSolver::DirichletSolver(int n, int m, double a, double b, double c, double d)
    : n_internal(n), m_internal(m),
      a_bound(a), b_bound(b), c_bound(c), d_bound(d),
      eps_precision(1e-6), eps_residual(1e-6), eps_exact_error(1e-6), max_iterations(10000),
      use_precision_stopping(true), use_residual_stopping(true), 
      use_error_stopping(false), use_max_iterations_stopping(true) {
    
    // Проверка инициализации Kokkos
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    
    // Инициализация сетки
    grid = std::make_unique<GridSystem>(m_internal, n_internal, a_bound, b_bound, c_bound, d_bound);
}

// Деструктор
DirichletSolver::~DirichletSolver() {
    // Освобождаем ресурсы
    grid.reset();
    solver.reset();
}

// Установка параметров сетки
void DirichletSolver::setGridParameters(int n, int m, double a, double b, double c, double d) {
    n_internal = n;
    m_internal = m;
    a_bound = a;
    b_bound = b;
    c_bound = c;
    d_bound = d;
    
    // Пересоздаем объект сетки с новыми параметрами
    grid = std::make_unique<GridSystem>(m_internal, n_internal, a_bound, b_bound, c_bound, d_bound);
}

// Установка параметров солвера
void DirichletSolver::setSolverParameters(double eps_p, double eps_r, double eps_e, int max_iter) {
    eps_precision = eps_p;
    eps_residual = eps_r;
    eps_exact_error = eps_e;
    max_iterations = max_iter;
}

// Установка колбэка для отслеживания итераций
void DirichletSolver::setIterationCallback(std::function<void(int, double, double, double)> callback) {
    iteration_callback = callback;
}

// Решение задачи Дирихле
SolverResults DirichletSolver::solve() {
    if (!grid) {
        throw std::runtime_error("Сетка не инициализирована");
    }
    
    // Создаем солвер
    solver = std::make_unique<MSGSolver>(grid->get_matrix(), grid->get_rhs(), 
                                        std::min({eps_precision, eps_residual, eps_exact_error}), max_iterations);
    
    // Устанавливаем точности для критериев останова на основе флагов
    if (use_precision_stopping) {
        solver->setPrecisionEps(eps_precision);
    } else {
        solver->setPrecisionEps(-1.0); // Отключаем этот критерий
    }
    
    if (use_residual_stopping) {
        solver->setResidualEps(eps_residual);
    } else {
        solver->setResidualEps(-1.0); // Отключаем этот критерий
    }
    
    if (use_error_stopping) {
        solver->setExactErrorEps(eps_exact_error);
    } else {
        solver->setExactErrorEps(-1.0); // Отключаем этот критерий
    }
    
    // Устанавливаем колбэк для отслеживания итераций, если он был задан
    if (iteration_callback) {
        solver->setIterationCallback(iteration_callback);
    }
    
    // Получаем истинное решение для сравнения
    true_solution = grid->get_true_solution_vector();
    
    // Решаем СЛАУ
    solution = solver->solve(true_solution);
    
    // Формируем результаты
    SolverResults results;
    results.solution = kokkosToStdVector(solution);
    results.true_solution = kokkosToStdVector(true_solution);
    
    // Вычисляем невязку
    KokkosVector residual = computeResidual(grid->get_matrix(), solution, grid->get_rhs());
    results.residual = kokkosToStdVector(residual);
    
    // Вычисляем ошибку (разница между точным и численным решением)
    results.error = computeError(solution);
    
    // Получаем векторы координат из GridSystem напрямую
    results.x_coords = grid->get_x_coords();
    results.y_coords = grid->get_y_coords();
    
    // Добавляем информацию о сходимости
    results.iterations = solver->getIterations();
    results.converged = solver->hasConverged();
    results.stop_reason = solver->getStopReasonText();
    
    // Добавляем нормы ошибок
    results.residual_norm = solver->getFinalResidualNorm();
    results.error_norm = solver->getFinalErrorNorm();
    
    // Вызываем обратный вызов завершения, если он установлен
    if (completion_callback) {
        completion_callback(results);
    }
    
    return results;
}

// Конвертация из Kokkos вектора в std::vector
std::vector<double> DirichletSolver::kokkosToStdVector(const KokkosVector& kv) const {
    std::vector<double> result(kv.extent(0));
    auto kv_host = Kokkos::create_mirror_view(kv);
    Kokkos::deep_copy(kv_host, kv);
    
    for (size_t i = 0; i < kv.extent(0); ++i) {
        result[i] = kv_host(i);
    }
    
    return result;
}

// Вычисление невязки Ax-b
KokkosVector DirichletSolver::computeResidual(const KokkosCrsMatrix& A, const KokkosVector& x, const KokkosVector& b) const {
    int n = b.extent(0);
    KokkosVector Ax("Ax", n);
    KokkosVector residual("residual", n);
    
    // Вычисляем Ax
    KokkosSparse::spmv("N", 1.0, A, x, 0.0, Ax);
    
    // Вычисляем r = Ax - b
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
        residual(i) = Ax(i) - b(i);
    });
    
    return residual;
}

// Вычисление разницы между численным решением и точным (ошибка)
std::vector<double> DirichletSolver::computeError(const KokkosVector& v) const {
    std::vector<double> error;
    
    if (true_solution.extent(0) > 0) {
        int n = v.extent(0);
        KokkosVector error_kokkos("error", n);
        
        // Вычисляем error = v - true_solution
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
            error_kokkos(i) = v(i) - true_solution(i);
        });
        
        error = kokkosToStdVector(error_kokkos);
    }
    
    return error;
}

// Получение вектора решения
std::vector<double> DirichletSolver::getSolution() const {
    return kokkosToStdVector(solution);
}

// Получение вектора точного решения
std::vector<double> DirichletSolver::getTrueSolution() const {
    return kokkosToStdVector(true_solution);
}

// Преобразование 1D вектора решения в 2D матрицу
std::vector<std::vector<double>> DirichletSolver::solutionToMatrix() const {
    std::vector<std::vector<double>> result(m_internal, std::vector<double>(n_internal));
    
    auto sol = kokkosToStdVector(solution);
    
    for (int j = 0; j < m_internal; ++j) {
        for (int i = 0; i < n_internal; ++i) {
            result[j][i] = sol[j * n_internal + i];
        }
    }
    
    return result;
}

// Генерация отчета о решении
std::string DirichletSolver::generateReport() const {
    if (!solver) {
        return "Решение еще не выполнено";
    }
    
    return solver->generateReport(n_internal, m_internal, a_bound, b_bound, c_bound, d_bound);
}

// Сохранение результатов в файл
bool DirichletSolver::saveResultsToFile(const std::string& filename) const {
    if (!solver) {
        return false;
    }
    
    SolverResults results;
    results.solution = kokkosToStdVector(solution);
    results.true_solution = kokkosToStdVector(true_solution);
    
    // Вычисляем невязку
    KokkosVector residual = computeResidual(grid->get_matrix(), solution, grid->get_rhs());
    results.residual = kokkosToStdVector(residual);
    
    // Вычисляем ошибку
    results.error = computeError(solution);
    
    // Добавляем информацию о сходимости
    results.iterations = solver->getIterations();
    results.converged = solver->hasConverged();
    results.stop_reason = solver->getStopReasonText();
    results.residual_norm = solver->getFinalResidualNorm();
    results.error_norm = solver->getFinalErrorNorm();
    
    return ResultsIO::saveResults(filename, results, n_internal, m_internal, 
                               a_bound, b_bound, c_bound, d_bound, solver->getName());
}

// Сохранение матрицы и вектора правой части в файл
bool DirichletSolver::saveMatrixAndRhsToFile(const std::string& filename) const {
    if (!grid) {
        return false;
    }
    
    return ResultsIO::saveMatrixAndRhs(filename, grid->get_matrix(), grid->get_rhs(), n_internal, m_internal);
}

// Реализация методов класса ResultsIO

bool ResultsIO::saveResults(const std::string& filename, const SolverResults& results,
                         int n, int m, double a, double b, double c, double d,
                         const std::string& solver_name) {
    std::ofstream file(filename);
    if (!file) {
        return false;
    }
    
    // Сохранение параметров задачи
    file << "PARAMETERS\n";
    file << n << " " << m << "\n";
    file << a << " " << b << " " << c << " " << d << "\n";
    file << solver_name << "\n";
    
    // Сохранение информации о сходимости
    file << "CONVERGENCE\n";
    file << results.iterations << "\n";
    file << (results.converged ? "1" : "0") << "\n";
    file << results.stop_reason << "\n";
    file << std::scientific << results.residual_norm << " " << results.error_norm << "\n";
    
    // Сохранение решения
    file << "SOLUTION\n";
    for (const auto& val : results.solution) {
        file << std::scientific << val << "\n";
    }
    
    // Сохранение точного решения
    file << "TRUE_SOLUTION\n";
    for (const auto& val : results.true_solution) {
        file << std::scientific << val << "\n";
    }
    
    // Сохранение невязки
    file << "RESIDUAL\n";
    for (const auto& val : results.residual) {
        file << std::scientific << val << "\n";
    }
    
    // Сохранение ошибки
    file << "ERROR\n";
    for (const auto& val : results.error) {
        file << std::scientific << val << "\n";
    }
    
    // Сохранение координат X
    file << "X_COORDS\n";
    for (const auto& val : results.x_coords) {
        file << std::scientific << val << "\n";
    }
    
    // Сохранение координат Y
    file << "Y_COORDS\n";
    for (const auto& val : results.y_coords) {
        file << std::scientific << val << "\n";
    }
    
    return true;
}

bool ResultsIO::loadResults(const std::string& filename, SolverResults& results,
                         int& n, int& m, double& a, double& b, double& c, double& d,
                         std::string& solver_name) {
    std::ifstream file(filename);
    if (!file) {
        return false;
    }
    
    std::string line;
    
    // Чтение параметров задачи
    if (!std::getline(file, line) || line != "PARAMETERS") {
        return false;
    }
    
    file >> n >> m;
    file >> a >> b >> c >> d;
    file.ignore();
    std::getline(file, solver_name);
    
    // Чтение информации о сходимости
    if (!std::getline(file, line) || line != "CONVERGENCE") {
        return false;
    }
    
    file >> results.iterations;
    int converged;
    file >> converged;
    results.converged = (converged == 1);
    file.ignore();
    std::getline(file, results.stop_reason);
    file >> results.residual_norm >> results.error_norm;
    
    // Чтение решения
    if (!std::getline(file, line) || !std::getline(file, line) || line != "SOLUTION") {
        return false;
    }
    
    results.solution.resize(n * m);
    for (int i = 0; i < n * m; ++i) {
        file >> results.solution[i];
    }
    
    // Чтение точного решения
    if (!std::getline(file, line) || !std::getline(file, line) || line != "TRUE_SOLUTION") {
        return false;
    }
    
    results.true_solution.resize(n * m);
    for (int i = 0; i < n * m; ++i) {
        file >> results.true_solution[i];
    }
    
    // Чтение невязки
    if (!std::getline(file, line) || !std::getline(file, line) || line != "RESIDUAL") {
        return false;
    }
    
    results.residual.resize(n * m);
    for (int i = 0; i < n * m; ++i) {
        file >> results.residual[i];
    }
    
    // Чтение ошибки
    if (!std::getline(file, line) || !std::getline(file, line) || line != "ERROR") {
        return false;
    }
    
    results.error.resize(n * m);
    for (int i = 0; i < n * m; ++i) {
        file >> results.error[i];
    }
    
    // Проверяем наличие координат X в файле
    if (std::getline(file, line) && std::getline(file, line) && line == "X_COORDS") {
        results.x_coords.resize(n * m);
        for (int i = 0; i < n * m; ++i) {
            file >> results.x_coords[i];
        }
    }
    
    // Проверяем наличие координат Y в файле
    if (std::getline(file, line) && std::getline(file, line) && line == "Y_COORDS") {
        results.y_coords.resize(n * m);
        for (int i = 0; i < n * m; ++i) {
            file >> results.y_coords[i];
        }
    }
    
    return true;
}

bool ResultsIO::saveMatrixAndRhs(const std::string& filename, const KokkosCrsMatrix& A,
                              const KokkosVector& b, int n, int m) {
    std::ofstream file(filename);
    if (!file) {
        return false;
    }
    
    // Получаем данные матрицы A
    auto row_map = Kokkos::create_mirror_view(A.graph.row_map);
    auto entries = Kokkos::create_mirror_view(A.graph.entries);
    auto values = Kokkos::create_mirror_view(A.values);
    
    Kokkos::deep_copy(row_map, A.graph.row_map);
    Kokkos::deep_copy(entries, A.graph.entries);
    Kokkos::deep_copy(values, A.values);
    
    // Получаем вектор правой части b
    auto b_host = Kokkos::create_mirror_view(b);
    Kokkos::deep_copy(b_host, b);
    
    // Размер матрицы и количество ненулевых элементов
    int num_rows = A.numRows();
    int nnz = A.nnz();
    
    // Сохраняем информацию о размере задачи
    file << "MATRIX_INFO\n";
    file << n << " " << m << "\n";
    file << num_rows << " " << nnz << "\n";
    
    // Сохраняем матрицу в формате CSR
    file << "MATRIX\n";
    for (int i = 0; i <= num_rows; ++i) {
        file << row_map(i) << "\n";
    }
    
    for (int i = 0; i < nnz; ++i) {
        file << entries(i) << "\n";
    }
    
    for (int i = 0; i < nnz; ++i) {
        file << std::scientific << values(i) << "\n";
    }
    
    // Сохраняем вектор правой части
    file << "RHS\n";
    for (int i = 0; i < num_rows; ++i) {
        file << std::scientific << b_host(i) << "\n";
    }
    
    return true;
}