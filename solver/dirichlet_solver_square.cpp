#include "dirichlet_solver_square.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// Объявления новых функций из default_functions.cpp
extern double custom_function_square(double x, double y);
extern double mu1_square(double x, double y);
extern double mu2_square(double x, double y);
extern double mu3_square(double x, double y);
extern double mu4_square(double x, double y);

// Объявления функций с тем же решением, что и G-образная область
extern double function2_square(double x, double y);
extern double solution2_square(double x, double y);
extern double mu1_square_solution2(double x, double y);
extern double mu2_square_solution2(double x, double y);
extern double mu3_square_solution2(double x, double y);
extern double mu4_square_solution2(double x, double y);

// Вспомогательная функция для граничных условий по умолчанию (нулевое граничное условие)
double default_boundary_value(double x, double y) {
    // Неиспользуемые параметры x и y для избежания предупреждений
    (void)x;
    (void)y;
    return 0.0;
}

// Конструктор с базовыми параметрами
DirichletSolverSquare::DirichletSolverSquare(int n, int m, double a, double b, double c, double d)
    : n_internal(n), m_internal(m),
      a_bound(a), b_bound(b), c_bound(c), d_bound(d),
      eps_precision(1e-6), eps_residual(1e-6), eps_exact_error(1e-6), max_iterations(10000),
      use_precision_stopping(true), use_residual_stopping(true), 
      use_error_stopping(false), use_max_iterations_stopping(true) {
    
    // Проверка инициализации Kokkos
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    
    // По умолчанию теперь используем функцию и точное решение как в G-образной области
    func = function2_square;
    mu1 = mu1_square_solution2;
    mu2 = mu2_square_solution2;
    mu3 = mu3_square_solution2;
    mu4 = mu4_square_solution2;
    exact_solution = solution2_square; // Устанавливаем точное решение
    
    // Инициализация сетки
    grid = std::make_unique<GridSystemSquare>(m_internal, n_internal, a_bound, b_bound, c_bound, d_bound,
                                           func, mu1, mu2, mu3, mu4);
}

// Конструктор с указанием функций правой части и точного решения
DirichletSolverSquare::DirichletSolverSquare(int n, int m, double a, double b, double c, double d,
                                           double (*f)(double, double), double (*sol)(double, double))
    : n_internal(n), m_internal(m),
      a_bound(a), b_bound(b), c_bound(c), d_bound(d),
      eps_precision(1e-6), eps_residual(1e-6), eps_exact_error(1e-6), max_iterations(10000),
      use_precision_stopping(true), use_residual_stopping(true), 
      use_error_stopping(false), use_max_iterations_stopping(true),
      func(f), exact_solution(sol) {
    
    // Проверка инициализации Kokkos
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    
    this->func = f; // Используем переданную функцию f
    this->exact_solution = sol;

    if (sol != nullptr) {
        // Если передано точное решение, используем его для mu, если они не заданы явно позже
        this->mu1 = sol;
        this->mu2 = sol;
        this->mu3 = sol;
        this->mu4 = sol;
    } else {
        // Если точное решение не передано, используем новые граничные условия по умолчанию
        // если пользователь не передал свои mu функции в другом конструкторе.
        // Если f передана, но sol нет, предполагаем, что пользователь хочет использовать стандартные mu_square
        // если только он не передаст свои mu функции через другой конструктор.
        this->mu1 = mu1_square; 
        this->mu2 = mu2_square;
        this->mu3 = mu3_square;
        this->mu4 = mu4_square;
    }
    
    // Инициализация сетки с функцией правой части, граничными условиями и функцией точного решения
    grid = std::make_unique<GridSystemSquare>(m_internal, n_internal, a_bound, b_bound, c_bound, d_bound,
                                           this->func, this->mu1, this->mu2, this->mu3, this->mu4,
                                           this->exact_solution); // Pass the exact solution function pointer
}

// Конструктор с указанием функций правой части и явных граничных условий
DirichletSolverSquare::DirichletSolverSquare(int n, int m, double a, double b, double c, double d,
                                           double (*f)(double, double),
                                           double (*mu1_func)(double, double),
                                           double (*mu2_func)(double, double),
                                           double (*mu3_func)(double, double),
                                           double (*mu4_func)(double, double))
    : n_internal(n), m_internal(m),
      a_bound(a), b_bound(b), c_bound(c), d_bound(d),
      eps_precision(1e-6), eps_residual(1e-6), eps_exact_error(1e-6), max_iterations(10000),
      use_precision_stopping(true), use_residual_stopping(true), 
      use_error_stopping(false), use_max_iterations_stopping(true),
      func(f), mu1(mu1_func), mu2(mu2_func), mu3(mu3_func), mu4(mu4_func) {
    
    // Проверка инициализации Kokkos
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    this->exact_solution = nullptr; // Явно указываем, что точного решения нет, если не передано
    
    // Инициализация сетки с функцией правой части и граничными условиями
    grid = std::make_unique<GridSystemSquare>(m_internal, n_internal, a_bound, b_bound, c_bound, d_bound,
                                           f, mu1, mu2, mu3, mu4);
}

// Деструктор
DirichletSolverSquare::~DirichletSolverSquare() {
    // Освобождаем ресурсы
    grid.reset();
    solver.reset();
}

// Установка параметров сетки
void DirichletSolverSquare::setGridParameters(int n, int m, double a, double b, double c, double d) {
    n_internal = n;
    m_internal = m;
    a_bound = a;
    b_bound = b;
    c_bound = c;
    d_bound = d;
    
    // Пересоздаем объект сетки с новыми параметрами
    // Сохраняем текущие указатели на функции, чтобы не потерять их, если они были установлены пользователем
    double (*current_func)(double, double) = this->func;
    double (*current_mu1)(double, double) = this->mu1;
    double (*current_mu2)(double, double) = this->mu2;
    double (*current_mu3)(double, double) = this->mu3;
    double (*current_mu4)(double, double) = this->mu4;
    double (*current_exact_solution)(double, double) = this->exact_solution;

    // Если какие-то из функций не были установлены (nullptr), используем значения по умолчанию
    if (!current_func) current_func = custom_function_square;
    if (!current_mu1) current_mu1 = mu1_square;
    if (!current_mu2) current_mu2 = mu2_square;
    if (!current_mu3) current_mu3 = mu3_square;
    if (!current_mu4) current_mu4 = mu4_square;
    // current_exact_solution остается nullptr, если не был задан

    // Use the constructor with boundary conditions but WITHOUT exact_solution
    grid = std::make_unique<GridSystemSquare>(
        m_internal, n_internal, 
        a_bound, b_bound, c_bound, d_bound,
        current_func, current_mu1, current_mu2, current_mu3, current_mu4);

    // Store the exact_solution for later use in the solver, but don't pass it to GridSystemSquare
    this->exact_solution = current_exact_solution;
}

// Установка параметров солвера
void DirichletSolverSquare::setSolverParameters(double eps_p, double eps_r, double eps_e, int max_iter) {
    eps_precision = eps_p;
    eps_residual = eps_r;
    eps_exact_error = eps_e;
    max_iterations = max_iter;
}

// Решение задачи Дирихле для квадратной области
SquareSolverResults DirichletSolverSquare::solve() {
    if (!grid) {
        throw std::runtime_error("Сетка не инициализирована");
    }
    // и use_... флаги в true. Это будет полностью переопределено ниже.
    solver = std::make_unique<MSGSolver>(grid->get_matrix(), grid->get_rhs(), 
    eps_precision, // Можно передать любой из eps, т.к. будет переопределено
    max_iterations); // Передаем актуальное значение max_iterations

    // Очищаем все критерии останова в MSGSolver, чтобы начать с чистого состояния.
    solver->clearStoppingCriteria();

// Включаем и настраиваем критерии останова на основе флагов use_..._stopping


    // Создаем солвер MSGSolver.
    // eps_precision (и другие eps_*) и max_iterations здесь являются членами DirichletSolverSquare.
    // max_iterations будет либо значением из SpinBox, либо INT_MAX, если критерий отключен.
    // eps_precision будет либо значением из SpinBox, либо 0.0, если критерий отключен.
    // Конструктор MSGSolver инициализирует свои внутренние eps_... значения на основе одного параметра eps,
    // и соответствующих значений eps_... / max_iterations из DirichletSolverSquare.

    if (use_precision_stopping) {
        solver->addPrecisionStoppingCriterion(eps_precision);
    }
    if (use_residual_stopping) {
        solver->addResidualStoppingCriterion(eps_residual);
    }
    if (use_error_stopping && hasTrueSolution()) {
        // Получаем вектор истинного решения из сетки
        // this->true_solution (член DirichletSolverSquare) будет обновлен.
        this->true_solution = grid->get_true_solution_vector(); 
        solver->addErrorStoppingCriterion(eps_exact_error, this->true_solution);
    }
    if (use_max_iterations_stopping) {
        solver->addMaxIterationsStoppingCriterion(max_iterations);
    }
    
    // Устанавливаем колбэк для отслеживания итераций, если он был задан
    if (iteration_callback) {
        solver->setIterationCallback(iteration_callback);
    }
    
    // Получаем истинное решение для сравнения, если оно было установлено через addErrorStoppingCriterion
    // или если hasTrueSolution() вернуло true ранее (this->true_solution уже должен быть установлен).
    // MSGSolver::solve() использует свой внутренний exact_solution, если он был установлен.
    solution = solver->solve();
    
    // Формируем результаты
    SquareSolverResults results;
    results.solution = kokkosToStdVector(solution);
    
    // Добавляем истинное решение, если оно доступно
    if (true_solution.extent(0) > 0) {
        results.true_solution = kokkosToStdVector(true_solution);
    }
    
    // Вычисляем невязку
    KokkosVector residual = computeResidual(grid->get_matrix(), solution, grid->get_rhs());
    results.residual = kokkosToStdVector(residual);
    
    // Вычисляем ошибку (разница между точным и численным решением), если есть точное решение
    if (true_solution.extent(0) > 0) {
        results.error = computeError(solution);
    }
    
    // Для квадратной области мы можем напрямую вычислить координаты
    int n_nodes = (n_internal - 1) * (m_internal - 1);
    results.x_coords.resize(n_nodes);
    results.y_coords.resize(n_nodes);
    
    double h_x = (b_bound - a_bound) / n_internal;
    double h_y = (d_bound - c_bound) / m_internal;
    
    int idx = 0;
    for (int j = 1; j < m_internal; ++j) {
        for (int i = 1; i < n_internal; ++i) {
            results.x_coords[idx] = a_bound + i * h_x;
            results.y_coords[idx] = c_bound + j * h_y;
            idx++;
        }
    }
    
    // Добавляем информацию о сходимости
    results.iterations = solver->getIterations();
    results.converged = solver->hasConverged();
    results.stop_reason = solver->getStopReasonText();
    
    // Добавляем нормы ошибок
    results.residual_norm = solver->getFinalResidualNorm();
    results.error_norm = solver->getFinalErrorNorm();
    results.precision = solver->getFinalPrecision();
    
    // Вычисляем ошибку относительно решения на более мелкой сетке, если флаг включен
    if (use_refined_grid_comparison) {
        double refined_grid_error = computeRefinedGridError();
        results.refined_grid_error = refined_grid_error;
        
        // Копируем результаты на мелкой сетке, если они доступны
        if (refined_grid_results) {
            results.refined_grid_solution = refined_grid_results->refined_grid_solution;
            results.solution_refined_diff = refined_grid_results->solution_refined_diff;
            results.refined_grid_x_coords = refined_grid_results->refined_grid_x_coords;
            results.refined_grid_y_coords = refined_grid_results->refined_grid_y_coords;
        }
    } else {
        results.refined_grid_error = -1.0; // Недоступно/не вычислено
    }
    
    // Вызываем обратный вызов завершения, если он установлен
    if (completion_callback) {
        completion_callback(results);
    }
    
    return results;
}

// Конвертация из Kokkos вектора в std::vector
std::vector<double> DirichletSolverSquare::kokkosToStdVector(const KokkosVector& kv) const {
    std::vector<double> result(kv.extent(0));
    auto kv_host = Kokkos::create_mirror_view(kv);
    Kokkos::deep_copy(kv_host, kv);
    
    for (size_t i = 0; i < kv.extent(0); ++i) {
        result[i] = kv_host(i);
    }
    
    return result;
}

// Вычисление невязки Ax-b
KokkosVector DirichletSolverSquare::computeResidual(const KokkosCrsMatrix& A, const KokkosVector& x, const KokkosVector& b) const {
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
std::vector<double> DirichletSolverSquare::computeError(const KokkosVector& v) const {
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
std::vector<double> DirichletSolverSquare::getSolution() const {
    return kokkosToStdVector(solution);
}

// Получение вектора точного решения
std::vector<double> DirichletSolverSquare::getTrueSolution() const {
    return kokkosToStdVector(true_solution);
}

// Преобразование 1D вектора решения в 2D матрицу
std::vector<std::vector<double>> DirichletSolverSquare::solutionToMatrix() const {
    // Для квадратной области матрица решения имеет простую структуру
    std::vector<std::vector<double>> result(m_internal + 1, std::vector<double>(n_internal + 1, 0.0));
    
    // Заполняем внутренние точки решения
    auto sol = kokkosToStdVector(solution);
    int idx = 0;
    
    for (int j = 1; j < m_internal; ++j) {
        for (int i = 1; i < n_internal; ++i) {
            if (idx < static_cast<int>(sol.size())) {
                result[j][i] = sol[idx++];
            }
        }
    }
    
    // Заполняем граничные точки, если известны граничные условия
    if (hasTrueSolution() || (mu1 != nullptr && mu2 != nullptr && mu3 != nullptr && mu4 != nullptr)) {
        double h_x = (b_bound - a_bound) / n_internal;
        double h_y = (d_bound - c_bound) / m_internal;
        
        // Левая и правая границы
        for (int j = 0; j <= m_internal; ++j) {
            double y = c_bound + j * h_y;
            // Левая граница (x = a_bound)
            if (mu1 != nullptr) {
                result[j][0] = mu1(a_bound, y);
            }
            // Правая граница (x = b_bound)
            if (mu2 != nullptr) {
                result[j][n_internal] = mu2(b_bound, y);
            }
        }
        
        // Нижняя и верхняя границы
        for (int i = 0; i <= n_internal; ++i) {
            double x = a_bound + i * h_x;
            // Нижняя граница (y = c_bound)
            if (mu3 != nullptr) {
                result[0][i] = mu3(x, c_bound);
            }
            // Верхняя граница (y = d_bound)
            if (mu4 != nullptr) {
                result[m_internal][i] = mu4(x, d_bound);
            }
        }
    }
    
    return result;
}

// Генерация отчета о решении
std::string DirichletSolverSquare::generateReport() const {
    if (!solver) {
        return "Решение еще не выполнено";
    }
    
    std::ostringstream oss;
    oss << "Отчет о решении уравнения Пуассона для квадратной области:\n";
    oss << "- Размер сетки: " << n_internal << "x" << m_internal << "\n";
    oss << "- Область: [" << a_bound << ", " << b_bound << "] x [" << c_bound << ", " << d_bound << "]\n";
    oss << "- Метод решения: " << getMethodName() << "\n";
    oss << "- Число итераций: " << solver->getIterations() << "\n";
    oss << "- Сходимость: " << (solver->hasConverged() ? "Да" : "Нет") << "\n";
    oss << "- Причина останова: " << solver->getStopReasonText() << "\n";
    oss << "- Норма невязки: " << std::scientific << solver->getFinalResidualNorm() << "\n";
    
    if (hasTrueSolution()) {
        oss << "- Норма ошибки: " << std::scientific << solver->getFinalErrorNorm() << "\n";
    }
    
    return oss.str();
}

// Сохранение результатов в файл
bool DirichletSolverSquare::saveResultsToFile(const std::string& filename) const {
    if (!solver) {
        return false;
    }
    
    SquareSolverResults results;
    results.solution = kokkosToStdVector(solution);
    
    if (hasTrueSolution()) {
        results.true_solution = kokkosToStdVector(true_solution);
    }
    
    // Вычисляем невязку
    KokkosVector residual = computeResidual(grid->get_matrix(), solution, grid->get_rhs());
    results.residual = kokkosToStdVector(residual);
    
    // Вычисляем ошибку, если есть точное решение
    if (hasTrueSolution()) {
        results.error = computeError(solution);
    }
    
    // Вычисляем координаты узлов для квадратной области
    int n_nodes = (n_internal - 1) * (m_internal - 1);
    results.x_coords.resize(n_nodes);
    results.y_coords.resize(n_nodes);
    
    double h_x = (b_bound - a_bound) / n_internal;
    double h_y = (d_bound - c_bound) / m_internal;
    
    int idx = 0;
    for (int j = 1; j < m_internal; ++j) {
        for (int i = 1; i < n_internal; ++i) {
            results.x_coords[idx] = a_bound + i * h_x;
            results.y_coords[idx] = c_bound + j * h_y;
            idx++;
        }
    }
    
    // Добавляем информацию о сходимости
    results.iterations = solver->getIterations();
    results.converged = solver->hasConverged();
    results.stop_reason = solver->getStopReasonText();
    results.residual_norm = solver->getFinalResidualNorm();
    results.error_norm = solver->getFinalErrorNorm();
    results.precision = solver->getFinalPrecision();
    
    return SquareResultsIO::saveResults(filename, results, n_internal, m_internal, 
                                      a_bound, b_bound, c_bound, d_bound, solver->getName());
}

// Сохранение матрицы и вектора правой части в файл
bool DirichletSolverSquare::saveMatrixAndRhsToFile(const std::string& filename) const {
    if (!grid) {
        return false;
    }
    
    return SquareResultsIO::saveMatrixAndRhs(filename, grid->get_matrix(), grid->get_rhs(), n_internal, m_internal);
}

// Реализация методов класса SquareResultsIO

bool SquareResultsIO::saveResults(const std::string& filename, const SquareSolverResults& results,
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
    
    // Сохранение точного решения (если доступно)
    file << "TRUE_SOLUTION\n";
    for (const auto& val : results.true_solution) {
        file << std::scientific << val << "\n";
    }
    
    // Сохранение невязки
    file << "RESIDUAL\n";
    for (const auto& val : results.residual) {
        file << std::scientific << val << "\n";
    }
    
    // Сохранение ошибки (если доступно)
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

bool SquareResultsIO::loadResults(const std::string& filename, SquareSolverResults& results,
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
    
    int n_nodes = (n - 1) * (m - 1);
    results.solution.resize(n_nodes);
    for (size_t i = 0; i < results.solution.size(); ++i) {
        file >> results.solution[i];
    }
    
    // Чтение точного решения
    if (!std::getline(file, line) || !std::getline(file, line) || line != "TRUE_SOLUTION") {
        return false;
    }
    
    results.true_solution.resize(n_nodes);
    for (size_t i = 0; i < results.true_solution.size(); ++i) {
        file >> results.true_solution[i];
    }
    
    // Чтение невязки
    if (!std::getline(file, line) || !std::getline(file, line) || line != "RESIDUAL") {
        return false;
    }
    
    results.residual.resize(n_nodes);
    for (size_t i = 0; i < results.residual.size(); ++i) {
        file >> results.residual[i];
    }
    
    // Чтение ошибки
    if (!std::getline(file, line) || !std::getline(file, line) || line != "ERROR") {
        return false;
    }
    
    results.error.resize(n_nodes);
    for (size_t i = 0; i < results.error.size(); ++i) {
        file >> results.error[i];
    }
    
    // Проверяем наличие координат X в файле
    if (std::getline(file, line) && std::getline(file, line) && line == "X_COORDS") {
        results.x_coords.resize(n_nodes);
        for (size_t i = 0; i < results.x_coords.size(); ++i) {
            file >> results.x_coords[i];
        }
    }
    
    // Проверяем наличие координат Y в файле
    if (std::getline(file, line) && std::getline(file, line) && line == "Y_COORDS") {
        results.y_coords.resize(n_nodes);
        for (size_t i = 0; i < results.y_coords.size(); ++i) {
            file >> results.y_coords[i];
        }
    }
    
    return true;
}

bool SquareResultsIO::saveMatrixAndRhs(const std::string& filename, const KokkosCrsMatrix& A,
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

// Вычисление ошибки с использованием решения на более мелкой сетке
double DirichletSolverSquare::computeRefinedGridError() {
    // Проверяем, есть ли уже решение на текущей сетке
    if (solution.extent(0) == 0) {
        // Если нет, решаем задачу
        this->solve();
    }
    std::vector<double> current_solution = kokkosToStdVector(solution);

    // Создаем решатель с теми же параметрами, но удвоенным размером сетки
    std::unique_ptr<DirichletSolverSquare> refined_solver;
    
    if (exact_solution == nullptr && mu1 && mu2 && mu3 && mu4) {
        // Создаем новый решатель с граничными условиями, если нет точного решения, но есть граничные условия
        refined_solver = std::make_unique<DirichletSolverSquare>(
            n_internal * 2, m_internal * 2,
            a_bound, b_bound, c_bound, d_bound,
            func, mu1, mu2, mu3, mu4
        );
    } else {
        // По умолчанию создаем решатель с точным решением, если оно есть
        refined_solver = std::make_unique<DirichletSolverSquare>(
            n_internal * 2, m_internal * 2,
            a_bound, b_bound, c_bound, d_bound,
            func, exact_solution
        );
    }
    
    // Настраиваем параметры решателя для более мелкой сетки
    refined_solver->setSolverParameters(eps_precision, eps_residual, eps_exact_error, max_iterations);
    refined_solver->setUsePrecisionStopping(use_precision_stopping);
    refined_solver->setUseResidualStopping(use_residual_stopping);
    refined_solver->setUseErrorStopping(use_error_stopping);
    refined_solver->setUseMaxIterationsStopping(use_max_iterations_stopping);
    refined_solver->setUseRefinedGridComparison(false); // Важно! Отключаем рекурсивное использование мелкой сетки
    
    // Решаем задачу на более мелкой сетке
    SquareSolverResults refined_results = refined_solver->solve();
    
    // Получаем координаты текущей сетки
    double h_x = (b_bound - a_bound) / n_internal;
    double h_y = (d_bound - c_bound) / m_internal;
    
    // Получаем координаты и решение на более мелкой сетке
    double refined_h_x = (b_bound - a_bound) / (2 * n_internal);
    double refined_h_y = (d_bound - c_bound) / (2 * m_internal);
    
    // Вычисляем максимальную разницу между решениями в соответствующих точках
    double max_error = 0.0;
    
    // Инициализируем или очищаем объект для хранения результатов мелкой сетки
    if (!refined_grid_results) {
        refined_grid_results = std::make_unique<SquareSolverResults>();
    } else {
        // Очищаем предыдущие результаты, чтобы избежать утечек памяти и некорректного поведения
        refined_grid_results->refined_grid_solution.clear();
        refined_grid_results->refined_grid_x_coords.clear();
        refined_grid_results->refined_grid_y_coords.clear();
        refined_grid_results->solution_refined_diff.clear();
    }
    
    // Сохраняем решение на мелкой сетке и координаты
    refined_grid_results->refined_grid_solution = refined_results.solution;
    refined_grid_results->refined_grid_x_coords = refined_results.x_coords;
    refined_grid_results->refined_grid_y_coords = refined_results.y_coords;
    
    // Вычисляем разницу между решениями во всех точках основной сетки
    refined_grid_results->solution_refined_diff.resize(current_solution.size());
    int idx_current = 0;
    
    // Обходим узлы текущей сетки
    for (int j = 1; j < m_internal; ++j) {
        for (int i = 1; i < n_internal; ++i) {
            if (idx_current >= static_cast<int>(current_solution.size())) {
                continue;  // Проверка выхода за границы
            }
            
            // Вычисляем координаты текущей точки
            double x = a_bound + i * h_x;
            double y = c_bound + j * h_y;
            
            // Находим соответствующий индекс в массиве решения на мелкой сетке
            // Для мелкой сетки индексы будут в 2 раза больше
            int refined_i = i * 2;
            int refined_j = j * 2;
            int refined_idx = (refined_j - 1) * (2 * n_internal - 1) + (refined_i - 1);
            
            if (refined_idx < static_cast<int>(refined_results.solution.size())) {
                double diff = std::abs(current_solution[idx_current] - refined_results.solution[refined_idx]);
                refined_grid_results->solution_refined_diff[idx_current] = diff;
                max_error = std::max(max_error, diff);
            }
            
            idx_current++;
        }
    }
    
    return max_error;
}