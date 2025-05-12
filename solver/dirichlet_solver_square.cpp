#include "dirichlet_solver_square.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <Kokkos_StdAlgorithms.hpp> // Для Kokkos::norm
#include <cmath> // для std::abs

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
    
    // Формируем результаты - объявляем в начале метода
    SquareSolverResults results;
    
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
    
    KokkosVector b_copy = grid->get_rhs(); // Копируем вектор правой части
    double norm_b = KokkosBlas::nrm2(b_copy);
    results.initial_residual_norm = norm_b; // Сохраняем норму правой части как начальную невязку
    
    solution = solver->solve();
    
    // Обновляем результаты (убираем повторное объявление)
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
            results.refined_grid_iterations = refined_grid_results->refined_grid_iterations;
            results.refined_grid_precision = refined_grid_results->refined_grid_precision;
            results.refined_grid_residual_norm = refined_grid_results->refined_grid_residual_norm;
            results.refined_grid_initial_residual_norm = refined_grid_results->refined_grid_initial_residual_norm;
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
    // Создаем новый экземпляр решателя для более мелкой сетки
    // Увеличиваем n_internal и m_internal вдвое
    int refined_n = n_internal * 2;
    int refined_m = m_internal * 2;

    // Сохраняем текущие параметры решателя, чтобы не изменять их для основного решения
    double current_eps_p = eps_precision;
    double current_eps_r = eps_residual;
    double current_eps_e = eps_exact_error;
    int current_max_iter = max_iterations;
    bool current_use_p = use_precision_stopping;
    bool current_use_r = use_residual_stopping;
    bool current_use_e = use_error_stopping;
    bool current_use_max_i = use_max_iterations_stopping;

    // Создаем новый решатель для мелкой сетки
    // Используем те же функции f, mu, exact_solution, что и для основного решателя
    auto refined_solver = std::make_unique<DirichletSolverSquare>(
        refined_n, refined_m, a_bound, b_bound, c_bound, d_bound,
        this->func, this->mu1, this->mu2, this->mu3, this->mu4
    );
    // Устанавливаем точное решение, если оно есть
    if (this->exact_solution) {
        refined_solver->exact_solution = this->exact_solution;
    }

    // Устанавливаем параметры решателя для мелкой сетки (можно использовать те же, что и для основной)
    refined_solver->setSolverParameters(current_eps_p, current_eps_r, current_eps_e, current_max_iter);
    refined_solver->setUsePrecisionStopping(current_use_p);
    refined_solver->setUseResidualStopping(current_use_r);
    refined_solver->setUseErrorStopping(current_use_e && refined_solver->hasTrueSolution());
    refined_solver->setUseMaxIterationsStopping(current_use_max_i);
    refined_solver->setUseRefinedGridComparison(false); // Отключаем рекурсивное сравнение

    // Решаем задачу на мелкой сетке
    SquareSolverResults refined_results_data = refined_solver->solve();

    // Сохраняем результаты решения на мелкой сетке в refined_grid_results
    if (!refined_grid_results) {
        refined_grid_results = std::make_unique<SquareSolverResults>();
    }
    *refined_grid_results = refined_results_data; // Копируем данные
    refined_grid_results->refined_grid_solution = refined_results_data.solution;
    refined_grid_results->refined_grid_x_coords.clear();
    refined_grid_results->refined_grid_y_coords.clear();

    double h_x_refined = (b_bound - a_bound) / refined_n;
    double h_y_refined = (d_bound - c_bound) / refined_m;
    for (int j = 1; j < refined_m; ++j) {
        for (int i = 1; i < refined_n; ++i) {
            refined_grid_results->refined_grid_x_coords.push_back(a_bound + i * h_x_refined);
            refined_grid_results->refined_grid_y_coords.push_back(c_bound + j * h_y_refined);
        }
    }

    // Получаем решение на основной сетке
    std::vector<double> current_sol_std = kokkosToStdVector(this->solution);

    // Интерполируем решение с основной сетки на узлы мелкой сетки
    // или выбираем значения из решения на мелкой сетке, соответствующие узлам основной
    // Для простоты, будем сравнивать значения в узлах, которые есть на обеих сетках.
    // То есть, каждый второй узел мелкой сетки должен совпадать с узлом основной сетки.

    std::vector<double> diff_vector;
    refined_grid_results->solution_refined_diff.clear();

    // Предполагаем, что solution (решение на основной сетке) имеет (n_internal-1)*(m_internal-1) точек
    // refined_results_data.solution (решение на мелкой сетке) имеет (refined_n-1)*(refined_m-1) точек

    // Сравниваем значения в узлах, которые присутствуют на обеих сетках
    // Узлы основной сетки (i, j) соответствуют узлам (2*i, 2*j) из grid2,
    // если нумерация узлов начинается с 1.
    // В реальности, если grid1 (N,M) и grid2 (2N, 2M), то узлы (i,j) из grid1
    // соответствуют узлам (2*(i+j)-1, 2*(j+1)-1) из grid2 (при правильной нумерации).
    // Если это не так, или если требуется более общее решение, нужна интерполяция.
    // Пока возвращаем -1.0, указывая, что вычисление не выполнено или не поддерживается в простом виде.

    double max_abs_diff = 0.0;
    int main_grid_idx = 0;
    for (int j_main = 0; j_main < m_internal - 1; ++j_main) {
        for (int i_main = 0; i_main < n_internal - 1; ++i_main) {
            // Координаты узла на основной сетке (не индекс, а номер узла)
            // int node_i_main = i_main + 1;
            // int node_j_main = j_main + 1;

            // Соответствующий индекс на мелкой сетке
            // Узел (i_main+1, j_main+1) на основной сетке соответствует
            // узлу (2*(i_main+1)-1, 2*(j_main+1)-1) в нумерации мелкой сетки, если она тоже с 1.
            // Или, если индексы с 0: узел (i_main, j_main) на основной (среди внутренних)
            // соответствует узлу (2*i_main + 1, 2*j_main + 1) на мелкой (среди внутренних).
            int refined_grid_i_idx = (i_main * 2) + 1; // Индекс столбца на мелкой сетке (0-based)
            int refined_grid_j_idx = (j_main * 2) + 1; // Индекс строки на мелкой сетке (0-based)

            // Преобразуем 2D индекс мелкой сетки в 1D индекс
            if (refined_grid_j_idx < refined_m - 1 && refined_grid_i_idx < refined_n - 1) {
                int refined_1d_idx = refined_grid_j_idx * (refined_n - 1) + refined_grid_i_idx;

                if (static_cast<size_t>(main_grid_idx) < current_sol_std.size() && 
                    static_cast<size_t>(refined_1d_idx) < refined_results_data.solution.size()) {
                    double diff = current_sol_std[main_grid_idx] - refined_results_data.solution[refined_1d_idx];
                    diff_vector.push_back(diff);
                    refined_grid_results->solution_refined_diff.push_back(diff); 
                    if (std::abs(diff) > max_abs_diff) {
                        max_abs_diff = std::abs(diff);
                    }
                } else {
                     // Добавляем 0 или NaN, если индексы выходят за пределы, для сохранения размера
                     refined_grid_results->solution_refined_diff.push_back(0.0); 
                }
            }
            main_grid_idx++;
        }
    }
    // Заполняем оставшуюся часть solution_refined_diff нулями или NaN, если ее размер меньше, чем у refined_grid_solution
    // Это не совсем корректно, так как diff осмыслен только для узлов основной сетки.
    // Правильнее было бы, чтобы solution_refined_diff имел размер основной сетки.
    // Переделываем: solution_refined_diff будет содержать разницу в тех же точках, что и основное решение.
    // refined_grid_results->solution_refined_diff теперь содержит разницу в точках основной сетки.

    // Норма разницы (например, L2 норма)
    double l2_norm_diff = 0.0;
    if (!diff_vector.empty()) {
        for (double val : diff_vector) {
            l2_norm_diff += val * val;
        }
        l2_norm_diff = std::sqrt(l2_norm_diff / diff_vector.size()); // RMSD
    }

    return l2_norm_diff; // Возвращаем норму разницы
}

// Добавим вспомогательную функцию для вычисления нормы разности решений
// Эта функция потребует значительной доработки для корректной интерполяции
double DirichletSolverSquare::calculateSolutionDifferenceNorm(
    const std::vector<double>& sol1, const GridSystemSquare& grid1,
    const std::vector<double>& sol2, const GridSystemSquare& grid2)
{
    // Get grid dimensions - since n and m are private and no getters are available,
    // we'll use the coordinate vectors size to determine the dimensions
    const auto& x_coords1 = grid1.get_x_coords();
    const auto& y_coords1 = grid1.get_y_coords();
    const auto& x_coords2 = grid2.get_x_coords();
    const auto& y_coords2 = grid2.get_y_coords();
    
    // Проверка: если размеры сеток не соотносятся как 1:2 или если массивы решений пусты
    // вернем -1.0, указывая, что вычисление не выполнено
    if (x_coords2.size() != 2 * x_coords1.size() || 
        y_coords2.size() != 2 * y_coords1.size() || 
        sol1.empty() || sol2.empty()) {
        return -1.0;
    }

    // Calculate dimensions based on coordinate vectors
    int N1 = static_cast<int>(std::sqrt(x_coords1.size()));
    int M1 = N1;  // Assuming square grid
    int N2 = static_cast<int>(std::sqrt(x_coords2.size()));
    int M2 = N2;  // Assuming square grid

    Kokkos::View<double*> diff_kokkos("difference", sol1.size());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> diff_host(diff_kokkos.data(), diff_kokkos.size());

    size_t k_diff = 0;
    double sum_sq_diff = 0.0;
    int valid_points_count = 0;

    for (int j = 0; j <= M1; ++j) { // Итерация по узлам грубой сетки
        for (int i = 0; i <= N1; ++i) {
            size_t idx1 = j * (N1 + 1) + i; // Индекс на грубой сетке

            // Соответствующий индекс на подробной сетке (каждый второй узел)
            int i_refined = i * 2;
            int j_refined = j * 2;
            size_t idx2 = j_refined * (N2 + 1) + i_refined;

            if (idx1 < sol1.size() && idx2 < sol2.size()) {
                double val1 = sol1[idx1];
                double val2 = sol2[idx2];
                diff_host(k_diff++) = val1 - val2;
                sum_sq_diff += (val1 - val2) * (val1 - val2);
                valid_points_count++;
            }
        }
    }
    
    if (valid_points_count == 0) return -1.0; // Не удалось сравнить точки

    // Вместо Kokkos::norm для std::vector можно использовать std::sqrt(sum_sq_diff)
    return std::sqrt(sum_sq_diff / valid_points_count); // Евклидова норма разности на узлах грубой сетки
}