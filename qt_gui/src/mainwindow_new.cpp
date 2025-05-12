#include "mainwindow_new.h"
#include <QVBoxLayout>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QDesktopServices>
#include <QUrl>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>     // Для std::abs, std::sqrt
#include <vector>
#include <algorithm> // Для std::max_element, std::distance
#include <limits>    // Для std::numeric_limits
#include <QStringList>
#include <QLocale>   // Для форматирования чисел

// Анонимное пространство имен для вспомогательных функций
namespace {

// Функция для поиска координат точки с максимальным абсолютным значением
std::pair<double, double> find_max_abs_value_coords(
    const std::vector<double>& values,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords)
{
    if (values.empty() || values.size() != x_coords.size() || values.size() != y_coords.size()) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }

    double max_abs_val = 0.0;
    int max_idx = -1;
    bool first = true;

    for (size_t i = 0; i < values.size(); ++i) {
        if (first || std::abs(values[i]) > max_abs_val) {
            max_abs_val = std::abs(values[i]);
            max_idx = static_cast<int>(i);
            first = false;
        }
    }

    if (max_idx != -1) {
        return {x_coords[max_idx], y_coords[max_idx]};
    }
    // Если все значения 0 или массив пуст (хотя это уже проверено)
    return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}

// Функция для форматирования числа с заданной точностью или "N/A"
QString formatDouble(double value, int precision = 2, char format = 'e') {
    if (value == -1.0 || std::isnan(value) || std::isinf(value)) {
        return "N/A";
    }
    return QLocale().toString(value, format, precision);
}

// Функция для построения строки с параметрами метода
QString build_method_parameters_string(const MainWindow::SolverParams& p, const QString& solverType) {
    QStringList criteria;
    if (p.use_precision) criteria << QString("Точность (eps=%1)").arg(formatDouble(p.eps_precision));
    if (p.use_residual) criteria << QString("Невязка (eps=%1)").arg(formatDouble(p.eps_residual));
    
    if (p.use_exact_error) {
         // Для G-образной области и тестовой задачи может быть точное решение
        if (solverType.contains("Тестовая задача") || solverType.contains("G-образная")) {
             criteria << QString("Погрешность (с точным реш., eps=%1)").arg(formatDouble(p.eps_exact_error));
        }
    }
    if (p.use_max_iterations) criteria << QString("Макс. итераций (%1)").arg(p.max_iterations);

    if (criteria.isEmpty()) {
        return "Критерии останова не заданы.";
    }
    return "Критерии: " + criteria.join(", ") + ".";
}

} // конец анонимного пространства имен

// Реализация класса SolverWorker
SolverWorker::SolverWorker(std::unique_ptr<DirichletSolver> solver)
    : solver(std::move(solver))
    , is_square_solver(false)
{
}

SolverWorker::SolverWorker(std::unique_ptr<DirichletSolverSquare> solver_sq)
    : solver_sq(std::move(solver_sq))
    , is_square_solver(true)
{
}

void SolverWorker::process()
{
    try {
        if (is_square_solver && solver_sq) {
            // Настройка callback для отслеживания итераций
            auto iter_callback = [this](int iter, double precision, double residual, double error) {
                emit iterationUpdate(iter, precision, residual, error);
                return true; // Продолжить итерации
            };
            solver_sq->setIterationCallback(iter_callback);
            
            // Решаем уравнение
            SquareSolverResults results_sq = solver_sq->solve();
            emit resultReadySquare(results_sq);
        } else if (solver) {
            // Настройка callback для отслеживания итераций
            auto iter_callback = [this](int iter, double precision, double residual, double error) {
                emit iterationUpdate(iter, precision, residual, error);
                return true; // Продолжить итерации
            };
            solver->setIterationCallback(iter_callback);
            
            // Решаем уравнение
            SolverResults results = solver->solve();
            emit resultReady(results);
        }
    } catch (const std::exception& e) {
        std::cerr << "Ошибка в процессе решения: " << e.what() << std::endl;
    }
    
    emit finished();
}

// Реализация класса MainWindow
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , isSolving(false)
    , solveSuccessful(false)
{
    // Инициализация Kokkos если нужно
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    
    // Создаем основной контейнер вкладок
    tabWidget = new QTabWidget(this);
    
    // Создаем компоненты вкладок
    solverTab = new SolverTabWidget();
    progressTab = new ProgressTabWidget();
    visualizationTab = new VisualizationTabWidget();
    visualization3DTab = new Visualization3DTabWidget();
    tableTab = new TableTabWidget();
    helpTab = new HelpTabWidget(this); // Создаем вкладку "Справка"
    
    // Добавляем вкладки в контейнер
    tabWidget->addTab(solverTab, "Настройки решателя");
    tabWidget->addTab(progressTab, "Прогресс");
    tabWidget->addTab(visualizationTab, "Визуализация 2D");
    tabWidget->addTab(visualization3DTab, "3D Визуализация");
    tabWidget->addTab(tableTab, "Таблица");
    tabWidget->addTab(helpTab, "Справка"); // Добавляем вкладку "Справка"
    
    // Устанавливаем контейнер вкладок как центральный виджет
    setCentralWidget(tabWidget);
    
    // Создаем меню
    QMenuBar* menuBar = new QMenuBar(this);
    QMenu* fileMenu = menuBar->addMenu("Файл");
    QMenu* exportMenu = menuBar->addMenu("Экспорт");
    QMenu* helpMenu = menuBar->addMenu("Справка");
    
    // Добавляем действия в меню Файл
    QAction* saveResultsAction = fileMenu->addAction("Сохранить результаты", this, &MainWindow::onSaveResultsButtonClicked);
    QAction* saveMatrixAction = fileMenu->addAction("Сохранить матрицу", this, &MainWindow::onSaveMatrixButtonClicked);
    fileMenu->addSeparator();
    QAction* exitAction = fileMenu->addAction("Выход", this, &QMainWindow::close);
    
    // Добавляем действия в меню Экспорт
    QAction* saveVisualizationAction = exportMenu->addAction("Сохранить визуализацию", this, &MainWindow::onSaveVisualizationButtonClicked);
    
    // Добавляем действия в меню Справка
    QAction* showReportAction = helpMenu->addAction("Показать отчёт", this, &MainWindow::onShowReportButtonClicked);
    
    setMenuBar(menuBar);
    
    // Соединяем сигналы и слоты
    connect(solverTab, &SolverTabWidget::solveButtonClicked, this, &MainWindow::onSolveButtonClicked);
    connect(solverTab, &SolverTabWidget::stopButtonClicked, this, &MainWindow::onStopButtonClicked);
    connect(tabWidget, &QTabWidget::currentChanged, this, &MainWindow::onTabChanged);
    connect(visualizationTab, &VisualizationTabWidget::chartTypeChanged, this, &MainWindow::onChartTypeChanged);
    connect(visualization3DTab, &Visualization3DTabWidget::showHeatMapClicked, this, &MainWindow::onShowHeatMapClicked);
    connect(tableTab, &TableTabWidget::exportCSVRequested, this, &MainWindow::onExportCSVRequested);
    
    // Устанавливаем параметры окна
    setWindowTitle("Решатель уравнения Пуассона");
    resize(1024, 768);
    
    // Устанавливаем начальные параметры решателя
    params.n_internal = 30;
    params.m_internal = 30;
    params.a_bound = 1.0;
    params.b_bound = 2.0;
    params.c_bound = 1.0;
    params.d_bound = 2.0;
    params.eps_precision = 1e-6;
    params.eps_residual = 1e-6;
    params.eps_exact_error = 1e-6;
    params.max_iterations = 10000;
    params.use_precision = true;
    params.use_residual = true;
    params.use_exact_error = false;
    params.use_max_iterations = true;
    params.use_refined_grid = false;
    params.solver_type = "Ступень 3";
}

MainWindow::~MainWindow()
{
    // Очищаем поток решателя
    cleanupThread();
    
    // Финализируем Kokkos, если мы его инициализировали
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
}

void MainWindow::onSolveButtonClicked()
{
    if (isSolving) {
        return;
    }
    
    // Получаем параметры решателя из UI
    params.n_internal = solverTab->getNInternal();
    params.m_internal = solverTab->getMInternal();
    params.a_bound = solverTab->getABound();
    params.b_bound = solverTab->getBBound();
    params.c_bound = solverTab->getCBound();
    params.d_bound = solverTab->getDBound();
    params.eps_precision = solverTab->getEpsPrecision();
    params.eps_residual = solverTab->getEpsResidual();
    params.eps_exact_error = solverTab->getEpsExactError();
    params.max_iterations = solverTab->getMaxIterations();
    params.use_precision = solverTab->getUsePrecision();
    params.use_residual = solverTab->getUseResidual();
    params.use_exact_error = solverTab->getUseExactError();
    params.use_max_iterations = solverTab->getUseMaxIterations();
    params.use_refined_grid = solverTab->getUseRefinedGrid();
    params.solver_type = solverTab->getSolverType();
    
    // Очищаем данные предыдущего решения
    results = SolverResults{};
    results_square = SquareSolverResults{};
    solveSuccessful = false;
    
    // Очищаем компоненты UI
    progressTab->clearProgress();
    visualizationTab->clear();
    visualization3DTab->clear();
    tableTab->setCSVData("");
    
    // Устанавливаем максимальное число итераций для прогресс-бара
    progressTab->setMaxIterations(params.max_iterations);
    
    // Переключаемся на вкладку "Прогресс"
    tabWidget->setCurrentWidget(progressTab);
    
    // Настраиваем UI для состояния "Решение"
    isSolving = true;
    solverTab->setSolveButtonEnabled(false);
    solverTab->setStopButtonEnabled(true);
    
    // Инициализируем решатель
    setupSolver();
    
    // Запускаем решение в отдельном потоке
    solverThread = new QThread;
    worker->moveToThread(solverThread);
    
    connect(solverThread, &QThread::started, worker, &SolverWorker::process);
    connect(worker, &SolverWorker::finished, this, &MainWindow::onSolverFinished);
    connect(worker, &SolverWorker::resultReady, this, &MainWindow::handleResults);
    connect(worker, &SolverWorker::resultReadySquare, this, &MainWindow::handleResultsSquare);
    connect(worker, &SolverWorker::iterationUpdate, this, &MainWindow::updateIterationInfo);
    connect(worker, &SolverWorker::finished, solverThread, &QThread::quit);
    connect(solverThread, &QThread::finished, this, [=]() {
        cleanupThread();
    });
    
    solverThread->start();
}

void MainWindow::onStopButtonClicked()
{
    if (!isSolving) {
        return;
    }
    
    // Останавливаем решатель
    if (worker) {
        if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
            if (auto* s = worker->getSolverSquare()) {
                s->requestStop();
            }
        } else {
            if (auto* s = worker->getSolver()) {
                s->requestStop();
            }
        }
    }
}

void MainWindow::setupSolver()
{
    // Создаем соответствующий объект решателя на основе выбранного типа
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        // Функции для граничных условий и правой части
        double (*f_func)(double, double) = nullptr;
        double (*exact_solution_func)(double, double) = nullptr;
        double (*mu1_func)(double, double) = nullptr;
        double (*mu2_func)(double, double) = nullptr;
        double (*mu3_func)(double, double) = nullptr;
        double (*mu4_func)(double, double) = nullptr;

        // Настраиваем функции на основе типа задачи
        if (params.solver_type.contains("тестовая", Qt::CaseInsensitive)) {
            // Функции для тестовой задачи (должна иметь точное решение)
            f_func = function2_square;
            mu1_func = mu1_square_solution2;
            mu2_func = mu2_square_solution2;
            mu3_func = mu3_square_solution2;
            mu4_func = mu4_square_solution2;
            exact_solution_func = solution2_square;
        } else if (params.solver_type.contains("основная", Qt::CaseInsensitive)) {
            // Функции для основной задачи (с граничными условиями, без точного решения)
            f_func = custom_function_square;
            mu1_func = mu1_square;
            mu2_func = mu2_square;
            mu3_func = mu3_square;
            mu4_func = mu4_square;
        }

        // Создаем квадратный решатель для задач ступени 2 с нужными функциями
        if (exact_solution_func != nullptr) {
            // Вариант с точным решением
            solver_square = std::make_unique<DirichletSolverSquare>(
                params.n_internal, params.m_internal, 
                params.a_bound, params.b_bound, 
                params.c_bound, params.d_bound,
                f_func, exact_solution_func
            );
        } else if (mu1_func != nullptr && mu2_func != nullptr && mu3_func != nullptr && mu4_func != nullptr) {
            // Вариант с граничными условиями
            solver_square = std::make_unique<DirichletSolverSquare>(
                params.n_internal, params.m_internal, 
                params.a_bound, params.b_bound, 
                params.c_bound, params.d_bound,
                f_func, mu1_func, mu2_func, mu3_func, mu4_func
            );
        } else {
            // Простой вариант без дополнительных функций
            solver_square = std::make_unique<DirichletSolverSquare>(
                params.n_internal, params.m_internal, 
                params.a_bound, params.b_bound, 
                params.c_bound, params.d_bound
            );
        }
        
        // Настраиваем параметры решателя
        solver_square->setSolverParameters(
            params.eps_precision, 
            params.eps_residual, 
            params.eps_exact_error, 
            params.max_iterations
        );
        
        // Устанавливаем критерии остановки
        solver_square->setUsePrecisionStopping(params.use_precision);
        solver_square->setUseResidualStopping(params.use_residual);
        solver_square->setUseErrorStopping(params.use_exact_error);
        solver_square->setUseMaxIterationsStopping(params.use_max_iterations);
        
        // Установка флага для использования уточненной сетки
        solver_square->setUseRefinedGridComparison(params.use_refined_grid);
        
        // Создаем рабочий объект для потока
        worker = new SolverWorker(std::move(solver_square));
    } else {
        // Создаем решатель для задач ступени 3 (G-образной области)
        solver = std::make_unique<DirichletSolver>(
            params.n_internal, params.m_internal, 
            params.a_bound, params.b_bound, 
            params.c_bound, params.d_bound
        );
        
        // Настраиваем параметры решателя
        solver->setSolverParameters(
            params.eps_precision, 
            params.eps_residual, 
            params.eps_exact_error, 
            params.max_iterations
        );
        
        // Устанавливаем критерии остановки
        solver->enablePrecisionStopping(params.use_precision);
        solver->enableResidualStopping(params.use_residual);
        solver->enableErrorStopping(params.use_exact_error);
        solver->enableMaxIterationsStopping(params.use_max_iterations);
        
        // Создаем рабочий объект для потока
        worker = new SolverWorker(std::move(solver));
    }
}

void MainWindow::handleResults(const SolverResults& res)
{
    // Сохраняем результаты
    this->results = res;
    solveSuccessful = true;
    
    // Обновляем визуализацию
    visualizationTab->updateChart(
        res.solution,
        res.x_coords,
        res.y_coords,
        res.true_solution,
        params.a_bound, params.b_bound,
        params.c_bound, params.d_bound
    );
    visualizationTab->setSolveSuccessful(true);
    
    // Обновляем 3D визуализацию
    visualization3DTab->createOrUpdate3DSurfaces(
        res.solution,
        res.true_solution,
        res.error,
        res.x_coords,
        res.y_coords,
        params.a_bound, params.b_bound,
        params.c_bound, params.d_bound,
        false, // is_square_solver
        false  // use_refined_grid
    );
    visualization3DTab->setSolveSuccessful(true);
    
    // Обновляем данные в таблице (используем прямое заполнение вместо CSV)
    tableTab->setResultsData(
        res.solution,
        res.true_solution,
        res.error,
        res.x_coords,
        res.y_coords,
        false // G-образная область не является квадратной сеткой
    );
    
    // Обновляем информацию о решении на вкладке прогресса
    progressTab->updateSolverFinished(
        solveSuccessful,
        res.iterations,
        res.residual_norm,
        res.error_norm,
        res.precision,
        res.converged,
        res.stop_reason
    );
    
    // Обновляем информацию о матрице - здесь нужно убрать обращение к полям matrix_size и nonzero_elements,
    // которых нет в структуре SolverResults
    QString matrixInfo = QString("Размер системы: %1x%1, Ненулевых элементов: %2")
                        .arg(res.solution.size())  // вместо matrix_size используем размер решения
                        .arg(0);  // nonzero_elements не доступно, выводим 0
    solverTab->updateMatrixInfo(matrixInfo);
    
    updateHelpTabInfo(); // Update help tab after results
}

void MainWindow::handleResultsSquare(const SquareSolverResults& results_sq)
{
    // Сохраняем результаты
    this->results_square = results_sq;
    solveSuccessful = true;
    
    // Обновляем визуализацию
    visualizationTab->updateChart(
        this->results_square.solution,
        this->results_square.x_coords,
        this->results_square.y_coords,
        this->results_square.true_solution,
        params.a_bound, params.b_bound,
        params.c_bound, params.d_bound
    );
    visualizationTab->setSolveSuccessful(true);
    
    // Обновляем 3D визуализацию
    if (params.use_refined_grid && !this->results_square.refined_grid_solution.empty()) {
        // Визуализация с учетом решения на уточненной сетке
        visualization3DTab->createOrUpdateRefinedGridSurfaces(
            this->results_square.solution,
            this->results_square.refined_grid_solution,
            this->results_square.solution_refined_diff,
            this->results_square.x_coords,
            this->results_square.y_coords,
            this->results_square.refined_grid_x_coords,
            this->results_square.refined_grid_y_coords,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound
        );
        
        // Устанавливаем информацию об ошибке на уточненной сетке
        progressTab->setRefinedGridError(this->results_square.refined_grid_error);
    } else {
        // Стандартная визуализация
        visualization3DTab->createOrUpdate3DSurfaces(
            this->results_square.solution,
            this->results_square.true_solution,
            this->results_square.error,
            this->results_square.x_coords,
            this->results_square.y_coords,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound,
            true, // is_square_solver
            false // use_refined_grid
        );
    }
    visualization3DTab->setSolveSuccessful(true);
    
    // Обновляем данные в таблице (используем прямое заполнение вместо CSV)
    tableTab->setResultsData(
        this->results_square.solution,
        this->results_square.true_solution,
        this->results_square.error,
        this->results_square.x_coords,
        this->results_square.y_coords,
        true // квадратная сетка
    );
    
    // Если есть решение на уточненной сетке, передаем его
    if (params.use_refined_grid && !this->results_square.refined_grid_solution.empty()) {
        tableTab->setRefinedGridData(
            this->results_square.refined_grid_solution,
            this->results_square.refined_grid_x_coords, 
            this->results_square.refined_grid_y_coords
        );
    }
    
    // Обновляем информацию о решении на вкладке прогресса
    progressTab->updateSolverFinished(
        solveSuccessful,
        this->results_square.iterations,
        this->results_square.residual_norm,
        this->results_square.error_norm,
        this->results_square.precision,
        this->results_square.converged,
        this->results_square.stop_reason
    );
    
    // Обновляем информацию о матрице - используем другие доступные поля
    int matrixSize = params.n_internal * params.m_internal;  // примерная оценка размера матрицы
    int nonZeroElements = matrixSize * 5;  // примерная оценка для 5-точечного шаблона
    
    QString matrixInfo = QString("Размер системы: %1x%1, Ненулевых элементов: ~%2")
                        .arg(matrixSize)
                        .arg(nonZeroElements);
    solverTab->updateMatrixInfo(matrixInfo);
    
    updateHelpTabInfo(); // Update help tab after results
}

void MainWindow::updateHelpTabInfo() {
    if (!helpTab) return;

    QString solverTypeStr;
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        if (params.solver_type.contains("тестовая", Qt::CaseInsensitive)) {
            solverTypeStr = "Ступень 2: Тестовая задача (квадрат)";
            
            // For test task in square domain
            if (solveSuccessful && !results_square.solution.empty()) {
                // Get coordinates of max error if available
                std::pair<double, double> max_err_coords = {0.0, 0.0};
                double max_error_val = 0.0;
                
                if (!results_square.error.empty() && 
                    !results_square.x_coords.empty() && 
                    !results_square.y_coords.empty()) {
                    
                    max_err_coords = find_max_abs_value_coords(
                        results_square.error, 
                        results_square.x_coords, 
                        results_square.y_coords
                    );
                    
                    auto it = std::max_element(
                        results_square.error.begin(), 
                        results_square.error.end(),
                        [](double a, double b) { return std::abs(a) < std::abs(b); }
                    );
                    
                    if (it != results_square.error.end()) {
                        max_error_val = *it;
                    }
                }
                
                helpTab->updateTestTaskInfo(
                    params.n_internal, params.m_internal,
                    "Метод сопряженных градиентов (CG)",
                    build_method_parameters_string(params, solverTypeStr),
                    params.eps_precision, params.max_iterations,
                    results_square.iterations, results_square.precision,
                    results_square.residual_norm, "Max-норма (L∞)",
                    results_square.error_norm,
                    max_err_coords.first, max_err_coords.second,
                    "Нулевое начальное приближение",
                    results_square.initial_residual_norm
                );
            }
            
        } else {
            solverTypeStr = "Ступень 2: Основная задача (квадрат)";
            
            // For main task in square domain
            if (solveSuccessful && !results_square.solution.empty()) {
                // Check if we have refined grid results
                bool hasRefinedResults = params.use_refined_grid && 
                                        !results_square.refined_grid_solution.empty();
                
                // Find coordinates of max deviation between main and refined grid solutions
                std::pair<double, double> max_diff_coords = {0.0, 0.0};
                if (hasRefinedResults && !results_square.solution_refined_diff.empty()) {
                    // We need to find which main grid point corresponds to the maximum difference
                    int max_diff_idx = -1;
                    double max_diff_val = 0.0;
                    
                    for (size_t i = 0; i < results_square.solution_refined_diff.size(); ++i) {
                        if (max_diff_idx == -1 || std::abs(results_square.solution_refined_diff[i]) > max_diff_val) {
                            max_diff_val = std::abs(results_square.solution_refined_diff[i]);
                            max_diff_idx = static_cast<int>(i);
                        }
                    }
                    
                    // If we found a valid maximum difference point
                    if (max_diff_idx >= 0 && max_diff_idx < static_cast<int>(results_square.x_coords.size()) &&
                        max_diff_idx < static_cast<int>(results_square.y_coords.size())) {
                        max_diff_coords.first = results_square.x_coords[max_diff_idx];
                        max_diff_coords.second = results_square.y_coords[max_diff_idx];
                    }
                }
                
                helpTab->updateMainTaskInfo(
                    params.n_internal, params.m_internal,
                    "Метод сопряженных градиентов (CG)",
                    build_method_parameters_string(params, solverTypeStr),
                    params.eps_precision, params.max_iterations,
                    results_square.iterations, results_square.precision,
                    results_square.residual_norm, "Max-норма (L∞)",
                    "Нулевое начальное приближение",
                    results_square.initial_residual_norm,
                    
                    // Refined grid parameters
                    hasRefinedResults ? "Метод сопряженных градиентов (CG) на подробной сетке" : "N/A",
                    hasRefinedResults ? "Подробная сетка" : "N/A",
                    params.eps_precision, params.max_iterations * 2, // Doubled for refined grid
                    results_square.refined_grid_iterations, 
                    hasRefinedResults ? results_square.precision : 0.0,
                    results_square.refined_grid_residual_norm, 
                    "Max-норма (L∞)",
                    results_square.refined_grid_error,
                    max_diff_coords.first, max_diff_coords.second,
                    "Нулевое начальное приближение",
                    results_square.refined_grid_initial_residual_norm
                );
            }
        }
    } else {
        solverTypeStr = "Ступень 3: G-образная область";
        
        // For G-shaped domain task
        if (solveSuccessful && !results.solution.empty()) {
            // Get coordinates of max error if available
            std::pair<double, double> max_err_coords = {0.0, 0.0};
            double max_error_val = 0.0;
            
            if (!results.error.empty() && 
                !results.x_coords.empty() && 
                !results.y_coords.empty()) {
                
                max_err_coords = find_max_abs_value_coords(
                    results.error, 
                    results.x_coords, 
                    results.y_coords
                );
                
                auto it = std::max_element(
                    results.error.begin(), 
                    results.error.end(),
                    [](double a, double b) { return std::abs(a) < std::abs(b); }
                );
                
                if (it != results.error.end()) {
                    max_error_val = *it;
                }
            }
            
            helpTab->updateShapedTestTaskInfo(
                params.n_internal, params.m_internal,
                "Метод сопряженных градиентов (CG)",
                build_method_parameters_string(params, solverTypeStr),
                params.eps_precision, params.max_iterations,
                results.iterations, results.precision,
                results.residual_norm, "Max-норма (L∞)",
                results.error_norm,
                max_err_coords.first, max_err_coords.second,
                "Нулевое начальное приближение",
                results.initial_residual_norm
            );
        }
    }
    
    // If we don't have results yet, clear help text
    if (!solveSuccessful) {
        helpTab->clearHelpText();
    }
}

void MainWindow::updateMainTaskInfo() {
    if (!helpTab) return;
    
    // This method should use helpTab instead of mainInfoTab
    // We'll simplify this method to just call updateHelpTabInfo
    // since that method already contains the necessary logic
    updateHelpTabInfo();
}

// Анонимное пространство имен для вспомогательных функций
namespace {

struct MaxDeviationInfo {
    double x = 0.0;
    double y = 0.0;
    double max_val = -std::numeric_limits<double>::infinity();
};

MaxDeviationInfo find_max_deviation_info(
    const std::vector<double>& values,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords)
{
    MaxDeviationInfo info;
    if (values.empty() || values.size() != x_coords.size() || values.size() != y_coords.size()) {
        return info; // Возвращаем значения по умолчанию, если данных нет или размеры не совпадают
    }

    double current_max_abs_val = -1.0; // Используем -1, так как абсолютное значение всегда >= 0
    int max_idx = -1;

    for (size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i]) > current_max_abs_val) {
            current_max_abs_val = std::abs(values[i]);
            max_idx = static_cast<int>(i);
        }
    }

    if (max_idx != -1) {
        info.x = x_coords[max_idx];
        info.y = y_coords[max_idx];
        info.max_val = values[max_idx]; // Возвращаем само значение, не абсолютное
    }
    return info;
}

} // конец анонимного пространства имен

void MainWindow::updateIterationInfo(int iteration, double precision, double residual, double error)
{
    // Обновляем информацию о текущей итерации на вкладке прогресса
    progressTab->updateIterationInfo(iteration, precision, residual, error);
}

void MainWindow::onSolverFinished()
{
    // Обновляем UI для состояния "Завершено"
    isSolving = false;
    solverTab->setSolveButtonEnabled(true);
    solverTab->setStopButtonEnabled(false);
    
    // Явно активируем ComboBox в табличной вкладке
    tableTab->setDataTypeComboEnabled(true);
    
    // Обновляем вкладку справки, если решение было успешным
    if (solveSuccessful) {
        updateHelpTabInfo(); // Ensure help tab is updated with final results
        tabWidget->setCurrentWidget(visualizationTab); // Optionally switch to visualization
    } else {
        helpTab->clearHelpText(); // Clear help text if solve was not successful or stopped
    }
}

void MainWindow::cleanupThread()
{
    if (solverThread) {
        if (solverThread->isRunning()) {
            solverThread->quit();
            solverThread->wait();
        }
        delete solverThread;
        solverThread = nullptr;
    }
    
    if (worker) {
        delete worker;
        worker = nullptr;
    }
}

void MainWindow::onSaveResultsButtonClicked()
{
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Ошибка", "Нет результатов для сохранения");
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this, "Сохранение результатов", "", "Текстовые файлы (*.txt)");
    if (fileName.isEmpty()) {
        return;
    }
    
    std::ofstream file(fileName.toStdString());
    if (!file.is_open()) {
        QMessageBox::critical(this, "Ошибка", "Не удалось открыть файл для записи");
        return;
    }
    
    file << "Результаты решения уравнения Пуассона\n";
    file << "=========================================\n";
    
    file << "Параметры сетки:\n";
    file << "Внутренние узлы: " << params.n_internal << " x " << params.m_internal << "\n";
    file << "Границы области: [" << params.a_bound << ", " << params.b_bound << "] x [" << params.c_bound << ", " << params.d_bound << "]\n\n";
    
    file << "Критерии останова:\n";
    if (params.use_precision) {
        file << "- Точность: " << params.eps_precision << "\n";
    }
    if (params.use_residual) {
        file << "- Невязка: " << params.eps_residual << "\n";
    }
    if (params.use_exact_error) {
        file << "- Ошибка: " << params.eps_exact_error << "\n";
    }
    if (params.use_max_iterations) {
        file << "- Макс. итераций: " << params.max_iterations << "\n";
    }
    file << "\n";
    
    file << "Результаты:\n";
    
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        file << "Тип решателя: Квадратная область\n";
        file << "Итерации: " << results_square.iterations << "\n";
        file << "Точность: " << results_square.precision << "\n";
        file << "Норма невязки: " << results_square.residual_norm << "\n";
        file << "Норма ошибки: " << results_square.error_norm << "\n";
        file << "Сходимость: " << (results_square.converged ? "Да" : "Нет") << "\n";
        file << "Причина остановки: " << results_square.stop_reason << "\n";
        
        if (params.use_refined_grid) {
            file << "Ошибка относительно решения на мелкой сетке: " << results_square.refined_grid_error << "\n";
        }
    } else {
        file << "Тип решателя: G-образная область\n";
        file << "Итерации: " << results.iterations << "\n";
        file << "Точность: " << results.precision << "\n";
        file << "Норма невязки: " << results.residual_norm << "\n";
        file << "Норма ошибки: " << results.error_norm << "\n";
        file << "Сходимость: " << (results.converged ? "Да" : "Нет") << "\n";
        file << "Причина остановки: " << results.stop_reason << "\n";
    }
    
    file.close();
    QMessageBox::information(this, "Успех", "Результаты сохранены в файл");
}

void MainWindow::onSaveMatrixButtonClicked()
{
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Ошибка", "Нет матрицы для сохранения");
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this, "Сохранение матрицы и вектора правой части", "", "Текстовые файлы (*.txt)");
    if (fileName.isEmpty()) {
        return;
    }
    
    bool result = false;
    
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        // Для квадратного решателя
        if (worker && worker->getSolverSquare()) {
            // Создаем новый солвер с теми же настройками, что и оригинальный
            auto solver_sq = std::make_unique<DirichletSolverSquare>(
                params.n_internal, params.m_internal,
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
            
            // Сохраняем матрицу и вектор правой части в файл
            result = solver_sq->saveMatrixAndRhsToFile(fileName.toStdString());
        }
    } else {
        // Для G-образного решателя
        if (worker && worker->getSolver()) {
            // Создаем новый солвер с теми же настройками, что и оригинальный
            auto solver = std::make_unique<DirichletSolver>(
                params.n_internal, params.m_internal,
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
            
            // Сохраняем матрицу и вектор правой части в файл
            result = solver->saveMatrixAndRhsToFile(fileName.toStdString());
        }
    }
    
    if (result) {
        QMessageBox::information(this, "Успех", "Матрица и вектор правой части сохранены в файл");
    } else {
        QMessageBox::critical(this, "Ошибка", "Не удалось сохранить матрицу и вектор правой части в файл");
    }
}

void MainWindow::onSaveVisualizationButtonClicked()
{
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Ошибка", "Нет визуализации для сохранения");
        return;
    }
    
    // Сохранение текущего вида графика в зависимости от активной вкладки
    int currentTabIndex = tabWidget->currentIndex();
    
    if (currentTabIndex == 2) { // 2D Визуализация
        QString fileName = QFileDialog::getSaveFileName(this, "Сохранение графика", "", "Изображения (*.png *.jpg)");
        if (fileName.isEmpty()) {
            return;
        }
        
        QPixmap pixmap = visualizationTab->grab();
        pixmap.save(fileName);
        
        QMessageBox::information(this, "Успех", "График сохранен в файл");
    } else if (currentTabIndex == 3) { // 3D Визуализация
        QMessageBox::information(this, "Информация", "Функция сохранения 3D визуализации в файл не реализована в данной версии.");
    } else {
        QMessageBox::warning(this, "Ошибка", "Текущая вкладка не содержит визуализации для сохранения");
    }
}

void MainWindow::onShowReportButtonClicked()
{
    QMessageBox::information(this, "Информация", "Функция показа отчета не реализована в данной версии.");
}

void MainWindow::onTabChanged(int index)
{
    // Выполняем действия при переключении вкладок
    if (index == 3) { // 3D Визуализация
        // Если решение успешно и мы переключились на вкладку 3D визуализации
        if (solveSuccessful) {
            // Обновляем 3D визуализацию, если необходимо (можно убрать, если это делается автоматически)
        }
    }
}

void MainWindow::onChartTypeChanged(int index)
{
    // Реагируем на изменение типа графика в 2D визуализации
    if (!solveSuccessful) {
        return;
    }
    
    // Обновляем график с нужным типом данных
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        // Для квадратного решателя
        if (index == 0) { // Решение
            visualizationTab->updateChart(
                results_square.solution,
                results_square.x_coords,
                results_square.y_coords,
                results_square.true_solution,
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
        } else if (index == 1) { // Ошибка
            visualizationTab->updateChart(
                results_square.error,
                results_square.x_coords,
                results_square.y_coords,
                std::vector<double>(),
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
        } else if (index == 2) { // Невязка
            visualizationTab->updateChart(
                results_square.residual,
                results_square.x_coords,
                results_square.y_coords,
                std::vector<double>(),
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
        }
    } else {
        // Для G-образного решателя
        if (index == 0) { // Решение
            visualizationTab->updateChart(
                results.solution,
                results.x_coords,
                results.y_coords,
                results.true_solution,  // изменено с true_solution_on_mesh
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
        } else if (index == 1) { // Ошибка
            visualizationTab->updateChart(
                results.error,
                results.x_coords,
                results.y_coords,
                std::vector<double>(),
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
        } else if (index == 2) { // Невязка
            visualizationTab->updateChart(
                results.residual,
                results.x_coords,
                results.y_coords,
                std::vector<double>(),
                params.a_bound, params.b_bound,
                params.c_bound, params.d_bound
            );
        }
    }
}

void MainWindow::onShowHeatMapClicked()
{
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Ошибка", "Нет данных для отображения тепловой карты");
        return;
    }
    
    QMessageBox::information(this, "Информация", "Функция отображения тепловой карты не реализована в данной версии.");
}

void MainWindow::onExportCSVRequested(int skipFactor)
{
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Ошибка", "Нет данных для экспорта");
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this, "Экспорт данных", "", "CSV файлы (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }
    
    QString csvData = generateCSVData(skipFactor);
    
    std::ofstream file(fileName.toStdString());
    if (!file.is_open()) {
        QMessageBox::critical(this, "Ошибка", "Не удалось открыть файл для записи");
        return;
    }
    
    file << csvData.toStdString();
    file.close();
    
    QMessageBox::information(this, "Успех", "Данные экспортированы в файл CSV");
}

QString MainWindow::generateCSVData(int skipFactor)
{
    if (params.solver_type.contains("основная", Qt::CaseInsensitive)) {
        return generateCSVForMainProblem(skipFactor);
    } else if (params.solver_type.contains("тестовая", Qt::CaseInsensitive)) {
        return generateCSVForTestProblem(skipFactor);
    } else {
        return generateCSVForGShapeProblem(skipFactor);
    }
}

QString MainWindow::generateCSVForTestProblem(int skipFactor)
{
    if (!solveSuccessful || results_square.solution.empty()) {
        return "";
    }
    
    std::stringstream ss;
    ss << "X,Y,Numerical Solution";
    
    bool hasTrueSolution = !results_square.true_solution.empty();
    bool hasError = !results_square.error.empty();
    
    if (hasTrueSolution) {
        ss << ",True Solution";
    }
    if (hasError) {
        ss << ",Error";
    }
    ss << "\n";
    
    const auto& solution = results_square.solution;
    const auto& x_coords = results_square.x_coords;
    const auto& y_coords = results_square.y_coords;
    const auto& true_sol = results_square.true_solution;
    const auto& error = results_square.error;
    
    for (size_t i = 0; i < solution.size(); i += skipFactor) {
        if (i < x_coords.size() && i < y_coords.size() && i < solution.size()) {
            ss << x_coords[i] << "," << y_coords[i] << "," 
               << solution[i];
            
            if (hasTrueSolution && i < true_sol.size()) {
                ss << "," << true_sol[i];
            } else if (hasTrueSolution) {
                ss << ",";
            }
            
            if (hasError && i < error.size()) {
                ss << "," << error[i];
            } else if (hasError) {
                ss << ",";
            }
            
            ss << "\n";
        }
    }
    
    // Добавляем секцию с информацией о типах данных для лучшей обработки в таблице
    ss << "\n# SECTIONS\n";
    ss << "# NUMERICAL_SOLUTION: Численное решение\n";
    if (hasTrueSolution) {
        ss << "# TRUE_SOLUTION: Точное решение\n";
    }
    if (hasError) {
        ss << "# ERROR: Ошибка\n";
    }
    
    return QString::fromStdString(ss.str());
}

QString MainWindow::generateCSVForMainProblem(int skipFactor)
{
    if (!solveSuccessful || results_square.solution.empty()) {
        return "";
    }
    
    std::stringstream ss;
    ss << "X,Y,Numerical Solution";
    
    bool hasTrueSolution = !results_square.true_solution.empty();
    bool hasError = !results_square.error.empty();
    bool hasRefinedGrid = !results_square.refined_grid_solution.empty();
    
    if (hasTrueSolution) {
        ss << ",True Solution";
    }
    if (hasError) {
        ss << ",Error";
    }
    ss << "\n";
    
    const auto& solution = results_square.solution;
    const auto& x_coords = results_square.x_coords;
    const auto& y_coords = results_square.y_coords;
    const auto& true_sol = results_square.true_solution;
    const auto& error = results_square.error;
    
    for (size_t i = 0; i < solution.size(); i += skipFactor) {
        if (i < x_coords.size() && i < y_coords.size() && i < solution.size()) {
            ss << x_coords[i] << "," << y_coords[i] << "," 
               << solution[i];
            
            if (hasTrueSolution && i < true_sol.size()) {
                ss << "," << true_sol[i];
            } else if (hasTrueSolution) {
                ss << ",";
            }
            
            if (hasError && i < error.size()) {
                ss << "," << error[i];
            } else if (hasError) {
                ss << ",";
            }
            
            ss << "\n";
        }
    }
    
    // Если есть решение на уточненной сетке, добавляем его как отдельную секцию
    if (hasRefinedGrid) {
        ss << "\n# REFINED_GRID_SOLUTION\n";
        ss << "X,Y,Refined Grid Solution\n";
        
        const auto& refined_solution = results_square.refined_grid_solution;
        const auto& refined_x_coords = results_square.refined_grid_x_coords;
        const auto& refined_y_coords = results_square.refined_grid_y_coords;
        
        for (size_t i = 0; i < refined_solution.size(); i += skipFactor) {
            if (i < refined_x_coords.size() && i < refined_y_coords.size() && i < refined_solution.size()) {
                ss << refined_x_coords[i] << "," << refined_y_coords[i] << "," 
                   << refined_solution[i] << "\n";
            }
        }
    }
    
    // Добавляем секцию с информацией о типах данных для лучшей обработки в таблице
    ss << "\n# SECTIONS\n";
    ss << "# NUMERICAL_SOLUTION: Численное решение\n";
    if (hasTrueSolution) {
        ss << "# TRUE_SOLUTION: Точное решение\n";
    }
    if (hasError) {
        ss << "# ERROR: Ошибка\n";
    }
    if (hasRefinedGrid) {
        ss << "# REFINED_GRID: Решение на уточненной сетке\n";
    }
    
    return QString::fromStdString(ss.str());
}

QString MainWindow::generateCSVForGShapeProblem(int skipFactor)
{
    if (!solveSuccessful || results.solution.empty()) {
        return "";
    }
    
    std::stringstream ss;
    ss << "X,Y,Numerical Solution";
    
    bool hasTrueSolution = !results.true_solution.empty();
    bool hasError = !results.error.empty();
    
    if (hasTrueSolution) {
        ss << ",True Solution";
    }
    if (hasError) {
        ss << ",Error";
    }
    ss << "\n";
    
    const auto& solution = results.solution;
    const auto& x_coords = results.x_coords;
    const auto& y_coords = results.y_coords;
    const auto& true_sol = results.true_solution;
    const auto& error = results.error;
    
    for (size_t i = 0; i < solution.size(); i += skipFactor) {
        if (i < x_coords.size() && i < y_coords.size() && i < solution.size()) {
            ss << x_coords[i] << "," << y_coords[i] << "," 
               << solution[i];
            
            if (hasTrueSolution && i < true_sol.size()) {
                ss << "," << true_sol[i];
            } else if (hasTrueSolution) {
                ss << ",";
            }
            
            if (hasError && i < error.size()) {
                ss << "," << error[i];
            } else if (hasError) {
                ss << ",";
            }
            
            ss << "\n";
        }
    }
    
    // Добавляем секцию с информацией о типах данных для лучшей обработки в таблице
    ss << "\n# SECTIONS\n";
    ss << "# NUMERICAL_SOLUTION: Численное решение\n";
    if (hasTrueSolution) {
        ss << "# TRUE_SOLUTION: Точное решение\n";
    }
    if (hasError) {
        ss << "# ERROR: Ошибка\n";
    }
    
    return QString::fromStdString(ss.str());
}