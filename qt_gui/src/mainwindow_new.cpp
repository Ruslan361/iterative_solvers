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
#include <QThread>
#include <QVariantMap>
#include <QAtomicInt> // For thread-safe stop flag
#include <QDebug> // For qDebug()
#include <Kokkos_Core.hpp> // For Kokkos::initialize and is_initialized
#include "dirichlet_solver.hpp" // For ResultsIO and DirichletSolverSquare

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
    if (value == -1.0 || std::isnan(value) || std::isinf(value)) { // Added isnan and isinf checks
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
SolverWorker::SolverWorker(std::unique_ptr<DirichletSolver> solver_arg, QObject *parent)
    : QObject(parent), m_solver(std::move(solver_arg)), m_solver_sq(nullptr), m_is_square_solver(false)
{
}

SolverWorker::SolverWorker(std::unique_ptr<DirichletSolverSquare> solver_sq_arg, QObject *parent)
    : QObject(parent), m_solver(nullptr), m_solver_sq(std::move(solver_sq_arg)), m_is_square_solver(true)
{
}

void SolverWorker::process()
{
    try {
        emit solverStageUpdate("Инициализация решателя...");
        if (m_is_square_solver && m_solver_sq) {
            // Настройка callback для отслеживания итераций
            auto iter_callback = [this](int iter, double precision, double residual, double error) {
                if (QThread::currentThread()->isInterruptionRequested()){ // Check for interruption before emitting
                    return false; // Stop solver's internal loop
                }
                emit iterationUpdate(iter, precision, residual, error);
                return !QThread::currentThread()->isInterruptionRequested(); // Continue if not interrupted
            };
            m_solver_sq->setIterationCallback(iter_callback);

            // Настройка callback для стадий решения (должна быть реализована в DirichletSolverSquare)
            m_solver_sq->setProgressCallback([this](const std::string& msg){
                if (QThread::currentThread()->isInterruptionRequested()){
                     // Optionally handle interruption here if progress callback can stop things
                    return;
                }
                emit solverStageUpdate(QString::fromStdString(msg));
            });
            
            emit solverStageUpdate("Запуск решения для квадратной области...");
            SquareSolverResults results_sq_data = m_solver_sq->solve();
            if (QThread::currentThread()->isInterruptionRequested()) {
                 emit solverStageUpdate("Решение для квадратной области прервано.");
                 emit finished();
                 return;
            }
            emit solverStageUpdate("Решение для квадратной области завершено.");
            emit resultReadySquare(results_sq_data);
        } else if (m_solver) {
            // Настройка callback для отслеживания итераций
            auto iter_callback = [this](int iter, double precision, double residual, double error) {
                if (QThread::currentThread()->isInterruptionRequested()){
                    return false; // Stop solver's internal loop
                }
                emit iterationUpdate(iter, precision, residual, error);
                return !QThread::currentThread()->isInterruptionRequested(); // Continue if not interrupted
            };
            m_solver->setIterationCallback(iter_callback);

            // Настройка callback для стадий решения (должна быть реализована в DirichletSolver)
            m_solver->setProgressCallback([this](const std::string& msg){
                 if (QThread::currentThread()->isInterruptionRequested()){
                    // Optionally handle interruption here
                    return;
                }
                emit solverStageUpdate(QString::fromStdString(msg));
            });
            
            emit solverStageUpdate("Запуск решения для G-образной области...");
            SolverResults results_data = m_solver->solve();
            if (QThread::currentThread()->isInterruptionRequested()) {
                emit solverStageUpdate("Решение для G-образной области прервано.");
                emit finished();
                return;
            }
            emit solverStageUpdate("Решение для G-образной области завершено.");
            emit resultReady(results_data);
        }
    } catch (const std::exception& e) {
        emit solverStageUpdate(QString("Ошибка в процессе решения: %1").arg(e.what()));
        std::cerr << "Ошибка в процессе решения: " << e.what() << std::endl;
    }
    
    emit finished();
}

// Implementation of PlottingWorker constructor
PlottingWorker::PlottingWorker(const PlottingWorker::MainWindowSolverParams& worker_params_arg,
                               const SquareSolverResults& results_square_arg,
                               const SolverResults& results_arg,
                               QObject *parent_arg)
    : QObject(parent_arg),
      params(worker_params_arg),
      results_square(results_square_arg),
      results(results_arg),
      stop_requested(0)
{
}

// Реализация PlottingWorker
void PlottingWorker::process() {
    emit progressUpdate("Начало подготовки данных для графиков...");

    if (stop_requested.loadAcquire()) { emit finished(); return; } // Changed to loadAcquire

    // 2D Plotting
    emit progressUpdate("Подготовка данных для 2D графика...");
    QVariantMap plot_2d_data;
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        plot_2d_data["solution"] = QVariant::fromValue(results_square.solution);
        plot_2d_data["x_coords"] = QVariant::fromValue(results_square.x_coords);
        plot_2d_data["y_coords"] = QVariant::fromValue(results_square.y_coords);
        plot_2d_data["true_solution"] = QVariant::fromValue(results_square.true_solution);
        plot_2d_data["error_data"] = QVariant::fromValue(results_square.error); // Added for chart type switching
        plot_2d_data["residual_data"] = QVariant::fromValue(results_square.residual); // Added for chart type switching
    } else {
        plot_2d_data["solution"] = QVariant::fromValue(results.solution);
        plot_2d_data["x_coords"] = QVariant::fromValue(results.x_coords);
        plot_2d_data["y_coords"] = QVariant::fromValue(results.y_coords);
        plot_2d_data["true_solution"] = QVariant::fromValue(results.true_solution);
        plot_2d_data["error_data"] = QVariant::fromValue(results.error); // Added
        plot_2d_data["residual_data"] = QVariant::fromValue(results.residual); // Added
    }
    plot_2d_data["a_bound"] = params.a_bound;
    plot_2d_data["b_bound"] = params.b_bound;
    plot_2d_data["c_bound"] = params.c_bound;
    plot_2d_data["d_bound"] = params.d_bound;
    plot_2d_data["solver_type"] = params.solver_type;


    emit progressUpdate("Данные для 2D графика подготовлены. Обновление...");
    if (stop_requested.loadAcquire()) { emit finished(); return; } // Changed to loadAcquire
    emit plotDataReady2D(plot_2d_data);
    // UI update for 2D plot will happen in MainWindow::handlePlotData2D
    emit progressUpdate("2D график должен быть обновлен.");

    if (stop_requested.loadAcquire()) { emit finished(); return; } // Changed to loadAcquire

    // 3D Plotting
    emit progressUpdate("Подготовка данных для 3D визуализации...");
    QVariantMap plot_3d_data;
    plot_3d_data["is_square_solver"] = params.solver_type.contains("ступень 2", Qt::CaseInsensitive);
    plot_3d_data["use_refined_grid"] = params.use_refined_grid;
    plot_3d_data["a_bound"] = params.a_bound;
    plot_3d_data["b_bound"] = params.b_bound;
    plot_3d_data["c_bound"] = params.c_bound;
    plot_3d_data["d_bound"] = params.d_bound;

    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        plot_3d_data["solution"] = QVariant::fromValue(results_square.solution);
        plot_3d_data["true_solution"] = QVariant::fromValue(results_square.true_solution);
        plot_3d_data["error"] = QVariant::fromValue(results_square.error);
        plot_3d_data["x_coords"] = QVariant::fromValue(results_square.x_coords);
        plot_3d_data["y_coords"] = QVariant::fromValue(results_square.y_coords);
        if (params.use_refined_grid && !results_square.refined_grid_solution.empty()) {
            plot_3d_data["refined_grid_solution"] = QVariant::fromValue(results_square.refined_grid_solution);
            plot_3d_data["solution_refined_diff"] = QVariant::fromValue(results_square.solution_refined_diff);
            plot_3d_data["refined_grid_x_coords"] = QVariant::fromValue(results_square.refined_grid_x_coords);
            plot_3d_data["refined_grid_y_coords"] = QVariant::fromValue(results_square.refined_grid_y_coords);
        }
    } else { // G-Shape
        plot_3d_data["solution"] = QVariant::fromValue(results.solution);
        plot_3d_data["true_solution"] = QVariant::fromValue(results.true_solution);
        plot_3d_data["error"] = QVariant::fromValue(results.error);
        plot_3d_data["x_coords"] = QVariant::fromValue(results.x_coords);
        plot_3d_data["y_coords"] = QVariant::fromValue(results.y_coords);
    }
    emit progressUpdate("Данные для 3D визуализации подготовлены. Обновление...");
    if (stop_requested.loadAcquire()) { emit finished(); return; } // Changed to loadAcquire
    emit plotDataReady3D(plot_3d_data);
    // UI update for 3D plot will happen in MainWindow::handlePlotData3D
    emit progressUpdate("3D визуализация должна быть обновлена.");

    emit progressUpdate("Построение графиков завершено.");
    emit finished();
}


// Реализация класса MainWindow
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , solverThread(nullptr) // Initialize pointers
    , worker(nullptr)
    , plottingWorker(nullptr)
    , plottingThread(nullptr)
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
    // Эти параметры могут быть загружены из настроек или установлены по умолчанию
    // Для примера, установим их здесь:
    params.n_internal = 30; // Примерное значение
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
    // Очищаем поток построения графиков
    cleanupPlottingThread();
    
    // Финализируем Kokkос, если мы его инициалировали
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
}

void MainWindow::onSolveButtonClicked()
{
    if (isSolving) { // If already solving, do nothing (or queue, but not requested)
        QMessageBox::information(this, "Информация", "Решение уже выполняется.");
        return;
    }

    // Stop and clean up previous plotting task if any
    if (plottingThread && plottingThread->isRunning()) {
        updateSolverProgress("Остановка предыдущего процесса построения графиков...");
        plottingWorker->requestStop();
        plottingThread->quit();
        plottingThread->wait(5000); // Wait max 5 seconds
        cleanupPlottingThread();
        updateSolverProgress("Предыдущий процесс построения графиков остановлен.");
    }
    cleanupPlottingThread(); // Ensure it's clean

    // Stop and clean up previous solver task if any (should not happen if isSolving is managed correctly)
    if (solverThread && solverThread->isRunning()) {
        updateSolverProgress("Остановка предыдущего процесса решения...");
        if(worker) { // Request solver to stop through its own mechanism if possible
            if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
                if (auto* s = worker->getSolverSquare()) s->requestStop();
            } else {
                if (auto* s = worker->getSolver()) s->requestStop();
            }
        }
        solverThread->requestInterruption(); // Request interruption
        solverThread->quit();
        solverThread->wait(5000); // Wait max 5 seconds
        cleanupThread();
        updateSolverProgress("Предыдущий процесс решения остановлен.");
    }
    cleanupThread(); // Ensure it's clean
    
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
    
    // Connect solver stage updates
    connect(worker, &SolverWorker::solverStageUpdate, this, &MainWindow::updateSolverProgress);

    connect(solverThread, &QThread::started, worker, &SolverWorker::process);
    connect(worker, &SolverWorker::finished, this, &MainWindow::onSolverFinished);
    // Corrected signal names for resultReady, resultReadySquare, and iterationUpdate
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
    if (!isSolving && !(plottingThread && plottingThread->isRunning())) {
        return;
    }
    
    updateSolverProgress("Получен запрос на остановку...");

    // Останавливаем решатель
    if (isSolving && worker) {
        updateSolverProgress("Остановка процесса решения...");
        if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
            if (auto* s = worker->getSolverSquare()) {
                s->requestStop(); // Internal stop mechanism of the solver
            }
        } else {
            if (auto* s = worker->getSolver()) {
                s->requestStop(); // Internal stop mechanism of the solver
            }
        }
        if (solverThread) {
            solverThread->requestInterruption(); // Interrupt the thread
        }
         // UI will be updated in onSolverFinished
    } else {
         updateSolverProgress("Процесс решения не активен или уже остановлен.");
    }

    // Останавливаем построение графиков, если оно идет
    if (plottingWorker && plottingThread && plottingThread->isRunning()) {
        updateSolverProgress("Остановка процесса построения графиков...");
        plottingWorker->requestStop();
        // plottingThread will quit and clean up via its finished signal
    } else {
        updateSolverProgress("Процесс построения графиков не активен.");
    }
    // Buttons will be re-enabled in onSolverFinished and onPlottingFinished
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
    solveSuccessful = true; // Mark solve as successful before starting plotting
    
    updateSolverProgress("Результаты решения (G-образная область) получены.");
    
    // Записываем в таблицу только если сетка не больше 50x50
    if (params.n_internal <= 50 && params.m_internal <= 50) {
        tableTab->setResultsData(
            res.solution,
            res.true_solution,
            res.error,
            res.x_coords,
            res.y_coords,
            false // G-образная область не является квадратной сеткой
        );
    } else {
        updateSolverProgress("Автоматическое создание таблицы пропущено: сетка больше 50x50.");
    }
    
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
    
    QString matrixInfo = QString("Размер системы (G-обр.): %1 узлов")
                        .arg(res.solution.size());
    solverTab->updateMatrixInfo(matrixInfo);

    // Запускаем построение графиков в отдельнем потоке
    startPlottingTasks();
}

void MainWindow::handleResultsSquare(const SquareSolverResults& results_sq_data)
{
    // Сохраняем результаты
    this->results_square = results_sq_data;
    solveSuccessful = true; // Mark solve as successful before starting plotting

    updateSolverProgress("Результаты решения (квадратная область) получены.");

    // Записываем в таблицу только если сетка не больше 50x50
    if (params.n_internal <= 50 && params.m_internal <= 50) {
        tableTab->setResultsData(
            this->results_square.solution,
            this->results_square.true_solution,
            this->results_square.error,
            this->results_square.x_coords,
            this->results_square.y_coords,
            true // квадратная сетка
        );
        if (params.use_refined_grid && !this->results_square.refined_grid_solution.empty()) {
            tableTab->setRefinedGridData(
                this->results_square.refined_grid_solution,
                this->results_square.refined_grid_x_coords, 
                this->results_square.refined_grid_y_coords
            );
        }
    } else {
        updateSolverProgress("Автоматическое создание таблицы пропущено: сетка больше 50x50.");
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
    
    int matrixSize = (params.n_internal -1) * (params.m_internal -1); 
    QString matrixInfo = QString("Размер системы (квадрат): %1 узлов")
                        .arg(matrixSize);
    solverTab->updateMatrixInfo(matrixInfo);

    // Запускаем построение графиков в отдельном потоке
    startPlottingTasks();
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
    // Также добавляем в общий лог прогресса, если нужно (опционально)
    // updateSolverProgress(QString("Итерация: %1, Точность: %L2, Невязка: %L3, Ошибка: %L4")
    //                      .arg(iteration).arg(precision, 0, 'e', 2).arg(residual, 0, 'e', 2).arg(error, 0, 'e', 2));
}

void MainWindow::onSolverFinished()
{
    // Этот слот теперь в основном управляет состоянием после завершения потока решателя.
    // Графики и отчеты обрабатываются после получения результатов.
    isSolving = false; 
    // Кнопки будут управляться более гранулярно, но можно оставить базовое управление здесь
    // solverTab->setSolveButtonEnabled(true); // Будет управляться в onPlottingFinished или если нет solveSuccessful
    // solverTab->setStopButtonEnabled(false);


    if (!solveSuccessful) { // Если решение не успешно (прервано или ошибка)
        updateSolverProgress("Процесс решения завершен (неуспешно или прерван).");
        solverTab->setSolveButtonEnabled(true); // Разрешить новый запуск
        solverTab->setStopButtonEnabled(false);
        helpTab->clearHelpText();
    } else {
        // Если успешно, то startPlottingTasks уже был вызван из handleResults/handleResultsSquare
        // Ничего специфичного здесь делать не нужно, кроме логгирования, если только
        updateSolverProgress("Процесс решения завершен успешно. Ожидание построения графиков...");
    }
    // Очистка потока решателя происходит в connect lambda в onSolveButtonClicked
}


void MainWindow::cleanupThread() // Renamed for clarity if it's specific to solver thread
{
    if (solverThread) {
        if (solverThread->isRunning()) {
            solverThread->requestInterruption(); // Ensure interruption is requested
            solverThread->quit();
            if (!solverThread->wait(3000)) { // Wait a bit
                updateSolverProgress("Поток решателя не завершился штатно, возможно потребуется принудительное завершение.");
                // solverThread->terminate(); // крайняя мера
            }
        }
        delete solverThread;
        solverThread = nullptr;
    }
    
    if (worker) {
        delete worker; // worker удаляется после того как поток завершен
        worker = nullptr;
    }
}

// Новые слоты и функции для управления потоком построения графиков

void MainWindow::updateSolverProgress(const QString& status) {
    // progressTab has no addProgressStep method, so we use qDebug
    qDebug() << status;
}

void MainWindow::startPlottingTasks() {
    if (!solveSuccessful) {
        updateSolverProgress("Построение графиков отменено: решение не было успешным.");
        solverTab->setSolveButtonEnabled(true); // Разрешить новый запуск
        solverTab->setStopButtonEnabled(false);
        return;
    }

    if (plottingThread && plottingThread->isRunning()) {
        updateSolverProgress("Остановка предыдущего процесса построения графиков перед запуском нового...");
        plottingWorker->requestStop();
        plottingThread->quit();
        plottingThread->wait(5000);
        cleanupPlottingThread();
    }
    cleanupPlottingThread(); // Double check

    updateSolverProgress("Запуск процесса построения графиков...");

    // Prepare worker params
    PlottingWorker::MainWindowSolverParams workerParams = params;
    plottingWorker = new PlottingWorker(workerParams, results_square, results);
    plottingThread = new QThread();
    plottingWorker->moveToThread(plottingThread);

    connect(plottingThread, &QThread::started, plottingWorker, &PlottingWorker::process);
    connect(plottingWorker, &PlottingWorker::progressUpdate, this, &MainWindow::updateSolverProgress);
    connect(plottingWorker, &PlottingWorker::plotDataReady2D, this, &MainWindow::handlePlotData2D);
    connect(plottingWorker, &PlottingWorker::plotDataReady3D, this, &MainWindow::handlePlotData3D);
    connect(plottingWorker, &PlottingWorker::finished, this, &MainWindow::onPlottingFinished);
    connect(plottingThread, &QThread::finished, this, &MainWindow::cleanupPlottingThread); // Связываем finished потока с его очисткой

    plottingThread->start();
    solverTab->setSolveButtonEnabled(false); // Блокируем кнопку "Решить" пока идет построение
    solverTab->setStopButtonEnabled(true); // Разрешаем остановку
}

void MainWindow::handlePlotData2D(const QVariantMap& data) {
    updateSolverProgress("Обновление 2D визуализации...");
    // Извлекаем данные из QVariantMap и обновляем visualizationTab
    // Это должно быть безопасным, так как выполняется в основном потоке
    QString solverType = data.value("solver_type").toString();
    std::vector<double> sol = data.value("solution").value<std::vector<double>>();
    std::vector<double> x_c = data.value("x_coords").value<std::vector<double>>();
    std::vector<double> y_c = data.value("y_coords").value<std::vector<double>>();
    std::vector<double> true_sol = data.value("true_solution").value<std::vector<double>>();
    // Данные для переключения типов графиков
    // std::vector<double> error_data = data.value("error_data").value<std::vector<double>>();
    // std::vector<double> residual_data = data.value("residual_data").value<std::vector<double>>();

    double a = data.value("a_bound").toDouble();
    double b = data.value("b_bound").toDouble();
    double c = data.value("c_bound").toDouble();
    double d = data.value("d_bound").toDouble();

    visualizationTab->updateChart(sol, x_c, y_c, true_sol, a, b, c, d);
    visualizationTab->setSolveSuccessful(true); // Убедимся, что вкладка знает об успехе
    updateSolverProgress("2D визуализация обновлена.");
}

void MainWindow::handlePlotData3D(const QVariantMap& data) {
    updateSolverProgress("Обновление 3D визуализации...");
    bool is_sq_solver = data.value("is_square_solver").toBool();
    bool use_ref_grid = data.value("use_refined_grid").toBool();
    // extract data
    std::vector<double> sol = data.value("solution").value<std::vector<double>>();
    std::vector<double> true_sol = data.value("true_solution").value<std::vector<double>>();
    std::vector<double> err = data.value("error").value<std::vector<double>>();
    std::vector<double> x_c = data.value("x_coords").value<std::vector<double>>();
    std::vector<double> y_c = data.value("y_coords").value<std::vector<double>>();
    double a = data.value("a_bound").toDouble();
    double b = data.value("b_bound").toDouble();
    double c = data.value("c_bound").toDouble();
    double d = data.value("d_bound").toDouble();

    // decimate for G-shaped if too many points
    if (!is_sq_solver) {
        size_t N = sol.size();
        if (N > 10000) {
            size_t factor = static_cast<size_t>(std::sqrt(static_cast<double>(N) / 10000.0)) + 1;
            std::vector<double> sol2, true2, err2, x2, y2;
            for (size_t i = 0; i < N; i += factor) {
                sol2.push_back(sol[i]);
                true2.push_back(true_sol.empty() ? 0.0 : true_sol[i]);
                err2.push_back(err.empty() ? 0.0 : err[i]);
                x2.push_back(x_c[i]);
                y2.push_back(y_c[i]);
            }
            sol.swap(sol); sol.swap(sol2); sol2.clear(); // Actually replace sol, see below
            // Actually assign sol=sol2 etc.
            sol = sol2; true_sol = true2; err = err2; x_c = x2; y_c = y2;
        }
    }
    
    if (is_sq_solver && use_ref_grid && data.contains("refined_grid_solution")) {
        std::vector<double> ref_sol = data.value("refined_grid_solution").value<std::vector<double>>();
        std::vector<double> sol_ref_diff = data.value("solution_refined_diff").value<std::vector<double>>();
        std::vector<double> ref_x_c = data.value("refined_grid_x_coords").value<std::vector<double>>();
        std::vector<double> ref_y_c = data.value("refined_grid_y_coords").value<std::vector<double>>();
        if (!ref_sol.empty()){
            visualization3DTab->createOrUpdateRefinedGridSurfaces(
                sol, ref_sol, sol_ref_diff, x_c, y_c, ref_x_c, ref_y_c, a, b, c, d
            );
        } else { // Fallback if refined solution is empty for some reason
             visualization3DTab->createOrUpdate3DSurfaces(sol, true_sol, err, x_c, y_c, a, b, c, d, is_sq_solver, false);
        }
    } else {
        visualization3DTab->createOrUpdate3DSurfaces(sol, true_sol, err, x_c, y_c, a, b, c, d, is_sq_solver, false);
    }
    // show initial zero plane for G-shaped
    if (!is_sq_solver) {
        visualization3DTab->onInitialApproximationVisibilityChanged(true);
    }
    visualization3DTab->setSolveSuccessful(true);
    updateSolverProgress("3D визуализация обновлена.");
}

void MainWindow::onPlottingFinished() {
    updateSolverProgress("Все процессы построения графиков завершены.");
    // Очистка потока построения графиков будет выполнена через cleanupPlottingThread,
    // который подключен к QThread::finished.

    // Теперь, когда графики построены, генерируем отчет
    if (solveSuccessful) {
        updateSolverProgress("Составление отчета...");
        updateHelpTabInfo(); // Generate report
        updateSolverProgress("Отчет сформирован и доступен на вкладке 'Справка'.");
        tabWidget->setCurrentWidget(visualizationTab); // Optionally switch to visualization
    } else {
        helpTab->clearHelpText();
    }
    
    // Разблокируем кнопку "Решить" и блокируем "Стоп" только если все завершено
    solverTab->setSolveButtonEnabled(true);
    solverTab->setStopButtonEnabled(false);
    tableTab->setDataTypeComboEnabled(true); // Активируем ComboBox в таблице
}

void MainWindow::cleanupPlottingThread() {
    if (plottingThread) {
        if (plottingThread->isRunning()) {
            // Поток должен был уже завершиться сам, но на всякий случай
            plottingWorker->requestStop(); // Попросим воркер остановиться
            plottingThread->quit();
            if (!plottingThread->wait(3000)) {
                 updateSolverProgress("Поток построения графиков не завершился штатно.");
                // plottingThread->terminate(); // крайняя мера
            }
        }
        delete plottingThread;
        plottingThread = nullptr;
    }
    if (plottingWorker) {
        delete plottingWorker;
        plottingWorker = nullptr;
    }
     updateSolverProgress("Ресурсы потока построения графиков очищены.");
}

// Слоты для сохранения и навигации
void MainWindow::onSaveResultsButtonClicked() {
    // Сохраняем результаты в CSV
    onExportCSVRequested(tableTab->getSkipFactor());
}

void MainWindow::onSaveMatrixButtonClicked() {
    QString baseName = QFileDialog::getSaveFileName(this, "Сохранить CSV матрицы и RHS (два файла)", QString(), "CSV файлы (*.csv)");
    if (baseName.isEmpty()) return;
    // Generate CSV contents for matrix and RHS (handles both square and G-shaped)
    QString matCsv = generateMatrixCSV();
    QString rhsCsv = generateRHSCSV();
    // Determine file paths
    QString matName = baseName + "_A.csv";
    QString rhsName = baseName + "_b.csv";
    // Save matrix CSV
    QFile matFile(matName);
    if (!matCsv.isEmpty() && matFile.open(QIODevice::WriteOnly)) {
        matFile.write(matCsv.toUtf8());
        matFile.close();
        updateSolverProgress(QString("Матрица сохранена: %1").arg(matName));
    } else {
        QMessageBox::warning(this, "Ошибка", "Не удалось сохранить матрицу.");
    }
    // Save RHS CSV
    QFile rhsFile(rhsName);
    if (!rhsCsv.isEmpty() && rhsFile.open(QIODevice::WriteOnly)) {
        rhsFile.write(rhsCsv.toUtf8());
        rhsFile.close();
        updateSolverProgress(QString("RHS сохранен: %1").arg(rhsName));
    } else {
        QMessageBox::warning(this, "Ошибка", "Не удалось сохранить RHS.");
    }
}

void MainWindow::onSaveVisualizationButtonClicked() {
    // Сохранение визуализации пока не реализовано
    QMessageBox::information(this, "Информация", "Сохранение визуализации не реализовано.");
}

void MainWindow::onShowReportButtonClicked() {
    // Переключаемся на вкладку "Справка"
    tabWidget->setCurrentWidget(helpTab);
}

void MainWindow::onTabChanged(int index) {
    // При смене вкладки можно выполнить дополнительные действия
    Q_UNUSED(index);
}

void MainWindow::onShowHeatMapClicked() {
    // Показ/скрытие тепловой карты
    visualization3DTab->toggleHeatMap();
}

void MainWindow::onExportCSVRequested(int skipFactor) {
    QString csv = generateCSVData(skipFactor);
    QString fileName = QFileDialog::getSaveFileName(this, "Сохранить CSV", QString(), "CSV файлы (*.csv)");
    if (!fileName.isEmpty()) {
        QFile file(fileName);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(csv.toUtf8());
            file.close();
            updateSolverProgress(QString("CSV сохранен: %1").arg(fileName));
        } else {
            QMessageBox::warning(this, "Ошибка", "Не удалось сохранить CSV.");
        }
    }
}

// Generate CSV based on solver type
QString MainWindow::generateCSVData(int skipFactor) {
    if (params.solver_type.contains("тестовая", Qt::CaseInsensitive)) {
        return generateCSVForTestProblem(skipFactor);
    } else if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        return generateCSVForMainProblem(skipFactor);
    } else {
        return generateCSVForGShapeProblem(skipFactor);
    }
}

// CSV for test problem (square domain with exact solution)
QString MainWindow::generateCSVForTestProblem(int skipFactor) {
    QString csv;
    QTextStream out(&csv);
    out << "x,y,solution,true_solution,error\n";

    // Check if primary data vectors are present and have consistent sizes.
    // If not, return CSV with headers only to prevent crashes.
    if (results_square.x_coords.empty() ||
        results_square.y_coords.size() != results_square.x_coords.size() ||
        results_square.solution.size() != results_square.x_coords.size()) {
        // Optionally, log this situation for debugging.
        // qDebug() << "CSV generation skipped for test problem: core data missing or mismatched.";
        return csv;
    }

    size_t n = results_square.x_coords.size(); // Use size_t for clarity and correctness with std::vector::size()
    skipFactor = qMax(1, skipFactor);

    for (size_t i = 0; i < n; i += skipFactor) { // Use size_t for loop index
        out << results_square.x_coords[i] << ","
            << results_square.y_coords[i] << ","
            << results_square.solution[i] << ",";

        // Check for true_solution data availability for the current index
        if (i < results_square.true_solution.size()) {
            out << results_square.true_solution[i];
        } else {
            // Output nothing (empty field) if data is not available for this index
        }
        out << ",";

        // Check for error data availability for the current index
        if (i < results_square.error.size()) {
            out << results_square.error[i];
        } else {
            // Output nothing (empty field) if data is not available for this index
        }
        out << "\n";
    }
    return csv;
}

// CSV for main square problem (no exact solution)
QString MainWindow::generateCSVForMainProblem(int skipFactor) {
    QString csv;
    QTextStream out(&csv);
    out << "x,y,solution,error\n";
    int n = results_square.x_coords.size();
    skipFactor = qMax(1, skipFactor);
    for (int i = 0; i < n; i += skipFactor) {
        out << results_square.x_coords[i] << ","
            << results_square.y_coords[i] << ","
            << results_square.solution[i] << ","
            << results_square.error[i] << "\n";
    }
    return csv;
}

// CSV for G-shaped domain problem
QString MainWindow::generateCSVForGShapeProblem(int skipFactor) {
    QString csv;
    QTextStream out(&csv);
    out << "x,y,solution,true_solution,error\n";
    int n = results.x_coords.size();
    skipFactor = qMax(1, skipFactor);
    for (int i = 0; i < n; i += skipFactor) {
        out << results.x_coords[i] << ","
            << results.y_coords[i] << ","
            << results.solution[i] << ","
            << results.true_solution[i] << ","
            << results.error[i] << "\n";
    }
    return csv;
}

// Export matrix A in CSV (row,col,value)
QString MainWindow::generateMatrixCSV() {
    QString csv;
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        // Square domain export with proper RHS for test/main tasks
        // Build solver with correct functions
        double (*f_func)(double, double) = nullptr;
        double (*exact_solution_func)(double, double) = nullptr;
        double (*mu1_func)(double, double) = nullptr;
        double (*mu2_func)(double, double) = nullptr;
        double (*mu3_func)(double, double) = nullptr;
        double (*mu4_func)(double, double) = nullptr;
        // Select functions based on task type
        if (params.solver_type.contains("тестовая", Qt::CaseInsensitive)) {
            f_func = function2_square;
            exact_solution_func = solution2_square;
            mu1_func = mu1_square_solution2;
            mu2_func = mu2_square_solution2;
            mu3_func = mu3_square_solution2;
            mu4_func = mu4_square_solution2;
        } else {
            f_func = custom_function_square;
            mu1_func = mu1_square;
            mu2_func = mu2_square;
            mu3_func = mu3_square;
            mu4_func = mu4_square;
        }
        // Construct solver
        DirichletSolverSquare tmp = (exact_solution_func)
            ? DirichletSolverSquare(params.n_internal, params.m_internal,
                                   params.a_bound, params.b_bound,
                                   params.c_bound, params.d_bound,
                                   f_func, exact_solution_func)
            : DirichletSolverSquare(params.n_internal, params.m_internal,
                                   params.a_bound, params.b_bound,
                                   params.c_bound, params.d_bound,
                                   f_func, mu1_func, mu2_func, mu3_func, mu4_func);
        const GridSystemSquare* grid = tmp.getGridSystem();
        const auto& A = grid->get_matrix();
        auto row_map = Kokkos::create_mirror_view(A.graph.row_map);
        auto entries = Kokkos::create_mirror_view(A.graph.entries);
        auto values = Kokkos::create_mirror_view(A.values);
        Kokkos::deep_copy(row_map, A.graph.row_map);
        Kokkos::deep_copy(entries, A.graph.entries);
        Kokkos::deep_copy(values, A.values);

        int nrows = A.numRows();
        // Build dense matrix
        std::vector<std::vector<double>> dense(nrows, std::vector<double>(nrows, 0.0));
        for (int i = 0; i < nrows; ++i) {
            int start = row_map(i);
            int end = row_map(i + 1);
            for (int k = start; k < end; ++k) {
                dense[i][entries(k)] = values(k);
            }
        }

        // Output CSV: each row as comma-separated values
        QTextStream out(&csv);
        for (int i = 0; i < nrows; ++i) {
            for (int j = 0; j < nrows; ++j) {
                out << dense[i][j];
                if (j + 1 < nrows) out << ',';
            }
            out << '\n';
        }
    } else {
        // G-shaped domain export: output sparse matrix triplets
        DirichletSolver tmp(
            params.n_internal, params.m_internal,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound);
        const GridSystem* grid = tmp.getGridSystem();
        const auto& A = grid->get_matrix();
        auto row_map = Kokkos::create_mirror_view(A.graph.row_map);
        auto entries = Kokkos::create_mirror_view(A.graph.entries);
        auto values = Kokkos::create_mirror_view(A.values);
        Kokkos::deep_copy(row_map, A.graph.row_map);
        Kokkos::deep_copy(entries, A.graph.entries);
        Kokkos::deep_copy(values, A.values);

        int nrows = A.numRows();
        QTextStream out(&csv);
        out << "row,col,value\n";
        for (int i = 0; i < nrows; ++i) {
            int start = row_map(i);
            int end = row_map(i + 1);
            for (int k = start; k < end; ++k) {
                out << i << ',' << entries(k) << ',' << values(k) << '\n';
            }
        }
    }
    return csv;
}

// Export RHS vector b in CSV (index,value)
QString MainWindow::generateRHSCSV() {
    QString csv;
    if (params.solver_type.contains("ступень 2", Qt::CaseInsensitive)) {
        // Square domain RHS with correct functions for test or main task
        double (*f_func)(double, double) = nullptr;
        double (*exact_solution_func)(double, double) = nullptr;
        double (*mu1_func)(double, double) = nullptr;
        double (*mu2_func)(double, double) = nullptr;
        double (*mu3_func)(double, double) = nullptr;
        double (*mu4_func)(double, double) = nullptr;
        if (params.solver_type.contains("тестовая", Qt::CaseInsensitive)) {
            f_func = function2_square;
            exact_solution_func = solution2_square;
            mu1_func = mu1_square_solution2;
            mu2_func = mu2_square_solution2;
            mu3_func = mu3_square_solution2;
            mu4_func = mu4_square_solution2;
        } else {
            f_func = custom_function_square;
            mu1_func = mu1_square;
            mu2_func = mu2_square;
            mu3_func = mu3_square;
            mu4_func = mu4_square;
        }
        DirichletSolverSquare tmp = (exact_solution_func)
            ? DirichletSolverSquare(params.n_internal, params.m_internal,
                                   params.a_bound, params.b_bound,
                                   params.c_bound, params.d_bound,
                                   f_func, exact_solution_func)
            : DirichletSolverSquare(params.n_internal, params.m_internal,
                                   params.a_bound, params.b_bound,
                                   params.c_bound, params.d_bound,
                                   f_func, mu1_func, mu2_func, mu3_func, mu4_func);
        const GridSystemSquare* grid = tmp.getGridSystem();
        auto b = grid->get_rhs();
        auto b_host = Kokkos::create_mirror_view(b);
        Kokkos::deep_copy(b_host, b);
        QTextStream out(&csv);
        out << "index,value\n";
        int n = b_host.extent(0);
        for (int i = 0; i < n; ++i) {
            out << i << "," << b_host(i) << "\n";
        }
    } else {
        // G-shaped domain RHS
        DirichletSolver tmp(
            params.n_internal, params.m_internal,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound);
        const GridSystem* grid = tmp.getGridSystem();
        auto b = grid->get_rhs();
        auto b_host = Kokkos::create_mirror_view(b);
        Kokkos::deep_copy(b_host, b);
        QTextStream out(&csv);
        out << "index,value\n";
        int n = b_host.extent(0);
        for (int i = 0; i < n; ++i) {
            out << i << "," << b_host(i) << "\n";
        }
    }
    return csv;
}

// Add extern declaration for mu4_square_solution2 (move here)
extern double mu4_square_solution2(double x, double y);

// Slot called when the 2D chart type is changed
void MainWindow::onChartTypeChanged(int index)
{
    Q_UNUSED(index);
    updateSolverProgress(QString("Тип 2D графика изменён: %1").arg(index));
}