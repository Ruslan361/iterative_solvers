#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QLineSeries>
#include <QValueAxis>
#include <QChart>
#include <QChartView>
#include <QMetaObject>
#include <QObject>
#include <QDateTime>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm> // For std::sort, std::unique, std::lower_bound
#include <map>
#include <set>       // For std::set to find unique coordinates easily
#include <cmath>     // For std::abs
#include <QDebug>       // For qDebug messages in stubs
#include <limits>       // For std::numeric_limits in stubs

#ifdef _OPENMP
#include <omp.h>
#endif

namespace { // Anonymous namespace for helper functions specific to this file

double exact_solution_exp_x2_y2(double x, double y) {
    return std::exp(x * x - y * y);
}

double rhs_for_exp_x2_y2(double x, double y) {
    // -(laplacian(exp(x*x - y*y)))
    // laplacian(u) = (d^2u/dx^2) + (d^2u/dy^2)
    // u = exp(x^2 - y^2)
    // du/dx = 2x * exp(x^2 - y^2)
    // d^2u/dx^2 = 2 * exp(x^2 - y^2) + (2x)^2 * exp(x^2 - y^2) = (2 + 4x^2) * exp(x^2 - y^2)
    // du/dy = -2y * exp(x^2 - y^2)
    // d^2u/dy^2 = -2 * exp(x^2 - y^2) + (-2y)^2 * exp(x^2 - y^2) = (-2 + 4y^2) * exp(x^2 - y^2)
    // laplacian(u) = (2 + 4x^2 - 2 + 4y^2) * exp(x^2 - y^2) = (4x^2 + 4y^2) * exp(x^2 - y^2)
    return -(4 * x * x + 4 * y * y) * std::exp(x * x - y * y);
}

} // end anonymous namespace

// Helper function to find the closest value in a sorted vector of unique coordinates
double find_closest_coord(const std::vector<double>& unique_coords, double target_val) {
    if (unique_coords.empty()) {
        // This case should ideally not be reached if there's valid solution data
        return target_val;
    }
    auto const it = std::lower_bound(unique_coords.begin(), unique_coords.end(), target_val);
    if (it == unique_coords.begin()) {
        return unique_coords.front();
    }
    if (it == unique_coords.end()) {
        return unique_coords.back();
    }
    double val_before = *(it - 1);
    double val_after = *it;
    if (std::abs(target_val - val_before) < std::abs(target_val - val_after)) {
        return val_before;
    }
    return val_after;
}

// Реализация класса SolverWorker
SolverWorker::SolverWorker(std::unique_ptr<DirichletSolver> solver)
    : solver(std::move(solver)), is_square_solver(false) { // <<< MODIFIED THIS LINE
    // Устанавливаем колбэк для отслеживания итераций
    this->solver->setIterationCallback([this](int iteration, double precision, double residual, double error) {
        // Отправляем сигнал о прогрессе
        emit iterationUpdate(iteration, precision, residual, error);
    });
}

// <<< ADD THIS CONSTRUCTOR IMPLEMENTATION
SolverWorker::SolverWorker(std::unique_ptr<DirichletSolverSquare> solver_sq)
    : solver_sq(std::move(solver_sq)), is_square_solver(true) {
    // Устанавливаем колбэк для отслеживания итераций
    this->solver_sq->setIterationCallback([this](int iteration, double precision, double residual, double error) {
        // Отправляем сигнал о прогрессе
        emit iterationUpdate(iteration, precision, residual, error);
    });
}

void SolverWorker::process() {
    try {
        if (is_square_solver) {
            // Выполняем решение в отдельном потоке для квадратного решателя
            SquareSolverResults results_sq = solver_sq->solve();
            // Отправляем сигнал с результатами
            emit resultReadySquare(results_sq);
        } else { // G-Shape Solver
            // Выполняем решение в отдельном потоке
            SolverResults results = solver->solve();
            // Отправляем сигнал с результатами
            emit resultReady(results);
        }
    } catch (const std::exception& e) {
        qDebug() << "Error in solver worker: " << e.what();
    }
    
    // Отправляем сигнал о завершении
    emit finished();
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , isSolving(false)
    , solveSuccessful(false)
    , graph3D(nullptr)
{
    // Инициализация Kokkос, если она еще не инициализирована
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    
    ui->setupUi(this);

    // Add solver type combobox
    ui->solverTypeComboBox->addItem("Ступень 3"); 
    ui->solverTypeComboBox->addItem("Основная задача (ступень 2)"); 
    ui->solverTypeComboBox->addItem("Тестовая задача (ступень 2)"); // New solver option

    // Подключаем сигнал смены вкладок
    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &MainWindow::onTabChanged);

    // Initialize slice controls
    sliceAxisLabel = ui->sliceAxisLabel;
    sliceAxisComboBox = ui->sliceAxisComboBox;
    sliceIndexLabel = ui->sliceIndexLabel;
    sliceIndexSpinBox = ui->sliceIndexSpinBox;
    sliceInfoLabel = ui->sliceInfoLabel;

    sliceAxisComboBox->addItem("Срез по X (фиксированный Y)");
    sliceAxisComboBox->addItem("Срез по Y (фиксированный X)");

    // Connect slice controls signals to slots
    connect(sliceAxisComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onSliceAxisChanged);
    connect(sliceIndexSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onSliceIndexChanged);

    // Инициализация классов визуализации
    heatMapGenerator = std::make_unique<HeatMapGenerator>(this);
    
    // Соединяем сигналы и слоты
    connect(ui->solveButton, &QPushButton::clicked, this, &MainWindow::onSolveButtonClicked);
    connect(ui->stopButton, &QPushButton::clicked, this, &MainWindow::onStopButtonClicked);
    connect(ui->saveResultsButton, &QPushButton::clicked, this, &MainWindow::onSaveResultsButtonClicked);
    connect(ui->saveMatrixButton, &QPushButton::clicked, this, &MainWindow::onSaveMatrixButtonClicked);
    connect(ui->saveVisualizationButton, &QPushButton::clicked, this, &MainWindow::onSaveVisualizationButtonClicked);
    connect(ui->showReportButton, &QPushButton::clicked, this, &MainWindow::onShowReportButtonClicked);
    connect(ui->chartTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            [this](int index) {
                if (!solveSuccessful) return;
                
                switch (index) {
                    case 0: // Решение
                        updateChart(results.solution);
                        break;
                    case 1: // Ошибка
                        updateChart(results.error);
                        break;
                    case 2: // Невязка
                        updateChart(results.residual);
                        break;
                }
            });
    
    // Устанавливаем начальные значения
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
    params.solver_type = "G-Shape Solver"; // <<< ADD THIS LINE
    
    // Обновляем UI
    ui->nSpinBox->setValue(params.n_internal);
    ui->mSpinBox->setValue(params.m_internal);
    ui->aLineEdit->setText(QString::number(params.a_bound));
    ui->bLineEdit->setText(QString::number(params.b_bound));
    ui->cLineEdit->setText(QString::number(params.c_bound));
    ui->dLineEdit->setText(QString::number(params.d_bound));
    ui->precisionSpinBox->setValue(params.eps_precision);
    ui->residualSpinBox->setValue(params.eps_residual);
    ui->exactErrorSpinBox->setValue(params.eps_exact_error);
    ui->maxIterSpinBox->setValue(params.max_iterations);
    ui->precisionCheckBox->setChecked(params.use_precision);
    ui->residualCheckBox->setChecked(params.use_residual);
    ui->exactErrorCheckBox->setChecked(params.use_exact_error);
    ui->maxIterCheckBox->setChecked(params.use_max_iterations);
    
    // Настраиваем начальное состояние кнопок
    ui->stopButton->setEnabled(false);
    ui->saveResultsButton->setEnabled(false);
    ui->saveMatrixButton->setEnabled(false);
    ui->saveVisualizationButton->setEnabled(false);
    ui->showReportButton->setEnabled(false);
    
    // Устанавливаем и настраиваем 3D визуализацию
    setup3DVisualization();
    
    // Настройка вкладки таблицы
    setupTableTab(); // <<< CALLING THE NEW METHOD HERE
}

MainWindow::~MainWindow() {
    // Убедимся, что поток корректно остановлен и удален
    cleanupThread();
    
    delete ui;
    
    // Освобождаем ресурсы 3D визуализации
    if (graph3D) {
        delete graph3D;
    }
    
    // Финализируем Kokkос если мы ее инициализировали
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
}

void MainWindow::cleanupThread() {
    if (solverThread) {
        if (solverThread->isRunning()) {
            solverThread->quit();
            solverThread->wait();
        }
        delete solverThread;
        solverThread = nullptr;
    }
    
    // Worker удаляется автоматически, благодаря Qt::DeleteOnQuit
    worker = nullptr;
}

void MainWindow::onSolveButtonClicked() {
    if (isSolving) {
        return;
    }
    
    try {
        // Очищаем историю итераций
        iterationHistory.clear();
        
        // Получаем параметры из UI
        params.n_internal = ui->nSpinBox->value();
        params.m_internal = ui->mSpinBox->value();
        params.a_bound = ui->aLineEdit->text().toDouble();
        params.b_bound = ui->bLineEdit->text().toDouble();
        params.c_bound = ui->cLineEdit->text().toDouble();
        params.d_bound = ui->dLineEdit->text().toDouble();
        params.eps_precision = ui->precisionSpinBox->value();
        params.eps_residual = ui->residualSpinBox->value();
        params.eps_exact_error = ui->exactErrorSpinBox->value();
        params.max_iterations = ui->maxIterSpinBox->value();
        params.use_precision = ui->precisionCheckBox->isChecked();
        params.use_residual = ui->residualCheckBox->isChecked();
        params.use_exact_error = ui->exactErrorCheckBox->isChecked();
        params.use_max_iterations = ui->maxIterCheckBox->isChecked();
        params.use_refined_grid = ui->refinedGridCheckBox->isChecked();
        params.solver_type = ui->solverTypeComboBox->currentText(); // <<< UNCOMMENTED
        // Ensure default solver_type is used if UI element is not available
        // if (params.solver_type.isEmpty()) { // Default to G-Shape if not set by UI // <<< REMOVE THIS BLOCK
        //     params.solver_type = "G-Shape Solver";
        // }
        
        // Проверяем, что хотя бы один критерий останова выбран
        if (!params.use_precision && !params.use_residual && 
            !params.use_exact_error && !params.use_max_iterations) {
            QMessageBox::warning(this, "Предупреждение", 
                              "Выберите хотя бы один критерий останова.");
            return;
        }
        
        // Обновляем интерфейс
        ui->solveButton->setEnabled(false);
        ui->stopButton->setEnabled(true);
        ui->progressBar->setValue(0);
        ui->tabWidget->setCurrentIndex(1); // Переходим на вкладку Прогресс
        ui->progressTextEdit->clear();
        ui->progressTextEdit->append("Настройка решателя...");
        
        // Создаем новый решатель
        setupSolver();
        
        // Устанавливаем флаг
        isSolving = true;
        solveSuccessful = false;
        
        // Создаем поток и рабочий объект
        solverThread = new QThread(this);
        
        if (params.solver_type == "Основная задача (ступень 2)" || params.solver_type == "Тестовая задача (ступень 2)") {
            worker = new SolverWorker(std::move(solver_square));
        } else { // Ступень 3
            worker = new SolverWorker(std::move(solver));
        }
        worker->moveToThread(solverThread);
        
        // Соединяем сигналы и слоты
        connect(solverThread, &QThread::started, worker, &SolverWorker::process);
        connect(worker, &SolverWorker::finished, this, &MainWindow::onSolverFinished);
        if (params.solver_type == "Основная задача (ступень 2)" || params.solver_type == "Тестовая задача (ступень 2)") {
            connect(worker, &SolverWorker::resultReadySquare, this, &MainWindow::handleResultsSquare);
        } else { // Ступень 3
            connect(worker, &SolverWorker::resultReady, this, &MainWindow::handleResults);
        }
        connect(worker, &SolverWorker::iterationUpdate, this, &MainWindow::updateIterationInfo);
        
        // Для автоматической очистки после завершения
        connect(worker, &SolverWorker::finished, worker, &QObject::deleteLater);
        connect(solverThread, &QThread::finished, solverThread, &QObject::deleteLater);
        
        // Выводим информацию о запуске
        ui->progressTextEdit->append("Настройка сетки...");
        ui->progressTextEdit->append(QString("Сетка: %1x%2").arg(params.n_internal).arg(params.m_internal));
        ui->progressTextEdit->append(QString("Область: [%1, %2] x [%3, %4]").arg(params.a_bound)
                                 .arg(params.b_bound).arg(params.c_bound).arg(params.d_bound));
        ui->progressTextEdit->append("Начинаем решение...\n");
        
        // Запускаем поток
        solverThread->start();
        
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Ошибка", QString("Произошла ошибка: %1").arg(e.what()));
        isSolving = false;
        ui->solveButton->setEnabled(true);
        ui->stopButton->setEnabled(false);
    }
}

void MainWindow::onStopButtonClicked() {
    if (!isSolving) {
        return;
    }
    
    ui->progressTextEdit->append("Остановка решения пользователем...");
    
    // Запрос на остановку решения (вместо просто остановки потока)
    if (worker) {
        if ((params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") && worker->getSolverSquare()) {
            worker->getSolverSquare()->requestStop();
            ui->progressTextEdit->append("Сигнал остановки отправлен квадратному решателю. Ожидаем завершения текущей итерации...");
        } else if (params.solver_type == "G-Shape Solver" && worker->getSolver()) { // Explicitly G-Shape Solver
            worker->getSolver()->requestStop();
            ui->progressTextEdit->append("Сигнал остановки отправлен решателю. Ожидаем завершения текущей итерации...");
        } else {
            // Если по какой-то причине нет доступа к солверу, останавливаем поток напрямую
            cleanupThread();
            ui->progressTextEdit->append("Решение остановлено принудительно.");
        }
    } else {
        // Обновляем интерфейс, но НЕ останавливаем поток - он завершится корректно сам
        // когда солвер проверит флаг остановки в следующей итерации
        ui->stopButton->setEnabled(false);
    }
}

void MainWindow::setupSolver() {
    // Создаем решатель с указанными параметрами
    
    // Устанавливаем критерии остановки
    double eps_precision = params.use_precision ? params.eps_precision : 0.0;
    double eps_residual = params.use_residual ? params.eps_residual : 0.0;
    double eps_exact_error = params.use_exact_error ? params.eps_exact_error : 0.0;
    int max_iterations = params.use_max_iterations ? params.max_iterations : INT_MAX;

    if (params.solver_type == "Основная задача (ступень 2)") {
        // Используем конструктор с передачей функций для правой части и граничных условий
        solver_square = std::make_unique<DirichletSolverSquare>(
            params.n_internal, params.m_internal,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound,
            custom_function_square,   // Функция правой части
            mu1_square,               // Граничное условие на левой границе
            mu2_square,               // Граничное условие на нижней границе 
            mu3_square,               // Граничное условие на правой границе
            mu4_square                // Граничное условие на верхней границе
        );
        solver_square->setSolverParameters(eps_precision, eps_residual, eps_exact_error, max_iterations);
        solver_square->setUsePrecisionStopping(params.use_precision);
        solver_square->setUseResidualStopping(params.use_residual);
        solver_square->setUseErrorStopping(params.use_exact_error);
        solver_square->setUseMaxIterationsStopping(params.use_max_iterations);
        solver_square->setUseRefinedGridComparison(params.use_refined_grid); // Добавленная строка
        
    } else if (params.solver_type == "Тестовая задача (ступень 2)") {
        // Создаем новый квадратный решатель с функцией std::exp(x*x - y*y) и соответствующей правой частью
        solver_square = std::make_unique<DirichletSolverSquare>(
            params.n_internal, params.m_internal,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound,
            rhs_for_exp_x2_y2,        // Правая часть для exp(x*x - y*y)
            exact_solution_exp_x2_y2  // Точное решение exp(x*x - y*y)
        );
        // Этот конструктор DirichletSolverSquare(..., f, sol) должен установить internal exact_solution
        // и использовать 'sol' для определения граничных условий.
        
        solver_square->setSolverParameters(eps_precision, eps_residual, eps_exact_error, max_iterations);
        solver_square->setUsePrecisionStopping(params.use_precision);
        solver_square->setUseResidualStopping(params.use_residual);
        solver_square->setUseErrorStopping(params.use_exact_error); // Будет работать, так как exact_solution установлено
        solver_square->setUseMaxIterationsStopping(params.use_max_iterations);
        solver_square->setUseRefinedGridComparison(params.use_refined_grid); // Добавленная строка
        
    } else { // Default to Ступень 3
        // Создаем решатель с указанными параметрами
        solver = std::make_unique<DirichletSolver>(
            params.n_internal, params.m_internal,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound
        );
        solver->setSolverParameters(eps_precision, eps_residual, eps_exact_error, max_iterations);
        // Устанавливаем флаги использования критериев останова
        solver->enablePrecisionStopping(params.use_precision);
        solver->enableResidualStopping(params.use_residual);
        solver->enableErrorStopping(params.use_exact_error);
        solver->enableMaxIterationsStopping(params.use_max_iterations);
    }
    
    // Обновляем информацию в UI о размере матрицы
    ui->matrixInfoLabel->setText("Подготовка матрицы...");
}

void MainWindow::handleResults(const SolverResults& res) {
    // Сохраняем результаты
    results = res;
    solveSuccessful = true;
}

// <<< ADD THIS SLOT IMPLEMENTATION
void MainWindow::handleResultsSquare(const SquareSolverResults& res_sq) {
    results_square = res_sq;
    // Adapt or copy relevant parts from 'results' to 'results_square' for UI updates if needed
    // For now, let's assume the UI elements can be updated using results_square directly
    // or we can populate a common structure if that's more convenient.
    // For simplicity, we'll try to use results_square directly where possible,
    // and map to 'results' for common UI updates.

    results.iterations = res_sq.iterations;
    results.converged = res_sq.converged;
    results.stop_reason = res_sq.stop_reason;
    results.residual_norm = res_sq.residual_norm;
    results.error_norm = res_sq.error_norm;
    results.precision = res_sq.precision; // Assuming SquareSolverResults has precision

    // The following might need careful handling if the structure of solution/coords differs significantly
    results.solution = res_sq.solution;
    results.true_solution = res_sq.true_solution;
    results.error = res_sq.error;
    results.residual = res_sq.residual; // Assuming this is available and compatible
    results.x_coords = res_sq.x_coords;
    results.y_coords = res_sq.y_coords;

    solveSuccessful = true;
    
    // Отображаем информацию об ошибке относительно решения на более мелкой сетке
    if (res_sq.refined_grid_error >= 0) {
        ui->refinedGridErrorLabel->setText(QString("Ошибка относительно решения на мелкой сетке: %1")
            .arg(res_sq.refined_grid_error, 0, 'e', 6));
    } else {
        ui->refinedGridErrorLabel->setText("Ошибка относительно решения на мелкой сетке: Н/Д");
    }
}

void MainWindow::updateIterationInfo(int iteration, double precision, double residual, double error) {
    // Сохраняем данные итерации в историю
    iterationHistory.push_back({iteration, precision, residual, error});
    
    // Обновляем прогресс-бар (примерно, т.к. мы не знаем точное число итераций заранее)
    int progressValue = std::min(100, static_cast<int>(iteration * 100.0 / params.max_iterations));
    ui->progressBar->setValue(progressValue);
    
    // Обновляем информацию о текущей итерации
    ui->iterationsLabel->setText(QString("Итераций: %1").arg(iteration));
    
    // Only show criteria that are non-negative
    if (precision >= 0) {
        ui->precisionLabel->setText(QString("Точность ||xn-x(n-1)||: %1").arg(precision, 0, 'e', 6));
    } else {
        ui->precisionLabel->setText(QString("Точность ||xn-x(n-1)||: не применимо"));
    }
    
    if (residual >= 0) {
        ui->residualNormLabel->setText(QString("Норма невязки: %1").arg(residual, 0, 'e', 6));
    } else {
        ui->residualNormLabel->setText(QString("Норма невязки: не применимо"));
    }
    
    if (error >= 0) {
        ui->errorNormLabel->setText(QString("Норма ошибки: %1").arg(error, 0, 'e', 6));
    } else {
        ui->errorNormLabel->setText(QString("Норма ошибки: не применимо"));
    }
    
    // Добавляем информацию в текстовое поле прогресса каждые 100 итераций или на первой итерации
    if (iteration % 100 == 0 || iteration == 1) {
        ui->progressTextEdit->append(QString("Итерация: %1").arg(iteration));
        
        // Only display criteria with non-negative values
        if (precision >= 0) {
            ui->progressTextEdit->append(QString("Точность ||x(n)-x(n-1)||: max-норма = %1").arg(precision, 0, 'e', 6));
        }
        
        if (residual >= 0) {
            ui->progressTextEdit->append(QString("Невязка ||Ax-b||: max-норма = %1").arg(residual, 0, 'e', 6));
        }
        
        if (error >= 0) {
            ui->progressTextEdit->append(QString("Ошибка ||u-x||: max-норма = %1").arg(error, 0, 'e', 6));
        }
        
        ui->progressTextEdit->append(""); // Empty line for better readability
    }
    
    // Обновляем график прогресса в реальном времени
    if (iterationHistory.size() > 1) {
        auto *series_precision = new QLineSeries();
        auto *series_residual = new QLineSeries();
        auto *series_error = new QLineSeries();
        
        for (const auto& data : iterationHistory) {
            // Only add points for criteria with non-negative values
            if (data.precision >= 0) {
                series_precision->append(data.iteration, std::log10(data.precision));
            }
            
            if (data.residual >= 0) {
                series_residual->append(data.iteration, std::log10(data.residual));
            }
            
            if (data.error >= 0) {
                series_error->append(data.iteration, std::log10(data.error));
            }
        }
        
        series_precision->setName("log10(Точность)");
        series_residual->setName("log10(Невязка)");
        series_error->setName("log10(Ошибка)");
        
        auto *chart = new QChart();
        
        // Only add series with data points
        if (!series_precision->points().isEmpty()) {
            chart->addSeries(series_precision);
        } else {
            delete series_precision;
        }
        
        if (!series_residual->points().isEmpty()) {
            chart->addSeries(series_residual);
        } else {
            delete series_residual;
        }
        
        if (!series_error->points().isEmpty()) {
            chart->addSeries(series_error);
        } else {
            delete series_error;
        }
        
        // Only create axes and attach series if chart has at least one series
        if (!chart->series().isEmpty()) {
            auto *axisX = new QValueAxis();
            axisX->setTitleText("Итерация");
            chart->addAxis(axisX, Qt::AlignBottom);
            
            auto *axisY = new QValueAxis();
            axisY->setTitleText("log10(Норма)");
            chart->addAxis(axisY, Qt::AlignLeft);
            
            // Attach each series to axes
            for (auto *series : chart->series()) {
                series->attachAxis(axisX);
                series->attachAxis(axisY);
            }
            
            chart->setTitle("Сходимость метода");
            chart->legend()->setVisible(true);
        }
        
        ui->progressChartView->setChart(chart);
        ui->progressChartView->setRenderHint(QPainter::Antialiasing);
    }
}

void MainWindow::onSolverFinished() {
    // Этот метод вызывается после завершения работы потока решателя
    isSolving = false;
    
    ui->solveButton->setEnabled(true);
    ui->stopButton->setEnabled(false);
    
    if (solveSuccessful) {
        ui->progressTextEdit->append("\\n*** Решение завершено ***");
        ui->progressTextEdit->append(QString("Выполнено итераций: %1").arg(results.iterations));
        ui->progressTextEdit->append(QString("Норма невязки: %1").arg(results.residual_norm, 0, 'e', 6));
        ui->progressTextEdit->append(QString("Норма ошибки: %1").arg(results.error_norm, 0, 'e', 6));
        ui->progressTextEdit->append(QString("Достигнутая точность: %1").arg(results.precision, 0, 'e', 6));
        ui->progressTextEdit->append(QString("Сходимость: %1").arg(results.converged ? "Да" : "Нет"));
        ui->progressTextEdit->append(QString("Причина остановки: %1").arg(results.stop_reason.c_str()));
        
        // Обновляем информацию в UI
        ui->progressBar->setValue(100);
        ui->iterationsLabel->setText(QString("Итераций: %1").arg(results.iterations));
        ui->precisionLabel->setText(QString("Точность ||xn-x(n-1)||: %1").arg(results.precision, 0, 'e', 6));
        ui->residualNormLabel->setText(QString("Норма невязки: %1").arg(results.residual_norm, 0, 'e', 6));
        ui->errorNormLabel->setText(QString("Норма ошибки: %1").arg(results.error_norm, 0, 'e', 6));
        ui->convergenceStatusLabel->setText(QString("Статус: %1").arg(results.converged ? "Сошелся" : "Не сошелся"));
        ui->stopReasonLabel->setText(QString("Причина остановки: %1").arg(results.stop_reason.c_str()));
        
        // Активируем кнопки для сохранения и отображения результатов
        ui->saveResultsButton->setEnabled(true);
        ui->saveMatrixButton->setEnabled(true);
        ui->saveVisualizationButton->setEnabled(true);
        ui->showReportButton->setEnabled(true);
        
        // Отображаем решение на 2D графике
        if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") {
            // For square solver
            updateChart(results_square.solution);
        } else { // G-Shape Solver
            // For G-shape solver
            updateChart(results.solution);
        }
        
        // Переходим на вкладку с графиком
        ui->tabWidget->setCurrentIndex(0);
    } else {
        ui->progressTextEdit->append("\n*** Решение не удалось или было прервано ***");
        
        // Очистка полей для предотвращения отображения старых данных
        ui->iterationsLabel->setText("Итераций: 0");
        ui->precisionLabel->setText("Точность ||xn-x(n-1)||: 0.000000e+00");
        ui->residualNormLabel->setText("Норма невязки: 0.000000e+00");
        ui->errorNormLabel->setText("Норма ошибки: 0.000000e+00");
        ui->convergenceStatusLabel->setText("Статус: Не выполнено");
        ui->stopReasonLabel->setText("Причина остановки: Решение не завершено");
        
        // Сброс кнопок
        ui->saveResultsButton->setEnabled(false);
        ui->saveMatrixButton->setEnabled(false);
        ui->saveVisualizationButton->setEnabled(false);
        ui->showReportButton->setEnabled(false);
        showHeatMapButton->setEnabled(false);
        decimationFactorSpinBox->setEnabled(false);
        decimationFactorButton->setEnabled(false);
    }
    
    // Очищаем поток
    solverThread = nullptr;
    worker = nullptr;
}

void MainWindow::updateChart(const std::vector<double>& dataValues) {
    if (!solveSuccessful || dataValues.empty()) {
        // Clear chart if no data or solution not successful
        QChart *chart = new QChart();
        ui->chartView->setChart(chart); // QChartView takes ownership and deletes the previous chart.
        updateSliceControls(); // Update to show no data
        return;
    }

    // Determine the source of x_coords and y_coords based on solver type
    const std::vector<double>* x_coords_ptr = nullptr;
    const std::vector<double>* y_coords_ptr = nullptr;
    const std::vector<double>* currentData = nullptr;
    const std::vector<double>* trueSolutionDataForPlot = nullptr;

    if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") {
        x_coords_ptr = &results_square.x_coords;
        y_coords_ptr = &results_square.y_coords;
    } else { // G-Shape Solver
        x_coords_ptr = &results.x_coords;
        y_coords_ptr = &results.y_coords;
    }


    // Extract unique sorted coordinates for slicing
    m_unique_x_coords.clear();
    m_unique_y_coords.clear();
    if (!x_coords_ptr->empty() && !y_coords_ptr->empty()) { // <<< MODIFIED THIS LINE
        std::set<double> unique_x_set(x_coords_ptr->begin(), x_coords_ptr->end()); // <<< MODIFIED THIS LINE
        m_unique_x_coords.assign(unique_x_set.begin(), unique_x_set.end());

        std::set<double> unique_y_set(y_coords_ptr->begin(), y_coords_ptr->end()); // <<< MODIFIED THIS LINE
        m_unique_y_coords.assign(unique_y_set.begin(), unique_y_set.end());
    }
    updateSliceControls(); // Update controls based on new data

    // Create series for the selected slice
    auto *series = new QLineSeries(); 
    auto *trueSeries = new QLineSeries();

    QString chartTitle = "";
    QString xAxisTitle = "";
    QString yAxisTitle = "Значение";

    // Determine which data to plot based on chartTypeComboBox
    // const std::vector<double>* currentData = nullptr; // <<< REMOVE THIS LINE
    // const std::vector<double>* trueSolutionDataForPlot = nullptr; // For results.true_solution // <<< REMOVE THIS LINE
    QString dataTypeString = "";

    int chartTypeIndex = ui->chartTypeComboBox->currentIndex();
    if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") {
        if (chartTypeIndex == 0) { // Решение
            currentData = &results_square.solution;
            if (!results_square.true_solution.empty() && results_square.true_solution.size() == results_square.solution.size()) {
                trueSolutionDataForPlot = &results_square.true_solution;
            }
            dataTypeString = "Решение (Квадрат)";
        } else if (chartTypeIndex == 1) { // Ошибка
            currentData = &results_square.error;
            dataTypeString = "Ошибка (Квадрат)";
        } else if (chartTypeIndex == 2) { // Невязка
            currentData = &results_square.residual; // Assuming SquareSolverResults has residual
            dataTypeString = "Невязка (Квадрат)";
        }
    } else { // G-Shape Solver
        if (chartTypeIndex == 0) { // Решение
            currentData = &results.solution;
            if (!results.true_solution.empty() && results.true_solution.size() == results.solution.size()) {
                trueSolutionDataForPlot = &results.true_solution;
            }
            dataTypeString = "Решение (Г-форма)";
        } else if (chartTypeIndex == 1) { // Ошибка
            currentData = &results.error;
            dataTypeString = "Ошибка (Г-форма)";
        } else if (chartTypeIndex == 2) { // Невязка
            currentData = &results.residual;
            dataTypeString = "Невязка (Г-форма)";
        }
    }


    if (!currentData || currentData->empty()) {
        ui->chartView->setChart(new QChart()); // Set a new empty chart.
        delete series; // Clean up allocated series
        delete trueSeries;
        return;
    }

    series->setName(QString("Численное %1").arg(dataTypeString));
    if (trueSolutionDataForPlot) {
        trueSeries->setName(QString("Истинное %1").arg(dataTypeString));
    }

    if (m_currentSliceAxis == 0 && !m_unique_x_coords.empty() && m_currentSliceIndex < m_unique_x_coords.size()) { // Slice along Y (fixed X)
        double fixed_x = m_unique_x_coords[m_currentSliceIndex];
        chartTitle = QString("%1 при X = %2 (срез по Y)").arg(dataTypeString).arg(fixed_x);
        xAxisTitle = "Y координата";

        for (size_t i = 0; i < x_coords_ptr->size(); ++i) { // <<< MODIFIED THIS LINE
            if (std::abs((*x_coords_ptr)[i] - fixed_x) < 1e-9) { // Compare doubles with tolerance // <<< MODIFIED THIS LINE
                if (i < currentData->size()) { // Ensure index is valid
                    series->append((*y_coords_ptr)[i], (*currentData)[i]); // <<< MODIFIED THIS LINE
                }
                if (trueSolutionDataForPlot && i < trueSolutionDataForPlot->size()) {
                    trueSeries->append((*y_coords_ptr)[i], (*trueSolutionDataForPlot)[i]); // <<< MODIFIED THIS LINE
                }
            }
        }
        // The old block for populating trueSeries by calling u(fixed_x, y_val) is removed.
        // True solution is now plotted using results.true_solution at the same points as the numerical solution.

    } else if (m_currentSliceAxis == 1 && !m_unique_y_coords.empty() && m_currentSliceIndex < m_unique_y_coords.size()) { // Slice along X (fixed Y)
        double fixed_y = m_unique_y_coords[m_currentSliceIndex];
        chartTitle = QString("%1 при Y = %2 (срез по X)").arg(dataTypeString).arg(fixed_y);
        xAxisTitle = "X координата";

        for (size_t i = 0; i < y_coords_ptr->size(); ++i) { // <<< MODIFIED THIS LINE
            if (std::abs((*y_coords_ptr)[i] - fixed_y) < 1e-9) { // Compare doubles with tolerance // <<< MODIFIED THIS LINE
                 if (i < currentData->size()) { // Ensure index is valid
                    series->append((*x_coords_ptr)[i], (*currentData)[i]); // <<< MODIFIED THIS LINE
                }
                if (trueSolutionDataForPlot && i < trueSolutionDataForPlot->size()) {
                    trueSeries->append((*x_coords_ptr)[i], (*trueSolutionDataForPlot)[i]); // <<< MODIFIED THIS LINE
                }
            }
        }
        // The old block for populating trueSeries by calling u(x_val, fixed_y) is removed.
    } else {
        // Fallback or no data for slicing
        chartTitle = QString("Нет данных для среза (%1)").arg(dataTypeString);
    }
    
    auto *chart = new QChart();

    chart->addSeries(series);
    bool trueSeriesAdded = false;
    if (trueSolutionDataForPlot && trueSeries->points().size() > 0) { // Check if true solution data was used and series has points
       chart->addSeries(trueSeries);
       trueSeriesAdded = true;
    } else {
        delete trueSeries; // trueSeries was allocated but not added to chart
        trueSeries = nullptr; 
    }

    auto *axisX = new QValueAxis();
    axisX->setTitleText(xAxisTitle);
    if (m_currentSliceAxis == 0 && !m_unique_y_coords.empty()) axisX->setRange(params.c_bound, params.d_bound);
    else if (m_currentSliceAxis == 1 && !m_unique_x_coords.empty()) axisX->setRange(params.a_bound, params.b_bound);
    axisX->setLabelFormat("%.2f");
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    if (trueSeriesAdded) trueSeries->attachAxis(axisX);

    auto *axisY = new QValueAxis();
    axisY->setTitleText(yAxisTitle);
    axisY->setLabelFormat("%.2e"); // Use scientific notation for Y-axis

    // MODIFIED: Calculate and set Y-axis range
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();
    bool dataFoundForRange = false;

    if (series->points().size() > 0) {
        dataFoundForRange = true;
        for (const QPointF &p : series->points()) {
            if (p.y() < minVal) minVal = p.y();
            if (p.y() > maxVal) maxVal = p.y();
        }
    }

    if (trueSeriesAdded && trueSeries->points().size() > 0) {
        dataFoundForRange = true; // Mark data found if true series contributes
        for (const QPointF &p : trueSeries->points()) {
            if (p.y() < minVal) minVal = p.y();
            if (p.y() > maxVal) maxVal = p.y();
        }
    }

    if (dataFoundForRange) {
        if (minVal == maxVal) { // Handle case with single point or all points having same Y value
            minVal -= 0.5; // Add some padding
            maxVal += 0.5;
        }
        // Ensure minVal is not greater than maxVal after adjustment, can happen if original was 0 and became -0.5, 0.5
        if (minVal > maxVal) std::swap(minVal, maxVal);

        double rangeValue = maxVal - minVal;
        double padding = rangeValue * 0.05; // 5% padding

        // Ensure padding is not zero if range is extremely small but non-zero, or if range became 1.0
        if (padding == 0.0 && rangeValue == 0.0) { // This case means minVal and maxVal were identical
             // minVal and maxVal already adjusted by +/- 0.5, so rangeValue is 1.0
             // padding will be 1.0 * 0.05 = 0.05, this condition might not be strictly needed with current logic
             // but as a safeguard:
            padding = 0.05; // Default padding if it ended up zero
        } else if (padding == 0.0 && rangeValue > 0.0) { // Range is positive but so small padding is zero
            padding = rangeValue * 0.05 + 1e-9; // Add a tiny bit if it's zero due to precision
        }


        axisY->setRange(minVal - padding, maxVal + padding);
    }
    // If no dataFoundForRange, Qt Charts will auto-scale Y-axis, or it will be a default (e.g. 0-1)

    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    if (trueSeriesAdded) trueSeries->attachAxis(axisY);

    chart->setTitle(chartTitle);
    chart->legend()->setVisible(true);

    ui->chartView->setChart(chart); // QChartView takes ownership and deletes the previous chart.
    ui->chartView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::updateChartErrorVsTrue(const std::vector<double>& error) {
    // This method will now be handled by updateChart by selecting "Ошибка" in chartTypeComboBox
    // For now, we can call updateChart directly if the chartType is Error
    if (ui->chartTypeComboBox->currentIndex() == 1) {
        // updateChart(results.error); // or simply results.error if called from handleResults
        if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") {
            updateChart(results_square.error);
        } else { // G-Shape Solver
            updateChart(results.error);
        }
    }
}

void MainWindow::updateChartResidual(const std::vector<double>& residual) {
    // This method will now be handled by updateChart by selecting "Невязка" in chartTypeComboBox
    // For now, we can call updateChart directly if the chartType is Residual
    if (ui->chartTypeComboBox->currentIndex() == 2) {
        // updateChart(results.residual); // or simply results.residual if called from handleResults
        if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") {
            updateChart(results_square.residual);
        } else { // G-Shape Solver
            updateChart(results.residual);
        }
    }
}

void MainWindow::onSliceAxisChanged(int index) {
    m_currentSliceAxis = index;
    m_currentSliceIndex = 0; // Reset slice index when axis changes
    updateSliceControls();
    // Trigger chart update based on the currently selected data type in chartTypeComboBox
    int chartTypeIndex = ui->chartTypeComboBox->currentIndex();
    const std::vector<double>* data_to_plot = nullptr;

    if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") { 
        if (chartTypeIndex == 0) {
            data_to_plot = &results_square.solution;
        } else if (chartTypeIndex == 1) { // Ошибка
            data_to_plot = &results_square.error; 
        } else if (chartTypeIndex == 2) { // Невязка
            data_to_plot = &results_square.residual;
        }
    } else { // G-Shape Solver
        if (chartTypeIndex == 0) {
            data_to_plot = &results.solution;
        } else if (chartTypeIndex == 1) {
            data_to_plot = &results.error;
        } else if (chartTypeIndex == 2) {
            data_to_plot = &results.residual;
        }
    }
    if(data_to_plot && !data_to_plot->empty()){
        updateChart(*data_to_plot);
    } else {
        // Clear chart if no data for this type
        QChart *chart = new QChart();
        ui->chartView->setChart(chart);
        updateSliceControls(); // Update to show no data
    }
}

void MainWindow::onSliceIndexChanged(int value) {
    m_currentSliceIndex = value;
    updateSliceControls(); // Update info label
    // Trigger chart update based on the currently selected data type in chartTypeComboBox
    int chartTypeIndex = ui->chartTypeComboBox->currentIndex();
    const std::vector<double>* data_to_plot = nullptr;

    if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") { 
        if (chartTypeIndex == 0) {
            data_to_plot = &results_square.solution;
        } else if (chartTypeIndex == 1) {
            data_to_plot = &results_square.error;
        } else if (chartTypeIndex == 2) {
            data_to_plot = &results_square.residual;
        }
    } else { // G-Shape Solver
        if (chartTypeIndex == 0) {
            data_to_plot = &results.solution;
        } else if (chartTypeIndex == 1) {
            data_to_plot = &results.error;
        } else if (chartTypeIndex == 2) {
            data_to_plot = &results.residual;
        }
    }
    if(data_to_plot && !data_to_plot->empty()){
        updateChart(*data_to_plot);
    } else {
        QChart *chart = new QChart();
        ui->chartView->setChart(chart);
        updateSliceControls(); 
    }
}

void MainWindow::updateSliceControls() {
    if (!solveSuccessful) {
        sliceIndexSpinBox->setEnabled(false);
        sliceAxisComboBox->setEnabled(false);
        sliceInfoLabel->setText("Нет данных для среза");
        sliceIndexSpinBox->setRange(0, 0);
        return;
    }

    sliceIndexSpinBox->setEnabled(true);
    sliceAxisComboBox->setEnabled(true);

    if (m_currentSliceAxis == 0) { // Slice along Y (fixed X)
        sliceIndexSpinBox->setRange(0, m_unique_x_coords.empty() ? 0 : m_unique_x_coords.size() - 1);
        if (!m_unique_x_coords.empty() && m_currentSliceIndex < m_unique_x_coords.size()) {
            sliceInfoLabel->setText(QString("X = %1").arg(m_unique_x_coords[m_currentSliceIndex]));
        } else {
            sliceInfoLabel->setText("X: нет данных");
        }
    } else { // Slice along X (fixed Y)
        sliceIndexSpinBox->setRange(0, m_unique_y_coords.empty() ? 0 : m_unique_y_coords.size() - 1);
        if (!m_unique_y_coords.empty() && m_currentSliceIndex < m_unique_y_coords.size()) {
            sliceInfoLabel->setText(QString("Y = %1").arg(m_unique_y_coords[m_currentSliceIndex]));
        } else {
            sliceInfoLabel->setText("Y: нет данных");
        }
    }
    sliceIndexSpinBox->setValue(m_currentSliceIndex); // Ensure spinbox reflects current index
}

// Метод для настройки 3D визуализации
void MainWindow::setup3DVisualization() {
    // Создаем виджет для 3D-визуализации
    visualization3DTab = new QWidget();
    
    // Создаем 3D-график
    graph3D = new Q3DSurface();
    
    // Создаем контейнер для 3D-графика
    QWidget *container = QWidget::createWindowContainer(graph3D);
    container->setMinimumSize(400, 300);
    container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    
    // Настраиваем оси
    graph3D->axisX()->setTitle("X");
    graph3D->axisY()->setTitle("Значение");
    graph3D->axisZ()->setTitle("Y");
    graph3D->axisX()->setTitleVisible(true);
    graph3D->axisY()->setTitleVisible(true);
    graph3D->axisZ()->setTitleVisible(true);
    graph3D->axisX()->setRange(params.a_bound, params.b_bound);
    graph3D->axisZ()->setRange(params.c_bound, params.d_bound);
    
    // Настраиваем камеру и вид
    graph3D->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetIsometricRight);
    graph3D->setHorizontalAspectRatio(1.0);
    graph3D->setShadowQuality(QAbstract3DGraph::ShadowQualityMedium);
    
    // Создаем элементы управления
    QGroupBox *controlsGroupBox = new QGroupBox("Управление визуализацией");
    QVBoxLayout *controlsLayout = new QVBoxLayout(controlsGroupBox);
    
    // Чекбоксы для выбора поверхностей
    showSolutionCheckBox = new QCheckBox("Показать численное решение");
    showTrueSolutionCheckBox = new QCheckBox("Показать точное решение");
    showErrorCheckBox = new QCheckBox("Показать ошибку");
    showHeatMapButton = new QPushButton("Показать тепловую карту ошибки");
    
    // Добавляем элементы управления прореживанием
    QHBoxLayout *decimationLayout = new QHBoxLayout();
    QLabel *decimationLabel = new QLabel("Коэффициент прореживания:");
    decimationFactorSpinBox = new QSpinBox();
    decimationFactorSpinBox->setMinimum(1);
    decimationFactorSpinBox->setMaximum(10);
    decimationFactorSpinBox->setValue(1);
    decimationFactorSpinBox->setToolTip("Значение 1 означает отображение всех точек, большие значения уменьшают количество отображаемых точек");
    decimationFactorButton = new QPushButton("Применить");
    
    decimationLayout->addWidget(decimationLabel);
    decimationLayout->addWidget(decimationFactorSpinBox);
    decimationLayout->addWidget(decimationFactorButton);
    
    // Соединяем кнопку прореживания с обработчиком
    connect(decimationFactorButton, &QPushButton::clicked, this, [this]() {
        if (!shapeRegion || !solveSuccessful) return; // Guard against no region or no results

        int decimationFactor = decimationFactorSpinBox ? decimationFactorSpinBox->value() : 1;
        if (params.solver_type == "Square Solver" || params.solver_type == "Square Solver (G-shaped solution)") {
            if (!results_square.solution.empty()) {
                 shapeRegion->createSurfaces(
                    results_square.solution,
                    results_square.true_solution,
                    results_square.error,
                    results_square.x_coords,
                    results_square.y_coords,
                    params.a_bound, params.b_bound,
                    params.c_bound, params.d_bound,
                    decimationFactor
                );
            }
        } else { // G-Shape Solver
            if (!results.solution.empty()) {
                shapeRegion->createSurfaces(
                    results.solution,
                    results.true_solution,
                    results.error,
                    results.x_coords,
                    results.y_coords,
                    params.a_bound, params.b_bound,
                    params.c_bound, params.d_bound,
                    decimationFactor
                );
            }
        }
    });
    
    // Устанавливаем начальные значения
    showSolutionCheckBox->setChecked(true);
    showTrueSolutionCheckBox->setChecked(false);
    showErrorCheckBox->setChecked(false);
    
    // Добавляем элементы управления в layout
    controlsLayout->addWidget(showSolutionCheckBox);
    controlsLayout->addWidget(showTrueSolutionCheckBox);
    controlsLayout->addWidget(showErrorCheckBox);
    controlsLayout->addLayout(decimationLayout);
    controlsLayout->addWidget(showHeatMapButton);
    
    // Соединяем сигналы и слоты для управления видимостью серий
    connect(showSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onSolutionSeriesVisibilityChanged);
    connect(showTrueSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onTrueSolutionSeriesVisibilityChanged);
    connect(showErrorCheckBox, &QCheckBox::toggled, this, &MainWindow::onErrorSeriesVisibilityChanged);
    connect(showHeatMapButton, &QPushButton::clicked, this, &MainWindow::onShowHeatMapClicked);
    
    // Создаем основной layout для вкладки 3D-визуализации
    QVBoxLayout *mainLayout = new QVBoxLayout(visualization3DTab);
    mainLayout->addWidget(container, 1);
    mainLayout->addWidget(controlsGroupBox, 0);
    
    // Добавляем вкладку в tabWidget
    ui->tabWidget->addTab(visualization3DTab, "3D Визуализация");
    
    // Изначально кнопки неактивны, т.к. нет результатов
    showHeatMapButton->setEnabled(false);
    decimationFactorButton->setEnabled(false);
}

// Обработка смены вкладок
void MainWindow::onTabChanged(int index) {
    // Проверка, является ли выбранная вкладка "3D Визуализация"
    if (index == vizTabIndex && solveSuccessful) {
        // Если перешли на вкладку 3D-визуализации и есть успешное решение,
        // создаем или обновляем 3D-поверхности
        createOrUpdate3DSurfaces();
    }
}

// Создание или обновление 3D-поверхностей на основе доступных результатов
void MainWindow::createOrUpdate3DSurfaces() {
    if (!solveSuccessful) {
        qDebug() << "createOrUpdate3DSurfaces: Нет доступных результатов.";
        return;
    }
    
    try {
        int decimationFactor = decimationFactorSpinBox ? decimationFactorSpinBox->value() : 1;
        
        if (params.solver_type == "Основная задача (ступень 2)" || params.solver_type == "Тестовая задача (ступень 2)") {
            // Для квадратных областей (ступень 2)
            if (!results_square.solution.empty()) {
                // Создаем объект региона, если он еще не создан или имеет неверный тип
                if (!shapeRegion || !dynamic_cast<SquareShapeRegion*>(shapeRegion.get())) {
                    shapeRegion = std::make_unique<SquareShapeRegion>(graph3D);
                }
                
                // Получаем указатель на SquareShapeRegion
                SquareShapeRegion* squareRegion = dynamic_cast<SquareShapeRegion*>(shapeRegion.get());
                
                if (params.solver_type == "Основная задача (ступень 2)" && params.use_refined_grid) {
                    // Проверяем наличие решения на мелкой сетке
                    if (!results_square.refined_grid_solution.empty() && 
                        !results_square.refined_grid_x_coords.empty() && 
                        !results_square.refined_grid_y_coords.empty() && 
                        !results_square.solution_refined_diff.empty()) {
                            
                        // Создаем поверхности с учетом решения на мелкой сетке
                        squareRegion->createSurfacesWithRefinedGrid(
                            results_square.solution,
                            results_square.refined_grid_solution,
                            results_square.solution_refined_diff,
                            results_square.x_coords,
                            results_square.y_coords,
                            results_square.refined_grid_x_coords,
                            results_square.refined_grid_y_coords,
                            params.a_bound, params.b_bound,
                            params.c_bound, params.d_bound,
                            decimationFactor
                        );
                        
                        // Настраиваем чекбоксы
                        showSolutionCheckBox->setChecked(true);
                        showSolutionCheckBox->setEnabled(true);
                        showSolutionCheckBox->setText("Показать численное решение v(N)(x,y)");
                        
                        showTrueSolutionCheckBox->setChecked(true);  // Устанавливаем в true для активации
                        showTrueSolutionCheckBox->setEnabled(true);
                        showTrueSolutionCheckBox->setText("Показать решение v2(N2)(x,y) на сетке с половинным шагом");
                        
                        showErrorCheckBox->setChecked(true);  // Устанавливаем в true для активации
                        showErrorCheckBox->setEnabled(true);
                        showErrorCheckBox->setText("Показать разность решений v(N) и v2(N2)");
                        
                        // Настраиваем обработчики сигналов чекбоксов
                        // Отключаем текущие соединения
                        disconnect(showTrueSolutionCheckBox, nullptr, nullptr, nullptr);
                        disconnect(showErrorCheckBox, nullptr, nullptr, nullptr);
                        
                        // Устанавливаем новые соединения используя lambda-выражения
                        connect(showTrueSolutionCheckBox, &QCheckBox::toggled, this, [squareRegion](bool visible) {
                            squareRegion->setRefinedGridSolutionVisible(visible);
                        });
                        
                        connect(showErrorCheckBox, &QCheckBox::toggled, this, [squareRegion](bool visible) {
                            squareRegion->setSolutionRefinedDiffVisible(visible);
                        });
                    } else {
                        // Если нет данных о решении на мелкой сетке, используем стандартный метод
                        squareRegion->createSurfaces(
                            results_square.solution,
                            results_square.true_solution,
                            results_square.error,
                            results_square.x_coords,
                            results_square.y_coords,
                            params.a_bound, params.b_bound,
                            params.c_bound, params.d_bound,
                            decimationFactor
                        );
                        
                        // Настраиваем чекбоксы
                        showSolutionCheckBox->setChecked(true);
                        showSolutionCheckBox->setEnabled(true);
                        showSolutionCheckBox->setText("Показать численное решение v(N)(x,y)");
                        
                        showTrueSolutionCheckBox->setChecked(false);
                        showTrueSolutionCheckBox->setEnabled(false);
                        showTrueSolutionCheckBox->setText("Показать решение на мелкой сетке (недоступно)");
                        
                        showErrorCheckBox->setChecked(false);
                        showErrorCheckBox->setEnabled(false);
                        showErrorCheckBox->setText("Показать разность решений (недоступно)");
                        
                        // Восстанавливаем стандартные обработчики событий
                        disconnect(showTrueSolutionCheckBox, nullptr, nullptr, nullptr);
                        disconnect(showErrorCheckBox, nullptr, nullptr, nullptr);
                        
                        connect(showTrueSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onTrueSolutionSeriesVisibilityChanged);
                        connect(showErrorCheckBox, &QCheckBox::toggled, this, &MainWindow::onErrorSeriesVisibilityChanged);
                    }
                } else if (params.solver_type == "Тестовая задача (ступень 2)") {
                    // Для тестовой задачи стандартное поведение
                    squareRegion->createSurfaces(
                        results_square.solution,
                        results_square.true_solution,
                        results_square.error,
                        results_square.x_coords,
                        results_square.y_coords,
                        params.a_bound, params.b_bound,
                        params.c_bound, params.d_bound,
                        decimationFactor
                    );
                    
                    // Настраиваем чекбоксы
                    showSolutionCheckBox->setChecked(true);
                    showSolutionCheckBox->setEnabled(true);
                    showSolutionCheckBox->setText("Показать численное решение");
                    
                    showTrueSolutionCheckBox->setChecked(false);
                    showTrueSolutionCheckBox->setEnabled(!results_square.true_solution.empty());
                    showTrueSolutionCheckBox->setText("Показать точное решение u*(x,y)");
                    
                    showErrorCheckBox->setChecked(false);
                    showErrorCheckBox->setEnabled(!results_square.error.empty());
                    showErrorCheckBox->setText("Показать разность точного и численного решения");
                    
                    // Восстанавливаем стандартные обработчики событий
                    disconnect(showTrueSolutionCheckBox, nullptr, nullptr, nullptr);
                    disconnect(showErrorCheckBox, nullptr, nullptr, nullptr);
                    
                    connect(showTrueSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onTrueSolutionSeriesVisibilityChanged);
                    connect(showErrorCheckBox, &QCheckBox::toggled, this, &MainWindow::onErrorSeriesVisibilityChanged);
                }
                
                showHeatMapButton->setEnabled(true);
                decimationFactorSpinBox->setEnabled(true);
                decimationFactorButton->setEnabled(true);
            }
        } else { // Ступень 3 - Г-образная область
            // Для Г-образного решателя
            if (!results.solution.empty()) {
                // Создаем объект региона, если он еще не создан или имеет неверный тип
                if (!shapeRegion || !dynamic_cast<GShapeRegion*>(shapeRegion.get())) {
                    shapeRegion = std::make_unique<GShapeRegion>(graph3D);
                }
                
                // Создаем поверхности
                shapeRegion->createSurfaces(
                    results.solution,
                    results.true_solution,
                    results.error,
                    results.x_coords,
                    results.y_coords,
                    params.a_bound, params.b_bound,
                    params.c_bound, params.d_bound,
                    decimationFactor
                );
                
                // Настраиваем элементы управления
                showSolutionCheckBox->setChecked(true);
                showSolutionCheckBox->setEnabled(true);
                showSolutionCheckBox->setText("Показать численное решение");
                
                showTrueSolutionCheckBox->setChecked(false);
                showTrueSolutionCheckBox->setEnabled(!results.true_solution.empty());
                showTrueSolutionCheckBox->setText("Показать точное решение");
                
                showErrorCheckBox->setChecked(false);
                showErrorCheckBox->setEnabled(!results.error.empty());
                showErrorCheckBox->setText("Показать разность решений");
                
                // Восстанавливаем стандартные обработчики событий
                disconnect(showTrueSolutionCheckBox, nullptr, nullptr, nullptr);
                disconnect(showErrorCheckBox, nullptr, nullptr, nullptr);
                
                connect(showTrueSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onTrueSolutionSeriesVisibilityChanged);
                connect(showErrorCheckBox, &QCheckBox::toggled, this, &MainWindow::onErrorSeriesVisibilityChanged);
                
                showHeatMapButton->setEnabled(true);
                decimationFactorSpinBox->setEnabled(true);
                decimationFactorButton->setEnabled(true);
            }
        }
    } catch (const std::exception& e) {
        qDebug() << "Ошибка при создании 3D поверхностей: " << e.what();
        QMessageBox::warning(this, "Ошибка визуализации", 
                            QString("При создании 3D-поверхностей произошла ошибка: %1").arg(e.what()));
    }
}

// Функция для генерации CSV-данных с учетом выбранной задачи и прореживания
QString MainWindow::generateCSVData(int skipFactor) {
    if (params.solver_type == "Тестовая задача (ступень 2)") {
        return generateCSVForTestProblem(skipFactor);
    } else if (params.solver_type == "Основная задача (ступень 2)") {
        return generateCSVForMainProblem(skipFactor);
    } else { // Ступень 3
        return generateCSVForGShapeProblem(skipFactor);
    }
    
    return QString();
}

// Генерация CSV для тестовой задачи (квадратная область)
QString MainWindow::generateCSVForTestProblem(int skipFactor) {
    if (!solveSuccessful || results_square.solution.empty()) {
        return QString();
    }
    
    QString csvData;
    QTextStream stream(&csvData);
    
    int n = params.n_internal;
    int m = params.m_internal;
    double a = params.a_bound;
    double b = params.b_bound;
    double c = params.c_bound;
    double d = params.d_bound;
    
    // Создаем заголовок с координатами x
    stream << "y\\x";
    for (int i = 0; i <= n + 1; i += skipFactor) {
        double xi = x(i, n, a, b);
        stream << "," << xi;
    }
    stream << "\n";
    
    // Заполняем таблицу значениями численного решения u*
    stream << "u* (численное решение)\n";
    for (int j = 0; j <= m + 1; j += skipFactor) {
        double yj = y(j, m, c, d);
        stream << yj;
        
        for (int i = 0; i <= n + 1; i += skipFactor) {
            int index = j * (n + 2) + i;
            if (index < results_square.solution.size()) {
                stream << "," << results_square.solution[index];
            } else {
                stream << ",0";
            }
        }
        stream << "\n";
    }
    
    // Заполняем таблицу значениями аналитического решения v
    stream << "\nv (аналитическое решение)\n";
    for (int j = 0; j <= m + 1; j += skipFactor) {
        double yj = y(j, m, c, d);
        stream << yj;
        
        for (int i = 0; i <= n + 1; i += skipFactor) {
            double xi = x(i, n, a, b);
            double exactValue = exact_solution_exp_x2_y2(xi, yj); // Используем функцию для exp(x²-y²)
            stream << "," << exactValue;
        }
        stream << "\n";
    }
    
    // Заполняем таблицу разности u*-v
    stream << "\nu*-v (разность решений)\n";
    for (int j = 0; j <= m + 1; j += skipFactor) {
        double yj = y(j, m, c, d);
        stream << yj;
        
        for (int i = 0; i <= n + 1; i += skipFactor) {
            double xi = x(i, n, a, b);
            int index = j * (n + 2) + i;
            double numericalValue = (index < results_square.solution.size()) ? results_square.solution[index] : 0;
            double exactValue = exact_solution_exp_x2_y2(xi, yj);
            stream << "," << (numericalValue - exactValue);
        }
        stream << "\n";
    }
    
    return csvData;
}

// Генерация CSV для основной задачи
QString MainWindow::generateCSVForMainProblem(int skipFactor) {
    if (!solveSuccessful || results_square.solution.empty()) {
        return QString();
    }
    
    QString csvData;
    QTextStream stream(&csvData);
    
    int n = params.n_internal;
    int m = params.m_internal;
    double a = params.a_bound;
    double b = params.b_bound;
    double c = params.c_bound;
    double d = params.d_bound;
    
    // Создаем заголовок с координатами x
    stream << "y\\x";
    for (int i = 0; i <= n + 1; i += skipFactor) {
        double xi = x(i, n, a, b);
        stream << "," << xi;
    }
    stream << "\n";
    
    // Заполняем таблицу значениями численного решения u*
    stream << "u* (численное решение)\n";
    for (int j = 0; j <= m + 1; j += skipFactor) {
        double yj = y(j, m, c, d);
        stream << yj;
        
        for (int i = 0; i <= n + 1; i += skipFactor) {
            int index = j * (n + 2) + i;
            if (index < results_square.solution.size()) {
                stream << "," << results_square.solution[index];
            } else {
                stream << ",NaN"; // Вне индексации
            }
        }
        stream << "\n";
    }
    
    // Заполняем таблицу значениями решения на подробной сетке (если есть)
    if (!results_square.refined_grid_solution.empty()) {
        stream << "\nv (численное решение на подробной сетке)\n";
        int refined_n = 2 * n; // Подробная сетка в 2 раза больше
        int refined_m = 2 * m;
        
        for (int j = 0; j <= refined_m + 1; j += skipFactor) {
            double yj = y(j, refined_m, c, d);
            stream << yj;
            
            for (int i = 0; i <= refined_n + 1; i += skipFactor) {
                int index = j * (refined_n + 2) + i;
                if (index < results_square.refined_grid_solution.size()) {
                    stream << "," << results_square.refined_grid_solution[index];
                } else {
                    stream << ",NaN";
                }
            }
            stream << "\n";
        }
        
        // Заполняем таблицу разности решений на общих узлах
        stream << "\nu*-v (разность решений на общих узлах)\n";
        for (int j = 0; j <= m + 1; j += skipFactor) {
            double yj = y(j, m, c, d);
            stream << yj;
            
            for (int i = 0; i <= n + 1; i += skipFactor) {
                int index = j * (n + 2) + i;
                double val1 = (index < results_square.solution.size()) ? results_square.solution[index] : 0;
                
                // Соответствующий индекс на подробной сетке
                int refined_i = i * 2;
                int refined_j = j * 2;
                int refined_index = refined_j * (2 * n + 2) + refined_i;
                double val2 = (refined_index < results_square.refined_grid_solution.size()) ? 
                             results_square.refined_grid_solution[refined_index] : 0;
                
                stream << "," << (val1 - val2);
            }
            stream << "\n";
        }
    }
    
    return csvData;
}

// Генерация CSV для Г-образной задачи
QString MainWindow::generateCSVForGShapeProblem(int skipFactor) {
    if (!solveSuccessful || results.solution.empty()) {
        return QString();
    }
    
    QString csvData;
    QTextStream stream(&csvData);
    
    int n = params.n_internal;
    int m = params.m_internal;
    double a = params.a_bound;
    double b = params.b_bound;
    double c = params.c_bound;
    double d = params.d_bound;
    
    // Создаем заголовок с координатами x
    stream << "y\\x";
    for (size_t i = 0; i < results.x_coords.size(); i += skipFactor) {
        stream << "," << results.x_coords[i];
    }
    stream << "\n";
    
    // Заполняем таблицу значениями численного решения u*
    stream << "u* (численное решение)\n";
    
    // Получаем уникальные координаты y
    std::set<double> unique_y_coords(results.y_coords.begin(), results.y_coords.end());
    std::vector<double> sorted_y_coords(unique_y_coords.begin(), unique_y_coords.end());
    std::sort(sorted_y_coords.begin(), sorted_y_coords.end());
    
    for (double yj : sorted_y_coords) {
        stream << yj;
        
        for (size_t i = 0; i < results.x_coords.size(); i += skipFactor) {
            double xi = results.x_coords[i];
            
            // Найдем индекс точки с координатами (xi, yj)
            size_t index = 0;
            bool found = false;
            
            for (size_t k = 0; k < results.x_coords.size(); ++k) {
                if (std::abs(results.x_coords[k] - xi) < 1e-9 && std::abs(results.y_coords[k] - yj) < 1e-9) {
                    index = k;
                    found = true;
                                       break;
                }
            }
            
            if (found && index < results.solution.size()) {
                stream << "," << results.solution[index];
            } else {
                stream << ",NaN"; // Нет точки или нет данных
            }
        }
        stream << "\n";
    }
    
    // Если есть данные об ошибке, добавим их
    if (!results.error.empty() && results.error.size() == results.solution.size()) {
        stream << "\nОшибка (error)\n";
        
        for (double yj : sorted_y_coords) {
            stream << yj;
            
            for (size_t i = 0; i < results.x_coords.size(); i += skipFactor) {
                double xi = results.x_coords[i];
                
                // Найдем индекс точки с координатами (xi, yj)
                size_t index = 0;
                bool found = false;
                
                for (size_t k = 0; k < results.x_coords.size(); ++k) {
                    if (std::abs(results.x_coords[k] - xi) < 1e-9 && std::abs(results.y_coords[k] - yj) < 1e-9) {
                        index = k;
                        found = true;
                        break;
                    }
                }
                
                if (found && index < results.error.size()) {
                    stream << "," << results.error[index];
                } else {
                    stream << ",NaN"; // Нет точки или нет данных
                }
            }
            stream << "\n";
        }
    }
    
    return csvData;
}

// Обработчик нажатия на кнопку экспорта в CSV
void MainWindow::onExportCSVButtonClicked() {
    int skipFactor = skipFactorSpinBox->value();
    
    QString csvData = generateCSVData(skipFactor);
    if (csvData.isEmpty()) {
        QMessageBox::warning(this, "Предупреждение", "Нет данных для экспорта. Выполните расчет.");
        return;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this, "Сохранить CSV", "", "CSV файлы (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }
    
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Ошибка", "Не удалось открыть файл для записи");
        return;
    }
    
    QTextStream stream(&file);
    #if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
        stream.setCodec("UTF-8");
    #endif
    stream << csvData;
    file.close();
    
    QMessageBox::information(this, "Экспорт завершен", "Данные успешно экспортированы в " + fileName);
}

// Обработчик нажатия на кнопку отображения таблицы
void MainWindow::onShowTableButtonClicked() {
    int skipFactor = skipFactorSpinBox->value();
    
    QString csvData = generateCSVData(skipFactor);
    if (csvData.isEmpty()) {
        QMessageBox::warning(this, "Предупреждение", "Нет данных для отображения. Выполните расчет.");
        return;
    }
    
    populateTableWithData(csvData);
}

// Обработчик нажатия на кнопку очистки таблицы
void MainWindow::onClearTableButtonClicked() {
    dataTable->setRowCount(0);
    dataTable->setColumnCount(0);
    tableInfoLabel->setText("Таблица очищена");
}

// Функция для заполнения таблицы данными из CSV-строки
void MainWindow::populateTableWithData(const QString& csvData) {
    QStringList rows = csvData.split('\n', Qt::SkipEmptyParts);
    if (rows.isEmpty()) {
        return;
    }
    
    // Разбираем первую строку для определения числа столбцов
    QStringList headers = rows[0].split(',');
    int columnCount = headers.size();
    
    // Настраиваем таблицу
    dataTable->setRowCount(rows.size() - 1); // Минус заголовок
    dataTable->setColumnCount(columnCount);
    
    // Устанавливаем заголовки
    dataTable->setHorizontalHeaderLabels(headers);
    
    // Заполняем данные
    for (int row = 1; row < rows.size(); ++row) {
        QStringList cells = rows[row].split(',');
        
        // Если строка содержит только один элемент, это может быть заголовок новой секции
        if (cells.size() == 1) {
            // Пропускаем строки с заголовками секций
            continue;
        }
        
        for (int col = 0; col < cells.size() && col < columnCount; ++col) {
            QTableWidgetItem* item = new QTableWidgetItem(cells[col]);
            
            // Устанавливаем выравнивание для чисел
            if (col > 0) { // Предполагаем, что первый столбец содержит метки
                item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            }
            
            dataTable->setItem(row - 1, col, item);
        }
    }
    
    // Подгоняем размеры столбцов под содержимое
    dataTable->resizeColumnsToContents();
    
    int skipFactor = skipFactorSpinBox->value();
    tableInfoLabel->setText(QString("Отображены данные с прореженностью: каждый %1-й элемент").arg(skipFactor));
}
