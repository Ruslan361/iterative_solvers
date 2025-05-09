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
        if (is_square_solver) { // <<< ADD THIS BLOCK
            // Выполняем решение в отдельном потоке для квадратного решателя
            SquareSolverResults results_sq = solver_sq->solve();
            // Отправляем сигнал с результатами
            emit resultReadySquare(results_sq);
        } else {
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
    ui->solverTypeComboBox->addItem("G-Shape Solver"); // <<< UNCOMMENTED
    ui->solverTypeComboBox->addItem("Square Solver"); // <<< UNCOMMENTED


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
        
        // Передаем ownership решателя рабочему классу
        // worker = new SolverWorker(std::move(solver)); // <<< REMOVE THIS LINE
        if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
            worker = new SolverWorker(std::move(solver_square));
        } else {
            worker = new SolverWorker(std::move(solver));
        }
        worker->moveToThread(solverThread);
        
        // Соединяем сигналы и слоты
        connect(solverThread, &QThread::started, worker, &SolverWorker::process);
        connect(worker, &SolverWorker::finished, this, &MainWindow::onSolverFinished);
        if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
            connect(worker, &SolverWorker::resultReadySquare, this, &MainWindow::handleResultsSquare);
        } else {
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
    if (worker) { // <<< MODIFIED THIS LINE
        if (params.solver_type == "Square Solver" && worker->getSolverSquare()) { // <<< ADD THIS BLOCK
            worker->getSolverSquare()->requestStop();
            ui->progressTextEdit->append("Сигнал остановки отправлен квадратному решателю. Ожидаем завершения текущей итерации...");
        } else if (worker->getSolver()) {
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
    // solver = std::make_unique<DirichletSolver>( // <<< REMOVE THIS LINE
    //     params.n_internal, params.m_internal, // <<< REMOVE THIS LINE
    //     params.a_bound, params.b_bound, // <<< REMOVE THIS LINE
    //     params.c_bound, params.d_bound // <<< REMOVE THIS LINE
    // ); // <<< REMOVE THIS LINE
    
    // Устанавливаем критерии остановки
    double eps_precision = params.use_precision ? params.eps_precision : 0.0;
    double eps_residual = params.use_residual ? params.eps_residual : 0.0;
    double eps_exact_error = params.use_exact_error ? params.eps_exact_error : 0.0;
    int max_iterations = params.use_max_iterations ? params.max_iterations : INT_MAX;

    if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
        solver_square = std::make_unique<DirichletSolverSquare>(
            params.n_internal, params.m_internal,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound
        );
        solver_square->setSolverParameters(eps_precision, eps_residual, eps_exact_error, max_iterations);
        // Enable all stopping criteria for square solver for now, can be refined
        solver_square->setUsePrecisionStopping(params.use_precision);
        solver_square->setUseResidualStopping(params.use_residual);
        solver_square->setUseErrorStopping(params.use_exact_error);
        solver_square->setUseMaxIterationsStopping(params.use_max_iterations);
    } else {
        solver = std::make_unique<DirichletSolver>(
            params.n_internal, params.m_internal,
            params.a_bound, params.b_bound,
            params.c_bound, params.d_bound
        );
        solver->setSolverParameters(eps_precision, eps_residual, eps_exact_error, max_iterations);
    }
    
    // Обновляем информацию в UI о размере матрицы
    ui->matrixInfoLabel->setText("Подготовка матрицы...");
}

void MainWindow::handleResults(SolverResults res) {
    // Сохраняем результаты
    results = res;
    solveSuccessful = true;
}

// <<< ADD THIS SLOT IMPLEMENTATION
void MainWindow::handleResultsSquare(SquareSolverResults res_sq) {
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
}

void MainWindow::updateIterationInfo(int iteration, double precision, double residual, double error) {
    // Сохраняем данные итерации в историю
    iterationHistory.push_back({iteration, precision, residual, error});
    
    // Обновляем прогресс-бар (примерно, т.к. мы не знаем точное число итераций заранее)
    int progressValue = std::min(100, static_cast<int>(iteration * 100.0 / params.max_iterations));
    ui->progressBar->setValue(progressValue);
    
    // Обновляем информацию о текущей итерации
    ui->iterationsLabel->setText(QString("Итераций: %1").arg(iteration));
    ui->precisionLabel->setText(QString("Точность ||xn-x(n-1)||: %1").arg(precision, 0, 'e', 6));
    ui->residualNormLabel->setText(QString("Норма невязки: %1").arg(residual, 0, 'e', 6));
    ui->errorNormLabel->setText(QString("Норма ошибки: %1").arg(error, 0, 'e', 6));
    
    // Добавляем информацию в текстовое поле прогресса каждые 100 итераций или на первой итерации
    if (iteration % 100 == 0 || iteration == 1) {
        ui->progressTextEdit->append(QString("Итерация: %1").arg(iteration));
        ui->progressTextEdit->append(QString("Точность ||x(n)-x(n-1)||: max-норма = %1").arg(precision, 0, 'e', 6));
        ui->progressTextEdit->append(QString("Невязка ||Ax-b||: max-норма = %1").arg(residual, 0, 'e', 6));
        ui->progressTextEdit->append(QString("Ошибка ||u-x||: max-норма = %1\n").arg(error, 0, 'e', 6));
    }
    
    // Обновляем график прогресса в реальном времени
    if (iterationHistory.size() > 1) {
        auto *series_precision = new QLineSeries();
        auto *series_residual = new QLineSeries();
        auto *series_error = new QLineSeries();
        
        for (const auto& data : iterationHistory) {
            series_precision->append(data.iteration, std::log10(data.precision));
            series_residual->append(data.iteration, std::log10(data.residual));
            series_error->append(data.iteration, std::log10(data.error));
        }
        
        series_precision->setName("log10(Точность)");
        series_residual->setName("log10(Невязка)");
        series_error->setName("log10(Ошибка)");
        
        auto *chart = new QChart();
        chart->addSeries(series_precision);
        chart->addSeries(series_residual);
        chart->addSeries(series_error);
        
        auto *axisX = new QValueAxis();
        axisX->setTitleText("Итерация");
        chart->addAxis(axisX, Qt::AlignBottom);
        series_precision->attachAxis(axisX);
        series_residual->attachAxis(axisX);
        series_error->attachAxis(axisX);
        
        auto *axisY = new QValueAxis();
        axisY->setTitleText("log10(Норма)");
        chart->addAxis(axisY, Qt::AlignLeft);
        series_precision->attachAxis(axisY);
        series_residual->attachAxis(axisY);
        series_error->attachAxis(axisY);
        
        chart->setTitle("Сходимость метода");
        chart->legend()->setVisible(true);
        
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
        ui->progressTextEdit->append("\n*** Решение завершено ***");
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
        
        // Создаем новый экземпляр решателя для использования в последующих операциях
        // setupSolver(); // <<< This might not be needed or might need adjustment based on solver type
        
        // Отображаем решение
        if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
            // For square solver, we might need a different updateChart or adapt existing one
            // For now, let's assume updateChart can handle results_square.solution
            // or that we've mapped it to results.solution in handleResultsSquare
            updateChart(results_square.solution);
            update3DSurfacesSquare(); // <<< CALL NEW METHOD FOR SQUARE SOLVER
            // Disable G-shape specific controls
            if (gshapeRegion) { 
                gshapeRegion->clearAllSurfaces(); 
            }
            showHeatMapButton->setEnabled(true); // Enable heatmap for square too (will need adaptation)
            decimationFactorSpinBox->setEnabled(false); // Decimation might not be relevant for square
            decimationFactorButton->setEnabled(false);
            showTrueSolutionCheckBox->setEnabled(false); // No true solution for this custom square problem
            showErrorCheckBox->setEnabled(false);      // No error surface if no true solution

        } else {
            updateChart(results.solution);
            update3DSurfaces(); // This is for G-shape
            decimationFactorSpinBox->setEnabled(true);
            decimationFactorButton->setEnabled(true);
            showTrueSolutionCheckBox->setEnabled(true);
            showErrorCheckBox->setEnabled(true);
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
    const std::vector<double>* x_coords_ptr = nullptr; // <<< ADD THIS LINE
    const std::vector<double>* y_coords_ptr = nullptr; // <<< ADD THIS LINE
    const std::vector<double>* currentData = nullptr; // <<< ADD THIS LINE
    const std::vector<double>* trueSolutionDataForPlot = nullptr; // <<< ADD THIS LINE

    if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
        x_coords_ptr = &results_square.x_coords;
        y_coords_ptr = &results_square.y_coords;
    } else {
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
    if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
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
    } else {
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
        // updateChart(results.error); // or simply results.error if called from handleResults // <<< REMOVE THIS LINE
        if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
            updateChart(results_square.error);
        } else {
            updateChart(results.error);
        }
    }
}

void MainWindow::updateChartResidual(const std::vector<double>& residual) {
    // This method will now be handled by updateChart by selecting "Невязка" in chartTypeComboBox
    // For now, we can call updateChart directly if the chartType is Residual
    if (ui->chartTypeComboBox->currentIndex() == 2) {
        // updateChart(results.residual); // or simply results.residual if called from handleResults // <<< REMOVE THIS LINE
        if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
            updateChart(results_square.residual);
        } else {
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

    if (params.solver_type == "Square Solver") { 
        if (chartTypeIndex == 0) {
            data_to_plot = &results_square.solution;
        } else if (chartTypeIndex == 1) { // Ошибка (для квадратного решателя пока нет)
            // data_to_plot = &results_square.error; // No error data for custom square problem yet
        } else if (chartTypeIndex == 2) { // Невязка
            // data_to_plot = &results_square.residual; // Assuming SquareSolverResults has residual
        }
    } else {
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

    if (params.solver_type == "Square Solver") { 
        if (chartTypeIndex == 0) {
            data_to_plot = &results_square.solution;
        } else if (chartTypeIndex == 1) {
            // data_to_plot = &results_square.error;
        } else if (chartTypeIndex == 2) {
            // data_to_plot = &results_square.residual;
        }
    } else {
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
    
    // Инициализируем объект GShapeRegion
    gshapeRegion = std::make_unique<GShapeRegion>(graph3D);
    
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
    // connect(decimationFactorButton, &QPushButton::clicked, this, &MainWindow::update3DSurfaces); // Old connection
    connect(decimationFactorButton, &QPushButton::clicked, this, [this]() { // New connection
        if (params.solver_type == "Square Solver") {
            update3DSurfacesSquare();
        } else {
            update3DSurfaces();
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
    connect(showHeatMapButton, &QPushButton::clicked, this, &MainWindow::onShowHeatmapClicked);
    
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

// Слот для обработки изменения видимости серии численного решения
void MainWindow::onSolutionSeriesVisibilityChanged(bool visible) {
    setNumericalSolutionVisible(visible);
}

// Слот для обработки изменения видимости серии точного решения
void MainWindow::onTrueSolutionSeriesVisibilityChanged(bool visible) {
    setTrueSolutionVisible(visible);
}

// Слот для обработки изменения видимости серии ошибки
void MainWindow::onErrorSeriesVisibilityChanged(bool visible) {
    setErrorSurfaceVisible(visible);
}

// Создание 2D матрицы точного решения
std::vector<std::vector<double>> MainWindow::createTrueSolutionMatrix() {
    std::vector<std::vector<double>> trueMatrix(params.m_internal, std::vector<double>(params.n_internal, std::numeric_limits<double>::quiet_NaN()));
    
    // Координаты "разделителей" для Г-образной области
    double x_split = (params.a_bound + params.b_bound) / 2.0;
    double y_split = (params.c_bound + params.d_bound) / 2.0;
    
    // Заполняем матрицу истинного решения только для точек внутри Г-образной области
    for (int i = 0; i < params.m_internal; ++i) {
        for (int j = 0; j < params.n_internal; ++j) {
            // Рассчитываем физические координаты точки
            double xCoord = x(j+1, params.n_internal, params.a_bound, params.b_bound);
            double yCoord = y(i+1, params.m_internal, params.c_bound, params.d_bound);
            
            // Определяем, находится ли точка в Г-образной области
            // Логика, аналогичная используемой в GShapeRegion и ранее в createGShapedSurface
            bool is_quadrant1 = (xCoord <= x_split && yCoord >= y_split); // Верхний-левый квадрант
            bool is_quadrant2 = (xCoord > x_split && yCoord > y_split);   // Верхний-правый квадрант
            bool is_quadrant4 = (xCoord > x_split && yCoord <= y_split);  // Нижний-правый квадрант
            
            bool isInGShapeDomain = is_quadrant1 || is_quadrant2 || is_quadrant4;

            if (isInGShapeDomain) {
                trueMatrix[i][j] = u(xCoord, yCoord);
            } else {
                trueMatrix[i][j] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    return trueMatrix;
}

// Создание 2D матрицы ошибки (разности решений)
std::vector<std::vector<double>> MainWindow::createErrorMatrix() {
    if (results.solution.empty()) {
        return {};
    }
    
    auto solutionMatrix = solutionTo2D();
    auto trueMatrix = createTrueSolutionMatrix();
    
    std::vector<std::vector<double>> errorMatrix(params.m_internal, std::vector<double>(params.n_internal));
    
    // Вычисляем ошибку как разность между численным и точным решениями
    for (int i = 0; i < params.m_internal; ++i) {
        for (int j = 0; j < params.n_internal; ++j) {
            if (std::isnan(trueMatrix[i][j])) {
                // Если точка вне области, то ошибка тоже NaN
                errorMatrix[i][j] = std::numeric_limits<double>::quiet_NaN();
            } else {
                // Иначе вычисляем абсолютную ошибку
                errorMatrix[i][j] = std::abs(solutionMatrix[i][j] - trueMatrix[i][j]);
            }
        }
    }
    
    return errorMatrix;
}


// Обновление 3D-поверхностей
void MainWindow::update3DSurfaces() {
    try {
        if (!solveSuccessful) {
            qDebug() << "update3DSurfaces: Решение не успешно.";
            // Очищаем серии, чтобы избежать отображения устаревших данных
            if (gshapeRegion) {
                gshapeRegion->clearAllSurfaces();
            }
            showHeatMapButton->setEnabled(false);
            return;
        }

        // Проверяем наличие данных координат и решения
        if (results.solution.empty() || results.x_coords.empty() || results.y_coords.empty() ||
            results.x_coords.size() != results.solution.size() || 
            results.y_coords.size() != results.solution.size()) {
            qDebug() << "update3DSurfaces: Отсутствуют или несогласованы данные решения/координат.";
            if (gshapeRegion) {
                gshapeRegion->clearAllSurfaces();
            }
            showHeatMapButton->setEnabled(false);
            return;
        }

        // Проверяем данные на наличие некорректных значений (NaN или Infinity)
        for (size_t i = 0; i < results.solution.size(); ++i) {
            if (std::isnan(results.solution[i]) || std::isinf(results.solution[i])) {
                qDebug() << "update3DSurfaces: Найдены некорректные значения в решении";
                if (gshapeRegion) {
                    gshapeRegion->clearAllSurfaces();
                }
                showHeatMapButton->setEnabled(false);
                return;
            }
        }

        // Получаем коэффициент прореживания из интерфейса или используем значение по умолчанию
        int decimationFactor = 1; // Значение по умолчанию
        if (decimationFactorSpinBox) {
            decimationFactor = std::max(1, decimationFactorSpinBox->value());
        }

        // Используем новый метод для создания Г-образной поверхности
        createGShapedSurface(
            results.solution,        // Численное решение
            results.true_solution,   // Точное решение
            results.error,           // Ошибка
            results.x_coords,        // X координаты
            results.y_coords,        // Y координаты
            decimationFactor,        // Коэффициент прореживания
            5                        // Количество строк для перемычки
        );

        // Делаем видимым только численное решение по умолчанию
        setNumericalSolutionVisible(true);
        setTrueSolutionVisible(false);
        setErrorSurfaceVisible(false);

        // Обновляем состояние элементов управления
        if (showSolutionCheckBox) showSolutionCheckBox->setChecked(true);
        if (showTrueSolutionCheckBox) showTrueSolutionCheckBox->setChecked(false);
        if (showErrorCheckBox) showErrorCheckBox->setChecked(false);
        showHeatMapButton->setEnabled(true);

        // Обновляем заголовок
        graph3D->setTitle("Численное решение на Г-образной области");
    } catch (const std::exception& e) {
        qDebug() << "ОШИБКА в update3DSurfaces: " << e.what();
        QMessageBox::critical(this, "Ошибка при обновлении 3D поверхности", 
                            QString("Произошла ошибка: %1").arg(e.what()));
    } catch (...) {
        qDebug() << "НЕИЗВЕСТНАЯ ОШИБКА в update3DSurfaces!";
        QMessageBox::critical(this, "Критическая ошибка", 
                            "Произошла неизвестная ошибка при обновлении 3D поверхности");
    }
}

// Слот для отображения тепловой карты ошибки
void MainWindow::onShowHeatmapClicked() {
    if (!solveSuccessful) {
        qDebug() << "onShowHeatmapClicked: Нет данных для отображения тепловой карты";
        return;
    }
    
    // Создаем матрицу ошибки
    // auto errorMatrix = createErrorMatrix(); // <<< REMOVE THIS LINE
    std::vector<std::vector<double>> errorMatrix; 
    if (params.solver_type == "Square Solver") { 
        // For square solver, we need a way to convert results_square.error to a 2D matrix
        // This might involve knowing n_internal and m_internal for the square case
        // For now, let's assume a similar structure or a new helper function
        // For simplicity, if results_square.error is flat, we need to reshape it.
        // This part needs to be implemented based on how SquareSolverResults are structured.
        // Placeholder for error matrix generation for square solver:
        if (!results_square.solution.empty() && params.n_internal > 0 && params.m_internal > 0 &&
            results_square.solution.size() == static_cast<size_t>(params.n_internal * params.m_internal)) {
            // For now, let's create a dummy error matrix for square, or disable heatmap if no error data
            // If results_square.error is available and has the same flat structure:
            if (!results_square.error.empty() && results_square.error.size() == results_square.solution.size()) {
                 errorMatrix.resize(params.m_internal, std::vector<double>(params.n_internal));
                 for (int r = 0; r < params.m_internal; ++r) {
                    for (int c = 0; c < params.n_internal; ++c) {
                        errorMatrix[r][c] = results_square.error[r * params.n_internal + c];
                    }
                }
            } else {
                // If no specific error data, create a matrix of zeros or NaNs for the heatmap for square
                // This is just a placeholder to allow the heatmap to be shown without actual error data.
                // Ideally, the heatmap for the square solver should be disabled if no error data is present.
                qDebug() << "onShowHeatmapClicked: No specific error data for square solver. Displaying solution magnitude or zeros.";
                errorMatrix.resize(params.m_internal, std::vector<double>(params.n_internal, 0.0));
                // As an alternative, one could display the solution itself as a heatmap if error is not applicable
                // for (int r = 0; r < params.m_internal; ++r) {
                //    for (int c = 0; c < params.n_internal; ++c) {
                //        errorMatrix[r][c] = results_square.solution[r * params.n_internal + c];
                //    }
                // }
            }
        } else {
             qDebug() << "onShowHeatmapClicked: Cannot create error/display matrix for square solver with current data.";
            QMessageBox::warning(this, "Тепловая карта", "Невозможно создать матрицу для тепловой карты квадратного решателя с текущими данными.");
            return;
        }
    } else {
        errorMatrix = createErrorMatrix(); // For G-Shape
    }
    
    // Используем новый класс HeatMapGenerator для отображения тепловой карты
    heatMapGenerator->showHeatMap(errorMatrix, "Тепловая карта ошибки", 20, 20);
}

// Метод для создания Г-образной поверхности (для вызова из UI)
void MainWindow::createGShapedSurface() {
    if (!solveSuccessful || results.solution.empty()) {
        qDebug() << "createGShapedSurface: Нет данных для отображения";
        return;
    }
    
    // Получаем коэффициент прореживания из спинбокса
    int decimationFactor = decimationFactorSpinBox ? decimationFactorSpinBox->value() : 1;
    
    // Используем полную версию метода с параметрами
    createGShapedSurface(
        results.solution,
        results.true_solution,
        results.error,
        results.x_coords,
        results.y_coords,
        decimationFactor,
        5 // Стандартное количество строк для перемычки
    );
}

// Метод для создания Г-образной поверхности с использованием нового класса GShapeRegion
void MainWindow::createGShapedSurface(
    const std::vector<double>& numericalSolution,
    const std::vector<double>& trueSolution,
    const std::vector<double>& errorValues,
    const std::vector<double>& xCoords,
    const std::vector<double>& yCoords,
    int decimationFactor,
    int connectorRows)
{
    try {
        if (!gshapeRegion || !graph3D) {
            qDebug() << "createGShapedSurface: нет объекта GShapeRegion или объекта graph3D";
            return;
        }

        if (numericalSolution.empty() || xCoords.empty() || yCoords.empty()) {
            qDebug() << "createGShapedSurface: отсутствуют исходные данные";
            return;
        }

        // Проверяем и очищаем данные от некорректных значений
        for (size_t i = 0; i < numericalSolution.size(); ++i) {
            if (std::isnan(numericalSolution[i]) || std::isinf(numericalSolution[i])) {
                qDebug() << "createGShapedSurface: Обнаружены некорректные значения в решении";
                QMessageBox::warning(this, "Предупреждение", 
                    "Некорректные значения в данных (NaN или бесконечность). Визуализация может быть некорректной.");
                break;
            }
        }

        // Используем указанный коэффициент прореживания или берем из интерфейса
        if (decimationFactor <= 0 && decimationFactorSpinBox) {
            decimationFactor = std::max(1, decimationFactorSpinBox->value());
        } else {
            decimationFactor = std::max(1, decimationFactor);
        }
        
        qDebug() << "createGShapedSurface: Создание поверхностей с коэффициентом прореживания = " << decimationFactor;

        // Создаем поверхности с помощью нового класса GShapeRegion
        bool success = false;
        
        // Оборачиваем вызов createSurfaces в отдельный try-catch для диагностики
        try {
            success = gshapeRegion->createSurfaces(
                numericalSolution,
                trueSolution,
                errorValues,
                xCoords,
                yCoords,
                params.a_bound,     // domainXMin
                params.b_bound,     // domainXMax
                params.c_bound,     // domainYMin
                params.d_bound,     // domainYMax
                decimationFactor,
                connectorRows
            );
        } catch (const std::exception& e) {
            qDebug() << "ОШИБКА при вызове gshapeRegion->createSurfaces: " << e.what();
            QMessageBox::critical(this, "Ошибка создания поверхностей", 
                QString("Произошла ошибка при создании поверхностей: %1").arg(e.what()));
            return;
        } catch (...) {
            qDebug() << "НЕИЗВЕСТНАЯ ОШИБКА при вызове gshapeRegion->createSurfaces";
            QMessageBox::critical(this, "Критическая ошибка", 
                "Произошла неизвестная ошибка при создании поверхностей");
            return;
        }

        if (success) {
            // Обновляем состояние элементов управления
            decimationFactorButton->setEnabled(true);
            showHeatMapButton->setEnabled(!errorValues.empty());

            // Устанавливаем видимость поверхностей согласно состоянию элементов управления
            gshapeRegion->setNumericalSolutionVisible(showSolutionCheckBox->isChecked());
            gshapeRegion->setTrueSolutionVisible(showTrueSolutionCheckBox->isChecked());
            gshapeRegion->setErrorSurfaceVisible(showErrorCheckBox->isChecked());

            // Обновляем заголовок графика
            graph3D->setTitle("Г-образная поверхность решения");
            
            qDebug() << "createGShapedSurface: Поверхности успешно созданы";
        } else {
            qDebug() << "createGShapedSurface: Не удалось создать поверхности";
            QMessageBox::warning(this, "Предупреждение", "Не удалось создать Г-образную поверхность");
        }
    } catch (const std::exception& e) {
        qDebug() << "ОШИБКА в createGShapedSurface: " << e.what();
        QMessageBox::critical(this, "Ошибка при создании 3D поверхности", 
                            QString("Произошла ошибка: %1").arg(e.what()));
    } catch (...) {
        qDebug() << "НЕИЗВЕСТНАЯ ОШИБКА в createGShapedSurface!";
        QMessageBox::critical(this, "Критическая ошибка", 
                            "Произошла неизвестная ошибка при создании 3D поверхности");
    }
}

// <<< ADD THIS METHOD IMPLEMENTATION
void MainWindow::update3DSurfacesSquare() {
    if (!solveSuccessful || results_square.solution.empty()) {
        qDebug() << "update3DSurfacesSquare: Решение не успешно или нет данных.";
        if (graph3D) {
            graph3D->seriesList().clear(); // Очищаем все предыдущие серии
        }
        return;
    }

    if (results_square.x_coords.empty() || results_square.y_coords.empty() ||
        results_square.x_coords.size() != results_square.solution.size() ||
        results_square.y_coords.size() != results_square.solution.size()) {
        qDebug() << "update3DSurfacesSquare: Отсутствуют или несогласованы данные решения/координат.";
        if (graph3D) {
            graph3D->seriesList().clear();
        }
        return;
    }

    if (!graph3D) {
        qDebug() << "update3DSurfacesSquare: graph3D не инициализирован.";
        return;
    }

    graph3D->seriesList().clear(); // Очищаем все предыдущие серии

    QSurfaceDataArray *dataArray = new QSurfaceDataArray;
    QSurfaceDataRow *dataRow;

    // Предполагается, что solution - это плоский вектор, а x_coords и y_coords
    // содержат соответствующие координаты для каждой точки решения.
    // Нам нужно преобразовать это в структуру сетки для QSurface3DSeries.

    // Сначала найдем уникальные отсортированные X и Y координаты
    std::vector<double> unique_x = results_square.x_coords;
    std::sort(unique_x.begin(), unique_x.end());
    unique_x.erase(std::unique(unique_x.begin(), unique_x.end()), unique_x.end());

    std::vector<double> unique_y = results_square.y_coords;
    std::sort(unique_y.begin(), unique_y.end());
    unique_y.erase(std::unique(unique_y.begin(), unique_y.end()), unique_y.end());

    // Создаем карту для быстрого поиска индекса решения по (x, y)
    std::map<std::pair<double, double>, double> solution_map;
    for (size_t i = 0; i < results_square.solution.size(); ++i) {
        // Используем некую точность для сравнения координат, если они могут быть неточными
        double rounded_x = std::round(results_square.x_coords[i] * 1e6) / 1e6;
        double rounded_y = std::round(results_square.y_coords[i] * 1e6) / 1e6;
        solution_map[{rounded_x, rounded_y}] = results_square.solution[i];
    }

    // Заполняем QSurfaceDataArray
    // Итерация по Y координатам (которые будут рядами на графике)
    for (double y_val : unique_y) {
        dataRow = new QSurfaceDataRow;
        // Итерация по X координатам (которые будут точками в ряду)
        for (double x_val : unique_x) {
            double sol_val = 0.0; // Значение по умолчанию, если точка не найдена
            double rounded_x_lookup = std::round(x_val * 1e6) / 1e6;
            double rounded_y_lookup = std::round(y_val * 1e6) / 1e6;
            auto it = solution_map.find({rounded_x_lookup, rounded_y_lookup});
            if (it != solution_map.end()) {
                sol_val = it->second;
            }
            // Для QSurface3DSeries: X это x_val, Y это sol_val, Z это y_val
            *dataRow << QVector3D(static_cast<float>(x_val), static_cast<float>(sol_val), static_cast<float>(y_val));
        }
        *dataArray << dataRow;
    }

    QSurface3DSeries *series = new QSurface3DSeries;
    series->dataProxy()->resetArray(dataArray);
    series->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
    series->setFlatShadingEnabled(true);

    graph3D->addSeries(series);

    // Устанавливаем диапазоны осей на основе фактических данных или параметров задачи
    graph3D->axisX()->setRange(static_cast<float>(params.a_bound), static_cast<float>(params.b_bound));
    graph3D->axisZ()->setRange(static_cast<float>(params.c_bound), static_cast<float>(params.d_bound));
    // Диапазон оси Y (значение решения) может быть установлен автоматически или вычислен
    // graph3D->axisY()->setAutoAdjustRange(true);

    // Настройка внешнего вида, если необходимо
    graph3D->setTitle("Численное решение (Квадратная область)");
    showSolutionCheckBox->setChecked(true); // Показываем численное решение
    showSolutionCheckBox->setEnabled(true);
    showTrueSolutionCheckBox->setChecked(false); // Нет точного решения для этой задачи
    showTrueSolutionCheckBox->setEnabled(false);
    showErrorCheckBox->setChecked(false); // Нет ошибки без точного решения
    showErrorCheckBox->setEnabled(false);

    // Управление видимостью для квадратного решателя (упрощенное)
    disconnect(showSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onSolutionSeriesVisibilityChanged);
    connect(showSolutionCheckBox, &QCheckBox::toggled, this, [this, series](bool visible){
        if (series) series->setVisible(visible);
    });
    // Для showTrueSolutionCheckBox и showErrorCheckBox обработчики не нужны, т.к. они отключены

    qDebug() << "update3DSurfacesSquare: Поверхность для квадратной области обновлена.";
}

// Методы для управления видимостью поверхностей
void MainWindow::setNumericalSolutionVisible(bool visible) {
    if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
        if (graph3D && !graph3D->seriesList().isEmpty()) {
            graph3D->seriesList().at(0)->setVisible(visible);
        }
    } else {
        if (gshapeRegion) {
            gshapeRegion->setNumericalSolutionVisible(visible);
        }
    }
}

void MainWindow::setTrueSolutionVisible(bool visible) {
    if (params.solver_type == "G-Shape Solver") { // <<< MODIFIED THIS LINE (was: if (gshapeRegion))
        if (gshapeRegion) { // <<< ADD THIS CHECK
            gshapeRegion->setTrueSolutionVisible(visible);
        }
    }
    // Для квадратного решателя эта функция не должна ничего делать, т.к. чекбокс отключен
}

void MainWindow::setErrorSurfaceVisible(bool visible) {
    if (params.solver_type == "G-Shape Solver") { // <<< MODIFIED THIS LINE (was: if (gshapeRegion))
         if (gshapeRegion) { // <<< ADD THIS CHECK
            gshapeRegion->setErrorSurfaceVisible(visible);
        }
    }
    // Для квадратного решателя эта функция не должна ничего делать, т.к. чекбокс отключен
}

// Stub implementations for missing functions
void MainWindow::onSaveResultsButtonClicked() {
    qDebug() << "onSaveResultsButtonClicked called - not implemented yet.";
    // TODO: Implement saving results functionality
    QMessageBox::information(this, "Not Implemented", "Saving results is not yet implemented.");
}

void MainWindow::onSaveMatrixButtonClicked() {
    qDebug() << "onSaveMatrixButtonClicked called - not implemented yet.";
    // TODO: Implement saving matrix functionality
    QMessageBox::information(this, "Not Implemented", "Saving matrix is not yet implemented.");
}

void MainWindow::onSaveVisualizationButtonClicked() {
    qDebug() << "onSaveVisualizationButtonClicked called - not implemented yet.";
    // TODO: Implement saving visualization functionality
    QMessageBox::information(this, "Not Implemented", "Saving visualization is not yet implemented.");
}

void MainWindow::onShowReportButtonClicked() {
    qDebug() << "onShowReportButtonClicked called - not implemented yet.";
    // TODO: Implement showing report functionality
    QMessageBox::information(this, "Not Implemented", "Showing report is not yet implemented.");
}

double MainWindow::u(double x_val, double y_val) {
    // qDebug() << "u(double, double) called - using placeholder. x:" << x_val << "y:" << y_val;
    // Placeholder for the true solution function u(x, y).
    // Replace with the actual formula for the true solution.
    // Example: return std::sin(M_PI * x_val) * std::cos(M_PI * y_val);
    return x_val * x_val + y_val * y_val; // A simple placeholder, e.g. x^2 + y^2
}

double MainWindow::x(int i, int n, double a_bound, double b_bound) {
    // qDebug() << "x(int, int, double, double) called - using placeholder. i:" << i << "n:" << n;
    // Calculates the x-coordinate of the i-th grid line (0-indexed internal node).
    // n is the number of internal divisions along x-axis (n_internal from params).
    // Grid has n+1 intervals, n internal nodes.
    if (n <= 0) return a_bound; // Should not happen with valid params.n_internal
    double hx = (b_bound - a_bound) / (static_cast<double>(n) + 1.0); // Step size
    return a_bound + static_cast<double>(i) * hx; // i is 1-indexed in some contexts, ensure consistency
                                                  // Assuming i here is 0 to n-1 for internal points
                                                  // Or 0 to n+1 for all points including boundary
                                                  // The call in createTrueSolutionMatrix uses i+1, so i is 0 to n_internal-1
}

double MainWindow::y(int j, int m, double c_bound, double d_bound) {
    // qDebug() << "y(int, int, double, double) called - using placeholder. j:" << j << "m:" << m;
    // Calculates the y-coordinate of the j-th grid line (0-indexed internal node).
    // m is the number of internal divisions along y-axis (m_internal from params).
    if (m <= 0) return c_bound; // Should not happen with valid params.m_internal
    double hy = (d_bound - c_bound) / (static_cast<double>(m) + 1.0); // Step size
    return c_bound + static_cast<double>(j) * hy; // Similar to x, j is 0 to m_internal-1 from createTrueSolutionMatrix context (j+1)
}

std::vector<std::vector<double>> MainWindow::solutionTo2D() {
    qDebug() << "solutionTo2D() called.";
    std::vector<std::vector<double>> matrix;

    if (params.n_internal <= 0 || params.m_internal <= 0) {
        qDebug() << "solutionTo2D: Invalid dimensions (n_internal or m_internal <= 0)."; // <<< MODIFIED THIS LINE
        return matrix; // Return empty matrix
    }

    matrix.resize(params.m_internal, std::vector<double>(params.n_internal));

    const std::vector<double>* sol_ptr = nullptr; // <<< ADD THIS LINE
    if (params.solver_type == "Square Solver") { // <<< ADD THIS BLOCK
        sol_ptr = &results_square.solution;
    } else {
        sol_ptr = &results.solution;
    }

    if (solveSuccessful && sol_ptr && !sol_ptr->empty()) { // <<< MODIFIED THIS LINE
        if (sol_ptr->size() == static_cast<size_t>(params.n_internal * params.m_internal)) { // <<< MODIFIED THIS LINE
            for (int r = 0; r < params.m_internal; ++r) {
                for (int c = 0; c < params.n_internal; ++c) {
                    matrix[r][c] = (*sol_ptr)[r * params.n_internal + c]; // <<< MODIFIED THIS LINE
                }
            }
            qDebug() << "solutionTo2D: Successfully reshaped solution for type: " << params.solver_type; // <<< MODIFIED THIS LINE
        } else {
            qDebug() << "solutionTo2D: Size mismatch for solver type " << params.solver_type << ". Filling with NaN."; // <<< MODIFIED THIS LINE
            for (int r = 0; r < params.m_internal; ++r) {
                std::fill(matrix[r].begin(), matrix[r].end(), std::numeric_limits<double>::quiet_NaN());
            }
        }
    } else {
        qDebug() << "solutionTo2D: No solution available or solution is empty for type " << params.solver_type << ". Filling with NaN."; // <<< MODIFIED THIS LINE
        for (int r = 0; r < params.m_internal; ++r) {
            std::fill(matrix[r].begin(), matrix[r].end(), std::numeric_limits<double>::quiet_NaN());
        }
    }
    return matrix;
}
