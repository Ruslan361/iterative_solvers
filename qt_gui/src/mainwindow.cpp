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
    : solver(std::move(solver)) {
    // Устанавливаем колбэк для отслеживания итераций
    this->solver->setIterationCallback([this](int iteration, double precision, double residual, double error) {
        // Отправляем сигнал о прогрессе
        emit iterationUpdate(iteration, precision, residual, error);
    });
}

void SolverWorker::process() {
    try {
        // Выполняем решение в отдельном потоке
        SolverResults results = solver->solve();
        
        // Отправляем сигнал с результатами
        emit resultReady(results);
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
                        updateChartErrorVsTrue(results.error);
                        break;
                    case 2: // Невязка
                        updateChartResidual(results.residual);
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
        worker = new SolverWorker(std::move(solver));
        worker->moveToThread(solverThread);
        
        // Соединяем сигналы и слоты
        connect(solverThread, &QThread::started, worker, &SolverWorker::process);
        connect(worker, &SolverWorker::finished, this, &MainWindow::onSolverFinished);
        connect(worker, &SolverWorker::resultReady, this, &MainWindow::handleResults);
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
    if (worker && worker->getSolver()) {
        worker->getSolver()->requestStop();
        ui->progressTextEdit->append("Сигнал остановки отправлен. Ожидаем завершения текущей итерации...");
    } else {
        // Если по какой-то причине нет доступа к солверу, останавливаем поток напрямую
        cleanupThread();
        ui->progressTextEdit->append("Решение остановлено принудительно.");
    }
    
    // Обновляем интерфейс, но НЕ останавливаем поток - он завершится корректно сам
    // когда солвер проверит флаг остановки в следующей итерации
    ui->stopButton->setEnabled(false);
}

void MainWindow::setupSolver() {
    // Создаем решатель с указанными параметрами
    solver = std::make_unique<DirichletSolver>(
        params.n_internal, params.m_internal,
        params.a_bound, params.b_bound,
        params.c_bound, params.d_bound
    );
    
    // Устанавливаем критерии остановки
    double eps_precision = params.use_precision ? params.eps_precision : 0.0;
    double eps_residual = params.use_residual ? params.eps_residual : 0.0;
    double eps_exact_error = params.use_exact_error ? params.eps_exact_error : 0.0;
    int max_iterations = params.use_max_iterations ? params.max_iterations : INT_MAX;
    
    solver->setSolverParameters(eps_precision, eps_residual, eps_exact_error, max_iterations);
    
    // Обновляем информацию в UI о размере матрицы
    ui->matrixInfoLabel->setText("Подготовка матрицы...");
}

void MainWindow::handleResults(SolverResults res) {
    // Сохраняем результаты
    results = res;
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
        setupSolver();
        
        // Отображаем решение
        updateChart(results.solution);
        
        // Обновляем 3D визуализацию
        update3DSurfaces();
        
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

void MainWindow::updateChart(const std::vector<double>& solution) {
    if (solution.empty()) {
        return;
    }
    
    // Создаем серию данных для графика с точным соответствием координат
    auto *series = new QScatterSeries();
    series->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series->setMarkerSize(7.0);
    series->setName("Численное решение");
    
    // Используем сохраненные координаты точек, если они доступны
    if (!results.x_coords.empty() && !results.y_coords.empty() && 
        results.x_coords.size() == solution.size() && results.y_coords.size() == solution.size()) {
        
        for (size_t i = 0; i < solution.size(); ++i) {
            series->append(results.x_coords[i], solution[i]);
        }
    }
    // Запасной вариант, если координаты не были сохранены
    else if (solver && solver->getGridSystem()) {
        const GridSystem* grid = solver->getGridSystem();
        
        // Для каждой точки решения получаем ее физические координаты
        for (size_t i = 0; i < solution.size(); ++i) {
            GridSystem::NodeCoordinates coords = grid->get_node_coordinates(i);
            series->append(coords.x, solution[i]);
        }
    } else {
        // Если совсем нет доступа к координатам, используем индексы
        for (size_t i = 0; i < solution.size(); ++i) {
            series->append(i, solution[i]);
        }
    }
    
    auto *chart = new QChart();
    chart->addSeries(series);
    
    // Добавляем серию для отображения точного решения, если оно доступно
    if (!results.x_coords.empty()) {
        auto* trueSeries = new QLineSeries();
        trueSeries->setName("Истинное решение");
        
        // Создаем гладкую кривую истинного решения для сравнения
        const double a = params.a_bound;
        const double b = params.b_bound;
        const int numPoints = 200; // Для гладкой кривой
        
        for (int i = 0; i < numPoints; ++i) {
            double xCoord = a + i * (b - a) / (numPoints - 1);
            double yCoord = u(xCoord, (params.c_bound + params.d_bound) / 2.0); // Берем среднюю точку по Y
            trueSeries->append(xCoord, yCoord);
        }
        
        chart->addSeries(trueSeries);
    }
    
    // Настраиваем оси
    auto *axisX = new QValueAxis();
    axisX->setTitleText("X координата");
    axisX->setRange(params.a_bound, params.b_bound);
    axisX->setLabelFormat("%.2f");
    chart->addAxis(axisX, Qt::AlignBottom);
    
    auto *axisY = new QValueAxis();
    axisY->setTitleText("Значение решения");
    axisY->setLabelFormat("%.2f");
    chart->addAxis(axisY, Qt::AlignLeft);
    
    // Привязываем серии к осям
    series->attachAxis(axisX);
    series->attachAxis(axisY);
    
    if (chart->series().size() > 1) {
        chart->series()[1]->attachAxis(axisX);
        chart->series()[1]->attachAxis(axisY);
    }
    
    chart->setTitle("Численное и точное решения задачи Дирихле");
    chart->legend()->setVisible(true);
    
    ui->chartView->setChart(chart);
    ui->chartView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::updateChartErrorVsTrue(const std::vector<double>& error) {
    if (error.empty()) {
        return;
    }
    
    auto *series = new QScatterSeries();
    series->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series->setMarkerSize(5.0);
    series->setName("Ошибка");
    
    // Используем сохраненные координаты точек, если они доступны
    if (!results.x_coords.empty() && !results.y_coords.empty() && 
        results.x_coords.size() == error.size() && results.y_coords.size() == error.size()) {
        
        for (size_t i = 0; i < error.size(); ++i) {
            series->append(results.x_coords[i], std::abs(error[i]));
        }
    } else {
        // Если нет координат, отображаем по индексам
        for (size_t i = 0; i < error.size(); ++i) {
            series->append(i, std::abs(error[i]));
        }
    }
    
    auto *chart = new QChart();
    chart->addSeries(series);
    
    // Настраиваем оси
    auto *axisX = new QValueAxis();
    if (!results.x_coords.empty()) {
        axisX->setTitleText("X координата");
        axisX->setRange(params.a_bound, params.b_bound);
        axisX->setLabelFormat("%.2f");
    } else {
        axisX->setTitleText("Индекс узла");
        axisX->setLabelFormat("%i");
    }
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    
    auto *axisY = new QValueAxis();
    axisY->setTitleText("Модуль ошибки");
    axisY->setLabelFormat("%.2e");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    
    chart->setTitle("Ошибка относительно истинного решения");
    
    ui->chartView->setChart(chart);
    ui->chartView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::updateChartResidual(const std::vector<double>& residual) {
    if (residual.empty()) {
        return;
    }
    
    auto *series = new QScatterSeries();
    series->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series->setMarkerSize(5.0);
    series->setName("Невязка");
    
    // Используем сохраненные координаты точек, если они доступны
    if (!results.x_coords.empty() && !results.y_coords.empty() && 
        results.x_coords.size() == residual.size() && results.y_coords.size() == residual.size()) {
        
        for (size_t i = 0; i < residual.size(); ++i) {
            series->append(results.x_coords[i], std::abs(residual[i]));
        }
    } else {
        // Если нет координат, отображаем по индексам
        for (size_t i = 0; i < residual.size(); ++i) {
            series->append(i, std::abs(residual[i]));
        }
    }
    
    auto *chart = new QChart();
    chart->addSeries(series);
    
    // Настраиваем оси
    auto *axisX = new QValueAxis();
    if (!results.x_coords.empty()) {
        axisX->setTitleText("X координата");
        axisX->setRange(params.a_bound, params.b_bound);
        axisX->setLabelFormat("%.2f");
    } else {
        axisX->setTitleText("Индекс узла");
        axisX->setLabelFormat("%i");
    }
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    
    auto *axisY = new QValueAxis();
    axisY->setTitleText("Модуль невязки");
    axisY->setLabelFormat("%.2e");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    
    chart->setTitle("Невязка решения");
    
    ui->chartView->setChart(chart);
    ui->chartView->setRenderHint(QPainter::Antialiasing);
}

std::vector<std::vector<double>> MainWindow::solutionTo2D() {
    if (results.solution.empty()) {
        return {};
    }
    
    // Создаем 2D-матрицу для визуализации
    std::vector<std::vector<double>> matrix(params.m_internal, std::vector<double>(params.n_internal));
    
    // Заполняем матрицу из одномерного вектора результата
    // Предполагаем, что нумерация в векторе идет по строкам
    for (int i = 0; i < params.m_internal; ++i) {
        for (int j = 0; j < params.n_internal; ++j) {
            size_t idx = i * params.n_internal + j;
            if (idx < results.solution.size()) {
                matrix[i][j] = results.solution[idx];
            }
        }
    }
    
    return matrix;
}

double MainWindow::y(int j, int m, double c_bound, double d_bound) {
    double k = (d_bound - c_bound) / (m + 1);
    return c_bound + j * k;
}

double MainWindow::x(int i, int n, double a_bound, double b_bound) {
    double h = (b_bound - a_bound) / (n + 1);
    return a_bound + i * h;
}

double MainWindow::u(double x_val, double y_val) {
    return exp(pow(x_val, 2) - pow(y_val, 2)); // Пример аналитического решения
}

void MainWindow::onSaveResultsButtonClicked() {
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Предупреждение", "Нет результатов для сохранения.");
        return;
    }
    
    QString filename = QFileDialog::getSaveFileName(this, 
                                                "Сохранить результаты", 
                                                "", 
                                                "Текстовые файлы (*.txt)");
    if (filename.isEmpty())
        return;
        
    try {
        bool success = solver->saveResultsToFile(filename.toStdString());
        if (success) {
            QMessageBox::information(this, "Успех", "Результаты успешно сохранены в файл.");
        } else {
            QMessageBox::warning(this, "Предупреждение", "Не удалось сохранить результаты.");
        }
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Ошибка", 
                           QString("Ошибка при сохранении результатов: %1").arg(e.what()));
    }
}

void MainWindow::onSaveMatrixButtonClicked() {
    QString filename = QFileDialog::getSaveFileName(this, 
                                                "Сохранить матрицу и вектор правой части", 
                                                "", 
                                                "Текстовые файлы (*.txt)");
    if (filename.isEmpty())
        return;
        
    try {
        bool success = solver->saveMatrixAndRhsToFile(filename.toStdString());
        if (success) {
            QMessageBox::information(this, "Успех", "Матрица и вектор правой части успешно сохранены в файл.");
        } else {
            QMessageBox::warning(this, "Предупреждение", "Не удалось сохранить матрицу и вектор правой части.");
        }
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Ошибка", 
                           QString("Ошибка при сохранении матрицы: %1").arg(e.what()));
    }
}

void MainWindow::onSaveVisualizationButtonClicked() {
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Предупреждение", "Нет результатов для визуализации.");
        return;
    }
    
    QString filename = QFileDialog::getSaveFileName(this, 
                                                "Сохранить данные для 3D визуализации", 
                                                "", 
                                                "Файлы данных (*.dat)");
    if (filename.isEmpty())
        return;
        
    try {
        auto matrix = solutionTo2D();
        bool success = ResultsIO::saveSolutionFor3D(filename.toStdString(), matrix,
                                               params.a_bound, params.b_bound,
                                               params.c_bound, params.d_bound);
        if (success) {
            QMessageBox::information(this, "Успех", 
                "Данные для 3D визуализации успешно сохранены в файл.\n"
                "Вы можете использовать gnuplot для отображения 3D поверхности:\n"
                "gnuplot -e \"splot '" + filename + "' with pm3d\"");
        } else {
            QMessageBox::warning(this, "Предупреждение", "Не удалось сохранить данные для визуализации.");
        }
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Ошибка", 
                           QString("Ошибка при сохранении данных: %1").arg(e.what()));
    }
}

void MainWindow::onShowReportButtonClicked() {
    if (!solveSuccessful) {
        QMessageBox::warning(this, "Предупреждение", "Нет результатов для отчета.");
        return;
    }
    
    try {
        // Формируем отчет согласно шаблону
        std::stringstream report;
        report << "Для решения тестовой задачи использовалась сетка-основа с числом разбиений по x n = " 
              << params.n_internal << " и числом разбиений по y m = " << params.m_internal << "\n";
        
        report << "Метод " << (solver ? solver->getMethodName() : "неизвестен") << "\n";
        
        report << "Область [" << params.a_bound << ", " << params.b_bound << "] x [" 
              << params.c_bound << ", " << params.d_bound << "]\n";
        
        // Критерии останова
        report << "Критерии останова:\n";
        if (params.use_precision)
            report << "- Точность: " << params.eps_precision << "\n";
        if (params.use_residual)
            report << "- Невязка: " << params.eps_residual << "\n";
        if (params.use_exact_error)
            report << "- Ошибка: " << params.eps_exact_error << "\n";
        if (params.use_max_iterations)
            report << "- Макс. итераций: " << params.max_iterations << "\n";
        
        report << "\nНа решение СЛАУ затрачено итераций N = " << results.iterations
              << ", достигнута точность epsilon = " << std::scientific << results.error_norm 
              << " - норма разницы между истинным решением и численным.\n";
        
        report << "Схема СЛАУ решена с невязкой норма ||R^N|| = " << std::scientific << results.residual_norm << "\n";
        report << "Для невязки СЛАУ использована норма max";
        
        // Отображаем отчет
        ui->reportTextEdit->setText(QString::fromStdString(report.str()));
        ui->tabWidget->setCurrentIndex(2); // Переходим на вкладку с отчетом
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Ошибка", 
                           QString("Ошибка при генерации отчета: %1").arg(e.what()));
    }
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
    connect(decimationFactorButton, &QPushButton::clicked, this, &MainWindow::update3DSurfaces);
    
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
    auto errorMatrix = createErrorMatrix();
    
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

// Методы для управления видимостью поверхностей
void MainWindow::setNumericalSolutionVisible(bool visible) {
    if (gshapeRegion) {
        gshapeRegion->setNumericalSolutionVisible(visible);
    }
}

void MainWindow::setTrueSolutionVisible(bool visible) {
    if (gshapeRegion) {
        gshapeRegion->setTrueSolutionVisible(visible);
    }
}

void MainWindow::setErrorSurfaceVisible(bool visible) {
    if (gshapeRegion) {
        gshapeRegion->setErrorSurfaceVisible(visible);
    }
}
