#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QLineSeries>
#include <QValueAxis>
#include <QChart>
#include <QChartView>
#include <QMetaObject>
#include <QDateTime>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <set>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    , solutionSeries(nullptr)
    , trueSolutionSeries(nullptr)
    , errorSeries(nullptr)
{
    // Инициализация Kokkos, если она еще не инициализирована
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    
    ui->setupUi(this);
    
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
    
    // Создаем серии для отображения поверхностей
    solutionSeries = new QSurface3DSeries();
    trueSolutionSeries = new QSurface3DSeries();
    errorSeries = new QSurface3DSeries();
    
    // Настраиваем серию численного решения
    solutionSeries->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
    solutionSeries->setFlatShadingEnabled(false);
    solutionSeries->setBaseColor(QColor(0, 0, 255)); // Синий цвет для численного решения
    solutionSeries->setMeshSmooth(true);
    solutionSeries->setName("Численное решение");
    
    // Настраиваем серию точного решения
    trueSolutionSeries->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
    trueSolutionSeries->setFlatShadingEnabled(false);
    trueSolutionSeries->setBaseColor(QColor(0, 255, 0)); // Зеленый цвет для точного решения
    trueSolutionSeries->setMeshSmooth(true);
    trueSolutionSeries->setName("Точное решение");
    trueSolutionSeries->setVisible(false); // Изначально скрыт
    
    // Настраиваем серию ошибки
    errorSeries->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
    errorSeries->setFlatShadingEnabled(false);
    errorSeries->setBaseColor(QColor(255, 0, 0)); // Красный цвет для ошибки
    errorSeries->setMeshSmooth(true);
    errorSeries->setName("Ошибка");
    errorSeries->setVisible(false); // Изначально скрыт
    
    // Добавляем серии в график
    graph3D->addSeries(solutionSeries);
    graph3D->addSeries(trueSolutionSeries);
    graph3D->addSeries(errorSeries);
    
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
    //graph3D->scene()->activeLight()->setIntensity(1.5f);
    
    // Создаем элементы управления
    QGroupBox *controlsGroupBox = new QGroupBox("Управление визуализацией");
    QVBoxLayout *controlsLayout = new QVBoxLayout(controlsGroupBox);
    
    // Чекбоксы для выбора поверхностей
    showSolutionCheckBox = new QCheckBox("Показать численное решение");
    showTrueSolutionCheckBox = new QCheckBox("Показать точное решение");
    showErrorCheckBox = new QCheckBox("Показать ошибку");
    showHeatMapButton = new QPushButton("Показать тепловую карту ошибки");
    
    // Добавляем кнопку для отображения точек (для отладки)
    QPushButton* showPointsButton = new QPushButton("Показать точки в узлах (отладка)");
    
    // Устанавливаем начальные значения
    showSolutionCheckBox->setChecked(true);
    showTrueSolutionCheckBox->setChecked(false);
    showErrorCheckBox->setChecked(false);
    
    // Добавляем элементы управления в layout
    controlsLayout->addWidget(showSolutionCheckBox);
    controlsLayout->addWidget(showTrueSolutionCheckBox);
    controlsLayout->addWidget(showErrorCheckBox);
    controlsLayout->addWidget(showHeatMapButton);
    controlsLayout->addWidget(showPointsButton);
    
    // Соединяем сигналы и слоты для управления видимостью серий
    connect(showSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onSolutionSeriesVisibilityChanged);
    connect(showTrueSolutionCheckBox, &QCheckBox::toggled, this, &MainWindow::onTrueSolutionSeriesVisibilityChanged);
    connect(showErrorCheckBox, &QCheckBox::toggled, this, &MainWindow::onErrorSeriesVisibilityChanged);
    connect(showHeatMapButton, &QPushButton::clicked, this, &MainWindow::onShowHeatmapClicked);
    connect(showPointsButton, &QPushButton::clicked, this, &MainWindow::showPointsIn3D);
    
    // Создаем основной layout для вкладки 3D-визуализации
    QVBoxLayout *mainLayout = new QVBoxLayout(visualization3DTab);
    mainLayout->addWidget(container, 1);
    mainLayout->addWidget(controlsGroupBox, 0);
    
    // Добавляем вкладку в tabWidget
    ui->tabWidget->addTab(visualization3DTab, "3D Визуализация");
    
    // Изначально кнопка тепловой карты неактивна, т.к. нет результатов
    showHeatMapButton->setEnabled(false);
}

// Слот для обработки изменения видимости серии численного решения
void MainWindow::onSolutionSeriesVisibilityChanged(bool visible) {
    if (solutionSeries) {
        solutionSeries->setVisible(visible);
    }
}

// Слот для обработки изменения видимости серии точного решения
void MainWindow::onTrueSolutionSeriesVisibilityChanged(bool visible) {
    if (trueSolutionSeries) {
        trueSolutionSeries->setVisible(visible);
    }
}

// Слот для обработки изменения видимости серии ошибки
void MainWindow::onErrorSeriesVisibilityChanged(bool visible) {
    if (errorSeries) {
        errorSeries->setVisible(visible);
    }
}

// Создание 2D матрицы точного решения
std::vector<std::vector<double>> MainWindow::createTrueSolutionMatrix() {
    std::vector<std::vector<double>> trueMatrix(params.m_internal, std::vector<double>(params.n_internal, std::numeric_limits<double>::quiet_NaN()));
    
    // Заполняем матрицу истинного решения только для точек внутри Г-образной области
    for (int i = 0; i < params.m_internal; ++i) {
        for (int j = 0; j < params.n_internal; ++j) {
            // Рассчитываем физические координаты точки
            double xCoord = x(j+1, params.n_internal, params.a_bound, params.b_bound);
            double yCoord = y(i+1, params.m_internal, params.c_bound, params.d_bound);
            
            // Определяем, находится ли точка в Г-образной области
            if (isInDomain(xCoord, yCoord)) {
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

// Создание массива данных для 3D-поверхности
QSurfaceDataArray* MainWindow::createSurfaceDataArray(const std::vector<std::vector<double>>& data) {
    QSurfaceDataArray* dataArray = new QSurfaceDataArray;
    
    // Проверка на пустые данные
    if (data.empty() || data[0].empty()) {
        return dataArray;
    }
    
    // Проверяем, есть ли у нас сохраненные координаты
    bool haveCoordinates = !results.x_coords.empty() && !results.y_coords.empty() && 
                          results.x_coords.size() == results.solution.size() && 
                          results.y_coords.size() == results.solution.size();
    
    if (haveCoordinates) {
        // Создаем равномерную сетку точек для Z-координат
        // Это обеспечит равномерное распределение точек без обрезаний
        std::set<double> uniqueYCoords;
        for (const auto& yCoord : results.y_coords) {
            uniqueYCoords.insert(yCoord);
        }
        
        // Для каждой уникальной Y-координаты создаем строку данных
        for (const auto& yPos : uniqueYCoords) {
            QSurfaceDataRow* newRow = new QSurfaceDataRow();
            
            // Собираем все точки с текущей Y-координатой
            for (size_t i = 0; i < results.solution.size(); ++i) {
                if (results.y_coords[i] == yPos) {
                    double xPos = results.x_coords[i];
                    double yValue = data[i / params.n_internal][i % params.n_internal];
                    
                    // Добавляем точку даже если она NaN - это может быть важно для сохранения формы
                    // Но если хотим фильтровать NaN, можно раскомментировать эту проверку
                    // if (!std::isnan(yValue)) {
                        newRow->append(QVector3D(xPos, yValue, yPos));
                    // }
                }
            }
            
            // Сортируем точки по X-координате
            if (!newRow->isEmpty()) {
                std::sort(newRow->begin(), newRow->end(), 
                      [](const QSurfaceDataItem& a, const QSurfaceDataItem& b) {
                          return a.x() < b.x();
                      });
                
                *dataArray << newRow;
            } else {
                delete newRow;
            }
        }
    } else {
        // Если нет сохраненных координат, создаем равномерную сетку
        int rows = data.size();
        int cols = data[0].size();
        
        // Создаем строки для каждой Z-координаты
        for (int i = 0; i < rows; ++i) {
            QSurfaceDataRow* newRow = new QSurfaceDataRow();
            
            // Вычисляем физическую Z-координату
            double zPos = y(i+1, params.m_internal, params.c_bound, params.d_bound);
            
            // Добавляем точки для этой строки
            for (int j = 0; j < cols; ++j) {
                // Вычисляем физическую X-координату
                double xPos = x(j+1, params.n_internal, params.a_bound, params.b_bound);
                double yValue = data[i][j];
                
                // Добавляем точку даже если она NaN
                newRow->append(QVector3D(xPos, yValue, zPos));
            }
            
            *dataArray << newRow;
        }
    }
    
    return dataArray;
}

// Обновление 3D-поверхностей
void MainWindow::update3DSurfaces() {
    if (!solveSuccessful) {
        qDebug() << "update3DSurfaces: Решение не успешно.";
        // Опционально очищаем серии, чтобы избежать отображения устаревших данных
        if(solutionSeries) solutionSeries->dataProxy()->resetArray(new QSurfaceDataArray());
        if(trueSolutionSeries) trueSolutionSeries->dataProxy()->resetArray(new QSurfaceDataArray());
        if(errorSeries) errorSeries->dataProxy()->resetArray(new QSurfaceDataArray());
        showHeatMapButton->setEnabled(false);
        return;
    }

    // Проверяем наличие данных координат и решения
    if (results.solution.empty() || results.x_coords.empty() || results.y_coords.empty() ||
        results.x_coords.size() != results.solution.size() || 
        results.y_coords.size() != results.solution.size()) {
        qDebug() << "update3DSurfaces: Отсутствуют или несогласованы данные решения/координат.";
        if(solutionSeries) solutionSeries->dataProxy()->resetArray(new QSurfaceDataArray());
        if(trueSolutionSeries) trueSolutionSeries->dataProxy()->resetArray(new QSurfaceDataArray());
        if(errorSeries) errorSeries->dataProxy()->resetArray(new QSurfaceDataArray());
        showHeatMapButton->setEnabled(false);
        return;
    }

    // Устанавливаем диапазоны осей на основе параметров (общий ограничивающий прямоугольник)
    graph3D->axisX()->setRange(params.a_bound, params.b_bound); // Физическая X
    graph3D->axisZ()->setRange(params.c_bound, params.d_bound); // Физическая Y (становится Z на графике)
    // Диапазон оси Y (значение) будет установлен автоматически Q3DSurface на основе данных

    // Создаем данные для поверхности решения
    QSurfaceDataArray* solutionData = createSurfaceDataArrayFromPoints(
        results.x_coords, results.y_coords, results.solution);
    solutionSeries->dataProxy()->resetArray(solutionData);

    // Создаем данные для поверхности точного решения
    if (!results.true_solution.empty() && results.true_solution.size() == results.x_coords.size()) {
        QSurfaceDataArray* trueData = createSurfaceDataArrayFromPoints(
            results.x_coords, results.y_coords, results.true_solution);
        trueSolutionSeries->dataProxy()->resetArray(trueData);
    } else {
        qDebug() << "update3DSurfaces: Данные точного решения отсутствуют или не соответствуют.";
        trueSolutionSeries->dataProxy()->resetArray(new QSurfaceDataArray()); // Очищаем их
    }

    // Создаем данные для поверхности ошибки
    if (!results.error.empty() && results.error.size() == results.x_coords.size()) {
        QSurfaceDataArray* errorData = createSurfaceDataArrayFromPoints(
            results.x_coords, results.y_coords, results.error);
        errorSeries->dataProxy()->resetArray(errorData);
    } else {
        qDebug() << "update3DSurfaces: Данные ошибки отсутствуют или не соответствуют.";
        errorSeries->dataProxy()->resetArray(new QSurfaceDataArray()); // Очищаем их
    }
    
    showHeatMapButton->setEnabled(true);
}


// Функция для отображения тепловой карты
void MainWindow::showHeatMap(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        qDebug() << "showHeatMap: Нет данных для отображения тепловой карты";
        return;
    }
    
    // Создаем новое окно для тепловой карты
    QDialog* heatmapDialog = new QDialog(this);
    heatmapDialog->setWindowTitle("Тепловая карта ошибки");
    heatmapDialog->setMinimumSize(600, 500);
    
    // Создаем сцену и вид для отображения тепловой карты
    QGraphicsScene* scene = new QGraphicsScene();
    QGraphicsView* view = new QGraphicsView(scene);
    view->setRenderHint(QPainter::Antialiasing);
    view->setDragMode(QGraphicsView::ScrollHandDrag);
    view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    
    // Находим минимальное и максимальное значение для цветовой шкалы
    double minValue = std::numeric_limits<double>::max();
    double maxValue = std::numeric_limits<double>::lowest();
    
    for (const auto& row : data) {
        for (const auto& val : row) {
            if (!std::isnan(val)) {
                minValue = std::min(minValue, val);
                maxValue = std::max(maxValue, val);
            }
        }
    }
    
    // Определяем размеры клеток тепловой карты
    int cellWidth = 20;
    int cellHeight = 20;
    int rows = data.size();
    int cols = data[0].size();
    
    // Создаем цветовую шкалу
    auto getColor = [minValue, maxValue](double value) -> QColor {
        if (std::isnan(value)) return Qt::transparent;
        
        // Нормализуем значение от 0 до 1
        double normalizedValue = (value - minValue) / (maxValue - minValue);
        
        // Интерполируем цвет от синего (холодный) к красному (горячий)
        if (normalizedValue <= 0.0)
            return Qt::blue;
        else if (normalizedValue <= 0.25)
            return QColor::fromRgbF(0, 0, 1.0 - normalizedValue * 4, 1.0); // Blue to Cyan
        else if (normalizedValue <= 0.5)
            return QColor::fromRgbF(0, normalizedValue * 4 - 1.0, 1.0, 1.0); // Cyan to Green
        else if (normalizedValue <= 0.75)
            return QColor::fromRgbF((normalizedValue - 0.5) * 4, 1.0, 1.0 - (normalizedValue - 0.5) * 4, 1.0); // Green to Yellow
        else if (normalizedValue < 1.0)
            return QColor::fromRgbF(1.0, 1.0 - (normalizedValue - 0.75) * 4, 0, 1.0); // Yellow to Red
        else
            return Qt::red;
    };
    
    // Рисуем тепловую карту
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double value = data[i][j];
            if (!std::isnan(value)) { // Пропускаем NaN значения (точки вне области)
                QColor color = getColor(value);
                QGraphicsRectItem* rect = new QGraphicsRectItem(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
                rect->setBrush(QBrush(color));
                rect->setPen(QPen(Qt::black, 0.5));
                rect->setToolTip(QString("X: %1, Y: %2, Значение: %3").arg(j).arg(i).arg(value, 0, 'e', 4));
                scene->addItem(rect);
            }
        }
    }
    
    // Устанавливаем размер сцены
    scene->setSceneRect(0, 0, cols * cellWidth, rows * cellHeight);
    
    // Создаем легенду
    QGroupBox* legendBox = new QGroupBox("Легенда");
    QGridLayout* legendLayout = new QGridLayout(legendBox);
    
    int legendSteps = 10;
    for (int i = 0; i <= legendSteps; ++i) {
        double value = minValue + (maxValue - minValue) * i / legendSteps;
        QColor color = getColor(value);
        
        // Цветной прямоугольник
        QLabel* colorLabel = new QLabel();
        colorLabel->setFixedSize(20, 20);
        colorLabel->setStyleSheet(QString("background-color: %1; border: 1px solid black;").arg(color.name()));
        
        // Значение
        QLabel* valueLabel = new QLabel(QString::number(value, 'e', 4));
        
        legendLayout->addWidget(colorLabel, i, 0);
        legendLayout->addWidget(valueLabel, i, 1);
    }
    
    // Создаем информационный блок с статистикой
    QGroupBox* statsBox = new QGroupBox("Статистика");
    QFormLayout* statsLayout = new QFormLayout(statsBox);
    
    statsLayout->addRow(new QLabel("Минимальное значение:"), new QLabel(QString::number(minValue, 'e', 6)));
    statsLayout->addRow(new QLabel("Максимальное значение:"), new QLabel(QString::number(maxValue, 'e', 6)));
    
    double average = 0.0;
    int count = 0;
    for (const auto& row : data) {
        for (const auto& val : row) {
            if (!std::isnan(val)) {
                average += val;
                count++;
            }
        }
    }
    average = (count > 0) ? average / count : 0.0;
    statsLayout->addRow(new QLabel("Среднее значение:"), new QLabel(QString::number(average, 'e', 6)));
    
    // Создаем компоновку для диалогового окна
    QVBoxLayout* mainLayout = new QVBoxLayout(heatmapDialog);
    
    QHBoxLayout* controlsLayout = new QHBoxLayout();
    controlsLayout->addWidget(legendBox);
    controlsLayout->addWidget(statsBox);
    
    mainLayout->addWidget(new QLabel("Тепловая карта ошибки:"));
    mainLayout->addWidget(view);
    mainLayout->addLayout(controlsLayout);
    
    // Добавляем кнопку экспорта
    QPushButton* exportButton = new QPushButton("Экспорт в PNG");
    connect(exportButton, &QPushButton::clicked, [scene, view]() {
        QString filename = QFileDialog::getSaveFileName(view, "Сохранить тепловую карту", 
                                                      "heatmap.png", "PNG (*.png)");
        if (!filename.isEmpty()) {
            QPixmap pixmap(scene->sceneRect().size().toSize());
            pixmap.fill(Qt::white);
            QPainter painter(&pixmap);
            scene->render(&painter);
            pixmap.save(filename);
        }
    });
    mainLayout->addWidget(exportButton);
    
    // Показываем диалог
    heatmapDialog->exec();
}

// Слот для отображения тепловой карты ошибки
void MainWindow::onShowHeatmapClicked() {
    if (!solveSuccessful) {
        qDebug() << "onShowHeatmapClicked: Нет данных для отображения тепловой карты";
        return;
    }
    
    // Создаем матрицу ошибки
    auto errorMatrix = createErrorMatrix();
    
    // Отображаем тепловую карту
    showHeatMap(errorMatrix);
}

// Проверка принадлежности точки Г-образной области
bool MainWindow::isInDomain(double x, double y) {
    // Параметры для Г-образной области (инвертированной L)
    // Предполагаем, что область состоит из трёх "квадрантов"
    
    // Координаты "разделителей" для Г-образной области
    // Эти координаты нужно настроить в зависимости от вашей конкретной геометрии
    double x_split = (params.a_bound + params.b_bound) / 2.0;
    double y_split = (params.c_bound + params.d_bound) / 2.0;
    
    // Проверяем, в каком "квадранте" находится точка
    bool is_in_q1 = (x <= x_split && y <= y_split);      // Нижний-левый квадрант
    bool is_in_q2 = (x > x_split && y <= y_split);       // Нижний-правый квадрант
    bool is_in_q3 = (x <= x_split && y > y_split);       // Верхний-левый квадрант
    // bool is_in_q4 = (x > x_split && y > y_split);     // Верхний-правый квадрант (исключён из области)
    
    // Точка принадлежит области, если находится в одном из трёх квадрантов
    return is_in_q1 || is_in_q2 || is_in_q3;
}

// Создание 3D-поверхности из векторов координат и значений
QSurfaceDataArray* MainWindow::createSurfaceDataArrayFromPoints(
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& values) {
    
    QSurfaceDataArray* dataArray = new QSurfaceDataArray;
    
    // Проверка входных данных
    if (x_coords.empty() || y_coords.empty() || values.empty() ||
        x_coords.size() != values.size() || y_coords.size() != values.size()) {
        qDebug() << "createSurfaceDataArrayFromPoints: Пустые или несоответствующие размеры входных векторов.";
        return dataArray; // Возвращаем пустой массив
    }
    
    // Группируем точки по физической Y-координате
    // Y-координата станет осью Z в Q3DSurface
    // Значение (решение, ошибка) станет осью Y в Q3DSurface
    // Физическая X-координата станет осью X в Q3DSurface
    std::map<double, std::vector<std::pair<double, double>>> rows_map; // Ключ: y_physical, Значение: список (x_physical, value)
    
    for (size_t i = 0; i < values.size(); ++i) {
        // NaN-значения для 'value' допустимы и создадут "дыры" в поверхности
        // Проверяем, что точка принадлежит Г-образной области
        if (isInDomain(x_coords[i], y_coords[i]) || true) {
            rows_map[y_coords[i]].push_back({x_coords[i], values[i]});
        }
    }
    
    // Для каждого уникального y_physical создаём QSurfaceDataRow, отсортированный по x_physical
    for (auto const& [y_phys_val, points_in_row_unsorted] : rows_map) {
        QSurfaceDataRow* newRow = new QSurfaceDataRow();
        
        // Копируем для сортировки (итератор map даёт константную ссылку)
        std::vector<std::pair<double, double>> sorted_points = points_in_row_unsorted;
        
        std::sort(sorted_points.begin(), sorted_points.end(),
                 [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                     return a.first < b.first; // Сортируем по x_physical
                 });
        
        for (const auto& point : sorted_points) {
            // QVector3D(plot_X_axis, plot_Y_axis, plot_Z_axis)
            // plot_X_axis = physical X
            // plot_Y_axis = value (solution, error)
            // plot_Z_axis = physical Y (y_phys_val)
            newRow->append(QVector3D(point.first, point.second, y_phys_val));
        }
        
        if (!newRow->isEmpty()) {
            *dataArray << newRow;
        } else {
            // Это не должно произойти, если rows_map[y_phys_val] непустой
            delete newRow;
        }
    }
    
    return dataArray;
}

// Функция для отображения точек в 3D без построения поверхности (для отладки)
void MainWindow::showPointsIn3D() {
    // Проверяем наличие данных
    if (!solveSuccessful || results.solution.empty() || 
        results.x_coords.empty() || results.y_coords.empty()) {
        qDebug() << "showPointsIn3D: Нет данных для отображения";
        return;
    }

    // Очистить текущие серии
    if (graph3D) {
        graph3D->removeSeries(solutionSeries);
        graph3D->removeSeries(trueSolutionSeries);
        graph3D->removeSeries(errorSeries);
    }

    // Создаем новую серию для точек (QSurface3DSeries вместо QScatter3DSeries)
    QSurface3DSeries* pointsSeries = new QSurface3DSeries();
    
    // Настраиваем отображение точек
    pointsSeries->setDrawMode(QSurface3DSeries::DrawSurface);
    pointsSeries->setFlatShadingEnabled(true);
    pointsSeries->setBaseColor(QColor(0, 0, 255)); // Синий цвет
    pointsSeries->setMeshSmooth(false);
    pointsSeries->setName("Точки в узлах");
    
    // Создаем массив данных для точек
    QSurfaceDataArray* pointsDataArray = new QSurfaceDataArray();
    
    // Группируем точки по их Y-координатам
    std::map<double, std::vector<std::pair<double, double>>> y_groups;
    
    for (size_t i = 0; i < results.solution.size(); ++i) {
        if (isInDomain(results.x_coords[i], results.y_coords[i]) && 
            !std::isnan(results.solution[i])) {
            // Группируем точки по Y-координате
            y_groups[results.y_coords[i]].push_back({results.x_coords[i], results.solution[i]});
        }
    }
    
    // Создаем ряды точек для каждой уникальной Y-координаты
    for (const auto& [y_coord, points] : y_groups) {
        QSurfaceDataRow* row = new QSurfaceDataRow(points.size());
        
        // Заполняем ряд точками
        for (size_t i = 0; i < points.size(); ++i) {
            (*row)[i] = QVector3D(points[i].first, points[i].second, y_coord);
        }
        
        // Добавляем ряд в массив данных
        *pointsDataArray << row;
    }
    
    // Устанавливаем данные в серию
    pointsSeries->dataProxy()->resetArray(pointsDataArray);
    
    // Добавляем серию в график
    graph3D->addSeries(pointsSeries);
    
    // Настраиваем отображение точек без сглаживания
    //graph3D->setMeshSmoothing(false);
    
    // Устанавливаем диапазоны осей
    graph3D->axisX()->setRange(params.a_bound, params.b_bound);
    graph3D->axisZ()->setRange(params.c_bound, params.d_bound);
    
    // Переходим на вкладку 3D
    ui->tabWidget->setCurrentIndex(3); // Предполагаем, что 3D вкладка имеет индекс 3
    
    // Обновляем заголовок
    graph3D->setTitle("Точки в узлах (режим отладки)");
}
