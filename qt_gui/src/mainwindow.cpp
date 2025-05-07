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
#include <iomanip>

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
}

MainWindow::~MainWindow() {
    // Убедимся, что поток корректно остановлен и удален
    cleanupThread();
    
    delete ui;
    
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
    }
    
    // Очищаем поток
    solverThread = nullptr;
    worker = nullptr;
}

void MainWindow::updateChart(const std::vector<double>& solution) {
    if (solution.empty()) {
        return;
    }
    
    // Преобразуем одномерный вектор в двумерную сетку для визуализации
    auto matrix = solutionTo2D();
    
    // Создаем серию данных для 3D поверхности, но визуализируем как 2D тепловую карту
    auto *series = new QLineSeries();
    
    // Линейная развертка 2D-массива в 1D для визуализации
    for (size_t i = 0; i < solution.size(); ++i) {
        series->append(i, solution[i]);
    }
    
    auto *chart = new QChart();
    chart->addSeries(series);
    
    // Настраиваем оси
    auto *axisX = new QValueAxis();
    axisX->setTitleText("Индекс узла");
    axisX->setLabelFormat("%i");
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    
    auto *axisY = new QValueAxis();
    axisY->setTitleText("Значение решения");
    axisY->setLabelFormat("%.2f");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    
    chart->setTitle("Численное решение задачи Дирихле");
    
    ui->chartView->setChart(chart);
    ui->chartView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::updateChartErrorVsTrue(const std::vector<double>& error) {
    if (error.empty()) {
        return;
    }
    
    auto *series = new QLineSeries();
    
    // Визуализируем ошибку
    for (size_t i = 0; i < error.size(); ++i) {
        series->append(i, std::abs(error[i]));
    }
    
    auto *chart = new QChart();
    chart->addSeries(series);
    
    // Настраиваем оси
    auto *axisX = new QValueAxis();
    axisX->setTitleText("Индекс узла");
    axisX->setLabelFormat("%i");
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
    
    auto *series = new QLineSeries();
    
    // Визуализируем невязку
    for (size_t i = 0; i < residual.size(); ++i) {
        series->append(i, std::abs(residual[i]));
    }
    
    auto *chart = new QChart();
    chart->addSeries(series);
    
    // Настраиваем оси
    auto *axisX = new QValueAxis();
    axisX->setTitleText("Индекс узла");
    axisX->setLabelFormat("%i");
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