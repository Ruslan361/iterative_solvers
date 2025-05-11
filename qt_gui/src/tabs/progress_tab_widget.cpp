#include "progress_tab_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QGridLayout>
#include <QHeaderView>
#include <QDateTime>
#include <QScrollBar>
#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QValueAxis>



ProgressTabWidget::ProgressTabWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

void ProgressTabWidget::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // Группа текущей итерации
    QGroupBox* currentIterationGroup = new QGroupBox("Текущая итерация");
    QGridLayout* currentIterationLayout = new QGridLayout(currentIterationGroup);
    
    // Прогресс-бар
    progressBar = new QProgressBar();
    progressBar->setRange(0, maxIterations);
    progressBar->setValue(0);
    
    // Метки для отображения текущей информации
    QLabel* iterationTitleLabel = new QLabel("Итерация:");
    iterationLabel = new QLabel("0");
    
    QLabel* precisionTitleLabel = new QLabel("Точность:");
    precisionLabel = new QLabel("0");
    
    QLabel* residualTitleLabel = new QLabel("Невязка:");
    residualLabel = new QLabel("0");
    
    QLabel* errorTitleLabel = new QLabel("Ошибка:");
    errorLabel = new QLabel("0");
    
    currentIterationLayout->addWidget(iterationTitleLabel, 0, 0);
    currentIterationLayout->addWidget(iterationLabel, 0, 1);
    currentIterationLayout->addWidget(precisionTitleLabel, 1, 0);
    currentIterationLayout->addWidget(precisionLabel, 1, 1);
    currentIterationLayout->addWidget(residualTitleLabel, 0, 2);
    currentIterationLayout->addWidget(residualLabel, 0, 3);
    currentIterationLayout->addWidget(errorTitleLabel, 1, 2);
    currentIterationLayout->addWidget(errorLabel, 1, 3);
    
    currentIterationLayout->addWidget(progressBar, 2, 0, 1, 4);
    
    // Группа результатов
    QGroupBox* resultsGroup = new QGroupBox("Результаты");
    QGridLayout* resultsLayout = new QGridLayout(resultsGroup);
    
    QLabel* totalIterationsTitleLabel = new QLabel("Всего итераций:");
    totalIterationsLabel = new QLabel("-");
    
    QLabel* finalPrecisionTitleLabel = new QLabel("Итоговая точность:");
    finalPrecisionLabel = new QLabel("-");
    
    QLabel* finalResidualTitleLabel = new QLabel("Итоговая невязка:");
    finalResidualLabel = new QLabel("-");
    
    QLabel* finalErrorTitleLabel = new QLabel("Итоговая ошибка:");
    finalErrorLabel = new QLabel("-");
    
    QLabel* convergenceTitleLabel = new QLabel("Сходимость:");
    convergenceLabel = new QLabel("-");
    
    QLabel* stopReasonTitleLabel = new QLabel("Причина остановки:");
    stopReasonLabel = new QLabel("-");
    
    QLabel* refinedGridErrorTitleLabel = new QLabel("Ошибка на уточненной сетке:");
    refinedGridErrorLabel = new QLabel("-");
    
    resultsLayout->addWidget(totalIterationsTitleLabel, 0, 0);
    resultsLayout->addWidget(totalIterationsLabel, 0, 1);
    resultsLayout->addWidget(finalPrecisionTitleLabel, 1, 0);
    resultsLayout->addWidget(finalPrecisionLabel, 1, 1);
    resultsLayout->addWidget(finalResidualTitleLabel, 0, 2);
    resultsLayout->addWidget(finalResidualLabel, 0, 3);
    resultsLayout->addWidget(finalErrorTitleLabel, 1, 2);
    resultsLayout->addWidget(finalErrorLabel, 1, 3);
    resultsLayout->addWidget(convergenceTitleLabel, 2, 0);
    resultsLayout->addWidget(convergenceLabel, 2, 1);
    resultsLayout->addWidget(stopReasonTitleLabel, 2, 2);
    resultsLayout->addWidget(stopReasonLabel, 2, 3);
    resultsLayout->addWidget(refinedGridErrorTitleLabel, 3, 0);
    resultsLayout->addWidget(refinedGridErrorLabel, 3, 1, 1, 3);
    
    // График сходимости
    QChartView* chartView = new QChartView();
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->chart()->setTitle("График сходимости");
    chartView->setMinimumHeight(200);
    
    // Таблица с историей итераций
    iterationTable = new QTableWidget();
    iterationTable->setColumnCount(4);
    QStringList headers;
    headers << "Итерация" << "Точность" << "Невязка" << "Ошибка";
    iterationTable->setHorizontalHeaderLabels(headers);
    iterationTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    iterationTable->verticalHeader()->setVisible(false);
    iterationTable->setMinimumHeight(150);
    
    // Добавляем все компоненты в главный layout
    mainLayout->addWidget(currentIterationGroup);
    mainLayout->addWidget(resultsGroup);
    mainLayout->addWidget(chartView, 1); // Даем графику больше места
    mainLayout->addWidget(iterationTable, 1);
}

void ProgressTabWidget::updateIterationInfo(int iteration, double precision, double residual, double error)
{
    // Обновляем информацию о текущей итерации
    iterationLabel->setText(QString::number(iteration));
    precisionLabel->setText(QString::number(precision, 'e', 6));
    residualLabel->setText(QString::number(residual, 'e', 6));
    errorLabel->setText(QString::number(error, 'e', 6));
    
    // Обновляем прогресс-бар
    progressBar->setValue(iteration);
    
    // Добавляем данные в историю итераций
    IterationData data = {iteration, precision, residual, error};
    iterationHistory.push_back(data);
    
    // Добавляем строку в таблицу
    int row = iterationTable->rowCount();
    iterationTable->insertRow(row);
    
    iterationTable->setItem(row, 0, new QTableWidgetItem(QString::number(iteration)));
    iterationTable->setItem(row, 1, new QTableWidgetItem(QString::number(precision, 'e', 6)));
    iterationTable->setItem(row, 2, new QTableWidgetItem(QString::number(residual, 'e', 6)));
    iterationTable->setItem(row, 3, new QTableWidgetItem(QString::number(error, 'e', 6)));
    
    // Прокручиваем таблицу к последней строке
    iterationTable->scrollToBottom();
    
    // Обновляем график
    updateChart();
}

void ProgressTabWidget::updateSolverFinished(bool success, int iterations, double residualNorm, double errorNorm, double precision, bool converged, const std::string& stopReason)
{
    // Обновляем информацию о результатах
    totalIterationsLabel->setText(QString::number(iterations));
    finalResidualLabel->setText(QString::number(residualNorm, 'e', 6));
    finalErrorLabel->setText(QString::number(errorNorm, 'e', 6));
    finalPrecisionLabel->setText(QString::number(precision, 'e', 6));
    
    convergenceLabel->setText(converged ? "Да" : "Нет");
    stopReasonLabel->setText(QString::fromStdString(stopReason));
    
    // Устанавливаем цвет для метки сходимости
    if (converged) {
        convergenceLabel->setStyleSheet("color: green; font-weight: bold;");
    } else {
        convergenceLabel->setStyleSheet("color: red; font-weight: bold;");
    }
    
    // Обновляем прогресс-бар
    progressBar->setValue(iterations);
    progressBar->setFormat(success ? "Завершено" : "Прервано");
    
    // Если решение не сошлось, устанавливаем красный цвет для прогресс-бара
    if (!converged) {
        progressBar->setStyleSheet("QProgressBar::chunk { background-color: #cc0000; }");
    } else {
        progressBar->setStyleSheet("QProgressBar::chunk { background-color: #00cc00; }");
    }
}

void ProgressTabWidget::clearProgress()
{
    // Очищаем информацию о текущей итерации
    iterationLabel->setText("0");
    precisionLabel->setText("0");
    residualLabel->setText("0");
    errorLabel->setText("0");
    
    // Очищаем информацию о результатах
    totalIterationsLabel->setText("-");
    finalResidualLabel->setText("-");
    finalErrorLabel->setText("-");
    finalPrecisionLabel->setText("-");
    convergenceLabel->setText("-");
    stopReasonLabel->setText("-");
    refinedGridErrorLabel->setText("-");
    
    // Сбрасываем прогресс-бар
    progressBar->setValue(0);
    progressBar->setFormat("%p%");
    progressBar->setStyleSheet("");
    
    // Очищаем таблицу
    iterationTable->setRowCount(0);
    
    // Очищаем историю итераций
    iterationHistory.clear();
    
    // Обновляем график
    updateChart();
}

void ProgressTabWidget::setMaxIterations(int max)
{
    maxIterations = max;
    progressBar->setMaximum(max);
}

void ProgressTabWidget::setRefinedGridError(double error)
{
    refinedGridErrorLabel->setText(QString::number(error, 'e', 6));
}

void ProgressTabWidget::updateChart()
{
    if (iterationHistory.empty()) {
        return;
    }

    // Находим QChartView в layout
    QChartView* chartView = nullptr;
    for (int i = 0; i < layout()->count(); ++i) {
        QLayoutItem* item = layout()->itemAt(i);
        if (QChartView* view = qobject_cast<QChartView*>(item->widget())) {
            chartView = view;
            break;
        }
    }

    if (!chartView) {
        return;
    }

    // Создаем новый график
    QChart* chart = new QChart();
    chart->setTitle("График сходимости");

    // Создаем серии данных
    QLineSeries* precisionSeries = new QLineSeries();
    precisionSeries->setName("Точность");

    QLineSeries* residualSeries = new QLineSeries();
    residualSeries->setName("Невязка");

    QLineSeries* errorSeries = new QLineSeries();
    errorSeries->setName("Ошибка");

    // Заполняем серии данными
    for (size_t i = 0; i < iterationHistory.size(); ++i) {
        const auto& data = iterationHistory[i];
        
        // Используем логарифмический масштаб для значений
        precisionSeries->append(data.iteration, data.precision > 0 ? log10(data.precision) : -16);
        residualSeries->append(data.iteration, data.residual > 0 ? log10(data.residual) : -16);
        errorSeries->append(data.iteration, data.error > 0 ? log10(data.error) : -16);
    }

    // Добавляем серии на график
    chart->addSeries(precisionSeries);
    chart->addSeries(residualSeries);
    chart->addSeries(errorSeries);

    // Создаем оси
    QValueAxis* axisX = new QValueAxis();
    axisX->setTitleText("Итерация");
    axisX->setLabelFormat("%d");
    
    QValueAxis* axisY = new QValueAxis();
    axisY->setTitleText("log10(значение)");
    axisY->setLabelFormat("%g");

    // Устанавливаем оси для графика
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    
    // Привязываем серии к осям
    precisionSeries->attachAxis(axisX);
    precisionSeries->attachAxis(axisY);
    residualSeries->attachAxis(axisX);
    residualSeries->attachAxis(axisY);
    errorSeries->attachAxis(axisX);
    errorSeries->attachAxis(axisY);

    // Устанавливаем диапазон осей
    auto last = iterationHistory.back();
    axisX->setRange(0, last.iteration + 5);
    axisY->setRange(-16, 0);

    // Обновляем график в виджете
    chartView->setChart(chart);
}
