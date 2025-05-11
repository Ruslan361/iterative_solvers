#include "visualization_tab_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <limits>
#include <algorithm>
#include <cmath>
#include <set>

VisualizationTabWidget::VisualizationTabWidget(QWidget *parent)
    : QWidget(parent)
    , currentABound(0.0)
    , currentBBound(1.0)
    , currentCBound(0.0)
    , currentDBound(1.0)
{
    setupUI();
}

void VisualizationTabWidget::setupUI()
{
    // Основной layout для вкладки
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Элементы управления для выбора типа графика и среза
    QGroupBox *controlsGroup = new QGroupBox("Управление визуализацией");
    QGridLayout *controlsLayout = new QGridLayout;
    
    // Тип графика
    QLabel *chartTypeLabel = new QLabel("Тип графика:");
    chartTypeComboBox = new QComboBox();
    chartTypeComboBox->addItem("Решение");
    chartTypeComboBox->addItem("Ошибка");
    chartTypeComboBox->addItem("Невязка");
    
    // Элементы управления срезом
    sliceAxisLabel = new QLabel("Ось среза:");
    sliceAxisComboBox = new QComboBox();
    sliceAxisComboBox->addItem("Срез по X (фиксированный Y)");
    sliceAxisComboBox->addItem("Срез по Y (фиксированный X)");
    
    sliceIndexLabel = new QLabel("Индекс среза:");
    sliceIndexSpinBox = new QSpinBox();
    sliceIndexSpinBox->setMinimum(0);
    sliceIndexSpinBox->setValue(0);
    
    sliceInfoLabel = new QLabel("Нет данных для среза");
    
    // Размещаем элементы управления в сетке
    controlsLayout->addWidget(chartTypeLabel, 0, 0);
    controlsLayout->addWidget(chartTypeComboBox, 0, 1);
    controlsLayout->addWidget(sliceAxisLabel, 1, 0);
    controlsLayout->addWidget(sliceAxisComboBox, 1, 1);
    controlsLayout->addWidget(sliceIndexLabel, 2, 0);
    controlsLayout->addWidget(sliceIndexSpinBox, 2, 1);
    controlsLayout->addWidget(sliceInfoLabel, 3, 0, 1, 2);
    
    controlsGroup->setLayout(controlsLayout);
    
    // График
    chartView = new QChartView();
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->setMinimumHeight(400);
    
    // Соединяем сигналы и слоты
    connect(chartTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &VisualizationTabWidget::onChartTypeChange);
    connect(sliceAxisComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &VisualizationTabWidget::onSliceAxisChanged);
    connect(sliceIndexSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &VisualizationTabWidget::onSliceIndexChanged);
    
    // Добавляем элементы в основной layout
    mainLayout->addWidget(controlsGroup);
    mainLayout->addWidget(chartView);
    
    setLayout(mainLayout);
}

void VisualizationTabWidget::updateChart(const std::vector<double>& dataValues, 
                                        const std::vector<double>& x_coords,
                                        const std::vector<double>& y_coords,
                                        const std::vector<double>& true_solution,
                                        double a_bound, double b_bound, 
                                        double c_bound, double d_bound)
{
    if (dataValues.empty() || x_coords.empty() || y_coords.empty()) {
        // Очищаем график если нет данных
        QChart *chart = new QChart();
        chartView->setChart(chart);
        return;
    }
    
    // Сохраняем данные для использования в дальнейшем
    currentSolution = dataValues;
    currentXCoords = x_coords;
    currentYCoords = y_coords;
    currentTrueSolution = true_solution;
    currentABound = a_bound;
    currentBBound = b_bound;
    currentCBound = c_bound;
    currentDBound = d_bound;
    
    // Извлекаем уникальные отсортированные координаты для срезов
    m_unique_x_coords.clear();
    m_unique_y_coords.clear();
    
    std::set<double> unique_x_set(x_coords.begin(), x_coords.end());
    m_unique_x_coords.assign(unique_x_set.begin(), unique_x_set.end());
    
    std::set<double> unique_y_set(y_coords.begin(), y_coords.end());
    m_unique_y_coords.assign(unique_y_set.begin(), unique_y_set.end());
    
    updateSliceControls();
    
    // Получаем индекс типа графика
    int chartTypeIndex = chartTypeComboBox->currentIndex();
    
    // Создаем серии для выбранного среза
    auto *series = new QLineSeries();
    auto *trueSeries = new QLineSeries();
    
    QString chartTitle = "";
    QString xAxisTitle = "";
    QString yAxisTitle = "Значение";
    
    // Определяем какие данные отображать в зависимости от типа графика
    const std::vector<double>* currentData = nullptr;
    const std::vector<double>* trueSolutionDataForPlot = nullptr;
    QString dataTypeString = "";
    
    if (chartTypeIndex == 0) { // Решение
        currentData = &currentSolution;
        if (!currentTrueSolution.empty() && currentTrueSolution.size() == currentSolution.size()) {
            trueSolutionDataForPlot = &currentTrueSolution;
        }
        dataTypeString = "Решение";
    } else if (chartTypeIndex == 1) { // Ошибка
        dataTypeString = "Ошибка";
        // TODO: Заполнение данных ошибки, если они доступны
    } else if (chartTypeIndex == 2) { // Невязка
        dataTypeString = "Невязка";
        // TODO: Заполнение данных невязки, если они доступны
    }
    
    if (!currentData || currentData->empty()) {
        // Очищаем график если нет данных для отображения
        chartView->setChart(new QChart());
        delete series;
        delete trueSeries;
        return;
    }
    
    series->setName(QString("Численное %1").arg(dataTypeString));
    if (trueSolutionDataForPlot) {
        trueSeries->setName(QString("Истинное %1").arg(dataTypeString));
    }
    
    if (m_currentSliceAxis == 0 && !m_unique_x_coords.empty() && 
        m_currentSliceIndex < static_cast<int>(m_unique_x_coords.size())) { 
        // Срез по Y (фиксированный X)
        double fixed_x = m_unique_x_coords[m_currentSliceIndex];
        chartTitle = QString("%1 при X = %2 (срез по Y)").arg(dataTypeString).arg(fixed_x);
        xAxisTitle = "Y координата";
        
        for (size_t i = 0; i < x_coords.size(); ++i) {
            if (std::abs(x_coords[i] - fixed_x) < 1e-9) {
                if (i < currentData->size()) {
                    series->append(y_coords[i], (*currentData)[i]);
                }
                if (trueSolutionDataForPlot && i < trueSolutionDataForPlot->size()) {
                    trueSeries->append(y_coords[i], (*trueSolutionDataForPlot)[i]);
                }
            }
        }
    } else if (m_currentSliceAxis == 1 && !m_unique_y_coords.empty() && 
               m_currentSliceIndex < static_cast<int>(m_unique_y_coords.size())) { 
        // Срез по X (фиксированный Y)
        double fixed_y = m_unique_y_coords[m_currentSliceIndex];
        chartTitle = QString("%1 при Y = %2 (срез по X)").arg(dataTypeString).arg(fixed_y);
        xAxisTitle = "X координата";
        
        for (size_t i = 0; i < y_coords.size(); ++i) {
            if (std::abs(y_coords[i] - fixed_y) < 1e-9) {
                if (i < currentData->size()) {
                    series->append(x_coords[i], (*currentData)[i]);
                }
                if (trueSolutionDataForPlot && i < trueSolutionDataForPlot->size()) {
                    trueSeries->append(x_coords[i], (*trueSolutionDataForPlot)[i]);
                }
            }
        }
    } else {
        // Fallback или нет данных для среза
        chartTitle = QString("Нет данных для среза (%1)").arg(dataTypeString);
    }
    
    auto *chart = new QChart();
    
    chart->addSeries(series);
    bool trueSeriesAdded = false;
    
    if (trueSolutionDataForPlot && trueSeries->points().size() > 0) {
        chart->addSeries(trueSeries);
        trueSeriesAdded = true;
    } else {
        delete trueSeries;
        trueSeries = nullptr;
    }
    
    auto *axisX = new QValueAxis();
    axisX->setTitleText(xAxisTitle);
    
    if (m_currentSliceAxis == 0 && !m_unique_y_coords.empty()) {
        axisX->setRange(c_bound, d_bound);
    }
    else if (m_currentSliceAxis == 1 && !m_unique_x_coords.empty()) {
        axisX->setRange(a_bound, b_bound);
    }
    
    axisX->setLabelFormat("%.2f");
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    if (trueSeriesAdded) {
        trueSeries->attachAxis(axisX);
    }
    
    auto *axisY = new QValueAxis();
    axisY->setTitleText(yAxisTitle);
    axisY->setLabelFormat("%.2e"); // Используем экспоненциальную запись для оси Y
    
    // Вычисляем и устанавливаем диапазон оси Y
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
        dataFoundForRange = true;
        for (const QPointF &p : trueSeries->points()) {
            if (p.y() < minVal) minVal = p.y();
            if (p.y() > maxVal) maxVal = p.y();
        }
    }
    
    if (dataFoundForRange) {
        if (minVal == maxVal) {
            minVal -= 0.5;
            maxVal += 0.5;
        }
        
        if (minVal > maxVal) {
            std::swap(minVal, maxVal);
        }
        
        double rangeValue = maxVal - minVal;
        double padding = rangeValue * 0.05;
        
        if (padding == 0.0 && rangeValue == 0.0) {
            padding = 0.05;
        } else if (padding == 0.0 && rangeValue > 0.0) {
            padding = rangeValue * 0.05 + 1e-9;
        }
        
        axisY->setRange(minVal - padding, maxVal + padding);
    }
    
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    if (trueSeriesAdded) {
        trueSeries->attachAxis(axisY);
    }
    
    chart->setTitle(chartTitle);
    chart->legend()->setVisible(true);
    
    chartView->setChart(chart);
}

void VisualizationTabWidget::clear()
{
    // Очищаем график и данные
    auto *chart = new QChart();
    chartView->setChart(chart);
    
    currentSolution.clear();
    currentError.clear();
    currentResidual.clear();
    currentTrueSolution.clear();
    currentXCoords.clear();
    currentYCoords.clear();
    
    m_unique_x_coords.clear();
    m_unique_y_coords.clear();
    
    updateSliceControls();
}

void VisualizationTabWidget::onSliceAxisChanged(int axis)
{
    m_currentSliceAxis = axis;
    m_currentSliceIndex = 0; // Сбрасываем индекс среза при смене оси
    updateSliceControls();
    
    // Обновляем график, если есть данные
    if (!currentSolution.empty() && !currentXCoords.empty() && !currentYCoords.empty()) {
        updateChart(currentSolution, currentXCoords, currentYCoords, currentTrueSolution, 
                   currentABound, currentBBound, currentCBound, currentDBound);
    }
}

void VisualizationTabWidget::onSliceIndexChanged(int value)
{
    m_currentSliceIndex = value;
    updateSliceControls();
    
    // Обновляем график, если есть данные
    if (!currentSolution.empty() && !currentXCoords.empty() && !currentYCoords.empty()) {
        updateChart(currentSolution, currentXCoords, currentYCoords, currentTrueSolution, 
                   currentABound, currentBBound, currentCBound, currentDBound);
    }
}

void VisualizationTabWidget::onChartTypeChange(int index)
{
    // Сообщаем об изменении типа графика
    emit chartTypeChanged(index);
    
    // Обновляем график, если есть данные
    if (!currentSolution.empty() && !currentXCoords.empty() && !currentYCoords.empty()) {
        updateChart(currentSolution, currentXCoords, currentYCoords, currentTrueSolution, 
                   currentABound, currentBBound, currentCBound, currentDBound);
    }
}

void VisualizationTabWidget::updateSliceControls()
{
    if (!solveSuccessful || (currentSolution.empty() && currentError.empty() && currentResidual.empty())) {
        sliceIndexSpinBox->setEnabled(false);
        sliceAxisComboBox->setEnabled(false);
        sliceInfoLabel->setText("Нет данных для среза");
        sliceIndexSpinBox->setRange(0, 0);
        return;
    }
    
    sliceIndexSpinBox->setEnabled(true);
    sliceAxisComboBox->setEnabled(true);
    
    if (m_currentSliceAxis == 0) { // Срез по Y (фиксированный X)
        sliceIndexSpinBox->setRange(0, m_unique_x_coords.empty() ? 0 : m_unique_x_coords.size() - 1);
        if (!m_unique_x_coords.empty() && m_currentSliceIndex < static_cast<int>(m_unique_x_coords.size())) {
            sliceInfoLabel->setText(QString("X = %1").arg(m_unique_x_coords[m_currentSliceIndex]));
        } else {
            sliceInfoLabel->setText("X: нет данных");
        }
    } else { // Срез по X (фиксированный Y)
        sliceIndexSpinBox->setRange(0, m_unique_y_coords.empty() ? 0 : m_unique_y_coords.size() - 1);
        if (!m_unique_y_coords.empty() && m_currentSliceIndex < static_cast<int>(m_unique_y_coords.size())) {
            sliceInfoLabel->setText(QString("Y = %1").arg(m_unique_y_coords[m_currentSliceIndex]));
        } else {
            sliceInfoLabel->setText("Y: нет данных");
        }
    }
    
    sliceIndexSpinBox->setValue(m_currentSliceIndex); // Обновляем значение в спинбоксе
}