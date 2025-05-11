#pragma once

#include <QWidget>
#include <QtCharts>
#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#include <memory>
#include <vector>
#include <string>

class VisualizationTabWidget : public QWidget {
    Q_OBJECT

public:
    explicit VisualizationTabWidget(QWidget *parent = nullptr);
    
    void updateChart(const std::vector<double>& dataValues, 
                    const std::vector<double>& x_coords,
                    const std::vector<double>& y_coords,
                    const std::vector<double>& true_solution = {},
                    double a_bound = 0.0, double b_bound = 1.0, 
                    double c_bound = 0.0, double d_bound = 1.0);
    
    void clear();
    
    bool getSolveSuccessful() const { return solveSuccessful; }
    void setSolveSuccessful(bool value) { solveSuccessful = value; }
    
signals:
    void chartTypeChanged(int index);
    
private slots:
    void onSliceAxisChanged(int axis);
    void onSliceIndexChanged(int value);
    void onChartTypeChange(int index);
    
private:
    void setupUI();
    void updateSliceControls();
    
private:
    QChartView *chartView;
    QComboBox *chartTypeComboBox;
    QLabel *sliceAxisLabel;
    QComboBox *sliceAxisComboBox;
    QLabel *sliceIndexLabel;
    QSpinBox *sliceIndexSpinBox;
    QLabel *sliceInfoLabel;
    
    std::vector<double> m_unique_x_coords;
    std::vector<double> m_unique_y_coords;
    int m_currentSliceAxis = 0; // 0 для среза по Y (фиксированный X), 1 для среза по X (фиксированный Y)
    int m_currentSliceIndex = 0;
    
    bool solveSuccessful = false;
    
    // Хранение текущих данных для различных типов графиков
    std::vector<double> currentSolution;
    std::vector<double> currentError;
    std::vector<double> currentResidual;
    std::vector<double> currentTrueSolution;
    std::vector<double> currentXCoords;
    std::vector<double> currentYCoords;
    double currentABound, currentBBound, currentCBound, currentDBound;
};