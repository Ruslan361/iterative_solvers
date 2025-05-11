#pragma once

#include <QWidget>
#include <QProgressBar>
#include <QLabel>
#include <QTableWidget>
#include <QString>
#include <QComboBox>

// Структура для хранения данных итерации
struct IterationData {
    int iteration;
    double precision;
    double residual;
    double error;
};

class ProgressTabWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ProgressTabWidget(QWidget *parent = nullptr);
    
    // Методы для обновления информации о прогрессе
    void updateIterationInfo(int iteration, double precision, double residual, double error);
    void updateSolverFinished(bool success, int iterations, double residualNorm, 
                             double errorNorm, double precision, bool converged,
                             const std::string& stopReason);
    
    void clearProgress();
    void setMaxIterations(int max);
    void setRefinedGridError(double error);

private:
    void setupUI();
    void updateChart();
    
    QProgressBar* progressBar;
    QLabel* iterationLabel;
    QLabel* precisionLabel;
    QLabel* residualLabel;
    QLabel* errorLabel;
    
    QLabel* totalIterationsLabel;
    QLabel* finalResidualLabel;
    QLabel* finalErrorLabel;
    QLabel* finalPrecisionLabel;
    QLabel* convergenceLabel;
    QLabel* stopReasonLabel;
    QLabel* refinedGridErrorLabel;
    
    QTableWidget* iterationTable;
    
    std::vector<IterationData> iterationHistory;
    int maxIterations = 10000;
};