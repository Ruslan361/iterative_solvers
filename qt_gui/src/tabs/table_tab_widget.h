#pragma once

#include <QWidget>
#include <QTableWidget>
#include <QSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QString>
#include <QFileDialog>
#include <QMessageBox>
#include <QComboBox>
#include <vector>

class TableTabWidget : public QWidget {
    Q_OBJECT

public:
    explicit TableTabWidget(QWidget *parent = nullptr);
    
    // Старый метод для работы с CSV данными
    void setCSVData(const QString& csvData);
    
    // Новый метод для заполнения таблицы данными напрямую из результатов
    void setResultsData(
        const std::vector<double>& solution,
        const std::vector<double>& true_solution,
        const std::vector<double>& error,
        const std::vector<double>& x_coords,
        const std::vector<double>& y_coords,
        bool is_square_grid = true
    );
    
    // Метод для установки данных для решения на уточненной сетке
    void setRefinedGridData(
        const std::vector<double>& refined_grid_solution,
        const std::vector<double>& refined_grid_x_coords,
        const std::vector<double>& refined_grid_y_coords
    );
    
    // Метод для управления активностью ComboBox выбора типа данных
    void setDataTypeComboEnabled(bool enabled);
    
    int getSkipFactor() const { return skipFactorSpinBox->value(); }
    
signals:
    void exportCSVRequested(int skipFactor);
    
private slots:
    void onShowTableButtonClicked();
    void onClearTableButtonClicked();
    void onExportButtonClicked();
    void onDataTypeChanged(int index);
    
private:
    void setupUI();
    
    // Метод для работы с CSV строкой
    void populateTableWithData(const QString& csvData);
    
    // Новые методы для прямой работы с данными
    void populateTableWithSolutionData(bool isNumericSolution);
    void populateTableWithErrorData();
    void populateTableWithRefinedGridData();
    void populateTableWithSolutionRefinedDiff(); // Новый метод для отображения разницы между решениями
    
private:
    QTableWidget *dataTable;
    QSpinBox *skipFactorSpinBox;
    QPushButton *exportCSVButton;
    QPushButton *showTableButton;
    QPushButton *clearTableButton;
    QLabel *tableInfoLabel;
    QComboBox *dataTypeComboBox;
    
    QString currentCSVData;
    
    // Хранение данных о результатах
    std::vector<double> m_solution;
    std::vector<double> m_true_solution;
    std::vector<double> m_error;
    std::vector<double> m_x_coords;
    std::vector<double> m_y_coords;
    std::vector<double> m_refined_solution;
    std::vector<double> m_refined_x_coords;
    std::vector<double> m_refined_y_coords;
    std::vector<double> m_solution_refined_diff; // Разница между основным и уточненным решением
    bool m_is_square_grid = true;
    bool m_has_results = false;
    bool m_has_refined_grid = false;
    
    enum DataType {
        NUMERICAL_SOLUTION = 0,
        EXACT_SOLUTION = 1,
        ERROR = 2,
        REFINED_GRID = 3,
        SOLUTION_REFINED_DIFF = 4 // Разница между основным и уточненным решением
    };
};