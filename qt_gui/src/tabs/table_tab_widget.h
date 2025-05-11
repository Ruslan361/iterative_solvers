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

class TableTabWidget : public QWidget {
    Q_OBJECT

public:
    explicit TableTabWidget(QWidget *parent = nullptr);
    
    void setCSVData(const QString& csvData);
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
    void populateTableWithData(const QString& csvData);
    
private:
    QTableWidget *dataTable;
    QSpinBox *skipFactorSpinBox;
    QPushButton *exportCSVButton;
    QPushButton *showTableButton;
    QPushButton *clearTableButton;
    QLabel *tableInfoLabel;
    QComboBox *dataTypeComboBox;
    
    QString currentCSVData;
    enum DataType {
        NUMERICAL_SOLUTION = 0,
        EXACT_SOLUTION = 1,
        ERROR = 2,
        REFINED_GRID = 3
    };
};