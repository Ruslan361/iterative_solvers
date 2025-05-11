#pragma once

#include <QWidget>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QGridLayout>

// Виджет настроек решателя
class SolverTabWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SolverTabWidget(QWidget *parent = nullptr);
    ~SolverTabWidget() = default;

    // Методы для получения параметров решателя
    int getNInternal() const;
    int getMInternal() const;
    double getABound() const;
    double getBBound() const;
    double getCBound() const;
    double getDBound() const;
    double getEpsPrecision() const;
    double getEpsResidual() const;
    double getEpsExactError() const;
    int getMaxIterations() const;
    bool getUsePrecision() const;
    bool getUseResidual() const;
    bool getUseExactError() const;
    bool getUseMaxIterations() const;
    bool getUseRefinedGrid() const;
    QString getSolverType() const;

    // Управление кнопками
    void setSolveButtonEnabled(bool enabled);
    void setStopButtonEnabled(bool enabled);
    
    // Обновление информации о матрице
    void updateMatrixInfo(const QString& info);

signals:
    void solveButtonClicked();
    void stopButtonClicked();

private slots:
    void onSolveButtonClicked();
    void onStopButtonClicked();
    void onSolverTypeChanged(int index);

private:
    void setupUI();
    void updateControlStates();

private:
    // Элементы UI для настройки параметров
    QSpinBox *nInternalSpinBox;
    QSpinBox *mInternalSpinBox;
    QDoubleSpinBox *aBoundSpinBox;
    QDoubleSpinBox *bBoundSpinBox;
    QDoubleSpinBox *cBoundSpinBox;
    QDoubleSpinBox *dBoundSpinBox;
    QDoubleSpinBox *epsPrecisionSpinBox;
    QDoubleSpinBox *epsResidualSpinBox;
    QDoubleSpinBox *epsExactErrorSpinBox;
    QSpinBox *maxIterationsSpinBox;
    QCheckBox *usePrecisionCheckBox;
    QCheckBox *useResidualCheckBox;
    QCheckBox *useExactErrorCheckBox;
    QCheckBox *useMaxIterationsCheckBox;
    QCheckBox *useRefinedGridCheckBox;
    QComboBox *solverTypeComboBox;
    
    // Кнопки управления
    QPushButton *solveButton;
    QPushButton *stopButton;
    
    // Информационные метки
    QLabel *matrixInfoLabel;
};