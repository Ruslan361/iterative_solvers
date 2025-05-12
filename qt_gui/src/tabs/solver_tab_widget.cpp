#include "solver_tab_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QLabel>
#include <QGridLayout>
#include <QSizePolicy>

SolverTabWidget::SolverTabWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

void SolverTabWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Группа настроек для выбора типа решателя
    QGroupBox *solverTypeGroup = new QGroupBox("Тип решения");
    QVBoxLayout *solverTypeLayout = new QVBoxLayout(solverTypeGroup);
    
    solverTypeComboBox = new QComboBox();
    solverTypeComboBox->addItem("Ступень 2: Тестовая задача");
    solverTypeComboBox->addItem("Ступень 2: Основная задача");
    solverTypeComboBox->addItem("Ступень 3: G-образная область");
    solverTypeLayout->addWidget(solverTypeComboBox);
    
    // Группа настроек для размера сетки
    QGroupBox *gridSizeGroup = new QGroupBox("Параметры сетки");
    QGridLayout *gridSizeLayout = new QGridLayout(gridSizeGroup);
    
    QLabel *nInternalLabel = new QLabel("Внутренние узлы по X:");
    nInternalSpinBox = new QSpinBox();
    nInternalSpinBox->setRange(3, 10000);
    nInternalSpinBox->setValue(30);
    
    QLabel *mInternalLabel = new QLabel("Внутренние узлы по Y:");
    mInternalSpinBox = new QSpinBox();
    mInternalSpinBox->setRange(3, 10000);
    mInternalSpinBox->setValue(30);
    
    gridSizeLayout->addWidget(nInternalLabel, 0, 0);
    gridSizeLayout->addWidget(nInternalSpinBox, 0, 1);
    gridSizeLayout->addWidget(mInternalLabel, 1, 0);
    gridSizeLayout->addWidget(mInternalSpinBox, 1, 1);
    
    // Группа настроек для границ области
    QGroupBox *boundaryGroup = new QGroupBox("Границы области");
    QGridLayout *boundaryLayout = new QGridLayout(boundaryGroup);
    
    QLabel *aBoundLabel = new QLabel("a (левая граница):");
    aBoundSpinBox = new QDoubleSpinBox();
    aBoundSpinBox->setRange(-1000.0, 1000.0);
    aBoundSpinBox->setValue(1.0);
    aBoundSpinBox->setDecimals(3);
    
    QLabel *bBoundLabel = new QLabel("b (правая граница):");
    bBoundSpinBox = new QDoubleSpinBox();
    bBoundSpinBox->setRange(-1000.0, 1000.0);
    bBoundSpinBox->setValue(2.0);
    bBoundSpinBox->setDecimals(3);
    
    QLabel *cBoundLabel = new QLabel("c (нижняя граница):");
    cBoundSpinBox = new QDoubleSpinBox();
    cBoundSpinBox->setRange(-1000.0, 1000.0);
    cBoundSpinBox->setValue(1.0);
    cBoundSpinBox->setDecimals(3);
    
    QLabel *dBoundLabel = new QLabel("d (верхняя граница):");
    dBoundSpinBox = new QDoubleSpinBox();
    dBoundSpinBox->setRange(-1000.0, 1000.0);
    dBoundSpinBox->setValue(2.0);
    dBoundSpinBox->setDecimals(3);
    
    boundaryLayout->addWidget(aBoundLabel, 0, 0);
    boundaryLayout->addWidget(aBoundSpinBox, 0, 1);
    boundaryLayout->addWidget(bBoundLabel, 0, 2);
    boundaryLayout->addWidget(bBoundSpinBox, 0, 3);
    boundaryLayout->addWidget(cBoundLabel, 1, 0);
    boundaryLayout->addWidget(cBoundSpinBox, 1, 1);
    boundaryLayout->addWidget(dBoundLabel, 1, 2);
    boundaryLayout->addWidget(dBoundSpinBox, 1, 3);
    
    // Группа настроек для критериев останова
    QGroupBox *stopCriteriaGroup = new QGroupBox("Критерии останова");
    QGridLayout *stopCriteriaLayout = new QGridLayout(stopCriteriaGroup);
    
    usePrecisionCheckBox = new QCheckBox("По точности:");
    epsPrecisionSpinBox = new QDoubleSpinBox();
    epsPrecisionSpinBox->setRange(1e-15, 1e-1);
    epsPrecisionSpinBox->setValue(1e-6);
    epsPrecisionSpinBox->setDecimals(15);
    epsPrecisionSpinBox->setSingleStep(1e-6);
    
    useResidualCheckBox = new QCheckBox("По невязке:");
    epsResidualSpinBox = new QDoubleSpinBox();
    epsResidualSpinBox->setRange(1e-15, 1e-1);
    epsResidualSpinBox->setValue(1e-6);
    epsResidualSpinBox->setDecimals(15);
    epsResidualSpinBox->setSingleStep(1e-6);
    
    useExactErrorCheckBox = new QCheckBox("По ошибке:");
    epsExactErrorSpinBox = new QDoubleSpinBox();
    epsExactErrorSpinBox->setRange(1e-15, 1e-1);
    epsExactErrorSpinBox->setValue(1e-6);
    epsExactErrorSpinBox->setDecimals(15);
    epsExactErrorSpinBox->setSingleStep(1e-6);
    
    useMaxIterationsCheckBox = new QCheckBox("По числу итераций:");
    maxIterationsSpinBox = new QSpinBox();
    maxIterationsSpinBox->setRange(1, 1000000);
    maxIterationsSpinBox->setValue(10000);
    
    // Установка начальных значений для чекбоксов
    usePrecisionCheckBox->setChecked(true);
    useResidualCheckBox->setChecked(true);
    useExactErrorCheckBox->setChecked(false);
    useMaxIterationsCheckBox->setChecked(true);
    
    stopCriteriaLayout->addWidget(usePrecisionCheckBox, 0, 0);
    stopCriteriaLayout->addWidget(epsPrecisionSpinBox, 0, 1);
    stopCriteriaLayout->addWidget(useResidualCheckBox, 1, 0);
    stopCriteriaLayout->addWidget(epsResidualSpinBox, 1, 1);
    stopCriteriaLayout->addWidget(useExactErrorCheckBox, 2, 0);
    stopCriteriaLayout->addWidget(epsExactErrorSpinBox, 2, 1);
    stopCriteriaLayout->addWidget(useMaxIterationsCheckBox, 3, 0);
    stopCriteriaLayout->addWidget(maxIterationsSpinBox, 3, 1);
    
    // Дополнительные опции
    QGroupBox *additionalOptionsGroup = new QGroupBox("Дополнительные опции");
    QVBoxLayout *additionalOptionsLayout = new QVBoxLayout(additionalOptionsGroup);
    
    useRefinedGridCheckBox = new QCheckBox("Использовать уточненную сетку");
    useRefinedGridCheckBox->setToolTip("Использовать двойную сетку для оценки погрешности решения");
    useRefinedGridCheckBox->setChecked(false);
    additionalOptionsLayout->addWidget(useRefinedGridCheckBox);
    
    // Информация о матрице
    matrixInfoLabel = new QLabel("Информация о матрице:");
    matrixInfoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    
    // Кнопки управления
    QHBoxLayout *buttonsLayout = new QHBoxLayout();
    
    solveButton = new QPushButton("Решить");
    stopButton = new QPushButton("Остановить");
    stopButton->setEnabled(false);
    
    buttonsLayout->addWidget(solveButton);
    buttonsLayout->addWidget(stopButton);
    
    // Добавляем группы в основной layout
    mainLayout->addWidget(solverTypeGroup);
    mainLayout->addWidget(gridSizeGroup);
    mainLayout->addWidget(boundaryGroup);
    mainLayout->addWidget(stopCriteriaGroup);
    mainLayout->addWidget(additionalOptionsGroup);
    mainLayout->addWidget(matrixInfoLabel);
    mainLayout->addLayout(buttonsLayout);
    mainLayout->addStretch();
    
    // Соединяем сигналы и слоты
    connect(solveButton, &QPushButton::clicked, this, &SolverTabWidget::onSolveButtonClicked);
    connect(stopButton, &QPushButton::clicked, this, &SolverTabWidget::onStopButtonClicked);
    connect(solverTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &SolverTabWidget::onSolverTypeChanged);
    onSolverTypeChanged(0);
}

void SolverTabWidget::onSolveButtonClicked()
{
    emit solveButtonClicked();
}

void SolverTabWidget::onStopButtonClicked()
{
    emit stopButtonClicked();
}

void SolverTabWidget::onSolverTypeChanged(int index)
{
    // Обновляем доступность опций в зависимости от выбранного типа решателя
    bool isTestTask = (index == 0); // "Ступень 2: Тестовая задача" - индекс 0
    bool isSquareMainTask = (index == 1); // "Ступень 2: Основная задача" - индекс 1
    bool isGShapeTask = (index == 2); // "Ступень 3" - индекс 2
    
    // Критерий по ошибке доступен только для тестовой задачи и G-образной области, где известно точное решение
    useExactErrorCheckBox->setEnabled(isTestTask || isGShapeTask);
    epsExactErrorSpinBox->setEnabled((isTestTask || isGShapeTask));
    
    // Уточненная сетка доступна только для основной задачи (ступень 2)

    if (isSquareMainTask)
    {
        useRefinedGridCheckBox->setCheckState(Qt::Checked);
    }
    else {
        useRefinedGridCheckBox->setCheckState(Qt::Unchecked);
    }
    useRefinedGridCheckBox->setEnabled(isSquareMainTask);
    // Обновляем диапазоны значений для параметров сетки
    if (isSquareMainTask || isTestTask) {
        // Для квадратной области рекомендуем равное количество узлов по X и Y
        if (nInternalSpinBox->value() != mInternalSpinBox->value()) {
            mInternalSpinBox->setValue(nInternalSpinBox->value());
        }
    }
}

// Геттеры для параметров решателя
int SolverTabWidget::getNInternal() const
{
    return nInternalSpinBox->value();
}

int SolverTabWidget::getMInternal() const
{
    return mInternalSpinBox->value();
}

double SolverTabWidget::getABound() const
{
    return aBoundSpinBox->value();
}

double SolverTabWidget::getBBound() const
{
    return bBoundSpinBox->value();
}

double SolverTabWidget::getCBound() const
{
    return cBoundSpinBox->value();
}

double SolverTabWidget::getDBound() const
{
    return dBoundSpinBox->value();
}

double SolverTabWidget::getEpsPrecision() const
{
    return epsPrecisionSpinBox->value();
}

double SolverTabWidget::getEpsResidual() const
{
    return epsResidualSpinBox->value();
}

double SolverTabWidget::getEpsExactError() const
{
    return epsExactErrorSpinBox->value();
}

int SolverTabWidget::getMaxIterations() const
{
    return maxIterationsSpinBox->value();
}

bool SolverTabWidget::getUsePrecision() const
{
    return usePrecisionCheckBox->isChecked();
}

bool SolverTabWidget::getUseResidual() const
{
    return useResidualCheckBox->isChecked();
}

bool SolverTabWidget::getUseExactError() const
{
    return useExactErrorCheckBox->isChecked();
}

bool SolverTabWidget::getUseMaxIterations() const
{
    return useMaxIterationsCheckBox->isChecked();
}

bool SolverTabWidget::getUseRefinedGrid() const
{
    return useRefinedGridCheckBox->isChecked();
}

QString SolverTabWidget::getSolverType() const
{
    return solverTypeComboBox->currentText();
}

void SolverTabWidget::setSolveButtonEnabled(bool enabled)
{
    solveButton->setEnabled(enabled);
}

void SolverTabWidget::setStopButtonEnabled(bool enabled)
{
    stopButton->setEnabled(enabled);
}

void SolverTabWidget::updateMatrixInfo(const QString& info)
{
    matrixInfoLabel->setText(info);
}
