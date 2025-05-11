#include "visualization_3d_tab_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QDebug>

Visualization3DTabWidget::Visualization3DTabWidget(QWidget *parent)
    : QWidget(parent)
    , graph3D(nullptr)
    , currentABound(0.0)
    , currentBBound(1.0)
    , currentCBound(0.0)
    , currentDBound(1.0)
    , isSquareSolver(false)
    , useRefinedGrid(false)
    , solveSuccessful(false)
{
    setupUI();
    
    // Инициализируем генератор тепловых карт
    heatMapGenerator = std::make_unique<HeatMapGenerator>(this);
}

Visualization3DTabWidget::~Visualization3DTabWidget()
{
    // Qt автоматически очищает graph3D вместе с container
}

void Visualization3DTabWidget::setupUI()
{
    // Основной layout для вкладки
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Создаем 3D-график
    graph3D = new Q3DSurface();
    
    // Создаем контейнер для 3D-графика
    container = QWidget::createWindowContainer(graph3D);
    container->setMinimumSize(400, 300);
    container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    
    // Настраиваем оси
    graph3D->axisX()->setTitle("X");
    graph3D->axisY()->setTitle("Значение");
    graph3D->axisZ()->setTitle("Y");
    graph3D->axisX()->setTitleVisible(true);
    graph3D->axisY()->setTitleVisible(true);
    graph3D->axisZ()->setTitleVisible(true);
    graph3D->axisX()->setRange(0.0, 1.0);
    graph3D->axisZ()->setRange(0.0, 1.0);
    
    // Настраиваем камеру и вид
    graph3D->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetIsometricRight);
    graph3D->setHorizontalAspectRatio(1.0);
    graph3D->setShadowQuality(QAbstract3DGraph::ShadowQualityMedium);
    
    // Создаем элементы управления
    QGroupBox *controlsGroupBox = new QGroupBox("Управление визуализацией");
    QVBoxLayout *controlsLayout = new QVBoxLayout(controlsGroupBox);
    
    // Чекбоксы для выбора поверхностей
    showSolutionCheckBox = new QCheckBox("Показать численное решение");
    showTrueSolutionCheckBox = new QCheckBox("Показать точное решение");
    showErrorCheckBox = new QCheckBox("Показать ошибку");
    showHeatMapButton = new QPushButton("Показать тепловую карту ошибки");
    
    // Добавляем элементы управления прореживанием
    QHBoxLayout *decimationLayout = new QHBoxLayout();
    QLabel *decimationLabel = new QLabel("Коэффициент прореживания:");
    decimationFactorSpinBox = new QSpinBox();
    decimationFactorSpinBox->setMinimum(1);
    decimationFactorSpinBox->setMaximum(10);
    decimationFactorSpinBox->setValue(1);
    decimationFactorSpinBox->setToolTip("Значение 1 означает отображение всех точек, большие значения уменьшают количество отображаемых точек");
    decimationFactorButton = new QPushButton("Применить");
    
    decimationLayout->addWidget(decimationLabel);
    decimationLayout->addWidget(decimationFactorSpinBox);
    decimationLayout->addWidget(decimationFactorButton);
    
    // Устанавливаем начальные значения
    showSolutionCheckBox->setChecked(true);
    showTrueSolutionCheckBox->setChecked(false);
    showErrorCheckBox->setChecked(false);
    
    // Добавляем элементы управления в layout
    controlsLayout->addWidget(showSolutionCheckBox);
    controlsLayout->addWidget(showTrueSolutionCheckBox);
    controlsLayout->addWidget(showErrorCheckBox);
    controlsLayout->addLayout(decimationLayout);
    controlsLayout->addWidget(showHeatMapButton);
    
    // Соединяем сигналы и слоты для управления видимостью серий
    connect(showSolutionCheckBox, &QCheckBox::toggled, 
            this, &Visualization3DTabWidget::onSolutionSeriesVisibilityChanged);
    connect(showTrueSolutionCheckBox, &QCheckBox::toggled, 
            this, &Visualization3DTabWidget::onTrueSolutionSeriesVisibilityChanged);
    connect(showErrorCheckBox, &QCheckBox::toggled, 
            this, &Visualization3DTabWidget::onErrorSeriesVisibilityChanged);
    connect(showHeatMapButton, &QPushButton::clicked, 
            this, &Visualization3DTabWidget::onShowHeatMapButtonClicked);
    connect(decimationFactorButton, &QPushButton::clicked, 
            this, &Visualization3DTabWidget::onDecimationFactorButtonClicked);
    
    // Добавляем компоненты в основной layout
    mainLayout->addWidget(container, 1);
    mainLayout->addWidget(controlsGroupBox, 0);
    
    // Изначально кнопки неактивны, т.к. нет результатов
    showHeatMapButton->setEnabled(false);
    decimationFactorSpinBox->setEnabled(false);
    decimationFactorButton->setEnabled(false);
    
    setLayout(mainLayout);
}

void Visualization3DTabWidget::createOrUpdate3DSurfaces(
    const std::vector<double>& solution,
    const std::vector<double>& true_solution,
    const std::vector<double>& error,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    double a_bound, double b_bound,
    double c_bound, double d_bound,
    bool is_square_solver,
    bool use_refined_grid)
{
    if (solution.empty() || x_coords.empty() || y_coords.empty()) {
        qDebug() << "createOrUpdate3DSurfaces: Нет данных для отображения";
        return;
    }
    
    // Сохраняем данные для использования в дальнейшем
    solveSuccessful = true;
    currentSolution = solution;
    currentTrueSolution = true_solution;
    currentError = error;
    currentXCoords = x_coords;
    currentYCoords = y_coords;
    currentABound = a_bound;
    currentBBound = b_bound;
    currentCBound = c_bound;
    currentDBound = d_bound;
    isSquareSolver = is_square_solver;
    useRefinedGrid = use_refined_grid;
    
    // Настраиваем диапазоны осей
    graph3D->axisX()->setRange(a_bound, b_bound);
    graph3D->axisZ()->setRange(c_bound, d_bound);
    
    try {
        int decimationFactor = decimationFactorSpinBox->value();
        
        if (is_square_solver) {
            // Для квадратных областей
            if (!shapeRegion || !dynamic_cast<SquareShapeRegion*>(shapeRegion.get())) {
                shapeRegion = std::make_unique<SquareShapeRegion>(graph3D);
            }
        } else {
            // Для Г-образной области
            if (!shapeRegion || !dynamic_cast<GShapeRegion*>(shapeRegion.get())) {
                shapeRegion = std::make_unique<GShapeRegion>(graph3D);
            }
        }
        
        // Создаем поверхности
        shapeRegion->createSurfaces(
            solution,
            true_solution,
            error,
            x_coords,
            y_coords,
            a_bound, b_bound,
            c_bound, d_bound,
            decimationFactor
        );
        
        // Настраиваем видимость в соответствии с чекбоксами
        shapeRegion->setNumericalSolutionVisible(showSolutionCheckBox->isChecked());
        shapeRegion->setTrueSolutionVisible(showTrueSolutionCheckBox->isChecked());
        shapeRegion->setErrorSurfaceVisible(showErrorCheckBox->isChecked());
        
        // Активируем кнопки
        showHeatMapButton->setEnabled(true);
        decimationFactorSpinBox->setEnabled(true);
        decimationFactorButton->setEnabled(true);
        
        // Настраиваем чекбоксы
        showSolutionCheckBox->setEnabled(true);
        showTrueSolutionCheckBox->setEnabled(!true_solution.empty());
        showErrorCheckBox->setEnabled(!error.empty());
        
    } catch (const std::exception& e) {
        qDebug() << "Ошибка при создании 3D поверхностей: " << e.what();
    }
}

void Visualization3DTabWidget::createOrUpdateRefinedGridSurfaces(
    const std::vector<double>& solution,
    const std::vector<double>& refined_grid_solution,
    const std::vector<double>& solution_refined_diff,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    const std::vector<double>& refined_grid_x_coords,
    const std::vector<double>& refined_grid_y_coords,
    double a_bound, double b_bound,
    double c_bound, double d_bound)
{
    if (solution.empty() || refined_grid_solution.empty() || solution_refined_diff.empty()) {
        qDebug() << "createOrUpdateRefinedGridSurfaces: Нет данных для отображения";
        return;
    }
    
    // Сохраняем базовые данные
    solveSuccessful = true;
    currentSolution = solution;
    currentXCoords = x_coords;
    currentYCoords = y_coords;
    currentABound = a_bound;
    currentBBound = b_bound;
    currentCBound = c_bound;
    currentDBound = d_bound;
    isSquareSolver = true;
    useRefinedGrid = true;
    
    // Настраиваем диапазоны осей
    graph3D->axisX()->setRange(a_bound, b_bound);
    graph3D->axisZ()->setRange(c_bound, d_bound);
    
    try {
        int decimationFactor = decimationFactorSpinBox->value();
        
        // Для квадратных областей с уточненной сеткой всегда используем SquareShapeRegion
        if (!shapeRegion || !dynamic_cast<SquareShapeRegion*>(shapeRegion.get())) {
            shapeRegion = std::make_unique<SquareShapeRegion>(graph3D);
        }
        
        // Получаем указатель на SquareShapeRegion
        auto* squareRegion = dynamic_cast<SquareShapeRegion*>(shapeRegion.get());
        
        // Отключаем текущие соединения перед любыми изменениями UI и созданием поверхностей
        disconnect(showTrueSolutionCheckBox, nullptr, nullptr, nullptr);
        disconnect(showErrorCheckBox, nullptr, nullptr, nullptr);
        
        // Устанавливаем новые соединения для специфического отображения с использованием лямбда-функций
        connect(showTrueSolutionCheckBox, &QCheckBox::toggled, this, 
                [squareRegion](bool visible) {
                    if (squareRegion) {
                        squareRegion->setRefinedGridSolutionVisible(visible);
                    }
                });
                
        connect(showErrorCheckBox, &QCheckBox::toggled, this, 
                [squareRegion](bool visible) {
                    if (squareRegion) {
                        squareRegion->setSolutionRefinedDiffVisible(visible);
                    }
                });
        
        // Создаем поверхности с учетом решения на мелкой сетке
        squareRegion->createSurfacesWithRefinedGrid(
            solution,
            refined_grid_solution,
            solution_refined_diff,
            x_coords,
            y_coords,
            refined_grid_x_coords,
            refined_grid_y_coords,
            a_bound, b_bound,
            c_bound, d_bound,
            decimationFactor
        );
        
        // Настраиваем чекбоксы ПОСЛЕ создания поверхностей и установки соединений
        showSolutionCheckBox->setText("Показать численное решение v(N)(x,y)");
        showTrueSolutionCheckBox->setText("Показать решение v2(N2)(x,y) на сетке с половинным шагом");
        showErrorCheckBox->setText("Показать разность решений v(N) и v2(N2)");
        
        showSolutionCheckBox->setEnabled(true);
        showTrueSolutionCheckBox->setEnabled(true);
        showErrorCheckBox->setEnabled(true);
        
        // Теперь устанавливаем состояние чекбоксов, что запустит обновленные обработчики
        showSolutionCheckBox->setChecked(true);
        showTrueSolutionCheckBox->setChecked(true);
        showErrorCheckBox->setChecked(true);
        
        // Активируем кнопки
        showHeatMapButton->setEnabled(true);
        decimationFactorSpinBox->setEnabled(true);
        decimationFactorButton->setEnabled(true);
        
    } catch (const std::exception& e) {
        qDebug() << "Ошибка при создании 3D поверхностей с уточненной сеткой: " << e.what();
    }
}

void Visualization3DTabWidget::clear()
{
    // Очищаем поверхности
    if (shapeRegion) {
        shapeRegion.reset();
    }
    
    // Очищаем данные
    currentSolution.clear();
    currentTrueSolution.clear();
    currentError.clear();
    currentXCoords.clear();
    currentYCoords.clear();
    
    // Сбрасываем UI
    showSolutionCheckBox->setChecked(true);
    showTrueSolutionCheckBox->setChecked(false);
    showErrorCheckBox->setChecked(false);
    
    showSolutionCheckBox->setText("Показать численное решение");
    showTrueSolutionCheckBox->setText("Показать точное решение");
    showErrorCheckBox->setText("Показать ошибку");
    
    showHeatMapButton->setEnabled(false);
    decimationFactorSpinBox->setEnabled(false);
    decimationFactorButton->setEnabled(false);
    
    solveSuccessful = false;
}

void Visualization3DTabWidget::onSolutionSeriesVisibilityChanged(bool visible)
{
    if (shapeRegion) {
        shapeRegion->setNumericalSolutionVisible(visible);
    }
}

void Visualization3DTabWidget::onTrueSolutionSeriesVisibilityChanged(bool visible)
{
    if (shapeRegion) {
        if (useRefinedGrid) {
            auto* squareRegion = dynamic_cast<SquareShapeRegion*>(shapeRegion.get());
            if (squareRegion) {
                squareRegion->setRefinedGridSolutionVisible(visible);
            }
        } else {
            shapeRegion->setTrueSolutionVisible(visible);
        }
    }
}

void Visualization3DTabWidget::onErrorSeriesVisibilityChanged(bool visible)
{
    if (shapeRegion) {
        if (useRefinedGrid) {
            auto* squareRegion = dynamic_cast<SquareShapeRegion*>(shapeRegion.get());
            if (squareRegion) {
                squareRegion->setSolutionRefinedDiffVisible(visible);
            }
        } else {
            shapeRegion->setErrorSurfaceVisible(visible);
        }
    }
}

void Visualization3DTabWidget::onDecimationFactorButtonClicked()
{
    if (!shapeRegion || !solveSuccessful) {
        return;
    }
    
    int decimationFactor = decimationFactorSpinBox->value();
    
    if (useRefinedGrid) {
        // Для случая с уточненной сеткой требуется специальное обновление
        // TODO: добавить поддержку обновления с уточненной сеткой
    } else {
        shapeRegion->createSurfaces(
            currentSolution,
            currentTrueSolution,
            currentError,
            currentXCoords,
            currentYCoords,
            currentABound, currentBBound,
            currentCBound, currentDBound,
            decimationFactor
        );
        
        // Восстанавливаем видимость поверхностей
        shapeRegion->setNumericalSolutionVisible(showSolutionCheckBox->isChecked());
        shapeRegion->setTrueSolutionVisible(showTrueSolutionCheckBox->isChecked());
        shapeRegion->setErrorSurfaceVisible(showErrorCheckBox->isChecked());
    }
}

void Visualization3DTabWidget::onShowHeatMapButtonClicked()
{
    // Генерируем сигнал для внешнего обработчика
    emit showHeatMapClicked();
}