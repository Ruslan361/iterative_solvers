#pragma once

#include <QWidget>
#include <QtDataVisualization/QtDataVisualization>
#include <QCheckBox>
#include <QPushButton>
#include <QSpinBox>
#include <memory>
#include <vector>
#include "shaperegion.h"
#include "gshaperegion.h"
#include "squareshaperegion.h"
#include "heatmapgenerator.h"


class Visualization3DTabWidget : public QWidget {
    Q_OBJECT

public:
    explicit Visualization3DTabWidget(QWidget *parent = nullptr);
    ~Visualization3DTabWidget();
    
    void createOrUpdate3DSurfaces(
        const std::vector<double>& solution,
        const std::vector<double>& true_solution,
        const std::vector<double>& error,
        const std::vector<double>& x_coords,
        const std::vector<double>& y_coords,
        double a_bound, double b_bound,
        double c_bound, double d_bound,
        bool is_square_solver = false,
        bool use_refined_grid = false);
        
    void createOrUpdateRefinedGridSurfaces(
        const std::vector<double>& solution,
        const std::vector<double>& refined_grid_solution,
        const std::vector<double>& solution_refined_diff,
        const std::vector<double>& x_coords,
        const std::vector<double>& y_coords,
        const std::vector<double>& refined_grid_x_coords,
        const std::vector<double>& refined_grid_y_coords,
        double a_bound, double b_bound,
        double c_bound, double d_bound);
        
    void clear();
    bool getSolveSuccessful() const { return solveSuccessful; }
    void setSolveSuccessful(bool value) { solveSuccessful = value; }
    
signals:
    void showHeatMapClicked();
    
public slots:
    void onSolutionSeriesVisibilityChanged(bool visible);
    void onTrueSolutionSeriesVisibilityChanged(bool visible);
    void onErrorSeriesVisibilityChanged(bool visible);
    
private slots:
    void onDecimationFactorButtonClicked();
    void onShowHeatMapButtonClicked();
    
private:
    void setupUI();
    
private:
    Q3DSurface *graph3D;
    QWidget *container;
    
    QCheckBox *showSolutionCheckBox;
    QCheckBox *showTrueSolutionCheckBox;
    QCheckBox *showErrorCheckBox;
    QPushButton *showHeatMapButton;
    QSpinBox *decimationFactorSpinBox;
    QPushButton *decimationFactorButton;
    
    // Для хранения текущих данных
    std::vector<double> currentSolution;
    std::vector<double> currentTrueSolution;
    std::vector<double> currentError;
    std::vector<double> currentXCoords;
    std::vector<double> currentYCoords;
    double currentABound, currentBBound, currentCBound, currentDBound;
    bool isSquareSolver;
    bool useRefinedGrid;
    bool solveSuccessful = false;
    
    // Класс для управления областью
    std::unique_ptr<ShapeRegion> shapeRegion;
    
    // Класс для генерации тепловых карт
    std::unique_ptr<HeatMapGenerator> heatMapGenerator;
};
