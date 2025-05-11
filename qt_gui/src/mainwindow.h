#pragma once

#include <QMainWindow>
#include <QtCharts>
#include <QSpinBox>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QGridLayout>
#include <QTimer>
#include <QTextEdit>
#include <QProgressBar>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QThread>
#include <memory>
#include <vector>
#include <string>
#include <fstream>

// Подключаем заголовки решателя
#include "dirichlet_solver.hpp"
#include "grid_system.h"
#include "dirichlet_solver_square.hpp"
#include "solver.hpp"
#include "dirichlet_solver_square.hpp" // Для SquareSolverResults
#include "dirichlet_solver.hpp"      // Для SolverResults

// Подключаем модуль 3D-визуализации Qt
#include <QtDataVisualization/QtDataVisualization>

// Подключаем классы для визуализации
#include "shaperegion.h"
#include "gshaperegion.h"
#include "squareshaperegion.h"
#include "heatmapgenerator.h"


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

// Класс рабочего потока для решателя
class SolverWorker : public QObject {
    Q_OBJECT

public:
    explicit SolverWorker(std::unique_ptr<DirichletSolver> solver);
    explicit SolverWorker(std::unique_ptr<DirichletSolverSquare> solver_sq);
    ~SolverWorker() = default;
    
    // Getter метод для доступа к solver
    DirichletSolver* getSolver() { return solver.get(); }
    DirichletSolverSquare* getSolverSquare() { return solver_sq.get(); }

public slots:
    void process();

signals:
    void resultReady(SolverResults results);
    void resultReadySquare(SquareSolverResults results);
    void finished();
    void iterationUpdate(int iteration, double precision, double residual, double error);

private:
    std::unique_ptr<DirichletSolver> solver;
    std::unique_ptr<DirichletSolverSquare> solver_sq;
    bool is_square_solver = false;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    
    // Метод для создания и отображения поверхности в соответствии с областью
    void createSurfaceForDomain();
    
    // Расширенный метод для создания поверхности в соответствии с областью
    void createSurfaceForDomain(
        const std::vector<double>& numericalSolution,
        const std::vector<double>& trueSolution,
        const std::vector<double>& errorValues,
        const std::vector<double>& xCoords,
        const std::vector<double>& yCoords,
        int decimationFactor
    );
    
public slots:
    // Слоты для управления видимостью поверхностей
    void setNumericalSolutionVisible(bool visible);
    void setTrueSolutionVisible(bool visible);
    void setErrorSurfaceVisible(bool visible);
    
private slots:
    void onSolveButtonClicked();
    void onStopButtonClicked();
    void handleResults(const SolverResults& results);
    void handleResultsSquare(const SquareSolverResults& results_sq);
    void updateIterationInfo(int iter, double res_norm, double err_norm, double prec_norm);
    void onSolverFinished();
    void onSaveResultsButtonClicked();
    void onSaveMatrixButtonClicked();
    void onSaveVisualizationButtonClicked();
    void onShowReportButtonClicked();
    void onShowHeatMapClicked(); // Добавлено объявление для функции с большой буквой M
    void onSolutionSeriesVisibilityChanged(bool visible);
    void onTrueSolutionSeriesVisibilityChanged(bool visible);
    void onErrorSeriesVisibilityChanged(bool visible);
    void onChartTypeChanged(int index);
    void onSliceAxisChanged(int axis);
    void onSliceIndexChanged(int value);
    void onTabChanged(int index);      // Слот для смены вкладок
    void createOrUpdate3DSurfaces(); // Слот для создания/обновления поверхностей
    
    // Новые слоты для работы с таблицами
    void onExportCSVButtonClicked();
    void onShowTableButtonClicked();
    void onClearTableButtonClicked();

private:
    Ui::MainWindow *ui;
    
    // Объект решателя
    std::unique_ptr<DirichletSolver> solver;
    std::unique_ptr<DirichletSolverSquare> solver_square;
    
    // Результаты решения
    SolverResults results;
    SquareSolverResults results_square;
    
    // Поток для решателя
    QThread* solverThread = nullptr;
    
    // Рабочий объект для решателя
    SolverWorker* worker = nullptr;
    
    // Флаг для отслеживания состояния решения
    bool isSolving;
    
    // Флаг для успешного завершения решения
    bool solveSuccessful;
    
    // Функции для работы с решателем
    void setupSolver();
    void updateChart(const std::vector<double>& solution);
    void updateChartErrorVsTrue(const std::vector<double>& error);
    void updateChartResidual(const std::vector<double>& residual);
    
    // Преобразование 1D вектора решения в 2D для визуализации
    std::vector<std::vector<double>> solutionTo2D();
    
    // Создание 2D матрицы истинного решения
    std::vector<std::vector<double>> createTrueSolutionMatrix();
    
    // Создание 2D матрицы ошибки (разности решений)
    std::vector<std::vector<double>> createErrorMatrix();
    
    // Вспомогательные функции
    double y(int j, int m, double c_bound, double d_bound);
    double x(int i, int n, double a_bound, double b_bound);
    double u(double x_val, double y_val);
    
    // Для обновления UI итераций
    struct IterationData {
        int iteration;
        double precision;
        double residual;
        double error;
    };
    std::vector<IterationData> iterationHistory;
    
    // Хранение параметров задачи
    struct SolverParams {
        int n_internal;
        int m_internal;
        double a_bound;
        double b_bound;
        double c_bound;
        double d_bound;
        double eps_precision;
        double eps_residual;
        double eps_exact_error;
        int max_iterations;
        bool use_precision;
        bool use_residual;
        bool use_exact_error;
        bool use_max_iterations;
        bool use_refined_grid; // Флаг для сравнения с решением на более мелкой сетке
        QString solver_type;
    } params;
    
    // Метод для очистки потока после завершения
    void cleanupThread();
    
    // Методы для 3D визуализации
    void setup3DVisualization();
    void update3DSurfaces();
    void update3DSurfacesSquare();
    
    // 3D визуализация объекты
    QWidget* visualization3DTab;
    Q3DSurface* graph3D;
    
    // Класс для управления областью (полиморфно)
    std::unique_ptr<ShapeRegion> shapeRegion;
    
    // Класс для генерации тепловых карт
    std::unique_ptr<HeatMapGenerator> heatMapGenerator;
    
    QCheckBox *showSolutionCheckBox;
    QCheckBox *showTrueSolutionCheckBox;
    QCheckBox *showErrorCheckBox;
    QPushButton *showHeatMapButton;
    QSpinBox *decimationFactorSpinBox;
    QPushButton *decimationFactorButton;
    
    // Индекс вкладки 3D визуализации
    static const int vizTabIndex = 3; // Индекс вкладки 3D визуализации (третья вкладка)

    // UI elements for 2D chart slicing
    QLabel *sliceAxisLabel;
    QComboBox *sliceAxisComboBox;
    QLabel *sliceIndexLabel;
    QSpinBox *sliceIndexSpinBox;
    QLabel *sliceInfoLabel;


    // Helper members for 2D chart slicing
    std::vector<double> m_unique_x_coords;
    std::vector<double> m_unique_y_coords;
    int m_currentSliceAxis = 0; // 0 for Y-slice (fixed X), 1 for X-slice (fixed Y)
    int m_currentSliceIndex = 0;

    void updateSliceControls(); // Helper to update spinbox range and info label

    // Объявления функций для квадратного решателя с G-образным решением
    void setupGShapeSolver();
    void updateChartGShape(const std::vector<double>& solution);
    std::vector<std::vector<double>> solutionTo2DGShape();
    std::vector<std::vector<double>> createTrueSolutionMatrixGShape();
    std::vector<std::vector<double>> createErrorMatrixGShape();
    
    // Функции для формирования таблиц и CSV
    void setupTableTab();
    QString generateCSVData(int skipFactor);
    QString generateCSVForTestProblem(int skipFactor);
    QString generateCSVForMainProblem(int skipFactor);
    QString generateCSVForGShapeProblem(int skipFactor);
    void populateTableWithData(const QString& csvData);
    
    // Элементы интерфейса для вкладки с таблицей
    QTableWidget* dataTable;
    QSpinBox* skipFactorSpinBox;
    QPushButton* exportCSVButton;
    QPushButton* showTableButton;
    QPushButton* clearTableButton;
    QLabel* tableInfoLabel;
};

// Forward declarations for external functions from default_functions.cpp
extern double custom_function_square(double x, double y);
extern double mu1_square(double x, double y);
extern double mu2_square(double x, double y);
extern double mu3_square(double x, double y);
extern double mu4_square(double x, double y);

// Functions for G-shaped solution in square domain
extern double function2_square(double x, double y);
extern double solution2_square(double x, double y);
extern double mu1_square_solution2(double x, double y);
extern double mu2_square_solution2(double x, double y);
extern double mu3_square_solution2(double x, double y);
extern double mu4_square_solution2(double x, double y);
