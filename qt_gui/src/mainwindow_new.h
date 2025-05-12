#pragma once

#include <QMainWindow>
#include <QTabWidget>
#include <QThread>
#include <memory>
#include <vector>
#include <string>

#include "dirichlet_solver.hpp"
#include "grid_system.h"
#include "dirichlet_solver_square.hpp"
#include "solver.hpp"

#include "tabs/solver_tab_widget.h"
#include "tabs/progress_tab_widget.h"
#include "tabs/visualization_tab_widget.h"
#include "tabs/visualization_3d_tab_widget.h"
#include "tabs/table_tab_widget.h"
#include "tabs/help_tab_widget.h" // Add include for HelpTabWidget

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

    // Moving SolverParams from private to public
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
        bool use_refined_grid;
        QString solver_type;
        QString solver_name; // Added for method name
    };

private slots:
    void onSolveButtonClicked();
    void onStopButtonClicked();
    void handleResults(const SolverResults& results);
    void handleResultsSquare(const SquareSolverResults& results_sq);
    void updateIterationInfo(int iteration, double precision, double residual, double error);
    void onSolverFinished();
    void onSaveResultsButtonClicked();
    void onSaveMatrixButtonClicked();
    void onSaveVisualizationButtonClicked();
    void onShowReportButtonClicked();
    void onTabChanged(int index);
    void onChartTypeChanged(int index);
    void onShowHeatMapClicked();
    void onExportCSVRequested(int skipFactor);
    
private:
    void setupSolver();
    void cleanupThread();
    QString generateCSVData(int skipFactor);
    QString generateCSVForTestProblem(int skipFactor);
    QString generateCSVForMainProblem(int skipFactor);
    QString generateCSVForGShapeProblem(int skipFactor);
    void updateHelpTabInfo(); // New private method
    void updateMainTaskInfo(); // Add missing method declaration

private:
    // Модульные компоненты
    SolverTabWidget *solverTab;
    ProgressTabWidget *progressTab;
    VisualizationTabWidget *visualizationTab;
    Visualization3DTabWidget *visualization3DTab;
    TableTabWidget *tableTab;
    HelpTabWidget *helpTab; // Add HelpTabWidget instance
    
    // Основной контейнер вкладок
    QTabWidget *tabWidget;
    
    // Объекты решателя
    std::unique_ptr<DirichletSolver> solver;
    std::unique_ptr<DirichletSolverSquare> solver_square;
    
    // Результаты
    SolverResults results;
    SquareSolverResults results_square;
    
    // Поток для решателя
    QThread* solverThread = nullptr;
    SolverWorker* worker = nullptr;
    
    // Флаги состояния
    bool isSolving = false;
    bool solveSuccessful = false;
    
    // Параметры решателя
    SolverParams params;
};

// Forward declarations для внешних функций из default_functions.cpp
extern double custom_function_square(double x, double y);
extern double mu1_square(double x, double y);
extern double mu2_square(double x, double y);
extern double mu3_square(double x, double y);
extern double mu4_square(double x, double y);

// Функции для G-образного решения в квадратной области
extern double function2_square(double x, double y);
extern double solution2_square(double x, double y);
extern double mu1_square_solution2(double x, double y);
extern double mu2_square_solution2(double x, double y);
extern double mu3_square_solution2(double x, double y);
extern double mu4_square_solution2(double x, double y);