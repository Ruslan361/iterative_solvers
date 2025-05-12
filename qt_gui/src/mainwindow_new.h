#pragma once

#include <QMainWindow>
#include <QTabWidget>
#include <QThread>
#include <memory>
#include <vector>
#include <string>

// Added Qt includes
#include <QObject>
#include <QString>
#include <QVariantMap>
#include <QAtomicInt>

#include "dirichlet_solver.hpp"
#include "grid_system.h"
#include "dirichlet_solver_square.hpp"
#include "solver.hpp"

#include "tabs/solver_tab_widget.h"
#include "tabs/progress_tab_widget.h"
#include "tabs/visualization_tab_widget.h"
#include "tabs/visualization_3d_tab_widget.h"
#include "tabs/table_tab_widget.h"
#include "tabs/help_tab_widget.h"

// Forward declarations
class SolverWorker;
class PlottingWorker;
// Ui::MainWindow is usually forward declared by Qt's uic if a .ui file is used.

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

// Define MainWindow class first.
// Its definition of SolverParams is needed for PlottingWorker::MainWindowSolverParams.
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    // Nested struct for solver parameters, used by MainWindow itself
    // and inherited by PlottingWorker's parameter struct.
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
        QString solver_name;
    };

    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onSolveButtonClicked();
    void onStopButtonClicked();
    void handleResults(const SolverResults& results_arg); // Renamed arg to avoid clash if any
    void handleResultsSquare(const SquareSolverResults& results_sq_arg); // Renamed arg
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
    void updateSolverProgress(const QString& status);
    void handlePlotData2D(const QVariantMap& data);
    void handlePlotData3D(const QVariantMap& data);
    void onPlottingFinished();
    
private:
    void setupSolver();
    void cleanupThread();
    void cleanupPlottingThread();
    void startPlottingTasks();
    QString generateCSVData(int skipFactor);
    QString generateCSVForTestProblem(int skipFactor);
    QString generateCSVForMainProblem(int skipFactor);
    QString generateCSVForGShapeProblem(int skipFactor);
    void updateHelpTabInfo();
    void updateMainTaskInfo();

private:
    // Ui::MainWindow *ui; // Assuming not present based on previous snippets and lack of .ui file in tree for mainwindow_new

    SolverTabWidget *solverTab;
    ProgressTabWidget *progressTab;
    VisualizationTabWidget *visualizationTab;
    Visualization3DTabWidget *visualization3DTab;
    TableTabWidget *tableTab;
    HelpTabWidget *helpTab;
    
    QTabWidget *tabWidget;
    
    std::unique_ptr<DirichletSolver> solver;
    std::unique_ptr<DirichletSolverSquare> solver_square;
    
    SolverResults results; // MainWindow's copy of results
    SquareSolverResults results_square; // MainWindow's copy of square results
    
    QThread* solverThread = nullptr;
    SolverWorker* worker = nullptr; // Uses forward-declared SolverWorker

    QThread* plottingThread = nullptr;
    PlottingWorker* plottingWorker = nullptr; // Uses forward-declared PlottingWorker
    
    bool isSolving = false;
    bool solveSuccessful = false;
    
    SolverParams params; // MainWindow's own instance of its SolverParams struct
};

// Define PlottingWorker class
// MainWindow::SolverParams is now defined.
class PlottingWorker : public QObject {
    Q_OBJECT

public:
    // Nested struct for parameters, inheriting from MainWindow's SolverParams
    struct MainWindowSolverParams : public MainWindow::SolverParams {
        // Constructor to allow conversion from SolverParams
        MainWindowSolverParams(const MainWindow::SolverParams& p) : MainWindow::SolverParams(p) {}
        MainWindowSolverParams() = default;
        // This struct is now fully defined here.
        // Add any PlottingWorker-specific fields if necessary, or leave empty if just inheriting.
    };

    explicit PlottingWorker(const MainWindowSolverParams& worker_params,
                            const SquareSolverResults& results_square,
                            const SolverResults& results,
                            QObject *parent = nullptr);
    ~PlottingWorker() = default;

    void requestStop() { stop_requested.storeRelease(1); }

public slots:
    void process();

signals:
    void progressUpdate(const QString& message);
    void plotDataReady2D(const QVariantMap& data);
    void plotDataReady3D(const QVariantMap& data);
    void finished();

private:
    MainWindowSolverParams params; // Member is of the fully defined nested type
    SquareSolverResults results_square;
    SolverResults results;
    QAtomicInt stop_requested; // Default initialized by QAtomicInt()
};

// Define SolverWorker class
class SolverWorker : public QObject {
    Q_OBJECT

public:
    // Removed WorkerSolverParams struct and old constructor
    explicit SolverWorker(std::unique_ptr<DirichletSolver> solver_arg, QObject *parent = nullptr);
    explicit SolverWorker(std::unique_ptr<DirichletSolverSquare> solver_sq_arg, QObject *parent = nullptr);
    ~SolverWorker() = default;

    void requestStop() { stop_requested.storeRelease(1); }

    // Methods to access solvers, used by MainWindow
    DirichletSolver* getSolver() { return m_solver.get(); }
    DirichletSolverSquare* getSolverSquare() { return m_solver_sq.get(); }

public slots:
    void process();

signals:
    void finished();
    void error(const QString& err);
    // Renamed signals to match usage in mainwindow_new.cpp
    void iterationUpdate(int iteration, double precision, double residual, double current_error);
    void resultReady(const SolverResults& results);
    void resultReadySquare(const SquareSolverResults& results_square);
    void solverStageUpdate(const QString& status);


private:
    // Removed WorkerSolverParams params;
    std::unique_ptr<DirichletSolver> m_solver;
    std::unique_ptr<DirichletSolverSquare> m_solver_sq;
    bool m_is_square_solver;
    QAtomicInt stop_requested;
};

// Forward declarations для внешних функций из default_functions.cpp
extern double custom_function_square(double x, double y);
extern double mu1_square(double x, double y);
extern double mu2_square(double x, double y);
extern double mu3_square(double x, double y);
extern double mu4_square(double x, double y);

extern double function2_square(double x, double y);
extern double solution2_square(double x, double y);
extern double mu1_square_solution2(double x, double y);
extern double mu2_square_solution2(double x, double y);
extern double mu3_square_solution2(double x, double y);
extern double mu4_square_solution2(double x, double y);