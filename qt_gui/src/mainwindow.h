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

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

// Класс рабочего потока для решателя
class SolverWorker : public QObject {
    Q_OBJECT

public:
    explicit SolverWorker(std::unique_ptr<DirichletSolver> solver);
    ~SolverWorker() = default;
    
    // Getter метод для доступа к solver
    DirichletSolver* getSolver() { return solver.get(); }

public slots:
    void process();

signals:
    void resultReady(SolverResults results);
    void finished();
    void iterationUpdate(int iteration, double precision, double residual, double error);

private:
    std::unique_ptr<DirichletSolver> solver;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onSolveButtonClicked();
    void onStopButtonClicked();
    void onSaveResultsButtonClicked();
    void onSaveMatrixButtonClicked();
    void onSaveVisualizationButtonClicked();
    void onShowReportButtonClicked();
    void updateIterationInfo(int iteration, double precision, double residual, double error);
    void handleResults(SolverResults results);
    void onSolverFinished();

private:
    Ui::MainWindow *ui;
    
    // Объект решателя
    std::unique_ptr<DirichletSolver> solver;
    
    // Результаты решения
    SolverResults results;
    
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
    } params;
    
    // Метод для очистки потока после завершения
    void cleanupThread();
};