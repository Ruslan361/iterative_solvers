#pragma once

#include <QWidget>
#include <QTextEdit>

// Виджет вкладки со справкой
class HelpTabWidget : public QWidget
{
    Q_OBJECT

public:
    explicit HelpTabWidget(QWidget *parent = nullptr);
    ~HelpTabWidget() = default;

    // Методы для обновления информации в справке
    void updateTestTaskInfo(
        int nGrid, int mGrid,
        const QString& method, const QString& parameters,
        double epsPrecision, int maxIterations,
        int usedIterations, double achievedPrecision,
        double residualNorm, const QString& normType,
        double taskError,
        double maxErrorX, double maxErrorY,
        const QString& initialApproximation,
        double initialResidualNorm
    );

    void updateMainTaskInfo(
        int nGrid, int mGrid,
        const QString& method, const QString& parameters,
        double epsPrecision, int maxIterations,
        int usedIterations, double achievedPrecision,
        double residualNorm, const QString& normType,
        const QString& initialApproximation,
        double initialResidualNorm,
        
        // Для сетки с половинным шагом
        const QString& refinedMethod, const QString& refinedParameters,
        double refinedEpsPrecision, int refinedMaxIterations,
        int refinedUsedIterations, double refinedAchievedPrecision,
        double refinedResidualNorm, const QString& refinedNormType,
        double taskError,
        double maxErrorX, double maxErrorY,
        const QString& refinedInitialApproximation,
        double refinedInitialResidualNorm
    );

    void updateShapedTestTaskInfo(
        int nGrid, int mGrid,
        const QString& method, const QString& parameters,
        double epsPrecision, int maxIterations,
        int usedIterations, double achievedPrecision,
        double residualNorm, const QString& normType,
        double taskError,
        double maxErrorX, double maxErrorY,
        const QString& initialApproximation,
        double initialResidualNorm
    );
    
    // Добавление нового метода для отображения ошибок решателя и результатов
    void updateSolverErrorsReport(const QString& errorMessage, bool hasErrors = true);
    
    void clearAllInfo(); // Clears data structures and calls regenerate for the active type or a default
    void clearHelpText(); // Clears the single helpTextEdit

private:
    void setupUI();
    // These will now update the single helpTextEdit with HTML
    void regenerateTestTaskText();
    void regenerateMainTaskText();
    void regenerateShapedTestTaskText();
    void regenerateSolverErrorReport(); // New method to regenerate solver error report

private:
    // Элементы UI
    QTextEdit *helpTextEdit; // Single QTextEdit for all help content

    // Данные для справки о тестовой задаче
    struct TestTaskInfo {
        int nGrid = 0;
        int mGrid = 0;
        QString method;
        QString parameters;
        double epsPrecision = 0.0;
        int maxIterations = 0;
        int usedIterations = 0;
        double achievedPrecision = 0.0;
        double residualNorm = 0.0;
        QString normType;
        double taskError = 0.0;
        double maxErrorX = 0.0;
        double maxErrorY = 0.0;
        QString initialApproximation;
        double initialResidualNorm = 0.0;
    } testTaskInfo;

    // Данные для справки об основной задаче
    struct MainTaskInfo {
        int nGrid = 0;
        int mGrid = 0;
        QString method;
        QString parameters;
        double epsPrecision = 0.0;
        int maxIterations = 0;
        int usedIterations = 0;
        double achievedPrecision = 0.0;
        double residualNorm = 0.0;
        QString normType;
        QString initialApproximation;
        double initialResidualNorm = 0.0;
        
        // Для сетки с половинным шагом
        QString refinedMethod;
        QString refinedParameters;
        double refinedEpsPrecision = 0.0;
        int refinedMaxIterations = 0;
        int refinedUsedIterations = 0;
        double refinedAchievedPrecision = 0.0;
        double refinedResidualNorm = 0.0;
        QString refinedNormType;
        double taskError = 0.0;
        double maxErrorX = 0.0;
        double maxErrorY = 0.0;
        QString refinedInitialApproximation;
        double refinedInitialResidualNorm = 0.0;
    } mainTaskInfo;

    // Данные для справки о тестовой задаче для Г-образной области
    TestTaskInfo shapedTestTaskInfo;
    
    // Данные для отчета об ошибках решателя
    struct SolverErrorReport {
        QString errorMessage;
        bool hasErrors = false;
    } solverErrorReport;
    
    // To keep track of which help to display if needed, or MainWindow handles this
    enum class CurrentHelpType { NONE, TEST, MAIN, SHAPED_TEST, SOLVER_ERROR };
    CurrentHelpType currentHelpType = CurrentHelpType::NONE; 
};