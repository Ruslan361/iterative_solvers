#include "help_tab_widget.h"

#include <QVBoxLayout>
#include <QLabel>
#include <QTextEdit>

HelpTabWidget::HelpTabWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    clearAllInfo();
}

void HelpTabWidget::setupUI()
{
    // Создаем главный компоновщик
    auto *mainLayout = new QVBoxLayout(this);
    
    // Создаем единое текстовое поле для отображения информации
    helpTextEdit = new QTextEdit(this);
    helpTextEdit->setReadOnly(true);
    helpTextEdit->setHtml(""); // Initialize with empty HTML content
    
    mainLayout->addWidget(helpTextEdit);
    
    setLayout(mainLayout);
}

void HelpTabWidget::clearAllInfo()
{
    // Очищаем данные для тестовой задачи
    testTaskInfo = TestTaskInfo();
    // Очищаем данные для основной задачи
    mainTaskInfo = MainTaskInfo();
    // Очищаем данные для тестовой задачи с Г-образной областью
    shapedTestTaskInfo = TestTaskInfo();
    
    // Очищаем текстовое поле
    clearHelpText();
    currentHelpType = CurrentHelpType::NONE; // Reset current help type
}

void HelpTabWidget::updateTestTaskInfo(
    int nGrid, int mGrid,
    const QString& method, const QString& parameters,
    double epsPrecision, int maxIterations,
    int usedIterations, double achievedPrecision,
    double residualNorm, const QString& normType,
    double taskError,
    double maxErrorX, double maxErrorY,
    const QString& initialApproximation,
    double initialResidualNorm)
{
    // Обновляем данные для тестовой задачи
    testTaskInfo.nGrid = nGrid;
    testTaskInfo.mGrid = mGrid;
    testTaskInfo.method = method;
    testTaskInfo.parameters = parameters;
    testTaskInfo.epsPrecision = epsPrecision;
    testTaskInfo.maxIterations = maxIterations;
    testTaskInfo.usedIterations = usedIterations;
    testTaskInfo.achievedPrecision = achievedPrecision;
    testTaskInfo.residualNorm = residualNorm;
    testTaskInfo.normType = normType;
    testTaskInfo.taskError = taskError;
    testTaskInfo.maxErrorX = maxErrorX;
    testTaskInfo.maxErrorY = maxErrorY;
    testTaskInfo.initialApproximation = initialApproximation;
    testTaskInfo.initialResidualNorm = initialResidualNorm;

    currentHelpType = CurrentHelpType::TEST;
    regenerateTestTaskText();
}

void HelpTabWidget::updateMainTaskInfo(
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
    double refinedInitialResidualNorm)
{
    // Обновляем данные для основной задачи
    mainTaskInfo.nGrid = nGrid;
    mainTaskInfo.mGrid = mGrid;
    mainTaskInfo.method = method;
    mainTaskInfo.parameters = parameters;
    mainTaskInfo.epsPrecision = epsPrecision;
    mainTaskInfo.maxIterations = maxIterations;
    mainTaskInfo.usedIterations = usedIterations;
    mainTaskInfo.achievedPrecision = achievedPrecision;
    mainTaskInfo.residualNorm = residualNorm;
    mainTaskInfo.normType = normType;
    mainTaskInfo.initialApproximation = initialApproximation;
    mainTaskInfo.initialResidualNorm = initialResidualNorm;
    
    // Для сетки с половинным шагом
    mainTaskInfo.refinedMethod = refinedMethod;
    mainTaskInfo.refinedParameters = refinedParameters;
    mainTaskInfo.refinedEpsPrecision = refinedEpsPrecision;
    mainTaskInfo.refinedMaxIterations = refinedMaxIterations;
    mainTaskInfo.refinedUsedIterations = refinedUsedIterations;
    mainTaskInfo.refinedAchievedPrecision = refinedAchievedPrecision;
    mainTaskInfo.refinedResidualNorm = refinedResidualNorm;
    mainTaskInfo.refinedNormType = refinedNormType;
    mainTaskInfo.taskError = taskError;
    mainTaskInfo.maxErrorX = maxErrorX;
    mainTaskInfo.maxErrorY = maxErrorY;
    mainTaskInfo.refinedInitialApproximation = refinedInitialApproximation;
    mainTaskInfo.refinedInitialResidualNorm = refinedInitialResidualNorm;

    currentHelpType = CurrentHelpType::MAIN;
    regenerateMainTaskText();
}

void HelpTabWidget::updateShapedTestTaskInfo(
    int nGrid, int mGrid,
    const QString& method, const QString& parameters,
    double epsPrecision, int maxIterations,
    int usedIterations, double achievedPrecision,
    double residualNorm, const QString& normType,
    double taskError,
    double maxErrorX, double maxErrorY,
    const QString& initialApproximation,
    double initialResidualNorm)
{
    // Обновляем данные для тестовой задачи с Г-образной областью
    shapedTestTaskInfo.nGrid = nGrid;
    shapedTestTaskInfo.mGrid = mGrid;
    shapedTestTaskInfo.method = method;
    shapedTestTaskInfo.parameters = parameters;
    shapedTestTaskInfo.epsPrecision = epsPrecision;
    shapedTestTaskInfo.maxIterations = maxIterations;
    shapedTestTaskInfo.usedIterations = usedIterations;
    shapedTestTaskInfo.achievedPrecision = achievedPrecision;
    shapedTestTaskInfo.residualNorm = residualNorm;
    shapedTestTaskInfo.normType = normType;
    shapedTestTaskInfo.taskError = taskError;
    shapedTestTaskInfo.maxErrorX = maxErrorX;
    shapedTestTaskInfo.maxErrorY = maxErrorY;
    shapedTestTaskInfo.initialApproximation = initialApproximation;
    shapedTestTaskInfo.initialResidualNorm = initialResidualNorm;

    currentHelpType = CurrentHelpType::SHAPED_TEST;
    regenerateShapedTestTaskText();
}

void HelpTabWidget::updateSolverErrorsReport(const QString& errorMessage, bool hasErrors) 
{
    // Обновляем данные для отчета об ошибках решателя
    solverErrorReport.errorMessage = errorMessage;
    solverErrorReport.hasErrors = hasErrors;

    currentHelpType = CurrentHelpType::SOLVER_ERROR;
    regenerateSolverErrorReport();
}

void HelpTabWidget::regenerateTestTaskText()
{
    QString html = "<h2>Справка для тестовой задачи</h2>";
    html += QString("<p>Для решения тестовой задачи использованы сетка с числом разбиений по x <b>n = «%1»</b> и числом разбиений по y <b>m = «%2»</b>, "
                    "метод <b>%3</b>, параметры <b>%4</b> (указать значения) "
                    "критерии остановки по точности <b>&epsilon;<sub>мет</sub> = «%5»</b> и по числу итераций <b>N<sub>max</sub> = «%6»</b>.</p>"
                    "<p>На решение схемы (СЛАУ) затрачено итераций <b>N = «%7»</b> и достигнута точность итерационного метода <b>&epsilon;(N) = «%8»</b>.</p>"
                    "<p>Схема (СЛАУ) решена с невязкой <b>|| R(N)|| = «%9»</b> (указать норму невязки) "
                    "для невязки СЛАУ использована норма <b>«%10»</b> (указать тип: евклидова норма, норма «max»).</p>"
                    "<p>Тестовая задача должна быть решена с погрешностью не более <b>&epsilon; = 0.5&sdot;10<sup>&ndash;6</sup></b>; "
                    "задача решена с погрешностью <b>&epsilon;<sub>1</sub> = «%11»</b>.</p>"
                    "<p>Максимальное отклонение точного и численного решений наблюдается в узле <b>x = «%12»</b>; <b>y = «%13»</b>.</p>"
                    "<p>В качестве начального приближения использовано <b>«%14»</b> "
                    "(указать, что использовано: интерполяция по x, интерполяция по y, иное)&raquo;.</p>"
                    "<p>Невязка СЛАУ на начальном приближении <b>|| R(0)|| = «%15»</b> "
                    "(указать норму невязки и тип нормы).</p>")
            .arg(testTaskInfo.nGrid)
            .arg(testTaskInfo.mGrid)
            .arg(testTaskInfo.method)
            .arg(testTaskInfo.parameters)
            .arg(QString::number(testTaskInfo.epsPrecision, 'e', 6))
            .arg(testTaskInfo.maxIterations)
            .arg(testTaskInfo.usedIterations)
            .arg(QString::number(testTaskInfo.achievedPrecision, 'e', 6))
            .arg(QString::number(testTaskInfo.residualNorm, 'e', 6))
            .arg(testTaskInfo.normType)
            .arg(QString::number(testTaskInfo.taskError, 'e', 6))
            .arg(QString::number(testTaskInfo.maxErrorX, 'f', 6))
            .arg(QString::number(testTaskInfo.maxErrorY, 'f', 6))
            .arg(testTaskInfo.initialApproximation)
            .arg(QString::number(testTaskInfo.initialResidualNorm, 'e', 6));
    
    helpTextEdit->setHtml(html);
}

void HelpTabWidget::regenerateMainTaskText()
{
    QString html = "<h2>Справка для основной задачи</h2>";
    html += QString("<p>Для решения основной задачи использована сетка с числом разбиений по x <b>n = «%1»</b> и числом разбиений по y <b>m = «%2»</b>, "
                   "метод <b>%3</b>, параметры <b>%4</b> (указать значения) "
                   "критерии остановки по точности <b>&epsilon;<sub>мет</sub> = «%5»</b> и по числу итераций <b>N<sub>max</sub> = «%6»</b>.</p>"
                   "<p>На решение схемы (СЛАУ) затрачено итераций <b>N = «%7»</b> и достигнута точность итерационного метода <b>&epsilon;(N) = «%8»</b>.</p>"
                   "<p>Схема (СЛАУ) решена с невязкой <b>|| R(N)|| = «%9»</b> (указать норму невязки) "
                   "использована норма <b>«%10»</b> (указать тип: евклидова норма, норма «max», иное).</p>"
                   "<p>В качестве начального приближения на основной сетке использовано "
                   "<b>&laquo;%11&raquo;</b> (указать, что именно: интерполяция по x, интерполяция по y, иное)&raquo;.</p>"
                   "<p>На основной сетке невязка СЛАУ на начальном приближении <b>|| R(0)|| = «%12»</b> "
                   "(указать норму невязки и тип нормы).</p>"
                   "<p>Для контроля точности использована сетка с половинным шагом, "
                   "метод <b>%13</b>, параметры <b>%14</b> (указать значения) "
                   "критерии остановки по точности <b>&epsilon;<sub>мет-2</sub> = «%15»</b> и по числу итераций "
                   "<b>N<sub>max-2</sub> = «%16»</b>.</p>"
                   "<p>На решение задачи (СЛАУ) затрачено итераций <b>N<sub>2</sub> = «%17»</b> и достигнута точность итерационного метода <b>&epsilon;(N<sub>2</sub>) = «%18»</b>.</p>"
                   "<p>Схема (СЛАУ) на сетке с половинным шагом решена с невязкой "
                   "<b>|| R(N<sub>2</sub>)|| = «%19»</b> (указать норму невязки) "
                   "использована норма <b>«%20»</b> (указать тип: евклидова норма, норма «max», иное).</p>"
                   "<p>Основная задача должна быть решена с точностью не хуже чем <b>&epsilon; = 0.5&sdot;10<sup>&ndash;6</sup></b>; "
                   "задача решена с точностью <b>&epsilon;<sub>2</sub> = «%21»</b>.</p>"
                   "<p>Максимальное отклонение численных решений на основной сетке и сетке с "
                   "половинным шагом наблюдается в узле <b>x = «%22»</b>; <b>y = «%23»</b>.</p>"
                   "<p>В качестве начального приближения на сетке с половинным шагом использовано <b>&laquo;%24&raquo;</b> (указать, что именно: интерполяция по x, "
                   "интерполяция по y, иное)&raquo;.</p>"
                   "<p>На сетке с половинным шагом невязка СЛАУ на начальном приближении "
                   "<b>|| R(0)|| = «%25»</b> (указать норму невязки и тип нормы).</p>")
            .arg(mainTaskInfo.nGrid)
            .arg(mainTaskInfo.mGrid)
            .arg(mainTaskInfo.method)
            .arg(mainTaskInfo.parameters)
            .arg(QString::number(mainTaskInfo.epsPrecision, 'e', 6))
            .arg(mainTaskInfo.maxIterations)
            .arg(mainTaskInfo.usedIterations)
            .arg(QString::number(mainTaskInfo.achievedPrecision, 'e', 6))
            .arg(QString::number(mainTaskInfo.residualNorm, 'e', 6))
            .arg(mainTaskInfo.normType)
            .arg(mainTaskInfo.initialApproximation)
            .arg(QString::number(mainTaskInfo.initialResidualNorm, 'e', 6))
            .arg(mainTaskInfo.refinedMethod)
            .arg(mainTaskInfo.refinedParameters)
            .arg(QString::number(mainTaskInfo.refinedEpsPrecision, 'e', 6))
            .arg(mainTaskInfo.refinedMaxIterations)
            .arg(mainTaskInfo.refinedUsedIterations)
            .arg(QString::number(mainTaskInfo.refinedAchievedPrecision, 'e', 6))
            .arg(QString::number(mainTaskInfo.refinedResidualNorm, 'e', 6))
            .arg(mainTaskInfo.refinedNormType)
            .arg(QString::number(mainTaskInfo.taskError, 'e', 6))
            .arg(QString::number(mainTaskInfo.maxErrorX, 'f', 6))
            .arg(QString::number(mainTaskInfo.maxErrorY, 'f', 6))
            .arg(mainTaskInfo.refinedInitialApproximation)
            .arg(QString::number(mainTaskInfo.refinedInitialResidualNorm, 'e', 6));
    
    helpTextEdit->setHtml(html);
}

void HelpTabWidget::regenerateShapedTestTaskText()
{
    QString html = "<h2>Г область - Справка для тестовой задачи</h2>";
    html += QString("<p>Для решения тестовой задачи использована сетка-основа "
                   "с числом разбиений по x <b>n = «%1»</b> и числом разбиений по y <b>m = «%2»</b>, "
                   "метод <b>%3</b>, параметры <b>%4</b> (указать значения) "
                   "критерии остановки по точности <b>&epsilon;<sub>мет</sub> = «%5»</b> и по числу итераций <b>N<sub>max</sub> = «%6»</b>.</p>"
                   "<p>На решение схемы (СЛАУ) затрачено итераций <b>N = «%7»</b> и достигнута точность итерационного метода <b>&epsilon;(N) = «%8»</b>.</p>"
                   "<p>Схема (СЛАУ) решена с невязкой <b>|| R (N) || = «%9»</b> (указать норму невязки) "
                   "для невязки СЛАУ использована норма <b>«%10»</b> "
                   "(указать тип: евклидова норма, норма «max»).</p>"
                   "<p>Тестовая задача должна быть решена с погрешностью не более <b>&epsilon; = 0.5&sdot;10<sup>&ndash;6</sup></b>; "
                   "задача решена с погрешностью <b>&epsilon;<sub>1</sub> = «%11»</b>.</p>"
                   "<p>Максимальное отклонение точного и численного решений наблюдается в узле <b>x = «%12»</b>; <b>y = «%13»</b>.</p>"
                   "<p>В качестве начального приближения использовано <b>&laquo;%14&raquo;</b> "
                   "(указать, что использовано: интерполяция по x, интерполяция по y, иное)&raquo;.</p>"
                   "<p>Невязка СЛАУ на начальном приближении <b>|| R (0) || = «%15»</b> "
                   "(указать норму невязки и тип нормы).</p>")
            .arg(shapedTestTaskInfo.nGrid)
            .arg(shapedTestTaskInfo.mGrid)
            .arg(shapedTestTaskInfo.method)
            .arg(shapedTestTaskInfo.parameters)
            .arg(QString::number(shapedTestTaskInfo.epsPrecision, 'e', 6))
            .arg(shapedTestTaskInfo.maxIterations)
            .arg(shapedTestTaskInfo.usedIterations)
            .arg(QString::number(shapedTestTaskInfo.achievedPrecision, 'e', 6))
            .arg(QString::number(shapedTestTaskInfo.residualNorm, 'e', 6))
            .arg(shapedTestTaskInfo.normType)
            .arg(QString::number(shapedTestTaskInfo.taskError, 'e', 6))
            .arg(QString::number(shapedTestTaskInfo.maxErrorX, 'f', 6))
            .arg(QString::number(shapedTestTaskInfo.maxErrorY, 'f', 6))
            .arg(shapedTestTaskInfo.initialApproximation)
            .arg(QString::number(shapedTestTaskInfo.initialResidualNorm, 'e', 6));
    
    helpTextEdit->setHtml(html);
}

void HelpTabWidget::regenerateSolverErrorReport() 
{
    QString html;
    
    if (solverErrorReport.hasErrors) {
        html = "<h2>Отчет об ошибках решателя</h2>";
        html += "<div style=\"color: red; background-color: #ffeeee; padding: 10px; border: 1px solid #ffcccc; border-radius: 5px;\">";
        html += "<pre>" + solverErrorReport.errorMessage.toHtmlEscaped() + "</pre>";
        html += "</div>";
        html += "<p>Обнаружены ошибки в решателе. Пожалуйста, исправьте указанные проблемы для корректной работы программы.</p>";
    } else {
        html = "<h2>Решатель успешно выполнил расчеты</h2>";
        html += "<div style=\"color: green; background-color: #eeffee; padding: 10px; border: 1px solid #ccffcc; border-radius: 5px;\">";
        html += "<pre>" + solverErrorReport.errorMessage.toHtmlEscaped() + "</pre>";
        html += "</div>";
        html += "<p>Решатель успешно завершил все расчеты.</p>";
    }
    
    helpTextEdit->setHtml(html);
}

void HelpTabWidget::clearHelpText() {
    helpTextEdit->clear();
}