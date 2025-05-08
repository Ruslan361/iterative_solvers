#include "heatmapgenerator.h"

HeatMapGenerator::HeatMapGenerator(QObject *parent)
    : QObject(parent)
{
}

HeatMapGenerator::~HeatMapGenerator()
{
}

void HeatMapGenerator::showHeatMap(
    const std::vector<std::vector<double>>& data,
    const QString& title,
    int cellWidth,
    int cellHeight)
{
    if (data.empty() || data[0].empty()) {
        qDebug() << "HeatMapGenerator::showHeatMap: Нет данных для отображения";
        return;
    }
    
    // Создаем новое окно для тепловой карты
    QDialog* heatmapDialog = new QDialog();
    heatmapDialog->setWindowTitle(title);
    heatmapDialog->setMinimumSize(600, 500);
    heatmapDialog->setAttribute(Qt::WA_DeleteOnClose); // Автоматическое удаление при закрытии
    
    // Создаем сцену и вид для отображения тепловой карты
    QGraphicsScene* scene = new QGraphicsScene(heatmapDialog);
    QGraphicsView* view = new QGraphicsView(scene, heatmapDialog);
    view->setRenderHint(QPainter::Antialiasing);
    view->setDragMode(QGraphicsView::ScrollHandDrag);
    view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    
    // Вычисляем статистику по данным
    double minValue, maxValue, average;
    int count;
    calculateStatistics(data, minValue, maxValue, average, count);
    
    // Определяем размеры ячеек тепловой карты
    int rows = data.size();
    int cols = data[0].size();
    
    // Рисуем тепловую карту
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double value = data[i][j];
            if (!std::isnan(value)) { // Пропускаем NaN значения (точки вне области)
                QColor color = getColorForValue(value, minValue, maxValue);
                QGraphicsRectItem* rect = new QGraphicsRectItem(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
                rect->setBrush(QBrush(color));
                rect->setPen(QPen(Qt::black, 0.5));
                rect->setToolTip(QString("X: %1, Y: %2, Значение: %3").arg(j).arg(i).arg(value, 0, 'e', 4));
                scene->addItem(rect);
            }
        }
    }
    
    // Устанавливаем размер сцены
    scene->setSceneRect(0, 0, cols * cellWidth, rows * cellHeight);
    
    // Создаем легенду
    QGroupBox* legendBox = new QGroupBox("Легенда", heatmapDialog);
    QGridLayout* legendLayout = new QGridLayout(legendBox);
    
    int legendSteps = 10;
    for (int i = 0; i <= legendSteps; ++i) {
        double value = minValue + (maxValue - minValue) * i / legendSteps;
        QColor color = getColorForValue(value, minValue, maxValue);
        
        // Цветной прямоугольник
        QLabel* colorLabel = new QLabel(legendBox);
        colorLabel->setFixedSize(20, 20);
        colorLabel->setStyleSheet(QString("background-color: %1; border: 1px solid black;").arg(color.name()));
        
        // Значение
        QLabel* valueLabel = new QLabel(QString::number(value, 'e', 4), legendBox);
        
        legendLayout->addWidget(colorLabel, i, 0);
        legendLayout->addWidget(valueLabel, i, 1);
    }
    
    // Создаем информационный блок с статистикой
    QGroupBox* statsBox = new QGroupBox("Статистика", heatmapDialog);
    QFormLayout* statsLayout = new QFormLayout(statsBox);
    
    statsLayout->addRow(new QLabel("Минимальное значение:", statsBox), 
                       new QLabel(QString::number(minValue, 'e', 6), statsBox));
    statsLayout->addRow(new QLabel("Максимальное значение:", statsBox), 
                       new QLabel(QString::number(maxValue, 'e', 6), statsBox));
    statsLayout->addRow(new QLabel("Среднее значение:", statsBox), 
                       new QLabel(QString::number(average, 'e', 6), statsBox));
    
    // Создаем компоновку для диалогового окна
    QVBoxLayout* mainLayout = new QVBoxLayout(heatmapDialog);
    
    QHBoxLayout* controlsLayout = new QHBoxLayout();
    controlsLayout->addWidget(legendBox);
    controlsLayout->addWidget(statsBox);
    
    mainLayout->addWidget(new QLabel("Тепловая карта:", heatmapDialog));
    mainLayout->addWidget(view);
    mainLayout->addLayout(controlsLayout);
    
    // Добавляем кнопку экспорта
    QPushButton* exportButton = new QPushButton("Экспорт в PNG", heatmapDialog);
    connect(exportButton, &QPushButton::clicked, [scene, view]() {
        QString filename = QFileDialog::getSaveFileName(view, "Сохранить тепловую карту", 
                                                     "heatmap.png", "PNG (*.png)");
        if (!filename.isEmpty()) {
            QPixmap pixmap(scene->sceneRect().size().toSize());
            pixmap.fill(Qt::white);
            QPainter painter(&pixmap);
            scene->render(&painter);
            pixmap.save(filename);
        }
    });
    mainLayout->addWidget(exportButton);
    
    // Показываем диалог (модально)
    heatmapDialog->exec();
}

QColor HeatMapGenerator::getColorForValue(double value, double minValue, double maxValue) const
{
    if (std::isnan(value)) {
        return Qt::transparent;
    }
    
    // Нормализуем значение от 0 до 1
    double normalizedValue = (minValue == maxValue) ? 0.5 : (value - minValue) / (maxValue - minValue);
    
    // Интерполируем цвет от синего (холодный) к красному (горячий)
    if (normalizedValue <= 0.0)
        return Qt::blue;
    else if (normalizedValue <= 0.25)
        return QColor::fromRgbF(0, 0, 1.0 - normalizedValue * 4, 1.0); // Blue to Cyan
    else if (normalizedValue <= 0.5)
        return QColor::fromRgbF(0, normalizedValue * 4 - 1.0, 1.0, 1.0); // Cyan to Green
    else if (normalizedValue <= 0.75)
        return QColor::fromRgbF((normalizedValue - 0.5) * 4, 1.0, 1.0 - (normalizedValue - 0.5) * 4, 1.0); // Green to Yellow
    else if (normalizedValue < 1.0)
        return QColor::fromRgbF(1.0, 1.0 - (normalizedValue - 0.75) * 4, 0, 1.0); // Yellow to Red
    else
        return Qt::red;
}

void HeatMapGenerator::calculateStatistics(
    const std::vector<std::vector<double>>& data,
    double& minValue, double& maxValue,
    double& average, int& count) const
{
    minValue = std::numeric_limits<double>::max();
    maxValue = std::numeric_limits<double>::lowest();
    average = 0.0;
    count = 0;
    
    for (const auto& row : data) {
        for (const auto& val : row) {
            if (!std::isnan(val)) {
                minValue = std::min(minValue, val);
                maxValue = std::max(maxValue, val);
                average += val;
                count++;
            }
        }
    }
    
    average = (count > 0) ? average / count : 0.0;
}