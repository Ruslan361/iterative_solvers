#pragma once

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGroupBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QColor>
#include <QBrush>
#include <QPen>
#include <QGraphicsRectItem>
#include <QPixmap>
#include <QPainter>
#include <QFileDialog>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

/**
 * @brief Класс для генерации и отображения тепловых карт
 * 
 * Данный класс предоставляет функциональность для создания и отображения 
 * тепловых карт (heatmap) на основе двумерных массивов данных. Поддерживает 
 * экспорт изображения и отображение статистики по данным.
 */
class HeatMapGenerator : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Конструктор
     * @param parent Родительский объект
     */
    explicit HeatMapGenerator(QObject *parent = nullptr);
    
    /**
     * @brief Деструктор
     */
    ~HeatMapGenerator();

    /**
     * @brief Создать и отобразить тепловую карту
     * @param data 2D-массив значений для отображения
     * @param title Заголовок окна с тепловой картой (по умолчанию "Тепловая карта")
     * @param cellWidth Ширина ячейки тепловой карты в пикселях (по умолчанию 20)
     * @param cellHeight Высота ячейки тепловой карты в пикселях (по умолчанию 20)
     */
    void showHeatMap(const std::vector<std::vector<double>>& data, 
                    const QString& title = "Тепловая карта",
                    int cellWidth = 20,
                    int cellHeight = 20);

private:
    /**
     * @brief Получить цвет для значения на тепловой карте
     * @param value Значение
     * @param minValue Минимальное значение в диапазоне
     * @param maxValue Максимальное значение в диапазоне
     * @return Цвет, соответствующий значению
     */
    QColor getColorForValue(double value, double minValue, double maxValue) const;
    
    /**
     * @brief Вычислить статистику по данным
     * @param data 2D-массив значений
     * @param minValue [out] Минимальное значение
     * @param maxValue [out] Максимальное значение
     * @param average [out] Среднее значение
     * @param count [out] Количество непустых (не NaN) значений
     */
    void calculateStatistics(const std::vector<std::vector<double>>& data, 
                           double& minValue, double& maxValue, 
                           double& average, int& count) const;
};