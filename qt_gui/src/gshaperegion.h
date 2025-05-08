#ifndef GSHAPEREGION_H
#define GSHAPEREGION_H

#include <QtDataVisualization/Q3DSurface>
#include <QtDataVisualization/QSurface3DSeries>
#include <QtDataVisualization/QSurfaceDataArray>
#include <QtDataVisualization/QSurfaceDataRow>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <QColor>
#include <QString>

/**
 * @brief Класс для представления и визуализации Г-образной области
 * 
 * Данный класс предоставляет методы для создания и управления 
 * Г-образной областью в 3D-пространстве. Он позволяет генерировать
 * поверхности для различных типов данных (решение, точное решение, ошибка)
 * и управлять их отображением.
 */
class GShapeRegion {
public:
    /**
     * @brief Структура для хранения серий для одной Г-образной поверхности
     */
    struct GShapeSeries {
        QSurface3DSeries* bigRect = nullptr;   ///< Верхняя часть Г-образной области
        QSurface3DSeries* smallRect = nullptr; ///< Правая нижняя часть Г-образной области
        QSurface3DSeries* connector = nullptr; ///< Соединительная часть между bigRect и smallRect
        
        /**
         * @brief Установить видимость всех частей поверхности
         * @param visible Значение видимости
         */
        void setVisible(bool visible) {
            if (bigRect) bigRect->setVisible(visible);
            if (smallRect) smallRect->setVisible(visible);
            if (connector) connector->setVisible(visible);
        }
        
        /**
         * @brief Очистить все серии
         * @param graph График, с которого нужно удалить серии (может быть nullptr)
         */
        void clear(Q3DSurface* graph = nullptr) {
            if (graph) {
                if (bigRect) graph->removeSeries(bigRect);
                if (smallRect) graph->removeSeries(smallRect);
                if (connector) graph->removeSeries(connector);
            }
            delete bigRect; // Deleting the series also deletes its proxy and data if owned
            delete smallRect;
            delete connector;
            bigRect = nullptr;
            smallRect = nullptr;
            connector = nullptr;
        }
    };

    /**
     * @brief Структура для хранения данных для одной Г-образной поверхности
     */
    struct GShapeDataArrays {
        QSurfaceDataArray* bigRectData = nullptr;   ///< Данные для верхней части
        QSurfaceDataArray* smallRectData = nullptr; ///< Данные для правой нижней части
        QSurfaceDataArray* connectorData = nullptr; ///< Данные для соединительной части
        // No destructor - ownership of non-null arrays is transferred or they are manually deleted
    };

    /**
     * @brief Конструктор
     * @param graph3D Указатель на объект 3D-графика
     */
    GShapeRegion(Q3DSurface* graph3D);

    /**
     * @brief Деструктор
     */
    ~GShapeRegion();

    /**
     * @brief Создать Г-образную поверхность для заданных данных
     * 
     * @param numericalSolution Численное решение
     * @param trueSolution Точное решение (может быть пустым)
     * @param errorValues Ошибки (может быть пустым)
     * @param xCoords X-координаты точек
     * @param yCoords Y-координаты точек
     * @param domainXMin Минимальное значение X для области
     * @param domainXMax Максимальное значение X для области
     * @param domainYMin Минимальное значение Y для области
     * @param domainYMax Максимальное значение Y для области
     * @param decimationFactor Коэффициент прореживания (1 = использовать все точки)
     * @param connectorRows Количество строк для соединительной части
     * @return true в случае успеха, false в случае ошибки
     */
    bool createSurfaces(
        const std::vector<double>& numericalSolution,
        const std::vector<double>& trueSolution,
        const std::vector<double>& errorValues,
        const std::vector<double>& xCoords,
        const std::vector<double>& yCoords,
        double domainXMin, double domainXMax, // Overall domain bounds for splitting
        double domainYMin, double domainYMax, // Overall domain bounds for splitting
        int decimationFactor = 1,
        int connectorRows = 5
    );

    /**
     * @brief Установить видимость поверхности численного решения
     * @param visible Значение видимости
     */
    void setNumericalSolutionVisible(bool visible);
    
    /**
     * @brief Установить видимость поверхности точного решения
     * @param visible Значение видимости
     */
    void setTrueSolutionVisible(bool visible);
    
    /**
     * @brief Установить видимость поверхности ошибки
     * @param visible Значение видимости
     */
    void setErrorSurfaceVisible(bool visible);
    
    /**
     * @brief Очистить все поверхности
     */
    void clearAllSurfaces();

private:
    Q3DSurface* m_graph3D; ///< Указатель на объект 3D-графика
    
    GShapeSeries m_solutionSeries;      ///< Серия для численного решения
    GShapeSeries m_trueSolutionSeries;  ///< Серия для точного решения
    GShapeSeries m_errorSeries;         ///< Серия для ошибки
    
    double m_currentDomainXMin = 0.0;
    double m_currentDomainXMax = 0.0;
    double m_currentDomainYMin = 0.0;
    double m_currentDomainYMax = 0.0;
    double m_currentValueMin = 0.0;
    double m_currentValueMax = 0.0;

    /**
     * @brief Создать массивы данных для одной Г-образной поверхности
     * 
     * @param values Значения (решение, ошибка и т.д.)
     * @param xCoords X-координаты точек
     * @param yCoords Y-координаты точек
     * @param xSplit X-координата разделителя области
     * @param ySplit Y-координата разделителя области
     * @param decimationFactor Коэффициент прореживания
     * @param connectorRows Количество строк для соединительной части
     * @return Структура с массивами данных для построения поверхности
     */
    GShapeDataArrays createDataArrays(
        const std::vector<double>& values,
        const std::vector<double>& xCoords,
        const std::vector<double>& yCoords,
        double xSplit,
        double ySplit,
        int decimationFactor,
        int connectorRows
    );
    
    /**
     * @brief Создать серии для поверхности из массивов данных
     * 
     * @param dataArrays Структура с массивами данных
     * @param color Цвет поверхности
     * @param seriesName Имя серии
     * @return Структура с сериями для отображения на графике
     */
    GShapeSeries createSeries(
        GShapeDataArrays& dataArrays, // Pass by non-const ref to allow nulling out transferred pointers
        const QColor& color, 
        const QString& seriesName
    );

    /**
     * @brief Обновить диапазоны осей в зависимости от значений
     * 
     * @param values Значения для анализа диапазонов
     */
    void updateAxesRanges(const std::vector<double>& values);
};

#endif // GSHAPEREGION_H