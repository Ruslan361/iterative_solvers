#ifndef SQUARESHAPEREGION_H
#define SQUARESHAPEREGION_H

#include "shaperegion.h"
#include <QtDataVisualization/Q3DSurface>
#include <QtDataVisualization/QSurface3DSeries>
#include <vector>
#include <QColor>

/**
 * @brief Класс для представления и визуализации квадратной области решения
 * 
 * Данный класс предназначен для создания и управления 3D-визуализацией решения
 * уравнения Дирихле на квадратной области.
 */
class SquareShapeRegion : public ShapeRegion {
public:
    /**
     * @brief Конструктор
     * @param graph3D Указатель на объект 3D-графика
     */
    SquareShapeRegion(Q3DSurface* graph3D);

    /**
     * @brief Деструктор
     */
    virtual ~SquareShapeRegion();

    /**
     * @brief Создать поверхности для квадратной области
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
     * @return true в случае успеха, false в случае ошибки
     */
    virtual bool createSurfaces(
        const std::vector<double>& numericalSolution,
        const std::vector<double>& trueSolution,
        const std::vector<double>& errorValues,
        const std::vector<double>& xCoords,
        const std::vector<double>& yCoords,
        double domainXMin, double domainXMax,
        double domainYMin, double domainYMax,
        int decimationFactor = 1
    ) override;

    /**
     * @brief Установить видимость поверхности численного решения
     * @param visible Значение видимости
     */
    virtual void setNumericalSolutionVisible(bool visible) override;
    
    /**
     * @brief Установить видимость поверхности точного решения
     * @param visible Значение видимости
     */
    virtual void setTrueSolutionVisible(bool visible) override;
    
    /**
     * @brief Установить видимость поверхности ошибки
     * @param visible Значение видимости
     */
    virtual void setErrorSurfaceVisible(bool visible) override;
    
    /**
     * @brief Очистить все поверхности
     */
    virtual void clearAllSurfaces() override;

protected:
    /**
     * @brief Обновить диапазоны осей на основе предоставленных значений
     * @param values Вектор значений
     */
    virtual void updateAxesRanges(const std::vector<double>& values) override;
    
    /**
     * @brief Динамическое обновление диапазонов осей на основе текущих видимых серий
     */
    virtual void updateDynamicAxesRanges() override;

private:
    // Серии для поверхностей
    QSurface3DSeries* m_solutionSeries = nullptr;
    QSurface3DSeries* m_trueSolutionSeries = nullptr;
    QSurface3DSeries* m_errorSeries = nullptr;
    
    // Последние загруженные данные для возможного обновления
    std::vector<double> m_lastNumericalSolutionData;
    std::vector<double> m_lastTrueSolutionData;
    std::vector<double> m_lastErrorData;
    
    /**
     * @brief Создать QSurface3DSeries для квадратной области
     * 
     * @param data Данные для визуализации
     * @param xCoords X-координаты
     * @param yCoords Y-координаты
     * @param color Цвет поверхности
     * @param name Название серии
     * @param decimationFactor Коэффициент прореживания
     * @return QSurface3DSeries* Указатель на созданную серию
     */
    QSurface3DSeries* createSquareSurface(
        const std::vector<double>& data,
        const std::vector<double>& xCoords,
        const std::vector<double>& yCoords,
        const QColor& color,
        const QString& name,
        int decimationFactor
    );
};

#endif // SQUARESHAPEREGION_H