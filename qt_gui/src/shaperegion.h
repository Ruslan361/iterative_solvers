#ifndef SHAPEREGION_H
#define SHAPEREGION_H

#include <QtDataVisualization/Q3DSurface>
#include <vector>

/**
 * @brief Абстрактный базовый класс для представления и визуализации различных областей
 * 
 * Данный класс предоставляет общий интерфейс для работы с разными типами
 * областей (квадратная, Г-образная и т.д.). Конкретные реализации областей
 * должны наследоваться от этого класса.
 */
class ShapeRegion {
public:
    /**
     * @brief Конструктор
     * @param graph3D Указатель на объект 3D-графика
     */
    ShapeRegion(Q3DSurface* graph3D);

    /**
     * @brief Виртуальный деструктор
     */
    virtual ~ShapeRegion();

    /**
     * @brief Создать поверхности для заданных данных
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
     * @param connectorRows Количество строк для соединительной части (используется в G-образной области)
     * @param initialApproximation Начальное приближение (может быть пустым)
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
        int decimationFactor = 1,
        int connectorRows = 4,
        const std::vector<double>& initialApproximation = std::vector<double>()
    ) = 0;

    /**
     * @brief Установить видимость поверхности численного решения
     * @param visible Значение видимости
     */
    virtual void setNumericalSolutionVisible(bool visible) = 0;
    
    /**
     * @brief Установить видимость поверхности точного решения
     * @param visible Значение видимости
     */
    virtual void setTrueSolutionVisible(bool visible) = 0;
    
    /**
     * @brief Установить видимость поверхности ошибки
     * @param visible Значение видимости
     */
    virtual void setErrorSurfaceVisible(bool visible) = 0;
    
    /**
     * @brief Установить видимость поверхности начального приближения (нулевой плоскости)
     * @param visible Значение видимости
     */
    virtual void setInitialApproximationVisible(bool visible) = 0;
    
    /**
     * @brief Очистить все поверхности
     */
    virtual void clearAllSurfaces() = 0;

    /**
     * @brief Проверяет, находится ли точка в области
     * @param i Координата i точки
     * @param j Координата j точки
     * @return true, если точка находится в области, false в противном случае
     */
    virtual bool pointInDomain(int i, int j) { return true; }

    /**
     * @brief Получает индекс точки в массиве решения
     * @param i Координата i точки
     * @param j Координата j точки
     * @return Индекс точки в массиве решения или -1, если точка не в области
     */
    virtual int getIndex(int i, int j) { return i * (i + 1) + j; }

protected:
    Q3DSurface* m_graph3D; ///< Указатель на объект 3D-графика
    
    // Текущие границы области
    double m_currentDomainXMin = 0.0;
    double m_currentDomainXMax = 0.0;
    double m_currentDomainYMin = 0.0;
    double m_currentDomainYMax = 0.0;
    
    // Текущие минимальные и максимальные значения для осей
    double m_currentValueMin = 0.0;
    double m_currentValueMax = 0.0;
    
    /**
     * @brief Обновить диапазоны осей на основе предоставленных значений
     * @param values Вектор значений
     */
    virtual void updateAxesRanges(const std::vector<double>& values) = 0;
    
    /**
     * @brief Динамическое обновление диапазонов осей на основе текущих видимых серий
     */
    virtual void updateDynamicAxesRanges() = 0;
};

#endif // SHAPEREGION_H