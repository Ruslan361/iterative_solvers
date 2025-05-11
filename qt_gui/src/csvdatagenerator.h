#ifndef CSVDATAGENERATOR_H
#define CSVDATAGENERATOR_H

#include <QString>
#include <QFile>
#include <QTextStream>

// Forward declarations
class ShapeRegion;
struct SolverResults;
struct SquareSolverResults;

/**
 * @brief The CSVDataGenerator class handles generation and saving of CSV data
 * 
 * This class provides functionality to generate CSV data for different problem types
 * and to save/load CSV data from files.
 */
class CSVDataGenerator {
public:
    CSVDataGenerator();
    
    /**
     * @brief Generate CSV data for the main problem
     * @param results The solver results containing solution, true solution, and error data
     * @param shapeRegion The region defining the problem domain
     * @param meshResolution The mesh resolution
     * @return The generated CSV data as a QString
     */
    QString generateCSVForMainProblem(
        const SolverResults& results,
        ShapeRegion* shapeRegion,
        int meshResolution);
    
    /**
     * @brief Generate CSV data for the G-shape problem
     * @param results_square The solver results for the square problem
     * @param meshResolution The mesh resolution
     * @return The generated CSV data as a QString
     */
    QString generateCSVForGShapeProblem(
        const SquareSolverResults& results_square,
        int meshResolution);
    
    /**
     * @brief Save CSV data to a file
     * @param data The CSV data to save
     * @param filePath The file path to save to
     * @return True if successful, false otherwise
     */
    bool saveCSVToFile(const QString& data, const QString& filePath);
    
    /**
     * @brief Load CSV data from a file
     * @param filePath The file path to load from
     * @return The loaded CSV data as a QString
     */
    QString loadCSVFromFile(const QString& filePath);
    
    /**
     * @brief Parse a CSV string into rows and columns
     * @param csvData The CSV data to parse
     * @return A QStringList of rows, each row is a QStringList of cells
     */
    QStringList parseCSV(const QString& csvData);
    
protected:
    /**
     * @brief Check if a point is in the domain
     * @param shapeRegion The region defining the problem domain
     * @param i The i coordinate
     * @param j The j coordinate
     * @return True if the point is in the domain, false otherwise
     */
    virtual bool pointInDomain(ShapeRegion* shapeRegion, int i, int j);
    
    /**
     * @brief Get the index of a point in the solution vector
     * @param shapeRegion The region defining the problem domain
     * @param i The i coordinate
     * @param j The j coordinate
     * @return The index of the point in the solution vector
     */
    virtual int getIndex(ShapeRegion* shapeRegion, int i, int j);

    /**
     * @brief Write CSV header with appropriate column names
     * @param stream The text stream to write to
     * @param hasTrueSolution Whether a true solution column should be included
     * @param hasError Whether an error column should be included
     * @param useRefinedName Use "refined_solution" instead of "true_solution" for the column name
     */
    void writeCSVHeader(QTextStream& stream, bool hasTrueSolution, bool hasError, bool useRefinedName = false);

    /**
     * @brief Write a data row to the CSV
     * @param stream The text stream to write to
     * @param x X coordinate
     * @param y Y coordinate
     * @param solution Solution value
     * @param trueSolution True or refined solution value (optional)
     * @param error Error value (optional)
     * @param hasTrueSolution Whether to include the true solution column
     * @param hasError Whether to include the error column
     */
    void writeCSVDataRow(QTextStream& stream, 
                         double x, 
                         double y, 
                         double solution, 
                         double trueSolution = 0.0, 
                         double error = 0.0,
                         bool hasTrueSolution = false,
                         bool hasError = false);

    /**
     * @brief Calculate grid coordinates from mesh resolution
     * @param i Index along x-axis
     * @param j Index along y-axis
     * @param meshResolution Mesh resolution
     * @return A pair of doubles representing (x, y) coordinates
     */
    std::pair<double, double> calculateGridCoordinates(int i, int j, int meshResolution);
};

#endif // CSVDATAGENERATOR_H