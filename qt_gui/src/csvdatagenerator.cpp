#include "csvdatagenerator.h"
#include "shaperegion.h"
#include <QDebug>
#include <QFileInfo>
#include <QDir>

// Define structures needed for implementation
// You might need to adjust these to match your actual structures
struct SolverResults {
    std::vector<double> solution;
    std::vector<double> true_solution;
    std::vector<double> error;
    // Add other members as needed
};

struct SquareSolverResults {
    std::vector<double> solution;
    std::vector<double> refined_grid_solution;
    std::vector<double> error;
    // Add other members as needed
};

CSVDataGenerator::CSVDataGenerator() {
    // Initialize if needed
}

void CSVDataGenerator::writeCSVHeader(QTextStream& stream, bool hasTrueSolution, bool hasError, bool useRefinedName) {
    stream << "x,y,solution";
    
    if (hasTrueSolution) {
        stream << "," << (useRefinedName ? "refined_solution" : "true_solution");
    }
    
    if (hasError) {
        stream << ",error";
    }
    
    stream << '\n';
}

void CSVDataGenerator::writeCSVDataRow(QTextStream& stream, 
                                       double x, 
                                       double y, 
                                       double solution, 
                                       double trueSolution, 
                                       double error,
                                       bool hasTrueSolution,
                                       bool hasError) {
    // Write coordinates and solution
    stream << x << "," << y << "," << solution;
    
    // Write true/refined solution if available
    if (hasTrueSolution) {
        stream << "," << trueSolution;
    }
    
    // Write error if available
    if (hasError) {
        stream << "," << error;
    }
    
    stream << '\n';
}

std::pair<double, double> CSVDataGenerator::calculateGridCoordinates(int i, int j, int meshResolution) {
    double x = static_cast<double>(i) / meshResolution;
    double y = static_cast<double>(j) / meshResolution;
    return {x, y};
}

QString CSVDataGenerator::generateCSVForMainProblem(
    const SolverResults& results,
    ShapeRegion* shapeRegion,
    int meshResolution) {
    
    QString csvData;
    QTextStream stream(&csvData, QIODevice::WriteOnly);
    
    bool hasTrueSolution = !results.true_solution.empty();
    bool hasError = !results.error.empty();
    
    // Write header
    writeCSVHeader(stream, hasTrueSolution, hasError);
    
    // Write data for each grid point
    for (int i = 0; i <= meshResolution; ++i) {
        for (int j = 0; j <= meshResolution; ++j) {
            auto [x, y] = calculateGridCoordinates(i, j, meshResolution);
            
            // Check if point is in domain
            bool isInDomain = !shapeRegion || pointInDomain(shapeRegion, i, j);
            
            if (isInDomain) {
                int index = shapeRegion ? getIndex(shapeRegion, i, j) : i * (meshResolution + 1) + j;
                
                double solutionValue = (index < results.solution.size()) ? results.solution[index] : 0.0;
                double trueSolutionValue = (hasTrueSolution && index < results.true_solution.size()) ? 
                                          results.true_solution[index] : 0.0;
                double errorValue = (hasError && index < results.error.size()) ? 
                                    results.error[index] : 0.0;
                
                writeCSVDataRow(stream, x, y, solutionValue, trueSolutionValue, errorValue, 
                               hasTrueSolution, hasError);
            }
        }
    }
    
    return csvData;
}

QString CSVDataGenerator::generateCSVForGShapeProblem(
    const SquareSolverResults& results_square,
    int meshResolution) {
    
    QString csvData;
    QTextStream stream(&csvData, QIODevice::WriteOnly);
    
    bool hasRefinedSolution = !results_square.refined_grid_solution.empty();
    bool hasError = !results_square.error.empty();
    
    // Write header
    writeCSVHeader(stream, hasRefinedSolution, hasError, true);
    
    // Write data for each grid point in the regular grid
    for (int i = 0; i <= meshResolution; ++i) {
        for (int j = 0; j <= meshResolution; ++j) {
            auto [x, y] = calculateGridCoordinates(i, j, meshResolution);
            
            int index = i * (meshResolution + 1) + j;
            int refined_index = index * 4; // Adjust as needed for your refined grid
            
            double solutionValue = (index < results_square.solution.size()) ? 
                                  results_square.solution[index] : 0.0;
            double refinedSolutionValue = (hasRefinedSolution && refined_index < results_square.refined_grid_solution.size()) ? 
                                         results_square.refined_grid_solution[refined_index] : 0.0;
            double errorValue = (hasError && index < results_square.error.size()) ? 
                               results_square.error[index] : 0.0;
            
            writeCSVDataRow(stream, x, y, solutionValue, refinedSolutionValue, errorValue, 
                           hasRefinedSolution, hasError);
        }
    }
    
    return csvData;
}

bool CSVDataGenerator::saveCSVToFile(const QString& data, const QString& filePath) {
    QFile file(filePath);
    
    // Ensure the directory exists
    QFileInfo fileInfo(filePath);
    QDir().mkpath(fileInfo.absolutePath());
    
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream stream(&file);
        // For Qt 6, we don't need setCodec anymore as UTF-8 is the default
        #if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
            stream.setCodec("UTF-8");
        #endif
        stream << data;
        file.close();
        return true;
    } else {
        qWarning() << "Failed to open file for writing:" << filePath << "Error:" << file.errorString();
        return false;
    }
}

QString CSVDataGenerator::loadCSVFromFile(const QString& filePath) {
    QFile file(filePath);
    QString csvData;
    
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream stream(&file);
        // For Qt 6, we don't need setCodec anymore as UTF-8 is the default
        #if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
            stream.setCodec("UTF-8");
        #endif
        csvData = stream.readAll();
        file.close();
    } else {
        qWarning() << "Failed to open file for reading:" << filePath << "Error:" << file.errorString();
    }
    
    return csvData;
}

QStringList CSVDataGenerator::parseCSV(const QString& csvData) {
    // Implementation for parseCSV if needed
    return QStringList(); // Placeholder implementation
}

bool CSVDataGenerator::pointInDomain(ShapeRegion* shapeRegion, int i, int j) {
    // Default implementation - based on the square shape
    // Override this in derived classes for complex shapes
    return true; // Assuming square domain by default
}

int CSVDataGenerator::getIndex(ShapeRegion* shapeRegion, int i, int j) {
    // Default implementation - based on the square shape
    // Override this in derived classes for complex shapes
    return i * (i + 1) + j; // Simple mapping for a square grid
}