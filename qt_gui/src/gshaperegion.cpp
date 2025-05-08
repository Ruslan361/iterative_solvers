#include "gshaperegion.h"
#include <QDebug>
#include <QMessageBox> // Required for QMessageBox, if you plan to use it here


// Constructor
GShapeRegion::GShapeRegion(Q3DSurface* graph3D) // Corrected constructor signature
    : m_graph3D(graph3D) { // Removed unused member initializations
    if (!m_graph3D) {
        qWarning() << "GShapeRegion initialized with a null Q3DSurface pointer.";
    }
}

// Destructor
GShapeRegion::~GShapeRegion() {
    clearAllSurfaces();
}

// Clears all surfaces
void GShapeRegion::clearAllSurfaces() {
    m_solutionSeries.clear(m_graph3D);
    m_trueSolutionSeries.clear(m_graph3D);
    m_errorSeries.clear(m_graph3D);
}

// Set visibility for numerical solution
void GShapeRegion::setNumericalSolutionVisible(bool visible) {
    m_solutionSeries.setVisible(visible);
}

// Set visibility for true solution
void GShapeRegion::setTrueSolutionVisible(bool visible) {
    m_trueSolutionSeries.setVisible(visible);
}

// Set visibility for error surface
void GShapeRegion::setErrorSurfaceVisible(bool visible) {
    m_errorSeries.setVisible(visible);
}

// Update axes ranges based on provided values
void GShapeRegion::updateAxesRanges(const std::vector<double>& values) {
    if (values.empty() || !m_graph3D) return;

    double valMin = std::numeric_limits<double>::max();
    double valMax = std::numeric_limits<double>::lowest();

    for (double val : values) {
        if (!std::isnan(val)) {
            valMin = std::min(valMin, val);
            valMax = std::max(valMax, val);
        }
    }
    
    // Update global min/max for values if this is the first set or they expand the range
    if (m_currentValueMin == 0.0 && m_currentValueMax == 0.0 || valMin < m_currentValueMin) {
        m_currentValueMin = valMin;
    }
    if (m_currentValueMin == 0.0 && m_currentValueMax == 0.0 || valMax > m_currentValueMax) {
        m_currentValueMax = valMax;
    }

    // Add a small margin to avoid data points being exactly on the edge of the axis range
    double margin = std::abs(m_currentValueMax - m_currentValueMin) * 0.1;
    if (margin == 0) margin = 0.1; // Handle case where all values are the same

    m_graph3D->axisX()->setRange(m_currentDomainXMin, m_currentDomainXMax);
    m_graph3D->axisZ()->setRange(m_currentDomainYMin, m_currentDomainYMax); // Z-axis in QtDataVis corresponds to Y in problem domain
    m_graph3D->axisY()->setRange(m_currentValueMin - margin, m_currentValueMax + margin); // Y-axis in QtDataVis is the value axis

    m_graph3D->axisX()->setTitle("X");
    m_graph3D->axisY()->setTitle("Value");
    m_graph3D->axisZ()->setTitle("Y");
}

// Create data arrays for the G-shape regions
GShapeRegion::GShapeDataArrays GShapeRegion::createDataArrays(
    const std::vector<double>& values,
    const std::vector<double>& xCoords,
    const std::vector<double>& yCoords,
    double xSplit,
    double ySplit,
    int decimationFactor,
    int connectorRows
) {
    GShapeDataArrays dataArrays;
    if (values.empty() || xCoords.empty() || yCoords.empty() || values.size() != xCoords.size() || values.size() != yCoords.size()) {
        qWarning() << "Input data for GShapeRegion::createDataArrays is invalid or mismatched.";
        return dataArrays; // Return empty arrays
    }

    dataArrays.bigRectData = new QSurfaceDataArray;
    dataArrays.smallRectData = new QSurfaceDataArray;
    dataArrays.connectorData = new QSurfaceDataArray;

    std::map<double, std::vector<std::pair<double, double>>> bigRectPoints; // y -> [(x, value)]
    std::map<double, std::vector<std::pair<double, double>>> smallRectPoints; // y -> [(x, value)]

    for (size_t i = 0; i < values.size(); ++i) {
        if (decimationFactor > 1 && i % decimationFactor != 0) continue; // Decimation

        double x = xCoords[i];
        double y = yCoords[i];
        double value = values[i];

        bool is_quadrant1 = (x <= xSplit && y >= ySplit);
        bool is_quadrant2 = (x > xSplit && y > ySplit);
        bool is_quadrant4 = (x > xSplit && y <= ySplit);

        if (is_quadrant1 || is_quadrant2) {
            bigRectPoints[y].push_back({x, value});
        } else if (is_quadrant4) {
            smallRectPoints[y].push_back({x, value});
        }
    }

    double min_y_big = std::numeric_limits<double>::max();
    if (!bigRectPoints.empty()) {
        for (const auto& pair : bigRectPoints) min_y_big = std::min(min_y_big, pair.first);
    } else {
        min_y_big = ySplit; // Default if no points
    }

    double max_y_small = std::numeric_limits<double>::lowest();
    if (!smallRectPoints.empty()) {
        for (const auto& pair : smallRectPoints) max_y_small = std::max(max_y_small, pair.first);
    } else {
        max_y_small = ySplit; // Default if no points
    }
    
    // Ensure min_y_big is actually greater than max_y_small for a valid connector
    if (min_y_big <= max_y_small && !bigRectPoints.empty() && !smallRectPoints.empty()) {
        // This can happen if the split is exactly on a grid line and points fall on both sides
        // or if the data doesn't perfectly form the G-shape as expected.
        // Attempt to find the true minimum Y in bigRect that is strictly greater than max_y_small
        double adjusted_min_y_big = std::numeric_limits<double>::max();
        bool found_better_min_y_big = false;
        for (const auto& pair : bigRectPoints) {
            if (pair.first > max_y_small) {
                adjusted_min_y_big = std::min(adjusted_min_y_big, pair.first);
                found_better_min_y_big = true;
            }
        }
        if (found_better_min_y_big) {
            min_y_big = adjusted_min_y_big;
        }
        // If still not resolved, the connector might be degenerate or incorrect.
        // For now, we proceed, but this indicates a potential issue with data or split logic.
    }

    std::map<double, std::vector<std::pair<double, double>>> connectorPoints; // y -> [(x, value)]

    if (min_y_big > max_y_small && connectorRows > 0 && !bigRectPoints.empty() && !smallRectPoints.empty()) {
        std::set<double> x_coords_for_connector;
        // Collect x-coordinates from the top edge of the small rectangle (at max_y_small)
        // that are to the right of xSplit.
        if (smallRectPoints.count(max_y_small)) {
            for (const auto& p : smallRectPoints.at(max_y_small)) {
                if (p.first >= xSplit) x_coords_for_connector.insert(p.first);
            }
        }
        // Collect x-coordinates from the bottom edge of the big rectangle (at min_y_big)
        // that are to the right of xSplit.
        if (bigRectPoints.count(min_y_big)) {
            for (const auto& p : bigRectPoints.at(min_y_big)) {
                if (p.first >= xSplit) x_coords_for_connector.insert(p.first);
            }
        }

        double y_step_connector = (min_y_big - max_y_small) / (connectorRows + 1);

        for (double x_conn : x_coords_for_connector) {
            double val_bottom = 0.0; bool found_bottom = false;
            double val_top = 0.0;    bool found_top = false;

            // Find value at (x_conn, max_y_small) from smallRectPoints
            if (smallRectPoints.count(max_y_small)) {
                for (const auto& p : smallRectPoints.at(max_y_small)) {
                    if (std::abs(p.first - x_conn) < 1e-9) { // Tolerance for float comparison
                        val_bottom = p.second;
                        found_bottom = true;
                        break;
                    }
                }
            }
            // If not found directly, try to interpolate from neighbors in the same row (max_y_small)
            if(!found_bottom && smallRectPoints.count(max_y_small)){
                const auto& row_points = smallRectPoints.at(max_y_small);
                if(row_points.size() >= 2){
                    std::vector<std::pair<double,double>> sorted_row = row_points;
                    std::sort(sorted_row.begin(), sorted_row.end());
                    auto it_upper = std::lower_bound(sorted_row.begin(), sorted_row.end(), std::make_pair(x_conn, 0.0));
                    if(it_upper != sorted_row.begin() && it_upper != sorted_row.end()){
                        auto it_lower = it_upper -1;
                        val_bottom = it_lower->second + (it_upper->second - it_lower->second) * (x_conn - it_lower->first) / (it_upper->first - it_lower->first);
                        found_bottom = true;
                    } else if (it_upper == sorted_row.begin() && it_upper != sorted_row.end()){ // x_conn is less than all x in row
                        val_bottom = sorted_row.front().second; // Extrapolate or use closest
                        found_bottom = true;
                    } else if (it_upper == sorted_row.end() && !sorted_row.empty()){ // x_conn is greater than all x in row
                        val_bottom = sorted_row.back().second; // Extrapolate or use closest
                        found_bottom = true;
                    }
                }
            }

            // Find value at (x_conn, min_y_big) from bigRectPoints
            if (bigRectPoints.count(min_y_big)) {
                for (const auto& p : bigRectPoints.at(min_y_big)) {
                    if (std::abs(p.first - x_conn) < 1e-9) {
                        val_top = p.second;
                        found_top = true;
                        break;
                    }
                }
            }
             if(!found_top && bigRectPoints.count(min_y_big)){
                const auto& row_points = bigRectPoints.at(min_y_big);
                 if(row_points.size() >= 2){
                    std::vector<std::pair<double,double>> sorted_row = row_points;
                    std::sort(sorted_row.begin(), sorted_row.end());
                    auto it_upper = std::lower_bound(sorted_row.begin(), sorted_row.end(), std::make_pair(x_conn, 0.0));
                    if(it_upper != sorted_row.begin() && it_upper != sorted_row.end()){
                        auto it_lower = it_upper -1;
                        val_top = it_lower->second + (it_upper->second - it_lower->second) * (x_conn - it_lower->first) / (it_upper->first - it_lower->first);
                        found_top = true;
                    } else if (it_upper == sorted_row.begin() && it_upper != sorted_row.end()){ 
                        val_top = sorted_row.front().second; 
                        found_top = true;
                    } else if (it_upper == sorted_row.end() && !sorted_row.empty()){ 
                        val_top = sorted_row.back().second; 
                        found_top = true;
                    }
                }
            }

            if (found_bottom && found_top) {
                connectorPoints[max_y_small].push_back({x_conn, val_bottom}); // Add bottom point of connector
                double val_step_connector = (val_top - val_bottom) / (connectorRows + 1);
                for (int k = 1; k <= connectorRows; ++k) {
                    double y_curr = max_y_small + k * y_step_connector;
                    double val_curr = val_bottom + k * val_step_connector;
                    connectorPoints[y_curr].push_back({x_conn, val_curr});
                }
                connectorPoints[min_y_big].push_back({x_conn, val_top}); // Add top point of connector
            }
        }
    }

    // Populate QSurfaceDataArrays
    auto populate_surface_data = [](QSurfaceDataArray* dataArray, const std::map<double, std::vector<std::pair<double, double>>>& pointMap) {
        for (auto const& [y_val, x_points] : pointMap) {
            if (x_points.empty()) continue;
            QSurfaceDataRow* row = new QSurfaceDataRow(x_points.size());
            std::vector<std::pair<double, double>> sorted_x_points = x_points;
            std::sort(sorted_x_points.begin(), sorted_x_points.end()); // Sort by x for correct row construction
            int idx = 0;
            for (const auto& p : sorted_x_points) {
                (*row)[idx++].setPosition(QVector3D(p.first, p.second, y_val)); // x, value, y
            }
            dataArray->append(row);
        }
    };

    populate_surface_data(dataArrays.bigRectData, bigRectPoints);
    populate_surface_data(dataArrays.smallRectData, smallRectPoints);
    populate_surface_data(dataArrays.connectorData, connectorPoints);
    
    return dataArrays;
}

// Create series for the G-shape regions
GShapeRegion::GShapeSeries GShapeRegion::createSeries(
    GShapeDataArrays& dataArrays, // Pass by non-const ref
    const QColor& color,
    const QString& seriesName
) {
    GShapeSeries series;
    if (!m_graph3D) return series;

    QLinearGradient grB(0, 0, 1, 100); // Example gradient, adjust as needed
    grB.setColorAt(0.0, Qt::black);
    grB.setColorAt(0.2, Qt::blue);
    grB.setColorAt(0.4, Qt::cyan);
    grB.setColorAt(0.6, Qt::green);
    grB.setColorAt(0.8, Qt::yellow);
    grB.setColorAt(1.0, Qt::red);

    if (dataArrays.bigRectData && !dataArrays.bigRectData->isEmpty()) {
        series.bigRect = new QSurface3DSeries;
        series.bigRect->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
        series.bigRect->setFlatShadingEnabled(false);
        series.bigRect->setBaseGradient(grB);
        series.bigRect->setColorStyle(Q3DTheme::ColorStyleRangeGradient);
        // series.bigRect->setBaseColor(color); // Keep or remove based on preference for gradient vs solid color
        series.bigRect->setName(seriesName + " - Upper Part");
        series.bigRect->dataProxy()->resetArray(dataArrays.bigRectData); // Ownership of bigRectData transferred
        dataArrays.bigRectData = nullptr; // Null out to prevent double deletion
        m_graph3D->addSeries(series.bigRect);
    } else {
        delete dataArrays.bigRectData; // Delete if empty or not used
        dataArrays.bigRectData = nullptr;
    }

    if (dataArrays.smallRectData && !dataArrays.smallRectData->isEmpty()) {
        series.smallRect = new QSurface3DSeries;
        series.smallRect->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
        series.smallRect->setFlatShadingEnabled(false);
        series.smallRect->setBaseGradient(grB);
        series.smallRect->setColorStyle(Q3DTheme::ColorStyleRangeGradient);
        // series.smallRect->setBaseColor(color); 
        series.smallRect->setName(seriesName + " - Lower Right Part");
        series.smallRect->dataProxy()->resetArray(dataArrays.smallRectData);
        dataArrays.smallRectData = nullptr;
        m_graph3D->addSeries(series.smallRect);
    } else {
        delete dataArrays.smallRectData;
        dataArrays.smallRectData = nullptr;
    }

    if (dataArrays.connectorData && !dataArrays.connectorData->isEmpty()) {
        series.connector = new QSurface3DSeries;
        series.connector->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
        series.connector->setFlatShadingEnabled(false);
        series.connector->setBaseGradient(grB);
        series.connector->setColorStyle(Q3DTheme::ColorStyleRangeGradient);
        // series.connector->setBaseColor(color.darker(120)); // Slightly different color for connector
        series.connector->setName(seriesName + " - Connector");
        series.connector->dataProxy()->resetArray(dataArrays.connectorData);
        dataArrays.connectorData = nullptr;
        m_graph3D->addSeries(series.connector);
    } else {
        delete dataArrays.connectorData;
        dataArrays.connectorData = nullptr;
    }
    return series;
}

// Create surfaces for the G-shape regions
bool GShapeRegion::createSurfaces(
    const std::vector<double>& numericalSolution,
    const std::vector<double>& trueSolution,
    const std::vector<double>& errorValues,
    const std::vector<double>& xCoords,
    const std::vector<double>& yCoords,
    double domainXMin, double domainXMax,
    double domainYMin, double domainYMax,
    int decimationFactor,
    int connectorRows
) {
    if (!m_graph3D) {
        qWarning() << "Graph3D is not initialized in GShapeRegion.";
        return false;
    }
    if (numericalSolution.empty() || xCoords.empty() || yCoords.empty()) {
        qWarning() << "No data provided for G-shaped surface construction.";
        return false;
    }

    clearAllSurfaces(); 

    m_currentDomainXMin = domainXMin;
    m_currentDomainXMax = domainXMax;
    m_currentDomainYMin = domainYMin;
    m_currentDomainYMax = domainYMax;
    m_currentValueMin = std::numeric_limits<double>::max(); 
    m_currentValueMax = std::numeric_limits<double>::lowest();

    double xSplit = (domainXMin + domainXMax) / 2.0;
    double ySplit = (domainYMin + domainYMax) / 2.0;

    // Create Numerical Solution Surface
    if (!numericalSolution.empty()) {
        GShapeDataArrays solData = createDataArrays(numericalSolution, xCoords, yCoords, xSplit, ySplit, decimationFactor, connectorRows);
        m_solutionSeries = createSeries(solData, QColor(Qt::blue), "Numerical Solution");
        // updateAxesRanges(numericalSolution); // Initial call, will be refined later
    }

    // Create True Solution Surface (if data provided)
    if (!trueSolution.empty()) {
        GShapeDataArrays trueSolData = createDataArrays(trueSolution, xCoords, yCoords, xSplit, ySplit, decimationFactor, connectorRows);
        m_trueSolutionSeries = createSeries(trueSolData, QColor(Qt::green), "True Solution");
        m_trueSolutionSeries.setVisible(false); 
        // updateAxesRanges(trueSolution); 
    }

    // Create Error Surface (if data provided)
    if (!errorValues.empty()) {
        GShapeDataArrays errorData = createDataArrays(errorValues, xCoords, yCoords, xSplit, ySplit, decimationFactor, connectorRows);
        m_errorSeries = createSeries(errorData, QColor(Qt::red), "Error");
        m_errorSeries.setVisible(false); 
        // updateAxesRanges(errorValues);
    }
    
    std::vector<double> all_values_for_range_check;
    if(!numericalSolution.empty()) all_values_for_range_check.insert(all_values_for_range_check.end(), numericalSolution.begin(), numericalSolution.end());
    if(!trueSolution.empty()) all_values_for_range_check.insert(all_values_for_range_check.end(), trueSolution.begin(), trueSolution.end());
    if(!errorValues.empty()) all_values_for_range_check.insert(all_values_for_range_check.end(), errorValues.begin(), errorValues.end());
    
    if (!all_values_for_range_check.empty()) {
        updateAxesRanges(all_values_for_range_check);
    } else if (!numericalSolution.empty()) { // Fallback if only numerical solution exists and others were empty vectors
         updateAxesRanges(numericalSolution);
    }


    m_graph3D->setTitle("G-Shaped Domain Solution");
    return true;
}