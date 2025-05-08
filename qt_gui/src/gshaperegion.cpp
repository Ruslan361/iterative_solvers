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
    updateDynamicAxesRanges();
}

// Set visibility for true solution
void GShapeRegion::setTrueSolutionVisible(bool visible) {
    m_trueSolutionSeries.setVisible(visible);
    updateDynamicAxesRanges();
}

// Set visibility for error surface
void GShapeRegion::setErrorSurfaceVisible(bool visible) {
    m_errorSeries.setVisible(visible);
    updateDynamicAxesRanges();
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
    // These m_currentValueMin/Max will store the overall range from all data passed during createSurfaces
    if (values.empty()) { // if called with empty values, don't alter min/max
        if (m_currentValueMin == std::numeric_limits<double>::max()){ // if it's the very first call and it's empty
             m_graph3D->axisY()->setRange(0,1); // Default Y range
        }
        // else keep existing m_currentValueMin/Max and current axis Y range
        return;
    }


    double current_call_valMin = std::numeric_limits<double>::max();
    double current_call_valMax = std::numeric_limits<double>::lowest();

    for (double val : values) {
        if (!std::isnan(val)) {
            current_call_valMin = std::min(current_call_valMin, val);
            current_call_valMax = std::max(current_call_valMax, val);
        }
    }
    
    if (m_currentValueMin == std::numeric_limits<double>::max() || current_call_valMin < m_currentValueMin) {
        m_currentValueMin = current_call_valMin;
    }
    if (m_currentValueMax == std::numeric_limits<double>::lowest() || current_call_valMax > m_currentValueMax) {
        m_currentValueMax = current_call_valMax;
    }

    // Add a small margin to avoid data points being exactly on the edge of the axis range
    double margin = std::abs(current_call_valMax - current_call_valMin) * 0.1;
    if (margin == 0) margin = std::max(0.1, std::abs(current_call_valMin*0.1)); // Handle case where all values are the same or zero
    if (current_call_valMin == std::numeric_limits<double>::max()){ // All NaNs or empty
         m_graph3D->axisY()->setRange(0,1);
    } else {
        m_graph3D->axisY()->setRange(current_call_valMin - margin, current_call_valMax + margin); // Y-axis in QtDataVis is the value axis
    }


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
        // Collect x-coordinates ONLY from the top edge of the small rectangle (at max_y_small)
        // that are to the right of xSplit. This ensures the connector's X-footprint
        // is determined by the small rectangle.
        if (smallRectPoints.count(max_y_small)) {
            for (const auto& p : smallRectPoints.at(max_y_small)) { // p is {x, value}
                if (p.first >= xSplit) { // p.first is X
                    x_coords_for_connector.insert(p.first);
                }
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
    m_currentValueMin = std::numeric_limits<double>::max(); // Reset for new data
    m_currentValueMax = std::numeric_limits<double>::lowest();

    // Store data for dynamic updates
    m_lastNumericalSolutionData.clear();
    m_lastTrueSolutionData.clear();
    m_lastErrorData.clear();

    if (!numericalSolution.empty()) m_lastNumericalSolutionData = numericalSolution;
    if (!trueSolution.empty()) m_lastTrueSolutionData = trueSolution;
    if (!errorValues.empty()) m_lastErrorData = errorValues;


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
    } else if (!numericalSolution.empty()) { // Fallback if only numerical solution exists and others were empty
         updateAxesRanges(numericalSolution);
    }


    m_graph3D->setTitle("G-Shaped Domain Solution");
    updateDynamicAxesRanges(); // Call this to set axes based on default visibility
    return true;
}

void GShapeRegion::updateDynamicAxesRanges() {
    if (!m_graph3D || !m_graph3D->axisY() || !m_graph3D->axisX() || !m_graph3D->axisZ()) return;

    std::vector<double> values_to_consider;
    bool sol_series_visible = m_solutionSeries.bigRect && m_solutionSeries.bigRect->isVisible();
    bool true_sol_series_visible = m_trueSolutionSeries.bigRect && m_trueSolutionSeries.bigRect->isVisible();
    bool error_series_visible = m_errorSeries.bigRect && m_errorSeries.bigRect->isVisible();

    if (sol_series_visible && !m_lastNumericalSolutionData.empty()) {
        values_to_consider.insert(values_to_consider.end(), m_lastNumericalSolutionData.begin(), m_lastNumericalSolutionData.end());
    }
    if (true_sol_series_visible && !m_lastTrueSolutionData.empty()) {
        values_to_consider.insert(values_to_consider.end(), m_lastTrueSolutionData.begin(), m_lastTrueSolutionData.end());
    }
    if (error_series_visible && !m_lastErrorData.empty()) {
        values_to_consider.insert(values_to_consider.end(), m_lastErrorData.begin(), m_lastErrorData.end());
    }

    double valMin = std::numeric_limits<double>::max();
    double valMax = std::numeric_limits<double>::lowest();

    if (!values_to_consider.empty()) {
        for (double val : values_to_consider) {
            if (!std::isnan(val)) {
                valMin = std::min(valMin, val);
                valMax = std::max(valMax, val);
            }
        }
    }

    // If no series are visible or they have no data, fall back to overall range or a default
    if (valMin == std::numeric_limits<double>::max()) { // No valid data points from visible series
        if (m_currentValueMin != std::numeric_limits<double>::max()) { // Use overall range if available
            valMin = m_currentValueMin;
            valMax = m_currentValueMax;
        } else { // Absolute fallback
            valMin = 0;
            valMax = 1;
        }
    }
    
    // If after all fallbacks, valMin is still at max (e.g. m_currentValueMin was never set), use default
    if (valMin == std::numeric_limits<double>::max()) {
        valMin = 0; valMax = 1;
    }


    double y_axis_min = valMin;
    double y_axis_max = valMax;
    double range = valMax - valMin;

    bool only_error_is_active = error_series_visible && !m_lastErrorData.empty() &&
                                !sol_series_visible && !true_sol_series_visible;
    if (error_series_visible && m_lastErrorData.empty()){ // if error series is visible but has no data
        only_error_is_active = false;
    }
    // Also consider cases where solution/true_solution series are visible but their data is empty
    if (sol_series_visible && m_lastNumericalSolutionData.empty()) sol_series_visible = false;
    if (true_sol_series_visible && m_lastTrueSolutionData.empty()) true_sol_series_visible = false;
    
    only_error_is_active = error_series_visible && !m_lastErrorData.empty() &&
                           !sol_series_visible && !true_sol_series_visible;


    if (only_error_is_active) {
        if (range == 0) { 
            double abs_val = std::abs(valMin);
            double gap = std::max(abs_val * 0.2, 1e-4); // Ensure a visible gap, at least 1e-4 or 20% of value
            if (abs_val < 1e-9) gap = 1e-3; // If value is ~0, use a fixed small gap like [-0.001, 0.001]
            y_axis_min = valMin - gap;
            y_axis_max = valMax + gap;
        } else {
            // Ensure a minimum visual range if errors are very close to zero or each other
            double min_visual_range_abs = 1e-4; // Absolute minimum range span
            double min_visual_range_rel_mag = std::max(std::abs(valMin), std::abs(valMax)) * 0.5; // 50% of max magnitude
            double required_span = std::max(min_visual_range_abs, min_visual_range_rel_mag);
            
            if (range < required_span && range > 0) { // range > 0 to avoid issues if valMin=valMax
                 double mid = (valMin + valMax) / 2.0;
                 y_axis_min = mid - required_span / 2.0;
                 y_axis_max = mid + required_span / 2.0;
            } else {
                double margin = range * 0.35; // Use a larger margin, e.g., 35% for errors
                y_axis_min = valMin - margin;
                y_axis_max = valMax + margin;
            }
        }
        // If values are extremely close to zero, ensure the range isn't overly magnified to noise
        // and also ensure that if min/max are slightly different but both tiny, the range reflects that.
        if (std::abs(y_axis_min) < 1e-7 && std::abs(y_axis_max) < 1e-7 && (y_axis_max - y_axis_min) < 1e-3) {
             if (y_axis_min == 0 && y_axis_max == 0) { // if strictly zero
                y_axis_min = -1e-3; y_axis_max = 1e-3;
             } else { // if very small, center around midpoint with a small fixed span
                double mid = (y_axis_min + y_axis_max) / 2.0;
                y_axis_min = mid - 5e-4; // e.g. total span of 1e-3
                y_axis_max = mid + 5e-4;
             }
        }
    } else { 
        double margin = range * 0.1;
        if (margin == 0) { // All values in 'values_to_consider' are the same
            margin = std::max(std::abs(valMin * 0.1), 0.1); 
        }
        y_axis_min = valMin - margin;
        y_axis_max = valMax + margin;
    }
    
    if (y_axis_min == y_axis_max) { // Final safety net if range is still zero
        y_axis_min -= 0.1; // Default small range
        y_axis_max += 0.1;
    }

    m_graph3D->axisX()->setRange(m_currentDomainXMin, m_currentDomainXMax);
    m_graph3D->axisZ()->setRange(m_currentDomainYMin, m_currentDomainYMax);
    m_graph3D->axisY()->setRange(y_axis_min, y_axis_max);
}