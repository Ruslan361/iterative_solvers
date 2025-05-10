#include "shaperegion.h"

// Constructor
ShapeRegion::ShapeRegion(Q3DSurface* graph3D)
    : m_graph3D(graph3D) {
}

// Destructor
ShapeRegion::~ShapeRegion() {
    // Base class destructor - concrete classes will handle their specific cleanup
}