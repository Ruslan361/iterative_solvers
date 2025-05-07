#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>
#include "grid_system.h"

// A matrix-free representation of the grid system matrix
// This class doesn't store the matrix elements but computes them on-the-fly
class MatrixFreeSystem {
private:
    int n, m;                     // Grid dimensions
    double a, b, c, d;            // Domain boundaries
    double x_step, y_step;        // Grid step sizes
    double A, x_k, y_k;           // Coefficients for the difference scheme
    
    // Right-hand side vector
    std::vector<double> rhs;
    
    // Helper functions for grid operations
    double function(double x, double y);
    double solution(double x, double y);
    bool is_left_boundary(int x, int y);
    bool is_right_boundary(int x, int y);
    bool is_top_boundary(int x, int y);
    bool is_bottom_boundary(int x, int y);
    bool is_boundary(int x, int y);
    double calculate_value(int x, int y, double x_k, double y_k);
    double calculate_x(int x);
    double calculate_y(int y);
    int calculate_position_in_template(int x, int y);
    int calculate_position_in_upper_area(int x, int y);
    int calculate_position_in_bottom_edge(int x, int y);
    
    // Calculate system size based on the grid dimensions
    int calculate_system_size();
    
    // Initialize the right-hand side vector
    void initialize_rhs();

public:
    // Constructor and destructor
    MatrixFreeSystem(int m, int n, double a, double b, double c, double d);
    ~MatrixFreeSystem() = default;
    
    // Get the right-hand side vector
    const std::vector<double>& get_rhs() const { return rhs; }
    
    // Calculate the true solution vector for comparison
    std::vector<double> get_true_solution_vector();
    
    // Matrix-vector product operation y = A*x
    // This is the key method that performs the matrix operation on-the-fly
    void apply(const std::vector<double>& x, std::vector<double>& y) const;
    
    // Matrix-vector product with overloaded operator
    std::vector<double> operator*(const std::vector<double>& x) const {
        std::vector<double> result(size());
        apply(x, result);
        return result;
    }
    
    // Get system size (number of equations)
    int size() const { return rhs.size(); }
    
    // Print information about the matrix-free system
    friend std::ostream &operator<<(std::ostream &os, const MatrixFreeSystem &grid);
};

// Matrix-free solver class for use with the MatrixFreeSystem
class MatrixFreeSolver {
private:
    const MatrixFreeSystem& system;  // Reference to the matrix-free system
    const std::vector<double>& b;    // Right-hand side vector
    double eps;                      // Convergence tolerance
    int maxIterations;               // Maximum number of iterations
    int iterations;                  // Performed iterations
    std::string name;                // Name of the solver method

    // Callback for tracking intermediate results
    std::function<void(int, double, double, double)> iteration_callback;
    
    // Callback for completion notification
    std::function<void(bool, const std::string&)> completion_callback;
    
    // Helper functions for vector operations
    double dot(const std::vector<double>& v1, const std::vector<double>& v2) const;
    double norm(const std::vector<double>& v) const;
    
    // Vector operations helpers
    void axpby(double alpha, const std::vector<double>& x, 
               double beta, const std::vector<double>& y, 
               std::vector<double>& result) const;

public:
    MatrixFreeSolver(const MatrixFreeSystem& system, 
                     const std::vector<double>& b, 
                     double eps = 1e-6, 
                     int maxIterations = 10000,
                     const std::string& name = "Matrix-free solver");
    
    virtual ~MatrixFreeSolver() = default;
    
    // Solve the system using the conjugate gradient method
    std::vector<double> solve(const std::vector<double>& true_solution);
    
    // Set callback for tracking iterations
    void setIterationCallback(std::function<void(int, double, double, double)> callback) {
        iteration_callback = callback;
    }
    
    // Set callback for completion notification
    void setCompletionCallback(std::function<void(bool, const std::string&)> callback) {
        completion_callback = callback;
    }
    
    // Common methods for all solvers
    int getIterations() const {
        return iterations;
    }
    
    std::string getName() const {
        return name;
    }
};