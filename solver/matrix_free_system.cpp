#include "matrix_free_system.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <exception>
#include <numeric>

// Grid system functions - these are identical to the original GridSystem implementations
double MatrixFreeSystem::function(double x, double y) {
    return 4 * (x * x + y * y) * std::exp(x * x - y * y);
}

double MatrixFreeSystem::solution(double x, double y) {
    return std::exp(x*x-y*y);
}

bool MatrixFreeSystem::is_left_boundary(int x, int y) {
    bool is_left_boundory_of_top = x == 0 && (y >= m / 2 && y <= m);
    bool is_left_boundory_of_bottom = x == n / 2 && (y >= 0 && y <= m / 2);
    return is_left_boundory_of_top || is_left_boundory_of_bottom;
}

bool MatrixFreeSystem::is_right_boundary(int x, int y) {
    return x == n;
}

bool MatrixFreeSystem::is_top_boundary(int x, int y) {
    return y == m;
}

bool MatrixFreeSystem::is_bottom_boundary(int x, int y) {
    bool is_bottom_boundory_of_right = y == 0 && (x >= n / 2 && x <= n);
    bool is_bottom_boundory_of_left = y == m / 2 && (x >= 0 && x <= n / 2);
    return is_bottom_boundory_of_right || is_bottom_boundory_of_left;
}

double MatrixFreeSystem::calculate_value(int x, int y, double x_k, double y_k) {
    double x_pos = calculate_x(x);
    double y_pos = calculate_y(y);
    double value = function(x_pos, y_pos);
    if (is_left_boundary(x - 1, y)) {
        value -= x_k * solution(calculate_x(x - 1), calculate_y(y));
    }
    if (is_right_boundary(x + 1, y)) {
        value -= x_k * solution(calculate_x(x + 1), calculate_y(y));
    }
    if (is_top_boundary(x, y + 1)) {
        value -= y_k * solution(calculate_x(x), calculate_y(y + 1));
    }
    if (is_bottom_boundary(x, y - 1)) {
        value -= y_k * solution(calculate_x(x), calculate_y(y - 1));
    }
    return value;
}

double MatrixFreeSystem::calculate_x(int x) {
    return a + x * x_step;
}

double MatrixFreeSystem::calculate_y(int y) {
    return c + y * y_step;
}

bool MatrixFreeSystem::is_boundary(int x, int y) {
    return is_left_boundary(x, y) || is_right_boundary(x, y) || is_top_boundary(x, y) || is_bottom_boundary(x, y);
}

int MatrixFreeSystem::calculate_position_in_template(int x, int y) {
    if (x < n / 2 && y < m / 2) {
        throw std::invalid_argument("Invalid position");
    }
    if (x == 0 || y == 0 || x == n || y == m) {
        throw std::invalid_argument("Invalid position");
    }
    if (y <= m / 2) {
        return calculate_position_in_bottom_edge(x, y);
    }
    int upper_position = calculate_position_in_upper_area(x, y);
    int bottom_position = calculate_position_in_bottom_edge(n - 1, m / 2);
    return upper_position + bottom_position + 1;
}

int MatrixFreeSystem::calculate_position_in_upper_area(int x, int y) {
    return (y - n / 2 - 1) * (n - 1) + x - 1;
}

int MatrixFreeSystem::calculate_position_in_bottom_edge(int x, int y) {
    return (n / 2 - 1) * (y - 1) + x - n / 2 - 1;
}

// Calculate the size of the system (number of grid points excluding boundaries)
int MatrixFreeSystem::calculate_system_size() {
    try {
        // Try to calculate the maximum position index
        return calculate_position_in_template(n - 1, m - 1) + 1;
    } catch (const std::exception& e) {
        // Fallback to an approximation
        return (n * m) / 2;
    }
}

// Initialize the right-hand side vector
void MatrixFreeSystem::initialize_rhs() {
    int system_size = calculate_system_size();
    rhs.resize(system_size, 0.0);
    
    // Fill the right-hand side vector for the bottom-right part
    for (int y = 1; y <= m / 2; ++y) {
        for (int x = n / 2 + 1; x < n; ++x) {
            if (!is_boundary(x, y)) {
                try {
                    int row = calculate_position_in_template(x, y);
                    if (row >= 0 && row < system_size) {
                        rhs[row] = calculate_value(x, y, x_k, y_k);
                    }
                } catch (const std::exception& e) {
                    // Skip invalid positions
                    continue;
                }
            }
        }
    }
    
    // Fill the right-hand side vector for the upper part
    for (int y = m / 2 + 1; y < m; ++y) {
        for (int x = 1; x < n; ++x) {
            if (!is_boundary(x, y)) {
                try {
                    int row = calculate_position_in_template(x, y);
                    if (row >= 0 && row < system_size) {
                        rhs[row] = calculate_value(x, y, x_k, y_k);
                    }
                } catch (const std::exception& e) {
                    // Skip invalid positions
                    continue;
                }
            }
        }
    }
}

// Constructor
MatrixFreeSystem::MatrixFreeSystem(int m, int n, double a, double b, double c, double d) {
    this->n = n;
    this->m = m;
    this->a = a;
    this->b = b;
    this->c = c;
    this->d = d;
    this->x_step = (b - a) / (n);
    this->y_step = (d - c) / (m);
    A = -2 * (1 / (x_step * x_step) + 1 / (y_step * y_step));
    x_k = 1 / (x_step * x_step);
    y_k = 1 / (y_step * y_step);
    
    // Initialize the right-hand side vector
    initialize_rhs();
}

// Get the true solution for comparison
std::vector<double> MatrixFreeSystem::get_true_solution_vector() {
    int system_size = size();
    std::vector<double> true_u(system_size, 0.0);
    
    // Fill values for the bottom-right part
    for (int y_idx = 1; y_idx <= m / 2; ++y_idx) {
        for (int x_idx = n / 2 + 1; x_idx < n; ++x_idx) {
            if (!is_boundary(x_idx, y_idx)) {
                try {
                    int row = calculate_position_in_template(x_idx, y_idx);
                    if (row >= 0 && row < system_size) {
                        true_u[row] = solution(calculate_x(x_idx), calculate_y(y_idx));
                    }
                } catch (const std::exception& e) {
                    continue;
                }
            }
        }
    }
    
    // Fill values for the upper part
    for (int y_idx = m / 2 + 1; y_idx < m; ++y_idx) {
        for (int x_idx = 1; x_idx < n; ++x_idx) {
            if (!is_boundary(x_idx, y_idx)) {
                try {
                    int row = calculate_position_in_template(x_idx, y_idx);
                    if (row >= 0 && row < system_size) {
                        true_u[row] = solution(calculate_x(x_idx), calculate_y(y_idx));
                    }
                } catch (const std::exception& e) {
                    continue;
                }
            }
        }
    }
    
    return true_u;
}

// This is the core of the matrix-free approach:
// Apply the stencil operator directly without storing the matrix
void MatrixFreeSystem::apply(const std::vector<double>& x, std::vector<double>& y) const {
    int system_size = size();
    
    // Reset output vector to zero
    std::fill(y.begin(), y.end(), 0.0);
    
    // Process the bottom-right part of the grid
    for (int y_idx = 1; y_idx <= m / 2; ++y_idx) {
        for (int x_idx = n / 2 + 1; x_idx < n; ++x_idx) {
            if (!is_boundary(x_idx, y_idx)) {
                try {
                    int row = calculate_position_in_template(x_idx, y_idx);
                    if (row >= 0 && row < system_size) {
                        // Apply the central coefficient (diagonal element)
                        y[row] += A * x[row];
                        
                        // Apply off-diagonal elements for the neighboring grid points
                        // Left neighbor
                        if (!is_left_boundary(x_idx - 1, y_idx)) {
                            try {
                                int col = calculate_position_in_template(x_idx - 1, y_idx);
                                if (col >= 0 && col < system_size) {
                                    y[row] += x_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                        
                        // Right neighbor
                        if (!is_right_boundary(x_idx + 1, y_idx)) {
                            try {
                                int col = calculate_position_in_template(x_idx + 1, y_idx);
                                if (col >= 0 && col < system_size) {
                                    y[row] += x_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                        
                        // Top neighbor
                        if (!is_top_boundary(x_idx, y_idx + 1)) {
                            try {
                                int col = calculate_position_in_template(x_idx, y_idx + 1);
                                if (col >= 0 && col < system_size) {
                                    y[row] += y_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                        
                        // Bottom neighbor
                        if (!is_bottom_boundary(x_idx, y_idx - 1)) {
                            try {
                                int col = calculate_position_in_template(x_idx, y_idx - 1);
                                if (col >= 0 && col < system_size) {
                                    y[row] += y_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    continue;
                }
            }
        }
    }
    
    // Process the upper part of the grid
    for (int y_idx = m / 2 + 1; y_idx < m; ++y_idx) {
        for (int x_idx = 1; x_idx < n; ++x_idx) {
            if (!is_boundary(x_idx, y_idx)) {
                try {
                    int row = calculate_position_in_template(x_idx, y_idx);
                    if (row >= 0 && row < system_size) {
                        // Apply the central coefficient (diagonal element)
                        y[row] += A * x[row];
                        
                        // Apply off-diagonal elements for the neighboring grid points
                        // Left neighbor
                        if (!is_left_boundary(x_idx - 1, y_idx)) {
                            try {
                                int col = calculate_position_in_template(x_idx - 1, y_idx);
                                if (col >= 0 && col < system_size) {
                                    y[row] += x_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                        
                        // Right neighbor
                        if (!is_right_boundary(x_idx + 1, y_idx)) {
                            try {
                                int col = calculate_position_in_template(x_idx + 1, y_idx);
                                if (col >= 0 && col < system_size) {
                                    y[row] += x_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                        
                        // Top neighbor
                        if (!is_top_boundary(x_idx, y_idx + 1)) {
                            try {
                                int col = calculate_position_in_template(x_idx, y_idx + 1);
                                if (col >= 0 && col < system_size) {
                                    y[row] += y_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                        
                        // Bottom neighbor
                        if (!is_bottom_boundary(x_idx, y_idx - 1)) {
                            try {
                                int col = calculate_position_in_template(x_idx, y_idx - 1);
                                if (col >= 0 && col < system_size) {
                                    y[row] += y_k * x[col];
                                }
                            } catch (const std::exception&) {
                                // Skip invalid positions
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    continue;
                }
            }
        }
    }
}

// Stream operator for printing information
std::ostream &operator<<(std::ostream &os, const MatrixFreeSystem &grid) {
    os << "MatrixFreeSystem Information:" << std::endl;
    os << "  Dimensions: " << grid.n << "x" << grid.m << std::endl;
    os << "  Domain: [" << grid.a << ", " << grid.b << "] x [" << grid.c << ", " << grid.d << "]" << std::endl;
    os << "  System size: " << grid.size() << std::endl;
    os << "  Memory savings: Matrix-free approach does not store matrix elements" << std::endl;
    return os;
}

// MatrixFreeSolver implementation

// Constructor
MatrixFreeSolver::MatrixFreeSolver(const MatrixFreeSystem& system,
                                   const std::vector<double>& b,
                                   double eps,
                                   int maxIterations,
                                   const std::string& name)
    : system(system), b(b), eps(eps), maxIterations(maxIterations),
      iterations(0), name(name) {}

// Helper method for vector dot product
double MatrixFreeSolver::dot(const std::vector<double>& v1, const std::vector<double>& v2) const {
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

// Helper method for vector norm
double MatrixFreeSolver::norm(const std::vector<double>& v) const {
    return std::sqrt(dot(v, v));
}

// Helper method for vector operation: result = alpha*x + beta*y
void MatrixFreeSolver::axpby(double alpha, const std::vector<double>& x,
                          double beta, const std::vector<double>& y,
                          std::vector<double>& result) const {
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = alpha * x[i] + beta * y[i];
    }
}

// Solve the system using the conjugate gradient method (suitable for symmetric matrices)
std::vector<double> MatrixFreeSolver::solve(const std::vector<double>& true_solution) {
    int n = system.size();
    
    // Initialize solution vector with zeros
    std::vector<double> x(n, 0.0);
    
    // Compute initial residual r = b - A*x
    std::vector<double> r(n);
    std::vector<double> Ax(n);
    system.apply(x, Ax);
    axpby(1.0, b, -1.0, Ax, r);
    
    // Initialize the search direction
    std::vector<double> p = r;
    
    // Calculate initial residual norm
    double r_norm = norm(r);
    double initial_r_norm = r_norm;
    
    // For storing previous solution for convergence check
    std::vector<double> prev_x = x;
    
    // For storing A*p result
    std::vector<double> Ap(n);
    
    // Iteration loop
    for (iterations = 0; iterations < maxIterations && r_norm > eps * initial_r_norm; ++iterations) {
        // Store previous solution
        prev_x = x;
        
        // Compute A*p
        system.apply(p, Ap);
        
        // Compute step size alpha
        double p_dot_Ap = dot(p, Ap);
        double r_dot_r = dot(r, r);
        double alpha = r_dot_r / p_dot_Ap;
        
        // Update solution x = x + alpha*p
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
        }
        
        // Compute residual r = r - alpha*A*p
        for (int i = 0; i < n; ++i) {
            r[i] -= alpha * Ap[i];
        }
        
        // Compute beta
        double new_r_dot_r = dot(r, r);
        double beta = new_r_dot_r / r_dot_r;
        
        // Update search direction p = r + beta*p
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        
        // Update residual norm
        r_norm = std::sqrt(new_r_dot_r);
        
        // Calculate difference between consecutive iterations
        std::vector<double> diff(n);
        for (int i = 0; i < n; ++i) {
            diff[i] = x[i] - prev_x[i];
        }
        double precision = norm(diff);
        
        // Calculate error compared to true solution (if provided)
        std::vector<double> error(n);
        for (int i = 0; i < n; ++i) {
            error[i] = x[i] - true_solution[i];
        }
        double error_norm = norm(error);
        
        // Calculate residual for reporting
        std::vector<double> residual(n);
        system.apply(x, Ax);
        for (int i = 0; i < n; ++i) {
            residual[i] = b[i] - Ax[i];
        }
        double residual_norm = norm(residual);
        
        // Report on current iteration
        if (iteration_callback) {
            iteration_callback(iterations, precision, residual_norm, error_norm);
        }
    }
    
    // Report completion
    bool converged = r_norm <= eps * initial_r_norm;
    std::string message = converged 
        ? "Converged successfully" 
        : "Failed to converge within maximum iterations";
        
    if (completion_callback) {
        completion_callback(converged, message);
    }
    
    return x;
}