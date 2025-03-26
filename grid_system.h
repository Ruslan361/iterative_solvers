#pragma once

#include <vector>
#include <ostream>
#include <iomanip>

class GridSystem
{
private:
    int n, m;
    double a, b, c, d;
    double x_step, y_step;
    double A, x_k, y_k;
    std::vector<double> rhs;
    std::vector<std::vector<double>> matrix;
    double function(double x, double y);
    // void rectangle_traversal(int x_start, int y_start, double x_step, double y_step, int n, int m);
    // std::vector<std::vector<double>> create_matrix(int pos, int n, int m, double a = 1, double b = 2, double c = 1, double d = 2);
    bool is_left_boundary(int x, int y);
    bool is_right_boundary(int x, int y);
    bool is_top_boundary(int x, int y);
    bool is_bottom_boundary(int x, int y);
    double calculate_value(int x, int y, double x_k, double y_k);
    double calculate_x(int x);
    double calculate_y(int y);
    void initiate_matrix();
    void apply_template(int amount_of_nodes, int x, int y);
    void boundary_in_matrix(int x, int y, int index);
    bool is_boundary(int x, int y);
    int calculate_position_in_template(int x, int y);
    int calculate_position_in_upper_area(int y, int x);
    int calculate_position_in_bottom_edge(int y, int x);
    std::vector<double> apply_template(int x, int y);
    public:
    std::vector<std::vector<double>> get_matrix();
    std::vector<double> get_rhs();
    friend std::ostream &operator<<(std::ostream &os, const GridSystem &grid)
    {
        for (int i = 0; i < grid.matrix.size(); i++)
        {
            for (int j = 0; j < grid.matrix[i].size(); j++)
            {
                os << std::setw(5) << grid.matrix[i][j];
            }
            os << "  ";
            os << std::setw(5) << grid.rhs[i] << std::endl;
        }
        return os;
    }
    GridSystem(int m, int n, double a, double b, double c, double d);
    ~GridSystem();
};

