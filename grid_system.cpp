#include "grid_system.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <vector>

#include <algorithm>
#include <exception>

double GridSystem::function(double x, double y) {
    return 4 * (x * x + y * y) * std::exp(x * x - y * y);
}

bool GridSystem::is_left_boundary(int x, int y)
{
    bool is_left_boundory_of_top = x == 0 && (y >= m / 2 && y <= m);
    bool is_left_boundory_of_bottom = x == n / 2 && (y >= 0 && y <= m / 2);
    return is_left_boundory_of_top || is_left_boundory_of_bottom;
}

bool GridSystem::is_right_boundary(int x, int y)
{
    return x == n;
}

bool GridSystem::is_top_boundary(int x, int y)
{
    return y == m;
}

bool GridSystem::is_bottom_boundary(int x, int y)
{
    bool is_bottom_boundory_of_right = y == 0 && (x >= n / 2 && x <= n);
    bool is_bottom_boundory_of_left = y == m / 2 && (x >= 0 && x <= n / 2);
    return is_bottom_boundory_of_right || is_bottom_boundory_of_left;
}

double GridSystem::calculate_value(int x, int y, double x_k, double y_k)
{
    double x_pos = calculate_x(x);
    double y_pos = calculate_y(y);
    double value = function(x_pos, y_pos);
    if (is_left_boundary(x - 1, y))
    {
        value -= x_k * function(calculate_x(x - 1), calculate_y(y));
    }
    if (is_right_boundary(x + 1, y))
    {
        value -= x_k * function(calculate_x(x + 1), calculate_y(y));
    }
    if (is_top_boundary(x, y + 1))
    {
        value -= y_k * function(calculate_x(x), calculate_y(y + 1));
    }
    if (is_bottom_boundary(x, y - 1))
    {
        value -= y_k * function(calculate_x(x), calculate_y(y - 1));
    }
    return value;
}

double GridSystem::calculate_x(int x)
{
    return a + x * x_step;
}

double GridSystem::calculate_y(int y)
{
    return c + y * y_step;
}

void GridSystem::initiate_matrix()
{
    int amount_of_nodes = calculate_position_in_template(n - 1, m - 1) + 1;

    for (int y = 1; y <= m / 2; ++y)
    {
        for (int x = n/2 + 1; x < n; ++x)
        {
            apply_template(amount_of_nodes, x, y);
        }   
    }

    for (int y = m/2 + 1; y < m; ++y)
    {
        for (int x = 1; x < n; ++x)
        {
            apply_template(amount_of_nodes, x, y);
        }   
    }
}

void GridSystem::apply_template(int amount_of_nodes, int x, int y)
{
    matrix.push_back(std::vector<double>(amount_of_nodes, 0));
    int position = calculate_position_in_template(x, y);
    int last_index = matrix.size() - 1;
    matrix[last_index][position] = A;
    boundary_in_matrix(x, y, last_index);
    rhs.push_back(calculate_value(x, y, x_k, y_k));
}

void GridSystem::boundary_in_matrix(int x, int y, int index)
{
    if (!is_left_boundary(x - 1, y))
    {
        int left_pos = calculate_position_in_template(x - 1, y);
        matrix[index][left_pos] = x_k;
    }
    if (!is_right_boundary(x + 1, y))
    {
        int right_pos = calculate_position_in_template(x + 1, y);
        matrix[index][right_pos] = x_k;
    }
    if (!is_top_boundary(x, y + 1))
    {
        int top_pos = calculate_position_in_template(x, y + 1);
        matrix[index][top_pos] = y_k;
    }
    if (!is_bottom_boundary(x, y - 1))
    {
        int bottom_pos = calculate_position_in_template(x, y - 1);
        matrix[index][bottom_pos] = y_k;
    }
}

bool GridSystem::is_boundary(int x, int y)
{
    return is_left_boundary(x, y) || is_right_boundary(x, y) || is_top_boundary(x, y) || is_bottom_boundary(x, y);
}

int GridSystem::calculate_position_in_template(int x, int y)
{
    if (x < n / 2 && y < m / 2)
    {
        throw std::invalid_argument("Invalid position");
    }
    if (x == 0 || y == 0 || x == n || y == m)
    {
        throw std::invalid_argument("Invalid position");
    }
    if (y <= m / 2)
    {   // 2 * 1 + 4 - 3 - 1 = 2
        return calculate_position_in_bottom_edge(x, y);
    }                                                   
    int upper_position = calculate_position_in_upper_area(x, y);
    int bottom_position = calculate_position_in_bottom_edge(n - 1, m / 2);
    return upper_position + bottom_position + 1;
}

int GridSystem::calculate_position_in_upper_area(int x, int y)
{
    return (y - n / 2 - 1) * (n - 1) + x - 1;
}

int GridSystem::calculate_position_in_bottom_edge(int x, int y)
{
    return (n / 2 - 1) * (y - 1) + x - n / 2 - 1;
}

std::vector<double> GridSystem::apply_template(int x, int y)
{
    int size = (m - 1) * (n / 2 - 1);
    std::vector<double> template_matrix = std::vector<double>(size, 0);
    return std::vector<double>();
}

std::vector<std::vector<double>> GridSystem::get_matrix()
{
    return std::vector<std::vector<double>>(matrix);
}

std::vector<double> GridSystem::get_rhs()
{
    return std::vector<double>(rhs);
}

GridSystem::GridSystem(int m, int n, double a, double b, double c, double d)
{
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
    initiate_matrix();
    // matrix = create_matrix(m, n, a, b, c, d);
}
GridSystem::~GridSystem()
{
}