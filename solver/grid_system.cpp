#include "grid_system.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <exception>

double GridSystem::function(double x, double y) {
    return 4 * (x * x + y * y) * std::exp(x * x - y * y);
}

double GridSystem::solution(double x, double y)
{
    return std::exp(x*x-y*y);
}

bool GridSystem::is_left_boundary(int x, int y) const
{
    bool is_left_boundory_of_top = x == 0 && (y >= m / 2 && y <= m);
    bool is_left_boundory_of_bottom = x == n / 2 && (y >= 0 && y <= m / 2);
    return is_left_boundory_of_top || is_left_boundory_of_bottom;
}

bool GridSystem::is_right_boundary(int x, int y) const
{
    // Unused parameter warning fixed with (void)y
    (void)y;
    return x == n;
}

bool GridSystem::is_top_boundary(int x, int y) const
{
    // Unused parameter warning fixed with (void)x
    (void)x;
    return y == m;
}

bool GridSystem::is_bottom_boundary(int x, int y) const
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
        value -= x_k * solution(calculate_x(x - 1), calculate_y(y));
    }
    if (is_right_boundary(x + 1, y))
    {
        value -= x_k * solution(calculate_x(x + 1), calculate_y(y));
    }
    if (is_top_boundary(x, y + 1))
    {
        value -= y_k * solution(calculate_x(x), calculate_y(y + 1));
    }
    if (is_bottom_boundary(x, y - 1))
    {
        value -= y_k * solution(calculate_x(x), calculate_y(y - 1));
    }
    return value;
}

double GridSystem::calculate_x(int x) const
{
    return a + x * x_step;
}

double GridSystem::calculate_y(int y) const
{
    return c + y * y_step;
}

bool GridSystem::is_boundary(int x, int y) const
{
    return is_left_boundary(x, y) || is_right_boundary(x, y) || is_top_boundary(x, y) || is_bottom_boundary(x, y);
}

int GridSystem::calculate_position_in_template(int x, int y) const
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
    {
        return calculate_position_in_bottom_edge(x, y);
    }
    int upper_position = calculate_position_in_upper_area(x, y);
    int bottom_position = calculate_position_in_bottom_edge(n - 1, m / 2);
    return upper_position + bottom_position + 1;
}

int GridSystem::calculate_position_in_upper_area(int x, int y) const
{
    return (y - n / 2 - 1) * (n - 1) + x - 1;
}

int GridSystem::calculate_position_in_bottom_edge(int x, int y) const
{
    return (n / 2 - 1) * (y - 1) + x - n / 2 - 1;
}

// Вспомогательный метод для добавления элемента в разреженную матрицу
void GridSystem::add_matrix_entry(int row, int col, double value)
{
    entries.push_back(col);
    values.push_back(value);
    row_map[row + 1]++;
}

// Финализация разреженной матрицы
void GridSystem::finalize_matrix()
{
    // Преобразуем счетчики элементов в каждой строке в смещения
    for (size_t i = 1; i < row_map.size(); ++i) {
        row_map[i] += row_map[i - 1];
    }
    
    // Создаем Kokkos::View для хранения данных матрицы
    int n_rows = row_map.size() - 1;
    int nnz = values.size();
    
    Kokkos::View<int*, memory_space> kokkos_row_map("row_map", n_rows + 1);
    Kokkos::View<int*, memory_space> kokkos_entries("entries", nnz);
    Kokkos::View<double*, memory_space> kokkos_values("values", nnz);
    
    // Копируем данные из std::vector в Kokkos::View
    for (int i = 0; i <= n_rows; ++i) {
        kokkos_row_map(i) = row_map[i];
    }
    
    for (int i = 0; i < nnz; ++i) {
        kokkos_entries(i) = entries[i];
        kokkos_values(i) = values[i];
    }
    
    // Создаем KokkosCrsMatrix
    matrix = KokkosCrsMatrix("A", n_rows, n_rows, nnz, kokkos_values, kokkos_row_map, kokkos_entries);
    
    // Создаем вектор правой части
    rhs = KokkosVector("rhs", n_rows);
    for (int i = 0; i < n_rows; ++i) {
        rhs(i) = rhs_values[i];
    }
}

void GridSystem::initiate_matrix()
{
    // Определяем количество узлов (размер матрицы)
    int amount_of_nodes = 0;
    try {
        amount_of_nodes = calculate_position_in_template(n - 1, m - 1) + 1;
    } catch (const std::exception& e) {
        std::cerr << "Error calculating template position: " << e.what() << std::endl;
        amount_of_nodes = (n * m) / 2; // Примерная оценка размера
    }
    
    // Инициализируем структуры для хранения разреженной матрицы
    row_map.resize(amount_of_nodes + 1, 0);
    
    // Предварительная оценка количества ненулевых элементов (5 элементов на строку)
    values.reserve(amount_of_nodes * 5);
    entries.reserve(amount_of_nodes * 5);
    rhs_values.resize(amount_of_nodes);
    
    // Инициализируем векторы координат
    node_x_coords.resize(amount_of_nodes, 0.0);
    node_y_coords.resize(amount_of_nodes, 0.0);
    
    // Заполняем матрицу для нижней правой части
    for (int y = 1; y <= m / 2; ++y)
    {
        for (int x = n/2 + 1; x < n; ++x)
        {
            if (!is_boundary(x, y)) {
                int row = calculate_position_in_template(x, y);
                
                // Сохраняем координаты узла
                node_x_coords[row] = calculate_x(x);
                node_y_coords[row] = calculate_y(y);
                
                // Диагональный элемент
                add_matrix_entry(row, row, A);
                
                // Соседние элементы, если они не на границе
                if (!is_left_boundary(x - 1, y))
                {
                    int left_pos = calculate_position_in_template(x - 1, y);
                    add_matrix_entry(row, left_pos, x_k);
                }
                
                if (!is_right_boundary(x + 1, y))
                {
                    int right_pos = calculate_position_in_template(x + 1, y);
                    add_matrix_entry(row, right_pos, x_k);
                }
                
                if (!is_top_boundary(x, y + 1))
                {
                    int top_pos = calculate_position_in_template(x, y + 1);
                    add_matrix_entry(row, top_pos, y_k);
                }
                
                if (!is_bottom_boundary(x, y - 1))
                {
                    int bottom_pos = calculate_position_in_template(x, y - 1);
                    add_matrix_entry(row, bottom_pos, y_k);
                }
                
                // Вектор правой части
                rhs_values[row] = calculate_value(x, y, x_k, y_k);
            }
        }   
    }
    
    // Заполняем матрицу для верхней части
    for (int y = m/2 + 1; y < m; ++y)
    {
        for (int x = 1; x < n; ++x)
        {
            if (!is_boundary(x, y)) {
                int row = calculate_position_in_template(x, y);
                
                // Сохраняем координаты узла
                node_x_coords[row] = calculate_x(x);
                node_y_coords[row] = calculate_y(y);
                
                // Диагональный элемент
                add_matrix_entry(row, row, A);
                
                // Соседние элементы, если они не на границе
                if (!is_left_boundary(x - 1, y))
                {
                    int left_pos = calculate_position_in_template(x - 1, y);
                    add_matrix_entry(row, left_pos, x_k);
                }
                
                if (!is_right_boundary(x + 1, y))
                {
                    int right_pos = calculate_position_in_template(x + 1, y);
                    add_matrix_entry(row, right_pos, x_k);
                }
                
                if (!is_top_boundary(x, y + 1))
                {
                    int top_pos = calculate_position_in_template(x, y + 1);
                    add_matrix_entry(row, top_pos, y_k);
                }
                
                if (!is_bottom_boundary(x, y - 1))
                {
                    int bottom_pos = calculate_position_in_template(x, y - 1);
                    add_matrix_entry(row, bottom_pos, y_k);
                }
                
                // Вектор правой части
                rhs_values[row] = calculate_value(x, y, x_k, y_k);
            }
        }   
    }
    
    // Финализируем матрицу
    finalize_matrix();
}

KokkosVector GridSystem::get_true_solution_vector() {
    if (matrix.numRows() == 0) {
        throw std::runtime_error("Matrix not initialized, cannot determine size for true solution vector.");
    }
    int amount_of_nodes = matrix.numRows();
    KokkosVector true_u("true_u", amount_of_nodes);
    auto u_host = Kokkos::create_mirror_view(true_u);

    // Используем сохраненные координаты для вычисления точного решения
    for(int i = 0; i < amount_of_nodes; ++i) {
        if (i < node_x_coords.size() && i < node_y_coords.size()) {
            double x_coord = node_x_coords[i];
            double y_coord = node_y_coords[i];
            
            // Вычисляем точное решение в данной точке
            u_host(i) = solution(x_coord, y_coord);
        } else {
            u_host(i) = 0.0; // Значение по умолчанию, если координаты недоступны
        }
    }

    Kokkos::deep_copy(true_u, u_host);
    return true_u;
}

GridSystem::GridSystem(int m, int n, double a, double b, double c, double d)
{
    // Инициализация Kokkos, если еще не инициализирована
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
    
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
    
    // Сборка разреженной матрицы и вектора правой части
    initiate_matrix();
}

GridSystem::~GridSystem()
{
    // При необходимости освобождаем ресурсы Kokkos
    // Если это последний объект, использующий Kokkos, можно вызвать finalize
    // Обычно это делается в конце main()
}

// Новый метод для получения координат узла по индексу в массиве решения
GridSystem::NodeCoordinates GridSystem::get_node_coordinates(int solution_index) const
{
    NodeCoordinates coords;
    coords.x = 0.0;
    coords.y = 0.0;
    
    // Проверяем корректность индекса
    if (solution_index < 0 || solution_index >= matrix.numRows()) {
        return coords; // Возвращаем нулевые координаты для некорректного индекса
    }
    
    // Определяем, находится ли точка в верхней или нижней области
    int bottom_max_index = 0;
    try {
        // Оценка максимального индекса для нижней части
        bottom_max_index = calculate_position_in_bottom_edge(n - 1, m / 2);
    } catch (const std::exception& e) {
        // В случае ошибки возвращаем нулевые координаты
        return coords;
    }
    
    if (solution_index <= bottom_max_index) {
        // Точка находится в нижней правой части
        // Найдем соответствующие индексы x, y
        for (int y = 1; y <= m / 2; ++y) {
            for (int x = n/2 + 1; x < n; ++x) {
                if (!is_boundary(x, y)) {
                    int idx = 0;
                    try {
                        idx = calculate_position_in_template(x, y);
                    } catch (const std::exception& e) {
                        continue; // Пропускаем, если точка вызывает исключение
                    }
                    
                    if (idx == solution_index) {
                        coords.x = calculate_x(x);
                        coords.y = calculate_y(y);
                        return coords;
                    }
                }
            }
        }
    } else {
        // Точка находится в верхней области
        for (int y = m/2 + 1; y < m; ++y) {
            for (int x = 1; x < n; ++x) {
                if (!is_boundary(x, y)) {
                    int idx = 0;
                    try {
                        idx = calculate_position_in_template(x, y);
                    } catch (const std::exception& e) {
                        continue; // Пропускаем, если точка вызывает исключение
                    }
                    
                    if (idx == solution_index) {
                        coords.x = calculate_x(x);
                        coords.y = calculate_y(y);
                        return coords;
                    }
                }
            }
        }
    }
    
    return coords;
}

std::ostream &operator<<(std::ostream &os, const GridSystem &grid)
{
    // Выводим информацию о матрице в сжатом виде
    os << "GridSystem Matrix Information:" << std::endl;
    os << "  Dimensions: " << grid.n << "x" << grid.m << std::endl;
    os << "  Domain: [" << grid.a << ", " << grid.b << "] x [" << grid.c << ", " << grid.d << "]" << std::endl;
    os << "  Matrix size: " << grid.matrix.numRows() << " rows x " << grid.matrix.numCols() << " columns" << std::endl;
    os << "  Non-zero elements: " << grid.matrix.nnz() << std::endl;
    os << "  Sparsity: " << (1.0 - (double)grid.matrix.nnz() / (grid.matrix.numRows() * grid.matrix.numCols())) * 100.0 << "%" << std::endl;
    
    // Для больших матриц вывод всех элементов нецелесообразен
    return os;
}