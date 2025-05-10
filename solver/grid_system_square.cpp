#include "grid_system_square.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <exception>

// Используем те же функции, что и в grid_system.cpp
// extern double function2(double x, double y); // Старая функция, больше не нужна здесь как extern
// extern double solution2(double x, double y); // Старое решение, больше не нужна здесь как extern

// Объявления новых функций из default_functions.cpp
extern double custom_function_square(double x, double y);
extern double mu1_square(double x, double y);
extern double mu2_square(double x, double y);
extern double mu3_square(double x, double y);
extern double mu4_square(double x, double y);

double GridSystemSquare::function(double x, double y) {
    return funcPtr(x, y);
}

double GridSystemSquare::solution(double x, double y) {
    return solPtr(x, y);
}

bool GridSystemSquare::is_left_boundary(int x, int y) const {
    // В квадратной области левая граница - это x = 0
    (void)y; // Устраняем предупреждение о неиспользуемой переменной
    return x == 0;
}

bool GridSystemSquare::is_right_boundary(int x, int y) const {
    // Правая граница - x = n
    (void)y;
    return x == n;
}

bool GridSystemSquare::is_top_boundary(int x, int y) const {
    // Верхняя граница - y = m
    (void)x;
    return y == m;
}

bool GridSystemSquare::is_bottom_boundary(int x, int y) const {
    // Нижняя граница - y = 0
    (void)x;
    return y == 0;
}

bool GridSystemSquare::is_boundary(int x, int y) const {
    return is_left_boundary(x, y) || is_right_boundary(x, y) || 
           is_top_boundary(x, y) || is_bottom_boundary(x, y);
}

double GridSystemSquare::calculate_value(int x, int y, double x_k, double y_k) {
    double x_pos = calculate_x(x);
    double y_pos = calculate_y(y);
    // ИСПРАВЛЕНИЕ: Меняем знак у функции источника при формировании правой части
    double value = -function(x_pos, y_pos); 
    
    // Учитываем граничные условия Дирихле, если соседняя точка находится на границе
    if (is_left_boundary(x - 1, y)) {
        double boundary_x = calculate_x(x - 1);
        double boundary_y = calculate_y(y);
        value -= x_k * boundary_value(boundary_x, boundary_y, 1);
    }
    
    if (is_right_boundary(x + 1, y)) {
        double boundary_x = calculate_x(x + 1);
        double boundary_y = calculate_y(y);
        value -= x_k * boundary_value(boundary_x, boundary_y, 2);
    }
    
    if (is_bottom_boundary(x, y - 1)) {
        double boundary_x = calculate_x(x);
        double boundary_y = calculate_y(y - 1);
        value -= y_k * boundary_value(boundary_x, boundary_y, 3);
    }
    
    if (is_top_boundary(x, y + 1)) {
        double boundary_x = calculate_x(x);
        double boundary_y = calculate_y(y + 1);
        value -= y_k * boundary_value(boundary_x, boundary_y, 4);
    }
    
    return value;
}

double GridSystemSquare::calculate_x(int x) const {
    return a + x * x_step;
}

double GridSystemSquare::calculate_y(int y) const {
    return c + y * y_step;
}

int GridSystemSquare::calculate_position_in_matrix(int x, int y) const {
    // Для квадратной области индекс в матрице вычисляется проще
    // Пропускаем граничные узлы, поэтому внутренних узлов будет (n-1)*(m-1)
    // Индексация с нуля для внутренних узлов
    if (is_boundary(x, y)) {
        throw std::invalid_argument("Invalid position: boundary point");
    }
    
    // Вычисляем позицию в матрице (нумерация по строкам)
    return (y - 1) * (n - 1) + (x - 1);
}

void GridSystemSquare::add_matrix_entry(int row, int col, double value) {
    entries.push_back(col);
    values.push_back(value);
    row_map[row + 1]++;
}

void GridSystemSquare::finalize_matrix() {
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

void GridSystemSquare::initiate_matrix() {
    // Для квадратной области количество внутренних узлов (размер матрицы)
    int amount_of_nodes = (n - 1) * (m - 1); // Внутренние узлы (без границ)
    
    // Инициализируем структуры для хранения разреженной матрицы
    row_map.resize(amount_of_nodes + 1, 0);
    
    // Предварительная оценка количества ненулевых элементов (5 элементов на строку)
    values.reserve(amount_of_nodes * 5);
    entries.reserve(amount_of_nodes * 5);
    rhs_values.resize(amount_of_nodes);
    
    // Инициализируем векторы координат
    node_x_coords.resize(amount_of_nodes, 0.0);
    node_y_coords.resize(amount_of_nodes, 0.0);
    
    // Заполняем матрицу для квадратной области
    for (int y = 1; y < m; ++y) {
        for (int x = 1; x < n; ++x) {
            if (!is_boundary(x, y)) {
                int row = calculate_position_in_matrix(x, y);
                
                // Сохраняем координаты узла
                node_x_coords[row] = calculate_x(x);
                node_y_coords[row] = calculate_y(y);
                
                // Диагональный элемент (центральная точка пятиточечного шаблона)
                add_matrix_entry(row, row, A);
                
                // Соседние элементы, если они не на границе
                if (!is_left_boundary(x - 1, y)) {
                    // Если левый сосед - внутренний узел
                    int left_pos = calculate_position_in_matrix(x - 1, y);
                    add_matrix_entry(row, left_pos, x_k);
                }
                
                if (!is_right_boundary(x + 1, y)) {
                    // Если правый сосед - внутренний узел
                    int right_pos = calculate_position_in_matrix(x + 1, y);
                    add_matrix_entry(row, right_pos, x_k);
                }
                
                if (!is_top_boundary(x, y + 1)) {
                    // Если верхний сосед - внутренний узел
                    int top_pos = calculate_position_in_matrix(x, y + 1);
                    add_matrix_entry(row, top_pos, y_k);
                }
                
                if (!is_bottom_boundary(x, y - 1)) {
                    // Если нижний сосед - внутренний узел
                    int bottom_pos = calculate_position_in_matrix(x, y - 1);
                    add_matrix_entry(row, bottom_pos, y_k);
                }
                
                // Вектор правой части (с учетом граничных условий)
                rhs_values[row] = calculate_value(x, y, x_k, y_k);
            }
        }
    }
    
    // Финализируем матрицу
    finalize_matrix();
}

// Реализация метода для получения граничных значений
double GridSystemSquare::boundary_value(double x, double y, int boundary_type)
{
    // Вызываем соответствующую функцию граничных условий
    switch (boundary_type) {
        case 1: // Левая граница (x=a)
            return mu1Ptr(x, y);
        case 2: // Правая граница (x=b)
            return mu2Ptr(x, y);
        case 3: // Нижняя граница (y=c)
            return mu3Ptr(x, y);
        case 4: // Верхняя граница (y=d)
            return mu4Ptr(x, y);
        default:
            return 0.0;
    }
}

// Конструктор с указанием функции правой части и точного решения
GridSystemSquare::GridSystemSquare(int m, int n, double a, double b, double c, double d,
                                 double (*f)(double, double), double (*sol)(double, double)) {
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
    this->x_step = (b - a) / n;
    this->y_step = (d - c) / m;
    
    // Коэффициенты для пятиточечной схемы
    A = -2 * (1 / (x_step * x_step) + 1 / (y_step * y_step));
    x_k = 1 / (x_step * x_step);
    y_k = 1 / (y_step * y_step);
    
    // Установка указателей на функции
    this->funcPtr = f;
    this->solPtr = sol; // Может быть nullptr
    
    // Если есть точное решение, используем его для граничных условий
    if (sol != nullptr) {
        this->mu1Ptr = sol;
        this->mu2Ptr = sol;
        this->mu3Ptr = sol;
        this->mu4Ptr = sol;
    } else {
        // Иначе устанавливаем новые граничные условия по умолчанию, если f не nullptr.
        // Это покрывает случай, когда пользователь передает f, но не sol и не mu.
        if (f != nullptr) { // Если f задана, а sol нет, используем mu_square по умолчанию
             this->mu1Ptr = mu1_square;
             this->mu2Ptr = mu2_square;
             this->mu3Ptr = mu3_square;
             this->mu4Ptr = mu4_square;
        } else { // Если и f не задана (например, базовый конструктор), то нужны какие-то стандартные mu.
            // В базовом конструкторе ниже мы устанавливаем funcPtr = custom_function_square и solPtr = nullptr,
            // поэтому этот блок else может быть не достигнут, если базовый конструктор правильно установит mu.
            // Для безопасности, можно установить их и здесь.
            this->mu1Ptr = mu1_square;
            this->mu2Ptr = mu2_square;
            this->mu3Ptr = mu3_square;
            this->mu4Ptr = mu4_square;
        }
    }
    
    // Сборка разреженной матрицы и вектора правой части
    initiate_matrix();
}

// Конструктор с указанием функции правой части и граничных условий
GridSystemSquare::GridSystemSquare(int m, int n, double a, double b, double c, double d,
                                 double (*f)(double, double),
                                 double (*mu1)(double, double), double (*mu2)(double, double),
                                 double (*mu3)(double, double), double (*mu4)(double, double),
                                 double (*exact_sol_func)(double, double)) { // Added exact_sol_func
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
    this->x_step = (b - a) / n;
    this->y_step = (d - c) / m;
    
    // Коэффициенты для пятиточечной схемы
    A = -2 * (1 / (x_step * x_step) + 1 / (y_step * y_step));
    x_k = 1 / (x_step * x_step);
    y_k = 1 / (y_step * y_step);
    
    // Установка указателей на функции
    this->funcPtr = f;
    this->solPtr = exact_sol_func;  // Use the passed exact_sol_func
    
    // Устанавливаем граничные условия
    this->mu1Ptr = mu1;
    this->mu2Ptr = mu2;
    this->mu3Ptr = mu3;
    this->mu4Ptr = mu4;
    
    // Сборка разреженной матрицы и вектора правой части
    initiate_matrix();
}

// Реализация get_true_solution_vector с проверкой на наличие точного решения
KokkosVector GridSystemSquare::get_true_solution_vector() {
    if (matrix.numRows() == 0) {
        throw std::runtime_error("Matrix not initialized, cannot determine size for true solution vector.");
    }
    
    if (solPtr == nullptr) {
        throw std::runtime_error("No true solution function provided.");
    }
    
    int amount_of_nodes = matrix.numRows();
    KokkosVector true_u("true_u", amount_of_nodes);
    auto u_host = Kokkos::create_mirror_view(true_u);

    // Используем сохраненные координаты для вычисления точного решения
    for (int i = 0; i < amount_of_nodes; ++i) {
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

GridSystemSquare::GridSystemSquare(int m, int n, double a, double b, double c, double d) {
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
    this->x_step = (b - a) / n;
    this->y_step = (d - c) / m;
    
    // Коэффициенты для пятиточечной схемы
    A = -2 * (1 / (x_step * x_step) + 1 / (y_step * y_step));
    x_k = 1 / (x_step * x_step);
    y_k = 1 / (y_step * y_step);
    
    // Установка указателей на функции по умолчанию для новой задачи
    this->funcPtr = custom_function_square; 
    this->solPtr = nullptr; // Для новой задачи точного решения нет
    
    // Устанавливаем граничные условия по умолчанию для новой задачи
    this->mu1Ptr = mu1_square;
    this->mu2Ptr = mu2_square;
    this->mu3Ptr = mu3_square;
    this->mu4Ptr = mu4_square;
    
    // Сборка разреженной матрицы и вектора правой части
    initiate_matrix();
}

GridSystemSquare::~GridSystemSquare() {
    // При необходимости освобождаем ресурсы Kokkos
    // Обычно Kokkos::finalize() вызывается в конце main()
}

GridSystemSquare::NodeCoordinates GridSystemSquare::get_node_coordinates(int solution_index) const {
    NodeCoordinates coords;
    coords.x = 0.0;
    coords.y = 0.0;
    
    // Проверяем корректность индекса
    if (solution_index < 0 || solution_index >= matrix.numRows()) {
        return coords; // Возвращаем нулевые координаты для некорректного индекса
    }
    
    // Для квадратной области координаты вычисляются проще
    // Восстанавливаем индексы x и y из линейного индекса
    int y = solution_index / (n - 1) + 1; // Прибавляем 1, так как индексация внутренних узлов с 1
    int x = solution_index % (n - 1) + 1;
    
    coords.x = calculate_x(x);
    coords.y = calculate_y(y);
    
    return coords;
}

std::ostream &operator<<(std::ostream &os, const GridSystemSquare &grid) {
    // Выводим информацию о матрице в сжатом виде
    os << "GridSystemSquare Matrix Information:" << std::endl;
    os << "  Dimensions: " << grid.n << "x" << grid.m << std::endl;
    os << "  Domain: [" << grid.a << ", " << grid.b << "] x [" << grid.c << ", " << grid.d << "]" << std::endl;
    os << "  Matrix size: " << grid.matrix.numRows() << " rows x " << grid.matrix.numCols() << " columns" << std::endl;
    os << "  Non-zero elements: " << grid.matrix.nnz() << std::endl;
    os << "  Sparsity: " << (1.0 - (double)grid.matrix.nnz() / (grid.matrix.numRows() * grid.matrix.numCols())) * 100.0 << "%" << std::endl;
    
    return os;
}