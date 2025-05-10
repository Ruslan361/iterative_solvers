#include <cmath>

// Функция правой части уравнения Пуассона для G-образной области (старая)
// Used as a default function in GridSystemSquare (originally)
double function2(double x, double y) {
    return 4 * (x * x + y * y) * std::exp(x * x - y * y);
}

// Точное решение уравнения для G-образной области (старое)
// Used as a default function in GridSystemSquare (originally)
double solution2(double x, double y) {
    return std::exp(x*x-y*y);
}

// Функция правой части уравнения Пуассона для квадратной области с тем же решением, что и в G-образной области
double function2_square(double x, double y) {
    return -4 * (x * x + y * y) * std::exp(x * x - y * y);  // Исправлен знак, должен быть минус
}

// Точное решение для квадратной области (такое же, как и для G-образной области)
double solution2_square(double x, double y) {
    return std::exp(x*x-y*y);
}

// Граничные условия для квадратной области с тем же решением, что и в G-образной области
// mu1: левая граница (x=a_bound)
double mu1_square_solution2(double x, double y) {
    // x фиксирован на a_bound (например, x=0 или x=1)
    // используем формулу точного решения: exp(x^2-y^2)
    return std::exp(x*x-y*y);
}

// mu2: правая граница (x=b_bound)
double mu2_square_solution2(double x, double y) {
    // x фиксирован на b_bound (например, x=1 или x=2)
    // используем формулу точного решения: exp(x^2-y^2)
    return std::exp(x*x-y*y);
}

// mu3: нижняя граница (y=c_bound)
double mu3_square_solution2(double x, double y) {
    // y фиксирован на c_bound (например, y=0 или y=1)
    // используем формулу точного решения: exp(x^2-y^2)
    return std::exp(x*x-y*y);
}

// mu4: верхняя граница (y=d_bound)
double mu4_square_solution2(double x, double y) {
    // y фиксирован на d_bound (например, y=1 или y=2)
    // используем формулу точного решения: exp(x^2-y^2)
    return std::exp(x*x-y*y);
}

// Новая функция правой части для квадратной области
double custom_function_square(double x, double y) {
    if (y == 0) return 0; // Avoid division by zero, though domain y in [1,2]
    return std::atan(x / y);
}

// Граничные условия для квадратной области [1,2]x[1,2]
// mu1: левая граница (x=1)
double mu1_square(double x, double y) {
    (void)x; // x is fixed at a_bound (1.0)
    (void)y; // y varies
    return 0.0;
}

// mu2: правая граница (x=2)
double mu2_square(double x, double y) {
    (void)x; // x is fixed at b_bound (2.0)
    (void)y; // y varies
    return 0.0;
}

// mu3: нижняя граница (y=1)
double mu3_square(double x, double y) {
    (void)y; // y is fixed at c_bound (1.0)
    // x varies
    return std::sin(M_PI * x) * std::sin(M_PI * x);
}

// mu4: верхняя граница (y=2)
double mu4_square(double x, double y) {
    (void)y; // y is fixed at d_bound (2.0)
    // x varies
    return std::cosh((x - 1.0) * (x - 2.0)) - 1.0;
}
