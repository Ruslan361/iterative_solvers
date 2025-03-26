#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <memory>
#include "grid_system.h"
#include "solver.hpp"
#include "zeidel_solver.hpp" 
#include "msg_solver.hpp"
#include <locale.h>

using namespace std;

/**
 *  Вычисляет невязку системы уравнений.
 * Arg: a Матрица коэффициентов системы.
 * Arg: v Вектор решений.
 * Arg: b Вектор свободных членов.
 * Returns: Вектор невязки.
 */
vector<double> compute_residual(const vector<vector<double>>& a, const vector<double>& v, const vector<double>& b)
{
    vector<double> residual(b.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (size_t j = 0; j < a[i].size(); ++j)
        {
            residual[i] += a[i][j] * v[j];
        }
        residual[i] -= b[i];
    }
    return residual;
}

/**
 *  Вычисляет координату y по индексу сеточной точки.
 * Arg: j Индекс точки по оси y.
 * Arg: m Количество внутренних узлов по оси y.
 * Arg: c_bound Левая граница по оси y.
 * Arg: d_bound Правая граница по оси y.
 * Returns: Координата y.
 */
double y(int j, int m, double c_bound, double d_bound)
{
    double k = (d_bound - c_bound) / (m + 1);
    return c_bound + j * k;
}

/**
 * Вычисляет координату x по индексу сеточной точки.
 * Arg: i Индекс точки по оси x.
 * Arg: n Количество внутренних узлов по оси x.
 * Arg: a_bound Левая граница по оси x.
 * Arg: b_bound Правая граница по оси x.
 * Returns: Координата x.
 */
double x(int i, int n, double a_bound, double b_bound)
{
    double h = (b_bound - a_bound) / (n + 1);
    return a_bound + i * h;
}

/**
 *  Истинное решение функции u(x, y).
 * Arg: x_val Координата x.
 * Arg: y_val Координата y.
 * Returns: Значение функции u в точке (x, y).
 */
double u(double x_val, double y_val)
{
    return pow(x_val, 3) + pow(y_val, 2) + 3;
}

/**
 * Вычисляет ошибку решения.
 * Arg: v Вектор решения.
 * Arg: n_internal Количество внутренних узлов по оси x.
 * Arg: m_internal Количество внутренних узлов по оси y.
 * Arg: a_bound Левая граница области по x.
 * Arg: b_bound Правая граница области по x.
 * Arg: c_bound Нижняя граница области по y.
 * Arg: d_bound Верхняя граница области по y.
 * Returns: Вектор ошибки.
 */
vector<double> compute_error(const vector<double>& v, int n_internal, int m_internal, double a_bound, double b_bound, double c_bound, double d_bound)
{
    vector<double> error(v.size(), 0.0);
    int h = (b_bound - a_bound) / (n_internal + 1);
    int k = (d_bound - c_bound) / (m_internal + 1);
    for (size_t i = 0; i < v.size(); ++i)
    {   
        int row = i / n_internal + 1;
        int col = i % n_internal + 1;
        error[i] = fabs(v[i] - u(x(col, n_internal, a_bound, b_bound), y(row, m_internal, c_bound, d_bound)));
    }
    return error;
}

/**
 *  Выводит вектор невязки и её нормы.
 * Arg: residual Вектор невязки.
 */
void print_residual(const vector<double>& residual)
{
    cout << "\nНевязка (r):\n";
    for (double r : residual)
    {
        cout << setw(15) << std::uppercase << std::scientific << r << "\n";
    }
    double maxResidual = *max_element(residual.begin(), residual.end(), [](double a, double b) {return fabs(a) < fabs(b);});
    cout << "Максимальная невязка: " << std::uppercase << std::scientific << maxResidual << "\n";

    double evk_norm = 0;
    for (double r : residual)
    {
        evk_norm += r * r;
    }
    evk_norm = sqrt(evk_norm);
    cout << "Евклидова норма невязки: " << std::uppercase << std::scientific << evk_norm << "\n";
}

/**
 * Выводит вектор ошибки и её нормы.
 * Arg: error Вектор ошибки.
 */
void print_error(const vector<double>& error)
{
    cout << "\nВектор погрешности:\n";
    for (double e : error)
    {
        cout << e << "\n";
    }
    double maxError = *max_element(error.begin(), error.end(), [](double a, double b) {return fabs(a) < fabs(b);});
    cout << "Максимальная погрешность: " << std::uppercase << std::scientific << maxError << "\n";
}

void print_separator() {
    cout << "\n_________________________________________________\n\n";
}

// Остальной код main.cpp остается без изменений
int main() {
    setlocale(LC_ALL, "RU");

    GridSystem grid(6, 6, 1, 2, 1, 2);
    std::cout << grid << std::endl;

    // Получаем матрицу и вектор правой части из объекта grid
    auto matrix = grid.get_matrix();
    auto rhs = grid.get_rhs();

    // Параметры для методов
    double eps = 1e-6;    // Точность
    int maxIterations = 100000; // Максимальное количество итераций

    // Выбор метода решения
    cout << "Выберите метод решения системы:\n";
    cout << "1. Метод Зейделя\n";
    cout << "2. Метод серединных градиентов\n";
    cout << "Ваш выбор: ";
    
    int choice;
    cin >> choice;
    
    // Используем умный указатель для автоматического управления памятью
    unique_ptr<Solver> solver;
    
    // В зависимости от выбора пользователя создаем соответствующий решатель
    switch (choice) {
        case 1:
            solver = make_unique<ZeidelSolver>(matrix, rhs, eps, maxIterations);
            break;
        case 2:
            solver = make_unique<MSGSolver>(matrix, rhs, eps, maxIterations);
            break;
        default:
            cout << "Неверный выбор. Используется метод Зейделя по умолчанию.\n";
            solver = make_unique<ZeidelSolver>(matrix, rhs, eps, maxIterations);
    }
    
    // Решаем систему выбранным методом
    vector<double> solution = solver->solve();
    
    // Вычисляем и выводим невязку
    vector<double> residual = compute_residual(matrix, solution, rhs);
    print_residual(residual);

    // Вычисляем и выводим ошибку
    int n_internal = 6; // Количество внутренних узлов по оси x
    int m_internal = 6; // Количество внутренних узлов по оси y
    double a_bound = 1.0; // Левая граница по x
    double b_bound = 2.0; // Правая граница по x
    double c_bound = 1.0; // Нижняя граница по y
    double d_bound = 2.0; // Верхняя граница по y
    vector<double> error = compute_error(solution, n_internal, m_internal, a_bound, b_bound, c_bound, d_bound);
    print_error(error);

    // Выводим результат
    cout << "\nРезультат (вектор решения " << solver->getName() << "):\n";
    print_separator();
    for (int i = 0; i < solution.size(); ++i) {
        cout << "v[" << i << "] = " << fixed << setprecision(6) << setw(15) << solution[i] << "\n";
    }
    print_separator();

    return 0;
}