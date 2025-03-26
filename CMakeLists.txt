cmake_minimum_required(VERSION 3.10)

# Название проекта
project(GridSystemSolver)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Добавление исходных файлов
set(SOURCES
    main.cpp
    grid_system.cpp
    zeidel_solver.cpp
    zeidel_solver.hpp
    msg_solver.cpp
    msg_solver.hpp
)

# Добавление заголовочных файлов
set(HEADERS
    grid_system.h
)

# Создание исполняемого файла
add_executable(solver ${SOURCES} ${HEADERS})

# Установка флагов компиляции
target_compile_options(solver PRIVATE -Wall -Wextra -Wpedantic -O2)

# Подключение необходимых библиотек
target_link_libraries(solver PRIVATE stdc++fs)