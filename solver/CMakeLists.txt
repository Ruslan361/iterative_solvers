cmake_minimum_required(VERSION 3.16)

# Название проекта
project(DirichletSolver)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Флаг, указывающий, является ли проект основным или подключаемым модулем
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(DIRICHLET_SOLVER_MAIN_PROJECT ON)
else()
    set(DIRICHLET_SOLVER_MAIN_PROJECT OFF)
endif()

# Включаем поддержку OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Сначала пытаемся найти Kokkos в системе
find_package(Kokkos QUIET)

# Если Kokkos не найден, скачиваем и собираем его
if(NOT Kokkos_FOUND)
  message(STATUS "Kokkos not found. Installing from source...")
  
  include(FetchContent)
  
  FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 4.0.01  # Используем стабильную версию
  )
  
  # Настраиваем опции сборки Kokkos, включая подмодуль KokkosSparse
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend")
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend")
  set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Disable CUDA backend")
  set(Kokkos_ENABLE_SPARSE ON CACHE BOOL "Enable KokkosSparse module")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries")
  
  FetchContent_MakeAvailable(kokkos)
  
  message(STATUS "Kokkos has been installed successfully")
endif()

# Добавление исходных файлов для библиотеки
set(DIRICHLET_SOLVER_SOURCES
    grid_system.cpp
    grid_system_square.cpp
    msg_solver.cpp
    dirichlet_solver.cpp
    dirichlet_solver_square.cpp
    default_functions.cpp
)

# Добавление заголовочных файлов
set(DIRICHLET_SOLVER_HEADERS
    grid_system.h
    grid_system_square.h
    solver.hpp
    msg_solver.hpp
    dirichlet_solver.hpp
    dirichlet_solver_square.hpp

)

# Создание библиотеки вместо исполняемого файла
add_library(dirichlet_solver ${DIRICHLET_SOLVER_SOURCES} ${DIRICHLET_SOLVER_HEADERS})

# Установка флагов компиляции
target_compile_options(dirichlet_solver PRIVATE -Wall -Wextra -Wpedantic -O2)

# Настройка экспорта заголовочных файлов
target_include_directories(dirichlet_solver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${Kokkos_INCLUDE_DIRS}
)

# Подключение Kokkos (работает и для установленного, и для скачанного)
target_link_libraries(dirichlet_solver PUBLIC Kokkos::kokkos)

# Подключение необходимых библиотек
target_link_libraries(dirichlet_solver PRIVATE stdc++fs)

# Подключение KokkosKernels: сперва пытаемся найти пакет
find_package(KokkosKernels QUIET)
if(NOT KokkosKernels_FOUND)
  message(STATUS "KokkosKernels not found. Installing from source...")
  include(FetchContent)
  FetchContent_Declare(
    kokkoskernels
    GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git
    GIT_TAG 4.0.01  # Укажите нужную стабильную версию
  )
  
  # Опции для KokkosKernels
  set(KokkosKernels_ENABLE_EXAMPLES OFF CACHE BOOL "")
  set(KokkosKernels_ENABLE_TESTS OFF CACHE BOOL "")
  
  # Для прямой передачи поставщика Kokkos
  set(KokkosKernels_ADD_DEFAULT_ETI ON CACHE BOOL "")  
  
  FetchContent_MakeAvailable(kokkoskernels)
  message(STATUS "KokkosKernels has been installed successfully")
endif()

# Условная линковка в зависимости от способа установки
if(KokkosKernels_FOUND)
  # Если KokkosKernels был найден через find_package
  target_link_libraries(dirichlet_solver PUBLIC KokkosKernels::kokkoskernels)
else()
  # Если KokkosKernels был установлен через FetchContent
  target_link_libraries(dirichlet_solver PUBLIC kokkoskernels)
endif()

# Вывод информации для отладки
message(STATUS "Kokkos include directories: ${Kokkos_INCLUDE_DIRS}")
message(STATUS "Kokkos libraries: ${Kokkos_LIBRARIES}")
message(STATUS "Kokkos_FOUND: ${Kokkos_FOUND}")
message(STATUS "KokkosKernels_FOUND: ${KokkosKernels_FOUND}")

# Если это основной проект, создаем исполняемый файл
if(DIRICHLET_SOLVER_MAIN_PROJECT)
  # Добавляем исполняемый файл консольной версии
  add_executable(solver main.cpp)
  target_link_libraries(solver PRIVATE dirichlet_solver)
endif()