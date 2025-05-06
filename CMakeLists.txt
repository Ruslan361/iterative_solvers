cmake_minimum_required(VERSION 3.16)

# Название проекта
project(GridSystemSolver)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
#    GIT_TAG 3.4.01  # Укажите нужную стабильную версию
# Добавление исходных файлов
set(SOURCES
    main.cpp
    grid_system.cpp
    msg_solver.cpp
)

# Добавление заголовочных файлов
set(HEADERS
    grid_system.h
    solver.hpp
    msg_solver.hpp
)

# Создание исполняемого файла
add_executable(solver ${SOURCES} ${HEADERS})

# Установка флагов компиляции
target_compile_options(solver PRIVATE -Wall -Wextra -Wpedantic -O2)

# Явно включаем директории с заголовочными файлами Kokkos
target_include_directories(solver PRIVATE ${Kokkos_INCLUDE_DIRS})

# Подключение Kokkos (работает и для установленного, и для скачанного)
target_link_libraries(solver PRIVATE Kokkos::kokkos)

# Подключение необходимых библиотек
target_link_libraries(solver PRIVATE stdc++fs)

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
  target_link_libraries(solver PRIVATE KokkosKernels::kokkoskernels)
else()
  # Если KokkosKernels был установлен через FetchContent
  # Проверим доступные цели
  get_property(all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  message(STATUS "Available targets: ${all_targets}")
  
  # Используем правильное имя цели для FetchContent версии
  target_link_libraries(solver PRIVATE kokkoskernels)
endif()

# Вывод информации для отладки
message(STATUS "Kokkos include directories: ${Kokkos_INCLUDE_DIRS}")
message(STATUS "Kokkos libraries: ${Kokkos_LIBRARIES}")
message(STATUS "Kokkos_FOUND: ${Kokkos_FOUND}")
message(STATUS "KokkosKernels_FOUND: ${KokkosKernels_FOUND}")