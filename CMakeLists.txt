cmake_minimum_required(VERSION 3.16)

project(DirichletSolverProject2 VERSION 1.0.0 LANGUAGES CXX)

# Установка глобальных параметров CMake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Определение флага для сборки Qt GUI
option(BUILD_QT_GUI "Build Qt GUI application" ON)

# Добавляем библиотеку решателя
add_subdirectory(solver)

# Проверяем, нужно ли собирать Qt GUI
if(BUILD_QT_GUI)
    # Проверяем наличие Qt
    #set(Qt6_DIR "/usr/lib/x86_64-linux-gnu/cmake/Qt6")
    # /usr/lib/qt6/bin/qmake
    find_package(Qt6 COMPONENTS Core Widgets Gui Charts QUIET)
    
    if(Qt6_FOUND)
        message(STATUS "Qt6 found, building GUI application")
        add_subdirectory(qt_gui)
    else()
        message(WARNING "Qt6 not found, GUI application will not be built")
    endif()
endif()

# Вывод сообщения для пользователя
message(STATUS "DirichletSolverProject2 configuration completed")
message(STATUS "  - Solver library will be built")
if(BUILD_QT_GUI AND Qt6_FOUND)
    message(STATUS "  - Qt GUI application will be built")
else()
    message(STATUS "  - Qt GUI application will NOT be built")
endif()
