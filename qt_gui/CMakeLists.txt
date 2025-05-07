cmake_minimum_required(VERSION 3.16)

project(DirichletSolverGUI VERSION 1.0.0 LANGUAGES CXX)

# Установка глобальных параметров CMake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Настройки для Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Указываем путь к Qt6 
set(CMAKE_PREFIX_PATH "/usr/lib/qt6")
# Можно также указать конкретно путь к директории с файлами конфигурации Qt6
# set(Qt6_DIR "/usr/lib/qt6/lib/cmake/Qt6")

# Находим Qt пакет с дополнительным модулем DataVisualization для 3D графиков
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui Charts DataVisualization)

# Добавляем исходные файлы GUI
set(GUI_SOURCES
    src/main.cpp
    src/mainwindow.cpp
)

# Заголовочные файлы GUI
set(GUI_HEADERS
    src/mainwindow.h
)

# UI файлы для Qt Designer
set(GUI_UI
    src/mainwindow.ui
)

# Ресурсы приложения
set(GUI_RESOURCES
    resources/resources.qrc
)

# Создаем исполняемый файл GUI
add_executable(dirichlet_gui 
    ${GUI_SOURCES} 
    ${GUI_HEADERS} 
    ${GUI_UI}
    ${GUI_RESOURCES}
)

# Подключаем необходимые модули Qt и библиотеку решателя
target_link_libraries(dirichlet_gui PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Charts
    Qt6::DataVisualization
    dirichlet_solver
)

# Если используем Windows, добавляем флаг для создания оконного приложения
if(WIN32)
    target_link_options(dirichlet_gui PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
endif()