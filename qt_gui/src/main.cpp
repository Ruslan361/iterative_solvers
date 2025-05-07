#include "mainwindow.h"

#include <QApplication>
#include <QFontDatabase>

int main(int argc, char *argv[])
{
    // Инициализация QApplication
    QApplication app(argc, argv);
    
    // Загрузка шрифтов (опционально)
    QFontDatabase::addApplicationFont(":/fonts/OpenSans-Regular.ttf");
    
    // Устанавливаем название приложения
    app.setApplicationName("Решатель задачи Дирихле");
    app.setOrganizationName("Университет");
    
    // Создаем и показываем главное окно
    MainWindow mainWindow;
    mainWindow.setWindowTitle("Решатель задачи Дирихле");
    mainWindow.show();
    
    // Запускаем главный цикл приложения
    return app.exec();
}