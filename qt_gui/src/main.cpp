#include <QApplication>
#include <QSurfaceFormat>
#include "mainwindow_new.h"

int main(int argc, char *argv[])
{
    // Устанавливаем атрибуты для OpenGL
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);
    
    QApplication a(argc, argv);
    
    // Создаем и показываем новое главное окно
    MainWindow w;
    w.show();
    
    return a.exec();
}