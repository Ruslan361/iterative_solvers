#include "table_tab_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QStandardPaths>

TableTabWidget::TableTabWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

void TableTabWidget::setupUI()
{
    // Основной layout для вкладки
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Группа для настроек таблицы
    QGroupBox *settingsGroup = new QGroupBox("Настройки отображения данных");
    QHBoxLayout *settingsLayout = new QHBoxLayout(settingsGroup);
    
    // Элементы управления для настройки отображения таблицы
    QLabel *skipFactorLabel = new QLabel("Коэффициент прореживания:");
    skipFactorSpinBox = new QSpinBox();
    skipFactorSpinBox->setRange(1, 100);
    skipFactorSpinBox->setValue(1);
    skipFactorSpinBox->setToolTip("1 означает отображение всех данных, большие значения уменьшают количество отображаемых строк");
    
    showTableButton = new QPushButton("Показать таблицу");
    clearTableButton = new QPushButton("Очистить таблицу");
    exportCSVButton = new QPushButton("Экспорт CSV");
    
    // Изначально кнопки неактивны
    showTableButton->setEnabled(false);
    clearTableButton->setEnabled(false);
    exportCSVButton->setEnabled(false);
    
    // Информация о таблице
    tableInfoLabel = new QLabel("Таблица не содержит данных");
    
    settingsLayout->addWidget(skipFactorLabel);
    settingsLayout->addWidget(skipFactorSpinBox);
    settingsLayout->addWidget(showTableButton);
    settingsLayout->addWidget(clearTableButton);
    settingsLayout->addWidget(exportCSVButton);
    settingsLayout->addWidget(tableInfoLabel);
    
    // Таблица для отображения данных
    dataTable = new QTableWidget();
    dataTable->setColumnCount(4); // Для x, y, численное решение, точное решение
    dataTable->setHorizontalHeaderLabels(QStringList() << "X" << "Y" << "Численное решение" << "Точное решение");
    dataTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    dataTable->setEditTriggers(QAbstractItemView::NoEditTriggers); // Запрещаем редактирование
    
    // Соединяем сигналы и слоты
    connect(showTableButton, &QPushButton::clicked, this, &TableTabWidget::onShowTableButtonClicked);
    connect(clearTableButton, &QPushButton::clicked, this, &TableTabWidget::onClearTableButtonClicked);
    connect(exportCSVButton, &QPushButton::clicked, this, &TableTabWidget::onExportButtonClicked);
    
    // Добавляем элементы в основной layout
    mainLayout->addWidget(settingsGroup);
    mainLayout->addWidget(dataTable);
    
    setLayout(mainLayout);
}

void TableTabWidget::setCSVData(const QString& csvData)
{
    currentCSVData = csvData;
    
    // Активируем кнопки, если есть данные
    bool hasData = !csvData.isEmpty();
    showTableButton->setEnabled(hasData);
    clearTableButton->setEnabled(hasData);
    exportCSVButton->setEnabled(hasData);
    
    if (hasData) {
        tableInfoLabel->setText("Данные готовы к отображению");
        
        // Подсчитываем количество строк в CSV
        int rows = csvData.count('\n');
        if (csvData.endsWith('\n')) {
            rows--; // Убираем пустую строку, если CSV заканчивается переносом строки
        }
        
        tableInfoLabel->setText(QString("Доступно %1 строк данных").arg(rows));
    } else {
        tableInfoLabel->setText("Таблица не содержит данных");
        
        // Очищаем таблицу
        dataTable->setRowCount(0);
    }
}

void TableTabWidget::populateTableWithData(const QString& csvData)
{
    if (csvData.isEmpty()) {
        return;
    }
    
    // Очищаем таблицу
    dataTable->setRowCount(0);
    
    // Разбиваем CSV на строки
    QStringList lines = csvData.split('\n', Qt::SkipEmptyParts);
    
    // Используем коэффициент прореживания для уменьшения количества данных
    int skipFactor = skipFactorSpinBox->value();
    
    // Проверяем первую строку, чтобы определить формат (количество столбцов)
    if (!lines.isEmpty()) {
        QStringList firstRow = lines.first().split(',');
        dataTable->setColumnCount(firstRow.size());
        
        // Устанавливаем заголовки в зависимости от количества столбцов
        QStringList headers;
        if (firstRow.size() == 3) {
            headers << "X" << "Y" << "Численное решение";
        } else if (firstRow.size() == 4) {
            headers << "X" << "Y" << "Численное решение" << "Точное решение";
        } else if (firstRow.size() == 5) {
            headers << "X" << "Y" << "Численное решение" << "Точное решение" << "Ошибка";
        } else {
            // Генерируем заголовки в общем случае
            for (int i = 0; i < firstRow.size(); ++i) {
                headers << QString("Столбец %1").arg(i + 1);
            }
        }
        dataTable->setHorizontalHeaderLabels(headers);
    }
    
    // Определяем количество строк после прореживания
    int rowCount = (lines.size() + skipFactor - 1) / skipFactor; // Округленное деление вверх
    dataTable->setRowCount(rowCount);
    
    // Заполняем таблицу данными
    int tableRow = 0;
    for (int i = 0; i < lines.size(); i += skipFactor) {
        if (tableRow >= rowCount) break; // Защита от переполнения
        
        QStringList columns = lines[i].split(',');
        for (int j = 0; j < columns.size(); ++j) {
            if (j < dataTable->columnCount()) {
                dataTable->setItem(tableRow, j, new QTableWidgetItem(columns[j]));
            }
        }
        tableRow++;
    }
    
    // Обновляем информацию о таблице
    tableInfoLabel->setText(QString("Отображено %1 из %2 строк с коэффициентом прореживания %3")
                            .arg(rowCount).arg(lines.size()).arg(skipFactor));
}

void TableTabWidget::onShowTableButtonClicked()
{
    populateTableWithData(currentCSVData);
}

void TableTabWidget::onClearTableButtonClicked()
{
    dataTable->setRowCount(0);
    tableInfoLabel->setText("Таблица очищена");
}

void TableTabWidget::onExportButtonClicked()
{
    emit exportCSVRequested(skipFactorSpinBox->value());
}