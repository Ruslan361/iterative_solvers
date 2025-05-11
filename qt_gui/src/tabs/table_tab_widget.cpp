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
    QVBoxLayout *settingsMainLayout = new QVBoxLayout(settingsGroup);
    
    // Верхний ряд настроек с ComboBox для выбора типа данных
    QHBoxLayout *dataTypeLayout = new QHBoxLayout();
    QLabel *dataTypeLabel = new QLabel("Тип данных:");
    dataTypeComboBox = new QComboBox();
    dataTypeComboBox->addItem("Численное решение");
    dataTypeComboBox->addItem("Точное решение");
    dataTypeComboBox->addItem("Ошибка");
    dataTypeComboBox->addItem("Решение на уточненной сетке");
    
    dataTypeLayout->addWidget(dataTypeLabel);
    dataTypeLayout->addWidget(dataTypeComboBox);
    dataTypeLayout->addStretch();
    
    // Второй ряд настроек для прореживания и кнопок
    QHBoxLayout *controlsLayout = new QHBoxLayout();
    
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
    
    controlsLayout->addWidget(skipFactorLabel);
    controlsLayout->addWidget(skipFactorSpinBox);
    controlsLayout->addWidget(showTableButton);
    controlsLayout->addWidget(clearTableButton);
    controlsLayout->addWidget(exportCSVButton);
    
    // Информация о таблице
    tableInfoLabel = new QLabel("Таблица не содержит данных");
    
    // Добавляем все ряды в основной layout группы настроек
    settingsMainLayout->addLayout(dataTypeLayout);
    settingsMainLayout->addLayout(controlsLayout);
    settingsMainLayout->addWidget(tableInfoLabel);
    
    // Таблица для отображения данных
    dataTable = new QTableWidget();
    dataTable->setEditTriggers(QAbstractItemView::NoEditTriggers); // Запрещаем редактирование
    
    // Соединяем сигналы и слоты
    connect(showTableButton, &QPushButton::clicked, this, &TableTabWidget::onShowTableButtonClicked);
    connect(clearTableButton, &QPushButton::clicked, this, &TableTabWidget::onClearTableButtonClicked);
    connect(exportCSVButton, &QPushButton::clicked, this, &TableTabWidget::onExportButtonClicked);
    connect(dataTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &TableTabWidget::onDataTypeChanged);
    
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
        // Подсчитываем количество строк в CSV
        int rows = csvData.count('\n');
        if (csvData.endsWith('\n')) {
            rows--; // Убираем пустую строку, если CSV заканчивается переносом строки
        }
        
        tableInfoLabel->setText(QString("Доступно %1 строк данных").arg(rows));
        
        // Анализируем данные, чтобы узнать, какие типы данных доступны
        bool hasExactSolution = csvData.contains("Точное решение");
        bool hasError = csvData.contains("Ошибка");
        bool hasRefinedGrid = csvData.contains("Уточненная сетка");
        
        // Активируем/деактивируем пункты ComboBox в зависимости от наличия данных
        dataTypeComboBox->setItemData(EXACT_SOLUTION, hasExactSolution ? QVariant(QVariant::Invalid) : QVariant(0), Qt::UserRole - 1);
        dataTypeComboBox->setItemData(ERROR, hasError ? QVariant(QVariant::Invalid) : QVariant(0), Qt::UserRole - 1);
        dataTypeComboBox->setItemData(REFINED_GRID, hasRefinedGrid ? QVariant(QVariant::Invalid) : QVariant(0), Qt::UserRole - 1);
    } else {
        tableInfoLabel->setText("Таблица не содержит данных");
        
        // Очищаем таблицу
        dataTable->setRowCount(0);
    }
}

void TableTabWidget::onDataTypeChanged(int index)
{
    // При смене типа данных, обновляем таблицу, если данные уже загружены
    if (!currentCSVData.isEmpty()) {
        populateTableWithData(currentCSVData);
    }
}

void TableTabWidget::populateTableWithData(const QString& csvData)
{
    if (csvData.isEmpty()) {
        return;
    }
    
    // Очищаем таблицу
    dataTable->setRowCount(0);
    dataTable->setColumnCount(0);
    
    // Разбиваем CSV на строки
    QStringList lines = csvData.split('\n', Qt::SkipEmptyParts);
    
    // Используем коэффициент прореживания для уменьшения количества данных
    int skipFactor = skipFactorSpinBox->value();
    
    // Определяем какой тип данных показывать
    int dataTypeIndex = dataTypeComboBox->currentIndex();
    QString dataTypeString;
    switch(dataTypeIndex) {
        case NUMERICAL_SOLUTION:
            dataTypeString = "Численное решение";
            break;
        case EXACT_SOLUTION:
            dataTypeString = "Точное решение";
            break;
        case ERROR:
            dataTypeString = "Ошибка";
            break;
        case REFINED_GRID:
            dataTypeString = "Решение на уточненной сетке";
            break;
    }
    
    // Разбираем данные CSV и находим нужные секции
    QStringList dataSection;
    bool inTargetSection = false;
    
    for (int i = 0; i < lines.size(); i++) {
        const QString& line = lines[i];
        if (line.contains(dataTypeString, Qt::CaseInsensitive)) {
            inTargetSection = true;
            continue; // Пропускаем строку с заголовком секции
        } else if (line.isEmpty() || (inTargetSection && (line.contains("решение", Qt::CaseInsensitive) || 
                   line.contains("ошибка", Qt::CaseInsensitive) || line.contains("сетка", Qt::CaseInsensitive)))) {
            inTargetSection = false;
        }
        
        if (inTargetSection) {
            dataSection.append(line);
        }
    }
    
    // Если нужная секция не найдена, выводим предупреждение
    if (dataSection.isEmpty()) {
        tableInfoLabel->setText(QString("Данные типа '%1' не найдены").arg(dataTypeString));
        return;
    }
    
    // Анализируем первую строку, чтобы определить количество X координат
    if (!dataSection.isEmpty()) {
        QStringList firstRow = dataSection[0].split(',', Qt::SkipEmptyParts);
        
        // Создаем формат таблицы: первая колонка для Y-координат, остальные для X-координат
        int numXCoords = firstRow.size() - 1; // Первый элемент - подпись строки или Y-координата
        
        // Создаем заголовки столбцов - первый пустой, остальные - X-координаты
        QStringList headers;
        headers << "yi/xj";
        
        for (int i = 1; i < firstRow.size(); i += skipFactor) {
            if (i < firstRow.size()) {
                headers << firstRow[i];
            }
        }
        
        // Устанавливаем количество столбцов и заголовки
        dataTable->setColumnCount(headers.size());
        dataTable->setHorizontalHeaderLabels(headers);
        
        // Фильтруем строки с данными по коэффициенту прореживания
        QStringList filteredDataRows;
        for (int i = 1; i < dataSection.size(); i += skipFactor) {
            if (i < dataSection.size()) {
                filteredDataRows << dataSection[i];
            }
        }
        
        // Устанавливаем количество строк
        dataTable->setRowCount(filteredDataRows.size());
        
        // Заполняем таблицу данными
        for (int row = 0; row < filteredDataRows.size(); row++) {
            QStringList columns = filteredDataRows[row].split(',');
            
            // Вертикальные заголовки - это первый элемент каждой строки (Y-координата)
            if (!columns.isEmpty()) {
                dataTable->setVerticalHeaderItem(row, new QTableWidgetItem(columns[0]));
            }
            
            // Заполняем остальные ячейки (значения для каждой X-координаты)
            int colIndex = 0;
            for (int col = 0; col < columns.size(); col++) {
                if (col == 0) {
                    // Первую колонку (Y-координату) помещаем в первую ячейку строки
                    dataTable->setItem(row, colIndex++, new QTableWidgetItem(columns[col]));
                } else if ((col - 1) % skipFactor == 0) { // Применяем прореживание к X координатам
                    if (colIndex < dataTable->columnCount()) {
                        dataTable->setItem(row, colIndex++, new QTableWidgetItem(columns[col]));
                    }
                }
            }
        }
    }
    
    // Подгоняем ширину столбцов под содержимое
    dataTable->resizeColumnsToContents();
    
    // Обновляем информацию о таблице
    tableInfoLabel->setText(QString("Тип данных: %1, отображено %2 строк с коэффициентом прореживания %3")
                            .arg(dataTypeString)
                            .arg(dataTable->rowCount())
                            .arg(skipFactor));
}

void TableTabWidget::onShowTableButtonClicked()
{
    populateTableWithData(currentCSVData);
}

void TableTabWidget::onClearTableButtonClicked()
{
    dataTable->setRowCount(0);
    dataTable->setColumnCount(0);
    tableInfoLabel->setText("Таблица очищена");
}

void TableTabWidget::onExportButtonClicked()
{
    emit exportCSVRequested(skipFactorSpinBox->value());
}