#include "table_tab_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QStandardPaths>
#include <set>

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
    dataTypeComboBox->addItem("Разница между решениями"); // Новый пункт
    dataTypeComboBox->setEnabled(true);
    
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
    
    // Активируем кнопки и ComboBox, если есть данные
    bool hasData = !csvData.isEmpty();
    showTableButton->setEnabled(hasData);
    clearTableButton->setEnabled(hasData);
    exportCSVButton->setEnabled(hasData);
    dataTypeComboBox->setEnabled(true); // Явно включаем ComboBox
    
    if (hasData) {
        // Подсчитываем количество строк в CSV
        int rows = csvData.count('\n');
        if (csvData.endsWith('\n')) {
            rows--; // Убираем пустую строку, если CSV заканчивается переносом строки
        }
        
        tableInfoLabel->setText(QString("Доступно %1 строк данных").arg(rows));
        
        // Анализируем данные только для информации
        bool hasNumericalSolution = true; 
        bool hasExactSolution = csvData.contains("Точное решение", Qt::CaseInsensitive);
        bool hasError = csvData.contains("Ошибка", Qt::CaseInsensitive) || csvData.contains("Error", Qt::CaseInsensitive);
        bool hasRefinedGrid = csvData.contains("Уточненная сетка", Qt::CaseInsensitive) || 
                              csvData.contains("Refined grid", Qt::CaseInsensitive) ||
                              csvData.contains("v2(2N)", Qt::CaseInsensitive);
        
        // Больше не отключаем пункты в ComboBox
    } else {
        tableInfoLabel->setText("Таблица не содержит данных");
        
        // Очищаем таблицу
        dataTable->setRowCount(0);
        dataTable->setColumnCount(0);
    }
}

void TableTabWidget::setResultsData(
    const std::vector<double>& solution,
    const std::vector<double>& true_solution,
    const std::vector<double>& error,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords,
    bool is_square_grid)
{
    // Сохраняем данные
    m_solution = solution;
    m_true_solution = true_solution;
    m_error = error;
    m_x_coords = x_coords;
    m_y_coords = y_coords;
    m_is_square_grid = is_square_grid;
    m_has_results = !solution.empty();
    
    // Активируем элементы управления
    bool hasData = m_has_results;
    showTableButton->setEnabled(hasData);
    clearTableButton->setEnabled(hasData);
    exportCSVButton->setEnabled(hasData);
    dataTypeComboBox->setEnabled(true); // Явно включаем ComboBox
    
    if (hasData) {
        // Больше не отключаем пункты в ComboBox, просто обновляем информацию
        
        // Обновляем информацию о таблице
        tableInfoLabel->setText(QString("Доступно %1 точек данных").arg(solution.size()));
        
        // Заполняем таблицу с текущим типом данных
        populateTableWithSolutionData(true);
    } else {
        tableInfoLabel->setText("Таблица не содержит данных");
        dataTable->setRowCount(0);
        dataTable->setColumnCount(0);
    }
}

void TableTabWidget::setRefinedGridData(
    const std::vector<double>& refined_grid_solution,
    const std::vector<double>& refined_grid_x_coords,
    const std::vector<double>& refined_grid_y_coords)
{
    // Сохраняем данные
    m_refined_solution = refined_grid_solution;
    m_refined_x_coords = refined_grid_x_coords;
    m_refined_y_coords = refined_grid_y_coords;
    m_has_refined_grid = !refined_grid_solution.empty();
    
    // Больше не активируем/деактивируем пункты меню, только сохраняем данные
    
    // Проверяем возможность расчета разницы между решениями
    if (m_has_results && !m_solution.empty() && m_has_refined_grid) {
        // Создаем разницу между решениями (для точек, общих для обеих сеток)
        m_solution_refined_diff.clear();
        
        // Для расчета разницы используем только узлы основной сетки
        // и ищем для них соответствующие значения в уточненной сетке
        m_solution_refined_diff.resize(m_solution.size(), 0.0);
        
        for (size_t i = 0; i < m_x_coords.size(); i++) {
            double x = m_x_coords[i];
            double y = m_y_coords[i];
            
            // Ищем такую же точку в уточненной сетке
            for (size_t j = 0; j < m_refined_x_coords.size(); j++) {
                if (std::abs(m_refined_x_coords[j] - x) < 1e-9 && 
                    std::abs(m_refined_y_coords[j] - y) < 1e-9) {
                    // Нашли совпадение, вычисляем разницу
                    m_solution_refined_diff[i] = m_solution[i] - m_refined_solution[j];
                    break;
                }
            }
        }
    }
}

void TableTabWidget::onDataTypeChanged(int index)
{
    // Выводим информацию для отладки
    qDebug() << "Выбран тип данных:" << index;
    
    // Мы больше не проверяем "активен" ли пункт, так как все пункты всегда активны
    // Вместо этого, нам нужно только обновить текущий выбранный индекс
    
    // При смене типа данных таблица НЕ обновляется автоматически
    // Пользователь должен нажать кнопку "Показать таблицу", чтобы загрузить данные
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

void TableTabWidget::populateTableWithSolutionData(bool isNumericSolution)
{
    if (!m_has_results) {
        tableInfoLabel->setText("Нет данных для отображения");
        return;
    }
    
    // Определяем, какие данные отображать
    const std::vector<double>& dataToShow = isNumericSolution ? m_solution : m_true_solution;
    if (dataToShow.empty()) {
        tableInfoLabel->setText(isNumericSolution ? 
            "Численное решение не доступно" : "Точное решение не доступно");
        dataTable->setRowCount(0);
        dataTable->setColumnCount(0);
        return;
    }
    
    int skipFactor = skipFactorSpinBox->value();
    
    if (m_is_square_grid) {
        // Для квадратной сетки организуем данные в виде таблицы X на Y
        
        // Получаем уникальные координаты X и Y
        std::set<double> uniqueX, uniqueY;
        for (size_t i = 0; i < m_x_coords.size(); i++) {
            uniqueX.insert(m_x_coords[i]);
            uniqueY.insert(m_y_coords[i]);
        }
        
        // Преобразуем в векторы для удобства индексации с прореживанием
        std::vector<double> xCoords(uniqueX.begin(), uniqueX.end());
        std::vector<double> yCoords(uniqueY.begin(), uniqueY.end());
        
        // Создаем прореженные координаты
        std::vector<double> xCoordsSkipped, yCoordsSkipped;
        for (size_t i = 0; i < xCoords.size(); i += skipFactor) {
            xCoordsSkipped.push_back(xCoords[i]);
        }
        for (size_t i = 0; i < yCoords.size(); i += skipFactor) {
            yCoordsSkipped.push_back(yCoords[i]);
        }
        
        // Устанавливаем размеры таблицы
        dataTable->setRowCount(yCoordsSkipped.size());
        dataTable->setColumnCount(xCoordsSkipped.size() + 2);  // +2 для столбца координат Y и индексов
        
        // Устанавливаем заголовки столбцов
        QStringList headers;
        headers << "Индекс" << "Y/X";
        for (const auto& x : xCoordsSkipped) {
            headers << QString::number(x, 'g', 6);
        }
        dataTable->setHorizontalHeaderLabels(headers);
        
        // Заполняем данные
        for (size_t rowIdx = 0; rowIdx < yCoordsSkipped.size(); rowIdx++) {
            double y = yCoordsSkipped[rowIdx];
            
            // Добавляем индекс строки
            dataTable->setItem(rowIdx, 0, new QTableWidgetItem(QString::number(rowIdx)));
            
            // Устанавливаем метку Y в первый столбец
            dataTable->setItem(rowIdx, 1, new QTableWidgetItem(QString::number(y, 'g', 6)));
            
            // Заполняем значения в строке
            for (size_t colIdx = 0; colIdx < xCoordsSkipped.size(); colIdx++) {
                double x = xCoordsSkipped[colIdx];
                
                // Ищем индекс точки с координатами (x, y)
                bool found = false;
                double value = 0.0;
                int nodeIndex = -1;
                
                for (size_t i = 0; i < m_x_coords.size(); i++) {
                    if (std::abs(m_x_coords[i] - x) < 1e-9 && std::abs(m_y_coords[i] - y) < 1e-9) {
                        if (i < dataToShow.size()) {
                            value = dataToShow[i];
                            found = true;
                            nodeIndex = i; // Сохраняем индекс узла в исходной сетке
                            break;
                        }
                    }
                }
                
                if (found) {
                    // Создаем текст с значением и индексом узла
                    QString cellText = QString("%1\n(Узел %2)").arg(value, 0, 'g', 9).arg(nodeIndex);
                    QTableWidgetItem* item = new QTableWidgetItem(cellText);
                    
                    // Сохраняем также индекс узла в данных элемента для возможного доступа
                    item->setData(Qt::UserRole, nodeIndex);
                    
                    dataTable->setItem(rowIdx, colIdx + 2, item);
                } else {
                    dataTable->setItem(rowIdx, colIdx + 2, new QTableWidgetItem("NaN"));
                }
            }
        }
    } else {
        // Для произвольной сетки просто выводим список точек
        int numRows = dataToShow.size() / skipFactor + (dataToShow.size() % skipFactor > 0 ? 1 : 0);
        dataTable->setRowCount(numRows);
        dataTable->setColumnCount(4); // Индекс, X, Y, Value
        
        QStringList headers;
        headers << "Индекс" << "X" << "Y" << (isNumericSolution ? "Численное решение" : "Точное решение");
        dataTable->setHorizontalHeaderLabels(headers);
        
        int rowIdx = 0;
        for (size_t i = 0; i < dataToShow.size(); i += skipFactor) {
            if (i < m_x_coords.size() && i < m_y_coords.size()) {
                dataTable->setItem(rowIdx, 0, new QTableWidgetItem(QString::number(i))); // Индекс узла
                dataTable->setItem(rowIdx, 1, new QTableWidgetItem(QString::number(m_x_coords[i], 'g', 6)));
                dataTable->setItem(rowIdx, 2, new QTableWidgetItem(QString::number(m_y_coords[i], 'g', 6)));
                dataTable->setItem(rowIdx, 3, new QTableWidgetItem(QString::number(dataToShow[i], 'g', 9)));
                rowIdx++;
            }
        }
    }
    
    // Подгоняем ширину столбцов под содержимое
    dataTable->resizeColumnsToContents();
    
    // Обновляем информацию о таблице
    tableInfoLabel->setText(QString("Отображено %1 точек с коэффициентом прореживания %2")
                        .arg(dataTable->rowCount() * (dataTable->columnCount() - 2))
                        .arg(skipFactor));
}

void TableTabWidget::populateTableWithErrorData()
{
    if (!m_has_results || m_error.empty()) {
        tableInfoLabel->setText("Данные об ошибках не доступны");
        dataTable->setRowCount(0);
        dataTable->setColumnCount(0);
        return;
    }
    
    int skipFactor = skipFactorSpinBox->value();
    
    if (m_is_square_grid) {
        // Для квадратной сетки - аналогично populateTableWithSolutionData
        
        // Получаем уникальные координаты X и Y
        std::set<double> uniqueX, uniqueY;
        for (size_t i = 0; i < m_x_coords.size(); i++) {
            uniqueX.insert(m_x_coords[i]);
            uniqueY.insert(m_y_coords[i]);
        }
        
        // Преобразуем в векторы для удобства индексации с прореживанием
        std::vector<double> xCoords(uniqueX.begin(), uniqueX.end());
        std::vector<double> yCoords(uniqueY.begin(), uniqueY.end());
        
        // Создаем прореженные координаты
        std::vector<double> xCoordsSkipped, yCoordsSkipped;
        for (size_t i = 0; i < xCoords.size(); i += skipFactor) {
            xCoordsSkipped.push_back(xCoords[i]);
        }
        for (size_t i = 0; i < yCoords.size(); i += skipFactor) {
            yCoordsSkipped.push_back(yCoords[i]);
        }
        
        // Устанавливаем размеры таблицы
        dataTable->setRowCount(yCoordsSkipped.size());
        dataTable->setColumnCount(xCoordsSkipped.size() + 2);  // +2 для столбца координат Y и индексов
        
        // Устанавливаем заголовки столбцов
        QStringList headers;
        headers << "Индекс" << "Y/X";
        for (const auto& x : xCoordsSkipped) {
            headers << QString::number(x, 'g', 6);
        }
        dataTable->setHorizontalHeaderLabels(headers);
        
        // Заполняем данные
        for (size_t rowIdx = 0; rowIdx < yCoordsSkipped.size(); rowIdx++) {
            double y = yCoordsSkipped[rowIdx];
            
            // Добавляем индекс строки
            dataTable->setItem(rowIdx, 0, new QTableWidgetItem(QString::number(rowIdx)));
            
            // Устанавливаем метку Y в первый столбец
            dataTable->setItem(rowIdx, 1, new QTableWidgetItem(QString::number(y, 'g', 6)));
            
            // Заполняем значения в строке
            for (size_t colIdx = 0; colIdx < xCoordsSkipped.size(); colIdx++) {
                double x = xCoordsSkipped[colIdx];
                
                // Ищем индекс точки с координатами (x, y)
                bool found = false;
                double value = 0.0;
                int nodeIndex = -1;
                
                for (size_t i = 0; i < m_x_coords.size(); i++) {
                    if (std::abs(m_x_coords[i] - x) < 1e-9 && std::abs(m_y_coords[i] - y) < 1e-9) {
                        if (i < m_error.size()) {
                            value = m_error[i];
                            found = true;
                            nodeIndex = i; // Сохраняем индекс узла в исходной сетке
                            break;
                        }
                    }
                }
                
                if (found) {
                    // Создаем текст с значением и индексом узла
                    QString cellText = QString("%1\n(Узел %2)").arg(value, 0, 'g', 9).arg(nodeIndex);
                    QTableWidgetItem* item = new QTableWidgetItem(cellText);
                    
                    // Сохраняем также индекс узла в данных элемента для возможного доступа
                    item->setData(Qt::UserRole, nodeIndex);
                    
                    dataTable->setItem(rowIdx, colIdx + 2, item);
                } else {
                    dataTable->setItem(rowIdx, colIdx + 2, new QTableWidgetItem("NaN"));
                }
            }
        }
    } else {
        // Для произвольной сетки просто выводим список точек
        int numRows = m_error.size() / skipFactor + (m_error.size() % skipFactor > 0 ? 1 : 0);
        dataTable->setRowCount(numRows);
        dataTable->setColumnCount(4); // Индекс, X, Y, Error
        
        QStringList headers;
        headers << "Индекс" << "X" << "Y" << "Ошибка";
        dataTable->setHorizontalHeaderLabels(headers);
        
        int rowIdx = 0;
        for (size_t i = 0; i < m_error.size(); i += skipFactor) {
            if (i < m_x_coords.size() && i < m_y_coords.size()) {
                dataTable->setItem(rowIdx, 0, new QTableWidgetItem(QString::number(i))); // Индекс узла
                dataTable->setItem(rowIdx, 1, new QTableWidgetItem(QString::number(m_x_coords[i], 'g', 6)));
                dataTable->setItem(rowIdx, 2, new QTableWidgetItem(QString::number(m_y_coords[i], 'g', 6)));
                dataTable->setItem(rowIdx, 3, new QTableWidgetItem(QString::number(m_error[i], 'g', 9)));
                rowIdx++;
            }
        }
    }
    
    // Подгоняем ширину столбцов под содержимое
    dataTable->resizeColumnsToContents();
    
    // Обновляем информацию о таблице
    tableInfoLabel->setText(QString("Ошибка: отображено %1 точек с коэффициентом прореживания %2")
                        .arg(dataTable->rowCount() * (dataTable->columnCount() - 2))
                        .arg(skipFactor));
}

void TableTabWidget::populateTableWithRefinedGridData()
{
    if (!m_has_refined_grid || m_refined_solution.empty()) {
        tableInfoLabel->setText("Решение на уточненной сетке не доступно");
        dataTable->setRowCount(0);
        dataTable->setColumnCount(0);
        return;
    }
    
    int skipFactor = skipFactorSpinBox->value();
    
    // Для решения на уточненной сетке всегда используем подход с квадратной сетки
    
    // Получаем уникальные координаты X и Y
    std::set<double> uniqueX, uniqueY;
    for (size_t i = 0; i < m_refined_x_coords.size(); i++) {
        uniqueX.insert(m_refined_x_coords[i]);
        uniqueY.insert(m_refined_y_coords[i]);
    }
    
    // Преобразуем в векторы для удобства индексации с прореживанием
    std::vector<double> xCoords(uniqueX.begin(), uniqueX.end());
    std::vector<double> yCoords(uniqueY.begin(), uniqueY.end());
    
    // Создаем прореженные координаты
    std::vector<double> xCoordsSkipped, yCoordsSkipped;
    for (size_t i = 0; i < xCoords.size(); i += skipFactor) {
        xCoordsSkipped.push_back(xCoords[i]);
    }
    for (size_t i = 0; i < yCoords.size(); i += skipFactor) {
        yCoordsSkipped.push_back(yCoords[i]);
    }
    
    // Устанавливаем размеры таблицы
    dataTable->setRowCount(yCoordsSkipped.size());
    dataTable->setColumnCount(xCoordsSkipped.size() + 2);  // +2 для столбца координат Y и индекса
    
    // Устанавливаем заголовки столбцов
    QStringList headers;
    headers << "Индекс" << "Y/X";
    for (const auto& x : xCoordsSkipped) {
        headers << QString::number(x, 'g', 6);
    }
    dataTable->setHorizontalHeaderLabels(headers);
    
    // Заполняем данные
    for (size_t rowIdx = 0; rowIdx < yCoordsSkipped.size(); rowIdx++) {
        double y = yCoordsSkipped[rowIdx];
        
        // Добавляем индекс строки
        dataTable->setItem(rowIdx, 0, new QTableWidgetItem(QString::number(rowIdx)));
        
        // Устанавливаем метку Y во второй столбец
        dataTable->setItem(rowIdx, 1, new QTableWidgetItem(QString::number(y, 'g', 6)));
        
        // Заполняем значения в строке
        for (size_t colIdx = 0; colIdx < xCoordsSkipped.size(); colIdx++) {
            double x = xCoordsSkipped[colIdx];
            
            // Ищем индекс точки с координатами (x, y)
            bool found = false;
            double value = 0.0;
            int nodeIndex = -1;
            
            for (size_t i = 0; i < m_refined_x_coords.size(); i++) {
                if (std::abs(m_refined_x_coords[i] - x) < 1e-9 && std::abs(m_refined_y_coords[i] - y) < 1e-9) {
                    if (i < m_refined_solution.size()) {
                        value = m_refined_solution[i];
                        found = true;
                        nodeIndex = i; // Сохраняем индекс узла в исходной сетке
                        break;
                    }
                }
            }
            
            if (found) {
                // Создаем текст с значением и индексом узла
                QString cellText = QString("%1\n(Узел %2)").arg(value, 0, 'g', 9).arg(nodeIndex);
                QTableWidgetItem* item = new QTableWidgetItem(cellText);
                
                // Сохраняем также индекс узла в данных элемента для возможного доступа
                item->setData(Qt::UserRole, nodeIndex);
                
                dataTable->setItem(rowIdx, colIdx + 2, item);
            } else {
                dataTable->setItem(rowIdx, colIdx + 2, new QTableWidgetItem("NaN"));
            }
        }
    }
    
    // Подгоняем ширину столбцов под содержимое
    dataTable->resizeColumnsToContents();
    
    // Обновляем информацию о таблице
    tableInfoLabel->setText(QString("Решение на уточненной сетке: отображено %1 точек с коэффициентом прореживания %2")
                        .arg(dataTable->rowCount() * (dataTable->columnCount() - 2))
                        .arg(skipFactor));
}

void TableTabWidget::populateTableWithSolutionRefinedDiff()
{
    if (!m_has_refined_grid || !m_has_results || m_solution_refined_diff.empty()) {
        tableInfoLabel->setText("Данные о разнице между решениями не доступны");
        dataTable->setRowCount(0);
        dataTable->setColumnCount(0);
        return;
    }
    
    int skipFactor = skipFactorSpinBox->value();
    
    if (m_is_square_grid) {
        // Для квадратной сетки организуем данные в виде таблицы X на Y
        
        // Получаем уникальные координаты X и Y
        std::set<double> uniqueX, uniqueY;
        for (size_t i = 0; i < m_x_coords.size(); i++) {
            uniqueX.insert(m_x_coords[i]);
            uniqueY.insert(m_y_coords[i]);
        }
        
        // Преобразуем в векторы для удобства индексации с прореживанием
        std::vector<double> xCoords(uniqueX.begin(), uniqueX.end());
        std::vector<double> yCoords(uniqueY.begin(), uniqueY.end());
        
        // Создаем прореженные координаты
        std::vector<double> xCoordsSkipped, yCoordsSkipped;
        for (size_t i = 0; i < xCoords.size(); i += skipFactor) {
            xCoordsSkipped.push_back(xCoords[i]);
        }
        for (size_t i = 0; i < yCoords.size(); i += skipFactor) {
            yCoordsSkipped.push_back(yCoords[i]);
        }
        
        // Устанавливаем размеры таблицы
        dataTable->setRowCount(yCoordsSkipped.size());
        dataTable->setColumnCount(xCoordsSkipped.size() + 2);  // +2 для столбца координат Y и индексов
        
        // Устанавливаем заголовки столбцов
        QStringList headers;
        headers << "Индекс" << "Y/X";
        for (const auto& x : xCoordsSkipped) {
            headers << QString::number(x, 'g', 6);
        }
        dataTable->setHorizontalHeaderLabels(headers);
        
        // Заполняем данные
        for (size_t rowIdx = 0; rowIdx < yCoordsSkipped.size(); rowIdx++) {
            double y = yCoordsSkipped[rowIdx];
            
            // Добавляем индекс строки
            dataTable->setItem(rowIdx, 0, new QTableWidgetItem(QString::number(rowIdx)));
            
            // Устанавливаем метку Y в первый столбец
            dataTable->setItem(rowIdx, 1, new QTableWidgetItem(QString::number(y, 'g', 6)));
            
            // Заполняем значения в строке
            for (size_t colIdx = 0; colIdx < xCoordsSkipped.size(); colIdx++) {
                double x = xCoordsSkipped[colIdx];
                
                // Ищем индекс точки с координатами (x, y)
                bool found = false;
                double value = 0.0;
                int nodeIndex = -1;
                
                for (size_t i = 0; i < m_x_coords.size(); i++) {
                    if (std::abs(m_x_coords[i] - x) < 1e-9 && std::abs(m_y_coords[i] - y) < 1e-9) {
                        if (i < m_solution_refined_diff.size()) {
                            value = m_solution_refined_diff[i];
                            found = true;
                            nodeIndex = i; // Сохраняем индекс узла в исходной сетке
                            break;
                        }
                    }
                }
                
                if (found) {
                    // Создаем текст с значением и индексом узла
                    QString cellText = QString("%1\n(Узел %2)").arg(value, 0, 'g', 9).arg(nodeIndex);
                    QTableWidgetItem* item = new QTableWidgetItem(cellText);
                    
                    // Сохраняем также индекс узла в данных элемента для возможного доступа
                    item->setData(Qt::UserRole, nodeIndex);
                    
                    dataTable->setItem(rowIdx, colIdx + 2, item);
                } else {
                    dataTable->setItem(rowIdx, colIdx + 2, new QTableWidgetItem("NaN"));
                }
            }
        }
    } else {
        // Для произвольной сетки просто выводим список точек
        int numRows = m_solution_refined_diff.size() / skipFactor + 
                      (m_solution_refined_diff.size() % skipFactor > 0 ? 1 : 0);
        dataTable->setRowCount(numRows);
        dataTable->setColumnCount(4); // Индекс, X, Y, Diff
        
        QStringList headers;
        headers << "Индекс" << "X" << "Y" << "Разница решений";
        dataTable->setHorizontalHeaderLabels(headers);
        
        int rowIdx = 0;
        for (size_t i = 0; i < m_solution_refined_diff.size(); i += skipFactor) {
            if (i < m_x_coords.size() && i < m_y_coords.size()) {
                dataTable->setItem(rowIdx, 0, new QTableWidgetItem(QString::number(i))); // Индекс узла
                dataTable->setItem(rowIdx, 1, new QTableWidgetItem(QString::number(m_x_coords[i], 'g', 6)));
                dataTable->setItem(rowIdx, 2, new QTableWidgetItem(QString::number(m_y_coords[i], 'g', 6)));
                dataTable->setItem(rowIdx, 3, new QTableWidgetItem(QString::number(m_solution_refined_diff[i], 'g', 9)));
                rowIdx++;
            }
        }
    }
    
    // Подгоняем ширину столбцов под содержимое
    dataTable->resizeColumnsToContents();
    
    // Обновляем информацию о таблице
    tableInfoLabel->setText(QString("Разница между решениями: отображено %1 точек с коэффициентом прореживания %2")
                        .arg(dataTable->rowCount() * (dataTable->columnCount() - 2))
                        .arg(skipFactor));
}

void TableTabWidget::onShowTableButtonClicked()
{
    // Если есть прямые данные о результатах, используем их
    if (m_has_results) {
        int dataTypeIndex = dataTypeComboBox->currentIndex();
        
        // Проверяем наличие запрошенных данных
        switch(dataTypeIndex) {
            case NUMERICAL_SOLUTION:
                // Численное решение почти всегда доступно, если m_has_results == true
                if (m_solution.empty()) {
                    tableInfoLabel->setText("Численное решение не доступно");
                    dataTable->setRowCount(0);
                    dataTable->setColumnCount(0);
                    return;
                }
                populateTableWithSolutionData(true);
                break;
                
            case EXACT_SOLUTION:
                // Проверка на наличие точного решения
                if (m_true_solution.empty()) {
                    tableInfoLabel->setText("Точное решение не доступно");
                    dataTable->setRowCount(0);
                    dataTable->setColumnCount(0);
                    return;
                }
                populateTableWithSolutionData(false);
                break;
                
            case ERROR:
                // Проверка на наличие данных ошибки
                if (m_error.empty()) {
                    tableInfoLabel->setText("Данные об ошибках не доступны");
                    dataTable->setRowCount(0);
                    dataTable->setColumnCount(0);
                    return;
                }
                populateTableWithErrorData();
                break;
                
            case REFINED_GRID:
                // Проверка на наличие решения на уточненной сетке
                if (!m_has_refined_grid || m_refined_solution.empty()) {
                    tableInfoLabel->setText("Решение на уточненной сетке не доступно");
                    dataTable->setRowCount(0);
                    dataTable->setColumnCount(0);
                    return;
                }
                populateTableWithRefinedGridData();
                break;
                
            case SOLUTION_REFINED_DIFF:
                // Проверка на наличие разницы между решениями
                if (!m_has_refined_grid || !m_has_results || m_solution_refined_diff.empty()) {
                    tableInfoLabel->setText("Данные о разнице между решениями не доступны");
                    dataTable->setRowCount(0);
                    dataTable->setColumnCount(0);
                    return;
                }
                populateTableWithSolutionRefinedDiff();
                break;
        }
    } else {
        // Иначе используем старый метод с CSV
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
            case SOLUTION_REFINED_DIFF:
                dataTypeString = "Разница между решениями";
                break;
        }
        
        if (currentCSVData.isEmpty() || 
            (dataTypeIndex > 0 && !currentCSVData.contains(dataTypeString, Qt::CaseInsensitive))) {
            tableInfoLabel->setText(QString("Данные типа '%1' не найдены").arg(dataTypeString));
            dataTable->setRowCount(0);
            dataTable->setColumnCount(0);
            return;
        }
        
        populateTableWithData(currentCSVData);
    }
}

void TableTabWidget::onClearTableButtonClicked()
{
    // Очищаем таблицу
    dataTable->setRowCount(0);
    dataTable->setColumnCount(0);
    tableInfoLabel->setText("Таблица очищена");
}

void TableTabWidget::onExportButtonClicked()
{
    // Сигнализируем о запросе на экспорт CSV с текущим коэффициентом прореживания
    emit exportCSVRequested(skipFactorSpinBox->value());
}

void TableTabWidget::setDataTypeComboEnabled(bool enabled)
{
    // Явно устанавливаем состояние активности для ComboBox
    if (dataTypeComboBox) {
        dataTypeComboBox->setEnabled(enabled);
    }
}
