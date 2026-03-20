# 🥐 Roadmap 2: Подготовка к WorldSkills — Анализ данных пекарни на Pandas

> **Контекст:** эталонное задание WorldSkills — анализ данных французской пекарни (3 CSV: продажи, продукты, клиенты).
>
> **Цель:** научиться за 8–12 недель выполнять полный цикл: загрузка → очистка → анализ → визуализация → прогноз → отчёт.
>
> **Эталон:** [worldskill.py](file:///f:/Dev/ws_1/worldskill.py) из `ws_1`.

---

## 📋 Оглавление

1. [Блок 0 — Окружение и структура данных](#блок-0--окружение-и-структура-данных)
2. [Блок 1 — Загрузка и исследование (EDA)](#блок-1--загрузка-и-исследование-eda)
3. [Блок 2 — Очистка и предобработка](#блок-2--очистка-и-предобработка)
4. [Блок 3 — Анализ трендов продаж](#блок-3--анализ-трендов-продаж)
5. [Блок 4 — Анализ продуктов и категорий](#блок-4--анализ-продуктов-и-категорий)
6. [Блок 5 — Анализ клиентов и сегментация](#блок-5--анализ-клиентов-и-сегментация)
7. [Блок 6 — Временные ряды и прогнозирование (ARIMA)](#блок-6--временные-ряды-и-прогнозирование-arima)
8. [Блок 7 — Кластеризация и рекомендации (KMeans)](#блок-7--кластеризация-и-рекомендации-kmeans)
9. [Блок 8 — Продвинутая аналитика (CLTV, Churn, Эластичность)](#блок-8--продвинутая-аналитика-cltv-churn-эластичность)
10. [Блок 9 — Визуализация и генерация отчётов (PDF/Excel)](#блок-9--визуализация-и-генерация-отчётов-pdfexcel)
11. [Блок 10 — Финальная сборка: полный пайплайн](#блок-10--финальная-сборка-полный-пайплайн)
12. [📚 Ресурсы](#-ресурсы)

---

## Блок 0 — Окружение и структура данных

> **Срок:** 1–2 дня

### Установка

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl xlsxwriter
```

### Структура эталонных данных

```
ws_1/
├── sales_transactions.csv   ← транзакции (transaction_id, customer_id, date, product_id, quantity, price, promotion_id)
├── products.csv             ← справочник (product_id, product_name, category, ingredients, price, cost, seasonality_score, active_status, release_date)
├── customers.csv            ← клиенты (customer_id, name, age, gender, zip_code, email, phone_number, member_status, join_date, last_purchase_date, total_spend, avg_order_value, frequency, preferred_category, churn_status)
└── worldskill.py            ← эталонное решение
```

### Что понять до старта

| Понятие | Зачем |
|---------|-------|
| Реляционные данные | 3 таблицы связаны по `product_id` и `customer_id` — нужен `merge` |
| «Грязные» данные | В датасетах: невалидные даты, отрицательные цены/количества, пропуски в `age`, `phone_number`, `promotion_id` |
| Бизнес-метрики | Revenue, AOV, Unit Margin, CLTV, Churn Rate, PED |

---

## Блок 1 — Загрузка и исследование (EDA)

> **Срок:** 3–4 дня  
> **Соответствует:** ЧАСТЬ 1 эталона (строки 12–43)

### Теория

- Чтение CSV в DataFrame.
- Автоматический анализ типов данных.
- Выявление аномалий: невалидные даты, отрицательные значения, несуществующие ID.

### 🔑 Ключевые методы

| Метод | Описание | Применение в задании |
|-------|----------|---------------------|
| `pd.read_csv()` | Чтение CSV | `df_sales = pd.read_csv('sales_transactions.csv')` |
| `.dtypes` | Типы данных столбцов | Определить, что `date` — object, а не datetime |
| `.isna().sum()` | Подсчёт пропусков | Найти NaN в `age`, `phone_number`, `promotion_id` |
| `pd.to_datetime(errors='coerce')` | Конвертация с обнаружением ошибок | Найти 10 невалидных дат (например, `2023-14-01`) |
| `(df['col'] < 0).sum()` | Подсчёт отрицательных | Найти 13 отрицательных quantity и 11 отрицательных price |
| `.isin()` | Проверка ссылочной целостности | `~df_sales['product_id'].isin(df_products['product_id'])` |

### 📝 Практические задания

1. Загрузи 3 CSV-файла и выведи `.info()` для каждого.
2. Посчитай количество невалидных дат через `pd.to_datetime(errors='coerce').isna().sum()`.
3. Найди отрицательные значения в `quantity` и `price`.
4. Проверь, все ли `product_id` и `customer_id` из продаж существуют в справочниках.
5. Запиши результат в текстовый файл (как в эталоне — `Session1_DataExploration.txt`).

### 🚀 Мини-проект: «Автоматический отчёт по качеству данных»

> Напиши функцию `generate_data_quality_report(df_sales, df_products, df_customers)`,  
> которая создаёт текстовый файл со всеми найденными аномалиями.

---

## Блок 2 — Очистка и предобработка

> **Срок:** 5–7 дней  
> **Соответствует:** ЧАСТЬ 2 эталона (строки 45–66)

### Теория

- Стратегии заполнения пропусков (среднее для числовых, константа для текстовых).
- Стандартизация строковых полей (телефоны, email).
- Преобразование типов (`object` → `datetime`).
- Обработка аномалий (отрицательные значения → абсолютные).

### 🔑 Ключевые методы

| Метод | Описание | Применение в задании |
|-------|----------|---------------------|
| `.fillna()` | Заполнение пропусков | `df['age'].fillna(df['age'].mean())` |
| `.str.replace(regex)` | Очистка строк по шаблону | `df['phone'].str.replace(r'[^0-9+]', '', regex=True)` — убрать скобки, пробелы, тире |
| `pd.to_datetime()` | Конвертация дат | `df['join_date'] = pd.to_datetime(df['join_date'])` |
| `.dropna(subset=)` | Удаление строк с NaN в конкретных столбцах | `df_sales.dropna(subset=['date'])` — удалить невалидные даты |
| `.abs()` | Абсолютные значения | `df['quantity'] = df['quantity'].abs()` |
| `.to_csv()` | Сохранение очищенных данных | `df.to_csv('customers_cleaned.csv', index=False)` |

### Что именно нужно очистить (по эталону)

```
customers.csv:
  ├── age:           NaN → fillna(mean)
  ├── phone_number:  NaN → "0", затем убрать все кроме цифр и "+"
  ├── join_date:     object → datetime
  └── last_purchase_date: object → datetime

sales_transactions.csv:
  ├── promotion_id:  NaN → "0"
  ├── date:          object → datetime (errors='coerce'), dropna
  ├── quantity:      отрицательные → abs()
  └── price:         отрицательные → abs()
```

### 📝 Практические задания

1. Заполни пропуски в `age` средним, в `phone_number` — нулём.
2. Стандартизируй телефоны: `(06) 99-20-58` → `0699205̀8` (regex).
3. Преобразуй `date`, `join_date`, `last_purchase_date` в datetime.
4. Удали строки с невалидными датами (NaT после coerce).
5. Исправь отрицательные `quantity` и `price` через `.abs()`.
6. Сохрани очищенные файлы.

### 🚀 Мини-проект: «Pipeline очистки данных»

> Создай класс `DataCleaner` с методами `clean_customers()`, `clean_sales()`,  
> который принимает сырые DataFrame и возвращает очищенные.

---

## Блок 3 — Анализ трендов продаж

> **Срок:** 5–7 дней  
> **Соответствует:** ЧАСТЬ 3 эталона (строки 68–108)

### Теория

- Создание вычисляемых столбцов (`revenue = quantity * price`).
- Группировка по временным периодам (`dt.to_period('M')`).
- Агрегация метрик: выручка, количество транзакций, средний чек (AOV).

### 🔑 Ключевые методы

| Метод | Описание | Применение |
|-------|----------|------------|
| `.dt.to_period('M')` | Преобразование даты в период (месяц) | `df['month_period'] = df['date'].dt.to_period('M')` |
| `.groupby().agg()` | Множественная агрегация | `df.groupby('month_period').agg({'revenue': 'sum', 'transaction_id': 'count'})` |
| `.rename(columns=)` | Переименование после агрегации | `.rename(columns={'transaction_id': 'num_transactions'})` |
| `.sort_values().head()` | ТОП-N записей | `monthly.sort_values('revenue', ascending=False).head(3)` |
| `plt.plot()` | Линейный график | Ежемесячная динамика Revenue и AOV |
| `plt.savefig()` | Сохранение графика | `plt.savefig('Session1_SalesTrends_Revenue.pdf')` |

### Формулы

```
revenue           = quantity × price
avg_order_value   = total_revenue / num_transactions
```

### 📝 Практические задания

1. Создай столбец `revenue` и `month_period`.
2. Агрегируй по месяцам: сумма выручки, количество транзакций.
3. Рассчитай AOV (средний чек) по месяцам.
4. Найди ТОП-3 месяца по выручке.
5. Построй 2 графика: Revenue и AOV по месяцам (сохрани в PDF).

### 🚀 Мини-проект: «Ежемесячный отчёт по продажам»

> Автоматизируй генерацию отчёта с графиками и ТОП-3 месяцами.  
> Добавь расчёт процентного изменения выручки (MoM growth).

---

## Блок 4 — Анализ продуктов и категорий

> **Срок:** 4–5 дней  
> **Соответствует:** ЧАСТЬ 4 эталона (строки 110–175)

### Теория

- Слияние таблиц (`merge`) для обогащения данных.
- Расчёт маржинальности (unit margin = avg_price − cost).
- Агрегация по продуктам и категориям.

### 🔑 Ключевые методы

| Метод | Описание | Применение |
|-------|----------|------------|
| `.merge(on=, how=)` | Соединение таблиц | `df_sales.merge(df_products, on='product_id', how='left')` |
| `.groupby([cols]).agg()` | Группировка по нескольким столбцам | `groupby(['product_id', 'product_name', 'category'])` |
| `.reset_index()` | Сброс мультииндекса | После groupby для получения плоского DataFrame |
| `.plot(kind='bar')` | Столбчатая диаграмма | Доход по категориям: Выпечка, Хлеб, Тарты |

### Формулы

```
unit_margin = (revenue / quantity) − cost
```

### 📝 Практические задания

1. Объедини `df_sales` с `df_products` по `product_id`.
2. Рассчитай: total quantity, total revenue по каждому продукту.
3. Добавь `cost` из справочника и вычисли unit margin.
4. Агрегируй выручку по категориям (Выпечка, Хлеб, Тарты).
5. Найди ТОП-3 продукта по количеству продаж.
6. Построй bar-chart дохода по категориям.

### 🚀 Мини-проект: «Карточка продукта»

> Для каждого из 10 продуктов сгенерируй мини-отчёт:  
> название, категория, кол-во продаж, выручка, маржа, сезонность.

---

## Блок 5 — Анализ клиентов и сегментация

> **Срок:** 4–5 дней  
> **Соответствует:** строки 143–163 эталона

### Теория

- Биннинг (разбиение на группы) непрерывных данных.
- Нормализованные частоты (`value_counts(normalize=True)`).
- Анализ по лояльности (member_status).

### 🔑 Ключевые методы

| Метод | Описание | Применение |
|-------|----------|------------|
| `pd.cut(bins, labels)` | Биннинг по возрасту | `pd.cut(df['age'], bins=[17,24,34,44,200], labels=['18-24','25-34','35-44','45+'])` |
| `.value_counts(normalize=True)` | Процентное распределение | Распределение по полу (M/F) |
| `.groupby('member_status')['total_spend'].mean()` | Средние расходы по лояльности | Basic vs Silver vs Gold |
| `sns.histplot()` | Гистограмма Seaborn | Распределение клиентов по возрастным группам |

### 📝 Практические задания

1. Создай столбец `age_group` с 4 группами: 18-24, 25-34, 35-44, 45+.
2. Построй гистограмму распределения клиентов по возрасту.
3. Рассчитай процентное распределение по полу.
4. Сравни средние расходы по уровням лояльности (Basic / Silver / Gold).
5. Найди самую популярную категорию по каждой возрастной группе.

### 🚀 Мини-проект: «Профиль целевой аудитории»

> Создай отчёт: кто наш типичный клиент? (возраст, пол, частота, предпочтения).

---

## Блок 6 — Временные ряды и прогнозирование (ARIMA)

> **Срок:** 7–10 дней  
> **Соответствует:** строки 164–186 эталона

### Теория

- `DatetimeIndex` и ресемплирование ежедневных данных.
- Модель ARIMA (AutoRegressive Integrated Moving Average).
- Метрика MAE (Mean Absolute Error).

### 🔑 Ключевые методы

| Метод | Описание | Применение |
|-------|----------|------------|
| `.resample('D').sum()` | Ежедневная агрегация | Заполнить пропущенные дни нулями |
| `ARIMA(data, order=(p,d,q))` | Построение модели | `ARIMA(daily_sales, order=(5,1,0))` |
| `.fit()` | Обучение модели | `model_fit = model.fit()` |
| `.get_forecast(steps=30)` | Прогноз на 30 дней | `forecast_result.predicted_mean` |
| `mean_absolute_error()` | Оценка качества | `mae = mean_absolute_error(actual, fitted)` |

### Пример из эталона

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

daily_sales = df_sales.groupby('date')['revenue'].sum()
daily_sales.index = pd.DatetimeIndex(daily_sales.index)
daily_sales = daily_sales.resample('D').sum()

model = ARIMA(daily_sales, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.get_forecast(steps=30)
mae = mean_absolute_error(daily_sales, model_fit.fittedvalues)
```

### 📝 Практические задания

1. Агрегируй продажи по дням, создай `DatetimeIndex`.
2. Ресемплируй с `.resample('D').sum()` для заполнения пробелов.
3. Обучи ARIMA(5,1,0) и получи `fittedvalues`.
4. Сделай прогноз на 30 дней вперёд.
5. Рассчитай MAE и сохрани прогноз в CSV.

### 🚀 Мини-проект: «30-дневный прогноз продаж»

> Повтори пайплайн из эталона: данные → ARIMA → прогноз → CSV + график.

---

## Блок 7 — Кластеризация и рекомендации (KMeans)

> **Срок:** 5–7 дней  
> **Соответствует:** строки 188–226 эталона

### Теория

- Расчёт признаков клиента (total_purchases, avg_purchase_value).
- Нормализация признаков (`StandardScaler`).
- Кластеризация KMeans (k=3).
- Рекомендация товаров: ТОП-10 товаров кластера минус уже купленные.

### 🔑 Ключевые методы

| Метод | Описание | Применение |
|-------|----------|------------|
| `.groupby().agg()` | Расчёт признаков клиента | `total_purchases`, `total_revenue` |
| `StandardScaler().fit_transform()` | Нормализация | Перед KMeans — обязательно |
| `KMeans(n_clusters=3).fit_predict()` | Кластеризация | Разделение клиентов на 3 группы |
| `.value_counts().head(10)` | ТОП-10 товаров кластера | Для рекомендаций |
| `set()` операции | Исключение купленных товаров | `candidates - bought_products` |

### Формулы

```
avg_purchase_value = total_revenue / total_purchases
recommendation     = top_products_of_cluster − already_bought
```

### 📝 Практические задания

1. Рассчитай `total_purchases` и `avg_purchase_value` по каждому клиенту.
2. Нормализуй признаки через `StandardScaler`.
3. Обучи KMeans(k=3) и присвой метки кластеров.
4. Найди ТОП-10 товаров для каждого кластера.
5. Для каждого клиента: выбери 3 рекомендации (не купленные ранее).
6. Сохрани в CSV: `customer_id, cluster_label, rec_1, rec_2, rec_3`.

### 🚀 Мини-проект: «Система рекомендаций»

> Повтори полный цикл: признаки → масштабирование → кластеризация → рекомендации → CSV.

---

## Блок 8 — Продвинутая аналитика (CLTV, Churn, Эластичность)

> **Срок:** 7–10 дней  
> **Соответствует:** строки 228–300 эталона

### 8.1 Product Performance & Profit Margin

| Метрика | Формула |
|---------|---------|
| `total_cost` | `total_quantity_sold × cost` |
| `profit_margin` | `(total_revenue − total_cost) / total_revenue` |

### 8.2 Price Elasticity of Demand (PED)

```
PED = (% изменение quantity) / (% изменение price)
|PED| > 1 → эластичный → рекомендация: снизить цену на 5%
|PED| ≤ 1 → неэластичный → рекомендация: повысить цену на 5%
```

### 8.3 Customer Lifetime Value (CLTV)

```
months_active      = (max_date − min_date).days / 30   (min = 1)
frequency_monthly  = total_transactions / months_active
CLTV               = avg_purchase_value × frequency_monthly × 36
```

### 8.4 Churn Analysis

```
churn_rate        = churned_customers / total_customers × 100
avg_cltv_churned  = среднее CLTV ушедших
avg_cltv_active   = среднее CLTV активных
```

### 🔑 Ключевые методы

| Метод | Описание |
|-------|----------|
| `.map()` | Маппинг cost по product_id: `df['product_id'].map(costs)` |
| `.agg(lambda x: ...)` | Кастомная агрегация: `'date': lambda x: (x.max()-x.min()).days` |
| `.apply(lambda x: max(x, 1))` | Минимальное значение months_active = 1 |
| `df[condition]['col'].mean()` | Среднее CLTV для churned vs active |

### 📝 Практические задания

1. Рассчитай `profit_margin` для каждого продукта.
2. Вычисли PED для каждого продукта и сформулируй ценовую рекомендацию.
3. Рассчитай CLTV каждого клиента по формуле из эталона.
4. Рассчитай churn_rate и сравни средний CLTV ушедших vs активных.
5. Сохрани всё в 4 CSV: Product_Performance, Price_Analysis, CLTV, Churn_Analysis.

### 🚀 Мини-проект: «Бизнес-дашборд пекарни»

> Объедини все метрики в один отчёт с выводами и рекомендациями.

---

## Блок 9 — Визуализация и генерация отчётов (PDF/Excel)

> **Срок:** 3–5 дней

### Графики из эталона

| Файл | Тип | Данные |
|------|-----|--------|
| `Session1_SalesTrends_Revenue.pdf` | Line plot | Ежемесячная выручка |
| `Session1_SalesTrends_AOV.pdf` | Line plot | Средний чек по месяцам |
| `Session1_ProductPerformance_Category.pdf` | Bar chart | Доход по категориям |
| `Session1_CustomerAnalysis_Age.pdf` | Histogram | Распределение по возрасту |

### 🔑 Ключевые методы

| Метод | Описание |
|-------|----------|
| `plt.figure(figsize=(10,5))` | Создание фигуры нужного размера |
| `plt.plot(x, y, marker='o')` | Линейный график с маркерами |
| `.plot(kind='bar', color=[...])` | Bar-chart из Pandas Series |
| `sns.histplot()` | Гистограмма через Seaborn |
| `plt.savefig('file.pdf')` | Сохранение в PDF |
| `plt.close()` | Закрытие фигуры (важно в скриптах!) |

### 📝 Практические задания

1. Воспроизведи все 4 графика из эталона.
2. Добавь оформление: заголовки, подписи осей, gridlines, цвета.
3. Объедини в функцию `generate_all_charts(...)`.
4. Добавь 2 своих графика: boxplot цен, heatmap корреляций.

---

## Блок 10 — Финальная сборка: полный пайплайн

> **Срок:** 7–10 дней

### Цель

Воспроизвести **весь** `worldskill.py` (300 строк) **своими руками**, без подглядывания.

### Чеклист выходных файлов

| Файл | Блок |
|------|------|
| `Session1_DataExploration.txt` | 1 — EDA |
| `customers_cleaned.csv` | 2 — Очистка |
| `sales_transactions_cleaned.csv` | 2 — Очистка |
| `Session1_SalesTrends_Revenue.pdf` | 3 — Тренды |
| `Session1_SalesTrends_AOV.pdf` | 3 — Тренды |
| `Session1_ProductPerformance_Category.pdf` | 4 — Продукты |
| `Session1_CustomerAnalysis_Age.pdf` | 5 — Клиенты |
| `Session1_SalesForecast.csv` | 6 — ARIMA |
| `Session5_Segmentation_and_Recommendations.csv` | 7 — KMeans |
| `Session5_Product_Performance.csv` | 8 — Маржа |
| `Session5_Price_Analysis.csv` | 8 — PED |
| `Session1_CLTV.csv` | 8 — CLTV |
| `Session1_Churn_Analysis.csv` | 8 — Churn |

### Пайплайн

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  LOAD    │ → │  CLEAN   │ → │ ANALYZE  │ → │  MODEL   │ → │  REPORT  │
│          │   │          │   │          │   │          │   │          │
│ 3× CSV   │   │ NaN/abs  │   │ Trends   │   │ ARIMA    │   │ 4× PDF   │
│ .dtypes  │   │ regex    │   │ Products │   │ KMeans   │   │ 7× CSV   │
│ anomaly  │   │ datetime │   │ Clients  │   │ PED/CLTV │   │ 1× TXT   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## 📚 Ресурсы

### Документация

| Ресурс | Ссылка |
|--------|--------|
| Pandas docs | [pandas.pydata.org/docs](https://pandas.pydata.org/docs/) |
| Matplotlib docs | [matplotlib.org](https://matplotlib.org/stable/contents.html) |
| Seaborn docs | [seaborn.pydata.org](https://seaborn.pydata.org/) |
| Statsmodels ARIMA | [ARIMA docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) |
| Scikit-learn KMeans | [KMeans docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) |

### Датасеты для практики (аналогичные)

| Датасет | Ссылка |
|---------|--------|
| Bakery Transactions | [Kaggle](https://www.kaggle.com/datasets/hosubjeong/bakery-sales) |
| Online Retail II | [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Superstore Sales | [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) |
| Store Sales Time Series | [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) |

---

## 🗓 График обучения (8–12 недель)

| Неделя | Блок | Тема |
|--------|------|------|
| 1 | 0 + 1 | Окружение + EDA |
| 2 | 2 | Очистка данных |
| 3 | 3 | Тренды продаж |
| 4 | 4 + 5 | Продукты + Клиенты |
| 5–6 | 6 | ARIMA-прогнозирование |
| 7 | 7 | KMeans + рекомендации |
| 8–9 | 8 | CLTV, Churn, PED |
| 10 | 9 | Визуализация и отчёты |
| 11–12 | 10 | **Финальная сборка** |

---

## ✅ Чеклист прогресса

- [ ] **Блок 0:** Окружение готово, данные загружены
- [ ] **Блок 1:** Умею находить аномалии, формировать EDA-отчёт
- [ ] **Блок 2:** Очищаю данные: NaN, regex, datetime, abs()
- [ ] **Блок 3:** Считаю Revenue, AOV, ТОП-3 месяца, строю графики
- [ ] **Блок 4:** Объединяю таблицы, считаю маржу, анализ по категориям
- [ ] **Блок 5:** Биннинг по возрасту, пол, лояльность
- [ ] **Блок 6:** ARIMA: обучение → прогноз → MAE → CSV
- [ ] **Блок 7:** KMeans: признаки → масштабирование → кластеры → рекомендации
- [ ] **Блок 8:** CLTV, Churn Rate, PED, Profit Margin
- [ ] **Блок 9:** 4 PDF-графика, оформление, автоматизация
- [ ] **Блок 10:** Полный пайплайн = 13 выходных файлов за один запуск

---

*Дорожная карта создана 18 марта 2026 г. на основе эталонного задания WorldSkills из `ws_1`.*
