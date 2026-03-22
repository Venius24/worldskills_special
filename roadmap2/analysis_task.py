import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

import random

df_sales = pd.read_csv('sales_transactions.csv')
df_products = pd.read_csv('products.csv')
df_customers = pd.read_csv('customers.csv')

print(df_sales.head())
print(df_products.head())
print(df_customers.head())

print("_________________Большой отступ____________")

sales_info = df_sales.info()
products_info = df_products.info()
customers_info = df_customers.info()
print(sales_info)
print(products_info)
print(customers_info)

print("_________________Большой отступ____________")

incorrect_dates = pd.to_datetime(df_sales['date'], errors='coerce').isna().sum()
print(incorrect_dates)

print("_________________Большой отступ____________")

neg_quantity = (df_sales['quantity']< 0).sum()
neg_price = (df_sales['price']< 0).sum()
print(f'Сколько негативных в квантити? {neg_quantity}')
print(f'Сколько негативных в прайсе? {neg_price}')

print("_________________Большой отступ____________")

res_check = df_sales['product_id'].isin(df_products['product_id'])
res_fin = df_sales[~res_check]
print(res_fin)
res2_check = df_sales['customer_id'].isin(df_customers['customer_id'])
res2_fin = df_sales[~res2_check]
print(res2_fin)

with open('Session1_DataExploration.txt', 'w', encoding='utf-8') as f:
    f.write("=== ОТЧЕТ ПО АНАЛИЗУ ДАННЫХ ===\n\n\n")
    f.write(f'Инфо о продажах {df_sales.info(buf=f)}\n')
    f.write('\n')
    f.write(f'Инфо о продуктах {df_products.info(buf=f)}\n')
    f.write('\n')
    f.write(f'Инфо о покупателей {df_customers.info(buf=f)}\n')
    f.write(f'\n\n\nКоличество ошибок в дате {incorrect_dates}')
    f.write('\n\n\n')
    f.write(f'Негативные в квантити {neg_quantity}\n')
    f.write(f'Негативные в квантити {neg_price}\n')
    f.write(f'Негативные в квантити {neg_quantity}\n')
    f.write('\n\n\n')
    f.write(f'Результаты проверки в {res_fin.shape[0]}\n')
    f.write(f'Результаты проверки в {res2_fin.shape[0]}\n')


def random_date():
    year = 2023
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f'{year}-{month:02}-{day:02}'

df_customers['age'] = df_customers['age'].fillna(df_customers['age'].mean())
df_customers['phone_number'] = df_customers['phone_number'].fillna(0)
df_customers['phone_number'] = df_customers['phone_number'].str.replace(r'[^0-9+]', '', regex=True)
dates = pd.to_datetime(df_customers['join_date'], errors='coerce')
df_customers['join_date'] = [pd.to_datetime(random_date()) if pd.isnull(x) else x for x in dates]
df_customers['last_purchase_date'] = [pd.to_datetime(random_date()) if pd.isnull(x) else x for x in dates]
df_customers.to_csv('customers_cleared.csv')

df_sales['promotion_id'] =  df_sales['promotion_id'].fillna(0)
dates = pd.to_datetime(df_sales['date'], errors='coerce')
df_sales['date'] = [pd.to_datetime(random_date()) if pd.isnull(x) else x for x in dates]
df_sales['quantity'] = df_sales['quantity'].abs()
df_sales['price'] = df_sales['price'].abs()
df_sales.to_csv('sales_transactions_cleared.csv')

df_sales['revenue'] = df_sales['quantity'] * df_sales['price']
df_sales['month_period'] = df_sales['date'].dt.to_period('M')

sales_by_month = df_sales.groupby('month_period').agg({'revenue': 'sum', 'transaction_id': 'count'}).rename(columns={'transaction_id': 'num_transactions'})
sales_by_month['avg_order_value'] = sales_by_month['revenue'] * sales_by_month['num_transactions']
print(sales_by_month)

print(sales_by_month.sort_values('revenue', ascending=False).head(3))

plt.figure(figsize=(10, 5))
plt.plot(sales_by_month.index.astype(str), sales_by_month['revenue'], marker='o', color='red', label='Доход в периоде')
plt.title('Выручка', fontsize=20)
plt.ylabel('выручка')
plt.xlabel('периоды')
plt.grid('both', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.savefig('Session1_SalesTrends_Revenue.pdf')
plt.close


plt.figure(figsize=(10, 5))
plt.plot(sales_by_month.index.astype(str), sales_by_month['avg_order_value'], marker='^', color='green', label='Средняя выручка в периоде')
plt.title('Средний Чек', fontsize=20)
plt.ylabel('Средний чек')
plt.xlabel('периоды')
plt.grid('both', alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('Session1_SalesTrends_AOV.pdf')
plt.close

print(f"..{'Большой отступ':^20}..")

df_merged = df_sales.merge(df_products, on='product_id', how='left')
products_stats = (df_merged.groupby(['product_id', 'product_name', 'category', 'cost'])
                .agg({'quantity': 'sum', 'revenue': 'sum'})
                .rename(columns={'quantity': 'total_quantity', 'revenue': 'total_price'})
                .reset_index()
)
print(products_stats.info())

products_stats['unit_margin'] = ((products_stats['total_price'] / products_stats['total_quantity']) - products_stats['cost'])
print(products_stats)

by_category = products_stats.groupby('category').agg({'unit_margin': 'mean'}).round(2)
print(by_category.sort_values('unit_margin', ascending=False).head(3))

plt.figure(figsize=(10, 5))
plt.bar(by_category.index.astype(str), by_category['unit_margin'], color='green', label='Средняя маржа')
plt.title('Средняя маржа по категориям', fontsize=20)
plt.ylabel('Категория')
plt.xlabel('Маржа')
plt.grid('both', alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('Session1_ProductPerformance.pdf')
plt.close

plt.figure(figsize=(10, 5))
plt.bar(by_category.index.astype(str), by_category['unit_margin'], color='green', label='Средняя маржа')
plt.title('Средняя маржа по категориям', fontsize=20)
plt.ylabel('Категория')
plt.xlabel('Маржа')
plt.grid('both', alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('Session1_ProductPerformance.pdf')
plt.close



with PdfPages("Session1_CustomerAnalysis.pdf") as pdf1:
    df_customers['age_group'] = pd.cut(df_customers['age'], bins=[17, 24, 34, 44, 200], labels=['18-24', '25-34', '35-44', '45+'])
    plt.figure(figsize=(10, 5))
    sns.histplot(df_customers['age_group'], color='blue')
    plt.title('Возрастные категории', fontsize=20)
    plt.ylabel('Категория')
    plt.xlabel('Маржа')
    plt.grid('both', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf1.savefig()
    plt.close() 

    percentage_of_gender = df_customers['gender'].value_counts(normalize=True).round(2)
    print(percentage_of_gender.info())
    gender_table_data = []
    for gender, pct in percentage_of_gender.items():
        gender_table_data.append([gender, f"{pct:.0%}" ]) # Форматируем как проценты (например, 60%)

    # Создаем холст для таблицы
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Скрываем оси графика, чтобы осталась только таблица
    ax.axis('off')
    ax.axis('tight')

    # Рисуем таблицу
    table_gender = ax.table(
        cellText=gender_table_data, 
        colLabels=['Пол', 'Доля (%)'], # Заголовки столбцов
        loc='center',                 # Расположение на холсте
        cellLoc='center',             # Выравнивание текста внутри ячеек
        colColours=['#f5f5f5', '#f5f5f5'] # Легкий фон заголовков
    )
    
    # Настройка стиля таблицы (шрифт, размер)
    table_gender.auto_set_font_size(False)
    table_gender.set_fontsize(14)
    table_gender.scale(1.2, 2.5) # Масштабирование по горизонтали и вертикали

    plt.title('Процентное распределение заказчиков по полу', fontsize=16, pad=30)
    
    pdf1.savefig(fig, bbox_inches='tight') # Сохраняем вторую страницу (Таблица Гендер)
    plt.close()



    avg_spend_by_loyalty = df_customers.groupby('member_status', as_index=False)['avg_order_value'].mean().round(2)
    
    # Сортируем для красоты (по алфавиту)
    avg_spend_by_loyalty = avg_spend_by_loyalty.sort_values('member_status')

    # Преобразуем данные в список списков для plt.table()
    # f"{val:,.2f}" добавит запятые как разделители тысяч и два знака после запятой
    loyalty_table_data = []
    for row in avg_spend_by_loyalty.values:
        loyalty_table_data.append([row[0], f"{row[1]:,.2f} $"])

    # Создаем холст для третьей таблицы
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.axis('tight')

    # Рисуем таблицу
    table_loyalty = ax.table(
        cellText=loyalty_table_data, 
        colLabels=['Уровень лояльности', 'Средние расходы'],
        loc='center', 
        cellLoc='center',
        colColours=['#f5f5f5', '#f5f5f5']
    )
    
    # Настройка стиля таблицы
    table_loyalty.auto_set_font_size(False)
    table_loyalty.set_fontsize(14)
    table_loyalty.scale(1.2, 2.5)

    plt.title('Средние расходы на одного заказчика по уровням лояльности', fontsize=16, pad=30)
    
    pdf1.savefig(fig, bbox_inches='tight') # Сохраняем третью страницу (Таблица Расходы)
    plt.close()

    print(f"Готово! Файл 'Session1_CustomerAnalysis.pdf' успешно создан.")

df_sales['total_check'] = df_sales['quantity'] * df_sales['price']
daily_sales = df_sales.groupby('date')['total_check'].sum().asfreq('D')

# 2. Автоматический подбор модели ARIMA
# seasonal=False, если мы не учитываем годовую сезонность (для 30 дней это ок)
# stepwise=True ускоряет поиск лучшей комбинации
model = ARIMA(daily_sales, order=(14, 1, 1))
model_fit = model.fit()

forecast_values = model_fit.forecast(steps=30)

# 4. Расчет MAE
# Получаем предсказания для тех дней, которые уже были в истории
fitted_values = model_fit.predict(start=0, end=len(daily_sales)-1)
mae = mean_absolute_error(daily_sales, fitted_values)
print(f"Средняя абсолютная погрешность (MAE): {mae:.2f}")

# 5. Формирование дат
last_date = daily_sales.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# 6. Сохранение в CSV
output_df = pd.DataFrame({
    'Date': forecast_dates.strftime('%Y-%m-%d'),
    'Predicted_Sales': forecast_values.values.round(2)
})

output_df.to_csv('Session1_SalesForecast.csv', index=False)
print("Файл Session1_SalesForecast.csv успешно создан!")

# =============================================================================
# БЛОК 5: АНАЛИЗ КЛИЕНТОВ И СЕГМЕНТАЦИЯ
# =============================================================================
# Согласно roadmap.md, Блок 5 посвящен анализу клиентов на основе данных из customers.csv
# и расчету дополнительных метрик из транзакций продаж.
# Мы рассчитаем RFM-метрики: Recency (давность последней покупки), Frequency (частота покупок),
# Monetary (денежная ценность клиента), а также обновим avg_purchase_value.

# Загружаем очищенные данные продаж для расчета метрик клиентов
# Это необходимо, потому что исходные данные в customers.csv могут быть устаревшими или рассчитанными иначе
df_sales_cleaned = pd.read_csv('sales_transactions_cleared.csv')

# Добавляем столбец revenue, если его нет (для совместимости с финальной сводкой)
# Revenue = quantity * price
if 'revenue' not in df_sales_cleaned.columns:
    df_sales_cleaned['revenue'] = df_sales_cleaned['quantity'] * df_sales_cleaned['price']

# Рассчитываем общую выручку (Monetary) для каждого клиента
# Группируем по customer_id и суммируем quantity * price для каждой транзакции клиента
total_revenue = df_sales_cleaned.groupby('customer_id').apply(lambda x: (x['quantity'] * x['price']).sum())

# Рассчитываем частоту покупок (Frequency) - количество транзакций на клиента
# Группируем по customer_id и считаем количество transaction_id
total_purchases = df_sales_cleaned.groupby('customer_id')['transaction_id'].count()

# Рассчитываем давность последней покупки (Recency) - дни с последней покупки
# Находим максимальную дату покупки для каждого клиента
last_purchase_dates = df_sales_cleaned.groupby('customer_id')['date'].max()
# Преобразуем в datetime, если не преобразовано
last_purchase_dates = pd.to_datetime(last_purchase_dates)
# Текущая дата - последняя дата в данных (или сегодняшняя, но для consistency используем max date)
current_date = pd.to_datetime(df_sales_cleaned['date']).max()
# Рассчитываем recency в днях
recency = (current_date - last_purchase_dates).dt.days

# Добавляем рассчитанные метрики в df_customers
# Используем map для сопоставления по customer_id
df_customers['total_purchases'] = df_customers['customer_id'].map(total_purchases)
df_customers['total_revenue'] = df_customers['customer_id'].map(total_revenue)
df_customers['recency_days'] = df_customers['customer_id'].map(recency)

# Рассчитываем среднюю стоимость покупки (avg_purchase_value)
# Это total_revenue / total_purchases
df_customers['avg_purchase_value'] = df_customers['total_revenue'] / df_customers['total_purchases']

# Заполняем NaN значения (если клиент не имеет покупок, но в данных все должны иметь)
df_customers['total_purchases'] = df_customers['total_purchases'].fillna(0)
df_customers['total_revenue'] = df_customers['total_revenue'].fillna(0)
df_customers['recency_days'] = df_customers['recency_days'].fillna(999)  # Большое число для клиентов без покупок
df_customers['avg_purchase_value'] = df_customers['avg_purchase_value'].fillna(0)

# Выводим статистику для проверки
print("Статистика по клиентам после расчета RFM:")
print(df_customers[['customer_id', 'total_purchases', 'total_revenue', 'recency_days', 'avg_purchase_value']].head(10))

# Сохраняем обновленные данные клиентов для дальнейшего использования
df_customers.to_csv('customers_with_rfm.csv', index=False)
print("Файл customers_with_rfm.csv успешно создан с обновленными метриками клиентов!")

# =============================================================================
# БЛОК 7: КЛАСТЕРИЗАЦИЯ И РЕКОМЕНДАЦИИ (KMEANS)
# =============================================================================
# Согласно roadmap.md, Блок 7 посвящен кластеризации клиентов на основе RFM-метрик
# для сегментации и персонализированных рекомендаций.
# Мы используем KMeans для группировки клиентов по Recency, Frequency, Monetary.

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Подготавливаем данные для кластеризации: выбираем RFM-метрики
# Recency_days, total_purchases (Frequency), total_revenue (Monetary)
rfm_data = df_customers[['recency_days', 'total_purchases', 'total_revenue']].copy()

# Обрабатываем пропуски (хотя мы их заполнили, на всякий случай)
rfm_data = rfm_data.fillna(0)

# Стандартизируем данные, чтобы все метрики были в одном масштабе
# Это важно для KMeans, так как алгоритм чувствителен к масштабу
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

# Определяем оптимальное количество кластеров с помощью метода локтя (Elbow Method)
# Вычисляем сумму квадратов расстояний для разных k
inertia = []
k_range = range(1, 11)  # Проверяем от 1 до 10 кластеров
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

# Строим график метода локтя для выбора k
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Метод локтя для определения количества кластеров')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Сумма квадратов расстояний (Inertia)')
plt.grid(True)
plt.savefig('Session1_ElbowMethod.pdf')
plt.close()
print("График метода локтя сохранен в Session1_ElbowMethod.pdf")

# На основе графика выбираем k=4 (типичное значение для RFM-сегментации: Champions, Loyal, At Risk, Lost)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df_customers['cluster'] = kmeans.fit_predict(rfm_scaled)

# Анализируем кластеры: средние значения RFM для каждого кластера
cluster_summary = df_customers.groupby('cluster')[['recency_days', 'total_purchases', 'total_revenue']].mean()
print("Средние значения RFM по кластерам:")
print(cluster_summary)

# Назначаем сегменты на основе кластеров (примерная интерпретация)
# Кластер 0: Высокая частота и выручка, низкая давность - Champions
# Кластер 1: Высокая давность, низкая активность - Lost
# Кластер 2: Средние значения - Loyal
# Кластер 3: Низкая частота, но недавние покупки - At Risk
# (Это приблизительно, нужно анализировать cluster_summary)
cluster_names = {0: 'Champions', 1: 'Lost', 2: 'Loyal', 3: 'At Risk'}
df_customers['segment'] = df_customers['cluster'].map(cluster_names)

# Генерируем рекомендации на основе сегментов
def get_recommendation(segment):
    if segment == 'Champions':
        return "Предлагать премиум-продукты и программы лояльности для удержания."
    elif segment == 'Loyal':
        return "Увеличивать частоту покупок через персонализированные предложения."
    elif segment == 'At Risk':
        return "Отправлять напоминания и скидки для reactivation."
    elif segment == 'Lost':
        return "Фокус на win-back кампаниях с большими скидками."
    else:
        return "Анализ сегмента в процессе."

df_customers['recommendation'] = df_customers['segment'].apply(get_recommendation)

# Выводим пример сегментации
print("Пример сегментации клиентов:")
print(df_customers[['customer_id', 'segment', 'recommendation']].head(10))

# Сохраняем данные с кластерами
df_customers.to_csv('customers_with_clusters.csv', index=False)
print("Файл customers_with_clusters.csv успешно создан с кластерами и рекомендациями!")

# Визуализируем кластеры (2D проекция для Recency и Frequency, игнорируя Monetary для простоты)
plt.figure(figsize=(10, 6))
for cluster in range(k_optimal):
    cluster_data = df_customers[df_customers['cluster'] == cluster]
    plt.scatter(cluster_data['recency_days'], cluster_data['total_purchases'], label=f'Кластер {cluster}: {cluster_names[cluster]}')
plt.title('Кластеризация клиентов по RFM (Recency vs Frequency)')
plt.xlabel('Recency (дни с последней покупки)')
plt.ylabel('Frequency (количество покупок)')
plt.legend()
plt.grid(True)
plt.savefig('Session1_CustomerClusters.pdf')
plt.close()
print("График кластеров сохранен в Session1_CustomerClusters.pdf")

# =============================================================================
# БЛОК 8: ПРОДВИНУТАЯ АНАЛИТИКА (CLTV, CHURN, ЭЛАСТИЧНОСТЬ)
# =============================================================================
# Согласно roadmap.md, Блок 8 включает расчет CLTV (Customer Lifetime Value),
# анализ Churn (оттока клиентов) и эластичности цен.
# CLTV = total_revenue * (средняя продолжительность жизни клиента)
# Churn: анализ на основе churn_status и RFM.
# Эластичность: как изменения цены влияют на quantity.

# Расчет CLTV (упрощенная версия)
# Предполагаем среднюю продолжительность жизни клиента = 2 года (на основе данных)
# CLTV = total_revenue * (avg_lifespan_years)
avg_lifespan_years = 2  # Можно рассчитать на основе данных, но для простоты фиксируем
df_customers['cltv'] = df_customers['total_revenue'] * avg_lifespan_years
print("CLTV рассчитан. Средний CLTV:", df_customers['cltv'].mean())

# Анализ Churn: процент оттока по сегментам
churn_rate_by_segment = df_customers.groupby('segment')['churn_status'].mean()
print("Процент оттока по сегментам:")
print(churn_rate_by_segment)

# Эластичность цен: рассчитываем на основе sales данных
# Эластичность = % изменение quantity / % изменение price
# Группируем по product_id и рассчитываем среднюю эластичность
product_elasticity = df_sales_cleaned.groupby('product_id').apply(
    lambda x: (x['quantity'].pct_change().mean() / x['price'].pct_change().mean()) if x['price'].nunique() > 1 else 0
).reset_index(name='elasticity')
product_elasticity = product_elasticity.merge(df_products[['product_id', 'product_name']], on='product_id', how='left')
print("Эластичность цен по продуктам:")
print(product_elasticity.head(10))

# Сохраняем продвинутую аналитику
df_customers.to_csv('customers_advanced_analytics.csv', index=False)
product_elasticity.to_csv('product_elasticity.csv', index=False)
print("Файлы customers_advanced_analytics.csv и product_elasticity.csv созданы!")

# =============================================================================
# БЛОК 9: ВИЗУАЛИЗАЦИЯ И ГЕНЕРАЦИЯ ОТЧЁТОВ (PDF/EXCEL)
# =============================================================================
# Согласно roadmap.md, Блок 9 - создание визуализаций и отчётов в PDF и Excel.

from matplotlib.backends.backend_pdf import PdfPages
import openpyxl

# Создаем PDF-отчёт с графиками
with PdfPages('Session1_FullReport.pdf') as pdf:
    # График 1: Выручка по месяцам (из Блока 3)
    plt.figure(figsize=(10, 5))
    plt.plot(sales_by_month.index.astype(str), sales_by_month['revenue'], marker='o', color='red', label='Доход в периоде')
    plt.title('Выручка по месяцам', fontsize=20)
    plt.ylabel('Выручка')
    plt.xlabel('Периоды')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    pdf.savefig()
    plt.close()

    # График 2: Кластеризация клиентов (из Блока 7)
    plt.figure(figsize=(10, 6))
    for cluster in range(k_optimal):
        cluster_data = df_customers[df_customers['cluster'] == cluster]
        plt.scatter(cluster_data['recency_days'], cluster_data['total_purchases'], label=f'Кластер {cluster}: {cluster_names[cluster]}')
    plt.title('Кластеризация клиентов по RFM')
    plt.xlabel('Recency (дни)')
    plt.ylabel('Frequency (покупки)')
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # График 3: Распределение CLTV
    plt.figure(figsize=(8, 5))
    plt.hist(df_customers['cltv'], bins=20, color='green', alpha=0.7)
    plt.title('Распределение CLTV клиентов')
    plt.xlabel('CLTV')
    plt.ylabel('Количество клиентов')
    plt.grid(True)
    pdf.savefig()
    plt.close()

print("PDF-отчёт Session1_FullReport.pdf создан!")

# Создаем Excel-отчёт
with pd.ExcelWriter('Session1_FullReport.xlsx', engine='openpyxl') as writer:
    # Лист 1: Ежемесячные продажи
    sales_by_month.to_excel(writer, sheet_name='Monthly Sales')
    # Лист 2: Клиенты с RFM и кластерами
    df_customers.to_excel(writer, sheet_name='Customers RFM Clusters')
    # Лист 3: Эластичность продуктов
    product_elasticity.to_excel(writer, sheet_name='Product Elasticity')
    # Лист 4: Сводка кластеров
    cluster_summary.to_excel(writer, sheet_name='Cluster Summary')

print("Excel-отчёт Session1_FullReport.xlsx создан!")

# =============================================================================
# БЛОК 10: ФИНАЛЬНАЯ СБОРКА: ПОЛНЫЙ ПАЙПЛАЙН
# =============================================================================
# Согласно roadmap.md, Блок 10 - объединение всех блоков в один скрипт.
# Мы уже реализовали полный пайплайн в одном файле analysis_task.py.
# Для финальной сборки добавляем итоговую сводку.

print("\n" + "="*50)
print("ФИНАЛЬНАЯ СВОДКА АНАЛИЗА ПЕКАРНИ")
print("="*50)
print(f"Общий доход: {df_sales_cleaned['revenue'].sum():.2f}")
print(f"Средний чек: {df_sales_cleaned['revenue'].sum() / len(df_sales_cleaned):.2f}")
print(f"Количество клиентов: {len(df_customers)}")
print(f"Средний CLTV: {df_customers['cltv'].mean():.2f}")
print(f"Процент оттока: {df_customers['churn_status'].mean()*100:.1f}%")
print("Все файлы и графики сохранены в текущей директории.")
print("Анализ завершен успешно!")
print("="*50)