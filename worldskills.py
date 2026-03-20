import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# ==========================================
# ЧАСТЬ 1: Загрузка и Исследование Данных [cite: 40-60]
# ==========================================
print(">>> НАЧАЛО: Загрузка данных...")
df_sales = pd.read_csv('sales_transactions.csv')
df_products = pd.read_csv('products.csv')
df_customers = pd.read_csv('customers.csv')

# --- Формирование отчета Session1_DataExploration.txt ---
with open('Session1_DataExploration.txt', 'w', encoding='utf-8') as f:
    f.write("Отчет по исследованию данных\n\n")
    
    # 1. Типы данных [cite: 53]
    f.write("1. Типы данных\n")
    for name, df in [("Продажи", df_sales), ("Продукты", df_products), ("Клиенты", df_customers)]:
        f.write(f"\nДатасет: {name}\n")
        f.write(str(df.dtypes) + "\n")

    # 2. Несоответствия и аномалии [cite: 54]
    f.write("\n2. Аномалии\n")
    
    # Недопустимые даты (пример: 2023-14-01) [cite: 55-56]
    # Пытаемся конвертировать, считаем ошибки (NaT)
    temp_dates = pd.to_datetime(df_sales['date'], errors='coerce')
    invalid_dates_count = temp_dates.isna().sum()
    f.write(f"Неправильная информация в следующих датах: {invalid_dates_count}\n")
    
    # Отрицательные значения [cite: 57]
    neg_qty = (df_sales['quantity'] < 0).sum()
    neg_price = (df_sales['price'] < 0).sum()
    f.write(f"Отрицательные количества: {neg_qty}\n")
    f.write(f"Отрицательные цены: {neg_price}\n")
    
    # Недопустимые ID [cite: 58]
    invalid_prod_ids = (~df_sales['product_id'].isin(df_products['product_id'])).sum()
    invalid_cust_ids = (~df_sales['customer_id'].isin(df_customers['customer_id'])).sum()
    f.write(f"Недопустимые ID продуктов в продажах: {invalid_prod_ids}\n")
    f.write(f"Недопустимые ID клиентов в продажах: {invalid_cust_ids}\n")

print("Файл Session1_DataExploration.txt создан.")


# ЧАСТЬ 2: Очистка и преобразование данных [cite: 61-78]

# 1. Customers: Заполнение пропусков [cite: 66, 67]
avg_age = df_customers['age'].mean()
df_customers['age'] = df_customers['age'].fillna(avg_age)
df_customers['phone_number'] = df_customers['phone_number'].fillna("0")

# 2. Customers: Стандартизация телефона (оставить только цифры и +) [cite: 73]
# Используем регулярное выражение [^0-9+] означает "все, кроме цифр и плюса"
df_customers['phone_number'] = df_customers['phone_number'].astype(str).str.replace(r'[^0-9+]', '', regex=True)

# 3. Customers: Даты [cite: 70]
df_customers['join_date'] = pd.to_datetime(df_customers['join_date'])
df_customers['last_purchase_date'] = pd.to_datetime(df_customers['last_purchase_date'])

df_sales['promotion_id'] = df_sales['promotion_id'].fillna("0")

df_sales['date'] = pd.to_datetime(df_sales['date'], errors='coerce')
df_sales = df_sales.dropna(subset=['date'])

df_sales['quantity'] = df_sales['quantity'].abs()
df_sales['price'] = df_sales['price'].abs()

df_customers.to_csv('customers_cleaned.csv', index=False)
df_sales.to_csv('sales_transactions_cleaned.csv', index=False)
print("Файлы customers_cleaned.csv и sales_transactions_cleaned.csv созданы.")

# ЧАСТЬ 3: Анализ тенденций продаж

# Агрегация по месяцам
df_sales['month_period'] = df_sales['date'].dt.to_period('M')
df_sales['revenue'] = df_sales['quantity'] * df_sales['price']

monthly_stats = df_sales.groupby('month_period').agg({
    'revenue': 'sum',
    'transaction_id': 'count',
}).rename(columns={'transaction_id': 'num_transactions'})

monthly_stats['avg_order_value'] = monthly_stats['revenue'] / monthly_stats['num_transactions']

# Топ-3 месяца [cite: 85]
top_3_months = monthly_stats.sort_values(by='revenue', ascending=False).head(3)

print("\nТОП-3 месяца по продажам")
print(top_3_months[['revenue']])

# ... (Код до этого момента уже выполнен)

# --- Визуализация: Общий ежемесячный доход (Revenue) ---
plt.figure(figsize=(10, 5))
plt.plot(monthly_stats.index.astype(str), monthly_stats['revenue'], marker='o', color='skyblue')
plt.title('Ежемесячный доход (Total Revenue)', fontsize=14)
plt.xlabel('Месяц')
plt.ylabel('Доход (€)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('Session1_SalesTrends_Revenue.pdf') 
plt.close() # Закрываем график, чтобы не мешал в консоли

# --- Визуализация: Средняя стоимость заказа (AOV) ---
plt.figure(figsize=(10, 5))
plt.plot(monthly_stats.index.astype(str), monthly_stats['avg_order_value'], marker='o', color='lightcoral')
plt.title('Средняя стоимость заказа (AOV)', fontsize=14)
plt.xlabel('Месяц')
plt.ylabel('AOV (€)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('Session1_SalesTrends_AOV.pdf')
plt.close()

print("Файлы Session1_SalesTrends_Revenue.pdf и Session1_SalesTrends_AOV.pdf созданы.")
print(f"Самые доходные месяцы: \n{top_3_months[['revenue']]}")

# ЧАСТЬ 4: Анализ продуктов [cite: 93-106]

# Объединение продаж с продуктами
df_merged = df_sales.merge(df_products, on='product_id', how='left')

# Расчет метрик по продуктам
product_perf = df_merged.groupby(['product_id', 'product_name', 'category']).agg({
    'quantity': 'sum',
    'revenue': 'sum'
}).reset_index()

# Добавляем себестоимость и маржу для каждого продукта (из справочника)
product_perf = product_perf.merge(df_products[['product_id', 'cost']], on='product_id')
# Маржа (прибыль/выручка) или абсолютная прибыль? 
# [cite: 98] говорит "цена - себестоимость" (unit margin), 
# но [cite: 187] говорит "(total_revenue - total_cost) / total_revenue". Используем данные для отчета PDF.
product_perf['unit_margin'] = (product_perf['revenue'] / product_perf['quantity']) - product_perf['cost']

# Доход по категориям [cite: 99]
cat_perf = product_perf.groupby('category')['revenue'].sum()

# Топ-3 продукта по количеству [cite: 101]
top_3_products = product_perf.sort_values(by='quantity', ascending=False).head(3)

print("\nТОП-3 продукта по количеству продаж")
print(top_3_products[['product_name', 'quantity', 'revenue']])

# ... (Код до этого момента уже выполнен)

# --- Визуализация: Доход по категориям (Category Revenue) ---
plt.figure(figsize=(8, 6))
# Используем cat_perf, рассчитанный ранее
cat_perf.sort_values(ascending=False).plot(kind='bar', color=['#4CAF50', '#FFC107', '#2196F3'])
plt.title('Доход по категориям продуктов', fontsize=14)
plt.xlabel('Категория')
plt.ylabel('Доход (€)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('Session1_ProductPerformance_Category.pdf')
plt.close()

print("Файл Session1_ProductPerformance_Category.pdf создан.")
print(f"ТОП-3 продукта по количеству продаж: \n{top_3_products[['product_name', 'quantity', 'revenue']]}")
