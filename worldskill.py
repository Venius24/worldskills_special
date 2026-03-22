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
import random

warnings.filterwarnings('ignore')

# ЧАСТЬ 1: Загрузка и Исследование Данных

print("Загрузка данных...")
df_sales = pd.read_csv('sales_transactions.csv')
df_products = pd.read_csv('products.csv')
df_customers = pd.read_csv('customers.csv')

with open('Session1_DataExploration.txt', 'w', encoding='utf-8') as f:
    f.write("Отчет по исследованию данных\n\n")
    
    f.write("1. Типы данных\n")
    for name, df in [("Продажи", df_sales), ("Продукты", df_products), ("Клиенты", df_customers)]:
        f.write(f"\nДатасет: {name}\n")
        f.write(str(df.dtypes) + "\n")

    f.write("\n2. Аномалии\n")
    
    temp_dates = pd.to_datetime(df_sales['date'], errors='coerce')
    invalid_dates_count = temp_dates.isna().sum()
    f.write(f"Неправильная информация в следующих датах: {invalid_dates_count}\n")
    
    neg_qty = (df_sales['quantity'] < 0).sum()
    neg_price = (df_sales['price'] < 0).sum()
    f.write(f"Отрицательные количества: {neg_qty}\n")
    f.write(f"Отрицательные цены: {neg_price}\n")
    
    invalid_prod_ids = (~df_sales['product_id'].isin(df_products['product_id'])).sum()
    invalid_cust_ids = (~df_sales['customer_id'].isin(df_customers['customer_id'])).sum()
    f.write(f"Недопустимые ID продуктов в продажах: {invalid_prod_ids}\n")
    f.write(f"Недопустимые ID клиентов в продажах: {invalid_cust_ids}\n")

print("Файл Session1_DataExploration.txt создан.")

# ЧАСТЬ 2: Очистка и преобразование данных

avg_age = df_customers['age'].mean()
df_customers['age'] = df_customers['age'].fillna(avg_age)
df_customers['phone_number'] = df_customers['phone_number'].fillna("0")

df_customers['phone_number'] = df_customers['phone_number'].astype(str).str.replace(r'[^0-9+]', '', regex=True)

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

df_sales['month_period'] = df_sales['date'].dt.to_period('M')
df_sales['revenue'] = df_sales['quantity'] * df_sales['price']

monthly_stats = df_sales.groupby('month_period').agg({
    'revenue': 'sum',
    'transaction_id': 'count',
}).rename(columns={'transaction_id': 'num_transactions'})

monthly_stats['avg_order_value'] = monthly_stats['revenue'] / monthly_stats['num_transactions']

top_3_months = monthly_stats.sort_values(by='revenue', ascending=False).head(3)

print("\nТОП-3 месяца по продажам")
print(top_3_months[['revenue']])

plt.figure(figsize=(10, 5))
plt.plot(monthly_stats.index.astype(str), monthly_stats['revenue'], marker='o', color='skyblue')
plt.title('Ежемесячный доход (Total Revenue)', fontsize=14)
plt.xlabel('Месяц')
plt.ylabel('Доход (€)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('Session1_SalesTrends_Revenue.pdf')
plt.close()

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

# ЧАСТЬ 4: Анализ продуктов

df_merged = df_sales.merge(df_products, on='product_id', how='left')

product_perf = df_merged.groupby(['product_id', 'product_name', 'category']).agg({
    'quantity': 'sum',
    'revenue': 'sum'
}).reset_index()

product_perf = product_perf.merge(df_products[['product_id', 'cost']], on='product_id')
product_perf['unit_margin'] = (product_perf['revenue'] / product_perf['quantity']) - product_perf['cost']

cat_perf = product_perf.groupby('category')['revenue'].sum()

top_3_products = product_perf.sort_values(by='quantity', ascending=False).head(3)

print("\nТОП-3 продукта по количеству продаж")
print(top_3_products[['product_name', 'quantity', 'revenue']])

plt.figure(figsize=(8, 6))
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

bins = [17, 24, 34, 44, 200]
labels = ['18-24', '25-34', '35-44', '45+']
df_customers['age_group'] = pd.cut(df_customers['age'], bins=bins, labels=labels)

plt.figure(figsize=(8, 6))
sns.histplot(df_customers['age_group'], kde=False, color='#FF9800')
plt.title('Распределение клиентов по возрасту', fontsize=14)
plt.xlabel('Возрастная группа')
plt.ylabel('Количество клиентов')
plt.tight_layout()
plt.savefig('Session1_CustomerAnalysis_Age.pdf')
plt.close()

gender_dist = df_customers['gender'].value_counts(normalize=True) * 100
gender_table = pd.DataFrame(gender_dist).reset_index().rename(columns={'index': 'gender', 'gender': 'percentage'})
loyalty_spend = df_customers.groupby('member_status')['total_spend'].mean().reset_index()

print("\n--- ОТЧЕТ: Анализ заказчиков ---")
print("\nРаспределение по полу (%):\n", gender_table)
print("\nСредние расходы по статусу лояльности:\n", loyalty_spend)

df_sales_cleaned = pd.read_csv('sales_transactions_cleaned.csv')
df_sales_cleaned['date'] = pd.to_datetime(df_sales_cleaned['date'])
df_sales_cleaned['revenue'] = df_sales_cleaned['quantity'] * df_sales_cleaned['price']

daily_sales = df_sales_cleaned.groupby('date')['revenue'].sum()
daily_sales.index = pd.DatetimeIndex(daily_sales.index)
daily_sales = daily_sales.resample('D').sum()

model = ARIMA(daily_sales, order=(5, 1, 0))
model_fit = model.fit()

forecast_result = model_fit.get_forecast(steps=30)
forecast_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=30)
predicted_values = forecast_result.predicted_mean

mae = mean_absolute_error(daily_sales, model_fit.fittedvalues)

df_forecast = pd.DataFrame({
    'Date': forecast_dates.strftime('%Y-%m-%d'),
    'Predicted_Sales': predicted_values.values.round(2)
})
df_forecast.to_csv('Session1_SalesForecast.csv', index=False)
print(f"\nФайл Session1_SalesForecast.csv создан. (MAE: {mae:.2f})")

cust_features = df_sales_cleaned.groupby('customer_id').agg(
    total_purchases=('transaction_id', 'count'),
    total_revenue=('revenue', 'sum')
)
cust_features['avg_purchase_value'] = cust_features['total_revenue'] / cust_features['total_purchases']

X = cust_features[['total_purchases', 'avg_purchase_value']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cust_features['cluster_label'] = kmeans.fit_predict(X_scaled) + 1

df_merged_cust = df_sales_cleaned.merge(cust_features[['cluster_label']], on='customer_id')
cluster_top_products = {}

for cluster in [1, 2, 3]:
    top_prods = df_merged_cust[df_merged_cust['cluster_label'] == cluster]['product_id'].value_counts().head(10).index.tolist()
    cluster_top_products[cluster] = top_prods

recommendations = []
all_prod_ids = set(df_products['product_id'].unique())

for cust_id in cust_features.index:
    cluster = cust_features.loc[cust_id, 'cluster_label']
    bought_products = set(df_sales_cleaned[df_sales_cleaned['customer_id'] == cust_id]['product_id'].unique())
    
    candidates = [p for p in cluster_top_products[cluster] if p not in bought_products]
    
    if len(candidates) < 3:
        unbought_overall = list(all_prod_ids - bought_products)
        random.shuffle(unbought_overall)
        candidates.extend(unbought_overall)
    
    recs = candidates[:3]
    recommendations.append([cust_id, cluster, recs[0], recs[1], recs[2]])

df_recs = pd.DataFrame(recommendations, columns=['customer_id', 'cluster_label', 'rec_1', 'rec_2', 'rec_3'])
df_recs.to_csv('Session5_Segmentation_and_Recommendations.csv', index=False)
print("Файл Session5_Segmentation_and_Recommendations.csv создан.")

prod_stats = df_merged.groupby('product_id').agg(
    total_quantity_sold=('quantity', 'sum'),
    total_revenue=('revenue', 'sum')
).reset_index()

costs = df_products.set_index('product_id')['cost']
prod_stats['total_cost'] = prod_stats['total_quantity_sold'] * prod_stats['product_id'].map(costs)

prod_stats['profit_margin'] = (prod_stats['total_revenue'] - prod_stats['total_cost']) / prod_stats['total_revenue']

prod_stats[['product_id', 'total_quantity_sold', 'total_revenue', 'profit_margin']].round(2).to_csv('Session5_Product_Performance.csv', index=False)

ped_data = []

for pid in df_products['product_id']:
    sub = df_sales_cleaned[df_sales_cleaned['product_id'] == pid]
    if sub.empty: continue
    
    price_summary = sub.groupby('price')['quantity'].sum().reset_index()
    
    if len(price_summary) >= 2:
        p1, q1 = price_summary.iloc[0]['price'], price_summary.iloc[0]['quantity']
        p2, q2 = price_summary.iloc[-1]['price'], price_summary.iloc[-1]['quantity']
        
        pct_change_q = (q2 - q1) / q1 if q1 != 0 else 0
        pct_change_p = (p2 - p1) / p1 if p1 != 0 else 0
        
        ped = pct_change_q / pct_change_p if pct_change_p != 0 else 0
    else:
        ped = 0

    if abs(ped) > 1:
        suggestion = -5
    else:
        suggestion = 5
        
    ped_data.append([pid, round(ped, 2), f"{suggestion}%"])

df_price_analysis = pd.DataFrame(ped_data, columns=['product_id', 'price_elasticity_of_demand', 'suggested_price_change'])
df_price_analysis.to_csv('Session5_Price_Analysis.csv', index=False)
print("Файлы Session5_Product_Performance.csv и Session5_Price_Analysis.csv созданы.")

cltv_stats = df_sales_cleaned.groupby('customer_id').agg(
    avg_purchase_value=('revenue', 'mean'),
    total_transactions=('transaction_id', 'count'),
    active_period=('date', lambda x: (x.max() - x.min()).days)
).reset_index()

cltv_stats['months_active'] = cltv_stats['active_period'] / 30
cltv_stats['months_active'] = cltv_stats['months_active'].apply(lambda x: max(x, 1))
cltv_stats['frequency_monthly'] = cltv_stats['total_transactions'] / cltv_stats['months_active']

cltv_stats['cltv'] = cltv_stats['avg_purchase_value'] * cltv_stats['frequency_monthly'] * 36

cltv_stats[['customer_id', 'cltv']].round(2).to_csv('Session1_CLTV.csv', index=False)
print("Файл Session1_CLTV.csv создан.")

df_churn = df_customers[['customer_id', 'churn_status']].merge(cltv_stats[['customer_id', 'cltv']], on='customer_id')

churn_rate = (df_churn['churn_status'].sum() / len(df_churn)) * 100

avg_cltv_churned = df_churn[df_churn['churn_status'] == 1]['cltv'].mean()
avg_cltv_active = df_churn[df_churn['churn_status'] == 0]['cltv'].mean()

df_churn_report = pd.DataFrame([{
    'churn_rate': round(churn_rate, 2),
    'avg_cltv_churned': round(avg_cltv_churned, 2),
    'avg_cltv_active': round(avg_cltv_active, 2)
}])

df_churn_report.to_csv('Session1_Churn_Analysis.csv', index=False)
print("Файл Session1_Churn_Analysis.csv создан.")
print("\nВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ УСПЕШНО.")