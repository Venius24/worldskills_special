import pandas as pd

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
    f.write(f'Результаты проверки в {res_fin}\n')
    f.write(f'Результаты проверки в {res2_fin}\n')