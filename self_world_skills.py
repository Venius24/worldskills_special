import csv


customers_file = 'customers.csv'
sales_file = 'sales_transactions.csv'
products_file = 'products.csv'
with open(customers_file, encoding='utf-8') as c:
    customers_reader = csv.reader(c)
    rows = list(customers_reader)
    for i in range(5   ):
        print(rows[i])
    
    for i in range(5):
        print(type(rows[i][0])) 