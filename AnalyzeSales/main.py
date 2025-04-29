import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

plt.style.use('ggplot')

sales_df = pd.read_csv('sales_data.csv')
print("\nDataset size:", sales_df.shape)

pprint(sales_df.head())
pprint(sales_df.info())
pprint(sales_df.describe())

print("\nData conversion...")
sales_df['Revenue'] = sales_df['Quantity'] * sales_df['Price']
sales_df['Date'] = pd.to_datetime(sales_df['Date'])
sales_df['Month'] = sales_df['Date'].dt.month_name()
sales_df['DayOfWeek'] = sales_df['Date'].dt.day_name()

print("\nRevenue by product category:")
category_sales = sales_df.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
pprint(category_sales)

print("\nRevenue by month:")
sales_df['Month_num'] = sales_df['Date'].dt.month
monthly_sales = sales_df.groupby(['Month_num', 'Month'])['Revenue'].sum().reset_index()
monthly_sales = monthly_sales.sort_values('Month_num')
pprint(monthly_sales)

print("\nTop 5 best-selling products:")
top_products = sales_df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(5)
pprint(top_products)

# Graph 1: Sales by category
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.bar(category_sales.index, category_sales.values, color='skyblue')
plt.title('Sales by product category')
plt.xticks(rotation=45)
plt.ylabel('Quantity')

# Graph 2: Sales for the month
plt.figure(figsize=(12, 6))
plt.bar(monthly_sales['Month'], monthly_sales['Revenue'], color='skyblue')
plt.title('Monthly sales')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Graph 3: Top 5 products
plt.figure(figsize=(10, 6))
plt.bar(top_products.index, top_products.values, color='teal')
plt.title('Top 5 best-selling products')
plt.xlabel('Product')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nCalculation of key indicators:")
total_sales = sales_df['Quantity'].sum()
total_revenue = sales_df['Revenue'].sum()
top_category = category_sales.index[0]
best_month = monthly_sales.loc[monthly_sales['Revenue'].idxmax(), 'Month']
top_product = top_products.index[0]

print(f"Total sales: {total_sales}")
print(f"Total revenue: ${total_revenue:.2f}")
print(f"Most popular category: {top_category}")
print(f"Best month: {best_month}")
print(f"Best-selling product: {top_product}")