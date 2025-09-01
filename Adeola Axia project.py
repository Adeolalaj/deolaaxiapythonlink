#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[32]:


import pandas as pd        
import matplotlib.pyplot as plt


# # Load the Datasets

# In[8]:


dataset = pd.read_excel("Python_SalesData.xlsx")
dataset


# In[21]:


import pandas as pd

dataset = pd.read_excel("Python_SalesData.xlsx", header=1)


# In[48]:


print(dataset.columns.tolist())


# In[47]:


print(dataset.head())


# In[46]:


dataset.info()


# # Handle Missing Values

# In[45]:


dataset.isnull().sum()


# # Remove Duplicates

# In[55]:


dataset.duplicated().any()


# In[54]:


duplicates_count = dataset.duplicated(subset=[
    'Order ID', 'Date', 'Product', 'Price', 'Quantity',
    'Purchase Type', 'Payment Method', 'Manager', 'City'
]).sum()

print("Number of duplicate rows:", duplicates_count)


# In[51]:


dataset.duplicated().sum()


# # Removing Extra Spaces

# In[44]:


dataset.columns = dataset.columns.str.strip()
dataset.head()


# # Convert to appropriate datatypes

# In[49]:


# Convert Order ID to integer
dataset['Order ID'] = pd.to_numeric(dataset['Order ID'], errors='coerce').astype('Int64')

# Keep Quantity as float (continuous measurement)
dataset['Quantity'] = pd.to_numeric(dataset['Quantity'], errors='coerce').astype('float')

# Convert Price to float
dataset['Price'] = pd.to_numeric(dataset['Price'], errors='coerce')

# Convert Date to datetime
dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')

# Convert selected columns to category
for col in ['Purchase Type', 'Payment Method', 'Manager', 'City']:
    dataset[col] = dataset[col].astype('category')

# Product as string
dataset['Product'] = dataset['Product'].astype('string')

#  Check result
print(dataset.dtypes)


# In[27]:


dataset = dataset.drop(columns=['Unnamed: 0'])


# In[56]:


# Create Revenue column
dataset["Revenue"] = dataset["Price"] * dataset["Quantity"]
dataset.head()


# # Cleaned Datasets

# In[57]:


dataset.head()


# # Visualizations

# In[58]:


#  Enhanced Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_theme(style="whitegrid")

# -----------------------------
# 1. DESCRIPTIVE STATISTICS
# -----------------------------
print("🔹 Dataset Shape:", dataset.shape)
print("\n🔹 Column Data Types:\n", dataset.dtypes)
print("\n🔹 Missing Values:\n", dataset.isna().sum())

print("\n🔹 Summary Statistics (Numeric):\n", dataset[['Price','Quantity']].describe())
print("\n🔹 Summary Statistics (Categorical):\n", dataset.describe(include='category'))

print("\n🔹 Skewness:\n", dataset[['Price','Quantity']].skew())
print("\n🔹 Kurtosis:\n", dataset[['Price','Quantity']].kurt())

# -----------------------------
# 2. DISTRIBUTIONS
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(dataset['Quantity'], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Quantity")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='Product', y='Price', data=dataset, palette="Set2")
plt.title("Price Distribution by Product")
plt.xticks(rotation=90)
plt.show()

# -----------------------------
# 3. GROUP-WISE SUMMARIES
# -----------------------------
dataset['Total_Sale'] = dataset['Price'] * dataset['Quantity']

# Product summary
product_summary = dataset.groupby('Product').agg({
    'Price': 'mean',
    'Quantity': 'mean',
    'Order ID': 'count'
}).rename(columns={'Order ID':'Num_Orders'})

print("\n🔹 Product Summary:\n", product_summary.head())

# Barplot for sales by product
plt.figure(figsize=(12,6))
sns.barplot(
    x='Product',
    y='Total_Sale',
    data=dataset,
    estimator=sum,
    palette="viridis"
)
plt.title("Total Sales by Product")
plt.xticks(rotation=90)
plt.show()

# City summary
city_summary = dataset.groupby('City').agg({
    'Total_Sale': 'sum',
    'Quantity': 'sum'
}).sort_values('Total_Sale', ascending=False)

print("\n🔹 City Summary:\n", city_summary)

plt.figure(figsize=(10,6))
sns.barplot(
    x=city_summary.index,
    y=city_summary['Total_Sale'],
    palette="coolwarm"
)
plt.title("Total Sales by City")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 4. TRENDS OVER TIME
# -----------------------------
sales_trend = dataset.groupby('Date')['Total_Sale'].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=sales_trend, x='Date', y='Total_Sale', marker="o")
plt.title("Daily Sales Trend")
plt.ylabel("Total Sales")
plt.xlabel("Date")
plt.show()

# -----------------------------
# 5. CROSS-TABS / PIVOTS
# -----------------------------
pivot = pd.pivot_table(
    dataset,
    values='Total_Sale',
    index='Payment Method',
    columns='Purchase Type',
    aggfunc='sum',
    fill_value=0
)

print("\n🔹 Sales Pivot (Payment Method vs Purchase Type):\n", pivot)

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")
plt.title("Sales Heatmap: Payment Method vs Purchase Type")
plt.show()


# # Project Questions 

# In[ ]:


1. What was the Most Preferred Payment Method


# In[59]:


# Count how many times each payment method was used
payment_counts = dataset['Payment Method'].value_counts()

print("Payment Method Counts:")
print(payment_counts)

# Plot it
payment_counts.plot(kind="bar", title="Most Preferred Payment Method", ylabel="Number of Orders", xlabel="Payment Method")


# In[60]:


get_ipython().set_next_input('2. Which one was the Most Selling Product by Quantity and by Revenue');get_ipython().run_line_magic('pinfo', 'Revenue')


# In[ ]:


2. Which one was the Most Selling Product by Quantity and by Revenue


# In[35]:


# 1. Most selling product by Quantity
product_quantity = dataset.groupby("Product")["Quantity"].sum().sort_values(ascending=False)
print("Most Selling Product by Quantity:")
print(product_quantity.head(1))  # Top product

# 2. Most selling product by Revenue (Revenue = Price × Quantity)
dataset["Revenue"] = dataset["Price"] * dataset["Quantity"]

product_revenue = dataset.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
print("\nMost Selling Product by Revenue:")
print(product_revenue.head(1))  # Top product

# Plot both for comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
product_quantity.head(5).plot(kind="bar", title="Top 5 Products by Quantity")
plt.show()

plt.figure(figsize=(10,5))
product_revenue.head(5).plot(kind="bar", title="Top 5 Products by Revenue")
plt.show()


# In[61]:


get_ipython().set_next_input('3. Which City had maximum revenue, and Which Manager earned maximum revenue');get_ipython().run_line_magic('pinfo', 'revenue')


# In[ ]:


3. Which City had maximum revenue, and Which Manager earned maximum revenue


# In[36]:


# 1. City with maximum revenue
city_revenue = dataset.groupby("City")["Revenue"].sum().sort_values(ascending=False)
print("City with Maximum Revenue:")
print(city_revenue.head(1))

# 2. Manager with maximum revenue
manager_revenue = dataset.groupby("Manager")["Revenue"].sum().sort_values(ascending=False)
print("\nManager with Maximum Revenue:")
print(manager_revenue.head(1))

# Plot top 5 for each
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
city_revenue.head(5).plot(kind="bar", title="Top 5 Cities by Revenue")
plt.show()

plt.figure(figsize=(10,5))
manager_revenue.head(5).plot(kind="bar", title="Top 5 Managers by Revenue")
plt.show()


# In[ ]:


get_ipython().set_next_input('4. What was the Average Revenue');get_ipython().run_line_magic('pinfo', 'Revenue')


# In[64]:


# Average Revenue across all transactions
average_revenue = dataset["Revenue"].mean()
print("Average Revenue:", average_revenue)


# In[ ]:


get_ipython().set_next_input('5. What was the Average Revenue of November and December');get_ipython().run_line_magic('pinfo', 'December')


# In[65]:


# Filter November and December
nov_dec_data = dataset[dataset["Date"].dt.month.isin([11, 12])]

# Average revenue for Nov and Dec combined
avg_revenue_nov_dec = nov_dec_data["Revenue"].mean()
print("Average Revenue (Nov & Dec combined):", avg_revenue_nov_dec)

# separate averages:
avg_revenue_by_month = nov_dec_data.groupby(nov_dec_data["Date"].dt.month)["Revenue"].mean()
print("\nAverage Revenue by Month:")
print(avg_revenue_by_month)


# In[ ]:


get_ipython().set_next_input('6. What was the Standard Deviation of Revenue and Quantity');get_ipython().run_line_magic('pinfo', 'Quantity')


# In[66]:


# Standard Deviation of Revenue and Quantity
std_revenue = dataset["Revenue"].std()
std_quantity = dataset["Quantity"].std()

print("Standard Deviation of Revenue:", std_revenue)
print("Standard Deviation of Quantity:", std_quantity)


# In[ ]:


get_ipython().set_next_input('7. What was the Variance of Revenue and Quantity');get_ipython().run_line_magic('pinfo', 'Quantity')


# In[67]:


# Variance of Revenue and Quantity
var_revenue = dataset["Revenue"].var()
var_quantity = dataset["Quantity"].var()

print("Variance of Revenue:", var_revenue)
print("Variance of Quantity:", var_quantity)


# In[ ]:


get_ipython().set_next_input('8. Was the revenue increasing or decreasing over the time');get_ipython().run_line_magic('pinfo', 'time')


# In[68]:


# Group revenue by month
monthly_revenue = dataset.groupby(dataset["Date"].dt.to_period("M"))["Revenue"].sum()

print("Monthly Revenue Trend:")
print(monthly_revenue)

#plot the trend
import matplotlib.pyplot as plt

monthly_revenue.plot(kind="line", marker="o", figsize=(8,5))
plt.title("Revenue Trend Over Time")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.grid(True)
plt.show()


# In[ ]:


get_ipython().set_next_input("9. What was the Average 'Quantity Sold' & 'Average Revenue' for each product");get_ipython().run_line_magic('pinfo', 'product')


# In[69]:


# Average Quantity and Revenue per Product
avg_product_stats = dataset.groupby("Product").agg({
    "Quantity": "mean",
    "Revenue": "mean"
}).reset_index()

print("Average Quantity Sold & Average Revenue per Product:")
print(avg_product_stats)


# In[ ]:


get_ipython().set_next_input('10. What was the total number of orders or sales made');get_ipython().run_line_magic('pinfo', 'made')


# In[70]:


# Total number of orders (rows in dataset)
total_orders = dataset.shape[0]

print("Total Number of Orders:", total_orders)


# In[ ]:




