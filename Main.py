

import pandas as pd
import os
import numpy as np
import seaborn as sns
from matplotlib import pylab as plt
!pip install scikit-learn


import warnings
warnings.filterwarnings('ignore')
sns.set_palette(['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])



files = [file for file in os.listdir('/')]
df = pd.DataFrame()
for i in files:
    data = pd.read_csv('/content/Dataset'+i)
    df = pd.concat([df,data],axis=0)
df.shape

#exclude header
df = df[df['Order ID'] != 'Order ID']
df = df.reset_index()
df = df.drop(columns='index')
df.sample(5)
#Check for data types
df.info()

#Check null values
df.isna().sum()
#Check null values
df[df.isna().any(axis=1)]
#Drop null vales
df = df.dropna()

#Correcting data types
df['Quantity Ordered'] = df['Quantity Ordered'].astype('int64')
df['Price Each'] = df['Price Each'].astype('float')
df['Order Date'] = pd.to_datetime(df['Order Date'])
#Adding new feature
def feature_extraction(data):

    # funtction to get the city in the data
    def get_city(address):
        return address.split(',')[1]

    # funtction to get the state in the data
    def get_state(address):
        return address.split(',')[2].split(' ')[1]

    # let's get the year data in order date column
    data['Year'] = data['Order Date'].dt.year

    # let's get the month data in order date column
    data['Month'] = data['Order Date'].dt.month

    # let's get the houe data in order date column
    data['Hour'] = data['Order Date'].dt.hour

    # let's get the minute data in order date column
    data['Minute'] = data['Order Date'].dt.minute

    # let's make the sales column by multiplying the quantity ordered colum with price each column
    data['Sales'] = data['Quantity Ordered'] * data['Price Each']

    # let's get the cities data in order date column
    data['Cities'] = data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")

    return data

df = feature_extraction(df)
df.sample(3)
df.describe()
#Select only for year 2019
df = df[df['Year']==2019]
df.info()

total_year_order = df.shape[0]
total_product_sold = df['Quantity Ordered'].sum()
total_year_sales = df['Sales'].sum()

print(f'Total orders in 2019 : {total_year_order:,} orders')
print(f'Total products sold in 2019 : {total_product_sold:,} items')
print(f'Total sales in 2019 : {total_year_sales:,} USD')

fig, ax = plt.subplots(3,2, figsize=(12,10))
sns.histplot(data=df,x='Quantity Ordered',kde=True,ax=ax[0,0])
sns.histplot(data=df,x='Price Each',kde=True,ax=ax[1,0],bins=50)
sns.histplot(data=df,x='Sales',kde=True,ax=ax[2,0],bins=50)

ax[0,0].set_title('Quantity Ordered')
ax[1,0].set_title('Price Each')
ax[2,0].set_title('Sales')

sns.boxplot(data=df,x='Quantity Ordered',ax=ax[0,1])
sns.boxplot(data=df,x='Price Each',ax=ax[1,1])
sns.boxplot(data=df,x='Sales',ax=ax[2,1])

ax[0,1].set_title('Quantity Ordered')
ax[1,1].set_title('Price Each')
ax[2,1].set_title('Sales')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))

df['Cities'].value_counts().plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
# sns.countplot(df['Cities'])
plt.title('Cities orders distribution',weight='bold',fontsize=20,pad=20)
plt.ylabel('Orders',fontsize=12)
plt.xlabel('City',fontsize=12)
plt.show()

plt.figure(figsize=(10,6))
df['Month'].value_counts().sort_index().plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Month orders distribution',weight='bold',fontsize=20,pad=20)
plt.ylabel('Orders',fontsize=12)
plt.xlabel('Month',fontsize=12)
plt.show()

plt.figure(figsize=(6,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f',cmap=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.show()

df_month = df.groupby('Month')['Sales'].sum()
plt.figure(figsize=(10,5))
df_month.plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Monthly Sales',weight='bold',fontsize=20,pad=20)
plt.ylabel('Sales (in million)')
plt.show()

df_city = df.groupby('Cities')['Sales'].sum()
plt.figure(figsize=(10,5))
df_city.sort_values(ascending=False).plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Cities Sales',weight='bold',fontsize=20)
plt.ylabel('Sales (in million)')

plt.show()

df_hour = df.groupby('Hour')['Quantity Ordered'].count()
plt.figure(figsize=(10,5))
plt.plot(df_hour.index,df_hour.values)
plt.title('Hour Sales',weight='bold',fontsize=20)
plt.grid(True)
plt.xticks(ticks=df_hour.index)
plt.ylabel('Sales (in million)')
plt.xlabel('Hour')


plt.show()

df_product = df.groupby('Product')['Quantity Ordered'].sum()
df_product = df_product.sort_values(ascending=False)

plt.figure(figsize=(10,6))
df_product.plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Product Quantity Orders',weight='bold',fontsize=20)

plt.ylabel('Quantity (pcs)')

plt.show()

from itertools import combinations
from collections import Counter

# drop it using duplicated() funct
data = df[df['Order ID'].duplicated(keep=False)]
# create a new column
data['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
# # let's make a new variable
data = data[['Order ID', 'Grouped']].drop_duplicates()
# # create a new variable for Counter
count = Counter()
# # make a for loop
for row in data['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))
# # and make another for loop
for key, value in count.most_common(10):
    print(key, value)

def proba_prod(product):
    product_size = df.shape[0]
    product_size1 = df[df.Product == product]
    product_size_ = product_size1.shape[0]
    prob_year = round(product_size_/product_size*100,2)

    product_month = []
    product_month1 = []
    prob_month = []
    for i in range(1,13):
        prod_size = df[df['Month']==i].shape[0]
        product_month.append(prod_size)
        prod_size1 = product_size1[product_size1['Month']==i].shape[0]
        product_month1.append(prod_size1)
    for a,b in zip(product_month1, product_month):
        prob = round(a/b,3)
        prob_month.append(prob)
    return np.array(prob_month),prob_year

fig, ax = plt.subplots(1,3,figsize=(15,5),sharey=True)
prod1 = 'USB-C Charging Cable'
ax[0].plot(range(1,13),(proba_prod(prod1)[0]*100),label='USB-C Charging Cable',color='r')
prod2 = 'Lightning Charging Cable'
ax[0].plot(range(1,13),(proba_prod(prod2)[0]*100),label='Lightning Charging Cable',color='b')
# ax[0].set_ylim(0,15)
ax[0].set_title(f'{prod1} & {prod2} \n Monthly Probability',weight='bold',fontsize=12,pad=10)
ax[0].grid()
ax[0].set_xticks(range(1,13))
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Probability (%)')
ax[0].legend()



prod1 = 'Google Phone'
ax[1].plot(range(1,13),(proba_prod(prod1)[0]*100),label='Google Phone',color='r')
prod2 = 'iPhone'
ax[1].plot(range(1,13),(proba_prod(prod2)[0]*100),label='iPhone',color='b')
# ax[1].set_ylim(0,6)
ax[1].set_title(f'{prod1} & {prod2} \n Monthly Probability',weight='bold',fontsize=12,pad=10)
ax[1].grid(True)
ax[1].set_xticks(range(1,13))
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Probability (%)')
ax[1].legend()

prod1 = 'Wired Headphones'
ax[2].plot(range(1,13),(proba_prod(prod1)[0]*100),label='Wired Headphones',color='r')
prod2 = 'Apple Airpods Headphones'
ax[2].plot(range(1,13),(proba_prod(prod2)[0]*100),label='Apple Airpods Headphones',color='b')
prod3 = 'Bose SoundSport Headphones'
ax[2].plot(range(1,13),(proba_prod(prod3)[0]*100),label='Bose SoundSport Headphones',color='g')
# ax[2].set_ylim(0,12)
ax[2].set_title(f'{prod1} & {prod2} \n {prod3} Monthly Probability',weight='bold',fontsize=12,pad=10)
ax[2].grid(True)
ax[2].set_xticks(range(1,13))
ax[2].set_xlabel('Month')
ax[2].set_ylabel('Probability (%)')
ax[2].legend()

plt.tight_layout()
plt.show()

prods =['USB-C Charging Cable', 'Lightning Charging Cable','Google Phone','iPhone',
        'Wired Headphones','Apple Airpods Headphones','Bose SoundSport Headphones']
for i in prods:
    print(f'Probability in year {i} : {proba_prod(i)[1]}')

print(f'Total orders in 2019 : {total_year_order:,} orders')
print(f'Total products sold in 2019 : {total_product_sold:,} items')
print(f'Total sales in 2019 : {total_year_sales:,} USD')

import pandas as pd
import os
import numpy as np
import seaborn as sns
from matplotlib import pylab as plt
!pip install scikit-learn

import warnings
warnings.filterwarnings('ignore')
sns.set_palette(['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])

from google.colab import drive
drive.mount('/content/drive')

files = [file for file in os.listdir('/content/drive/MyDrive/Dataset')]
df = pd.DataFrame()
for i in files:
    data = pd.read_csv('/content/drive/MyDrive/Dataset/'+i)
    df = pd.concat([df,data],axis=0)
df.shape

#Exclude header for each dataset inside dataframe
df = df[df['Order ID'] != 'Order ID']

df = df.reset_index()
df = df.drop(columns='index')
df.sample(5)

df.info()
#Check null values
df.isna().sum()
#Check null values
df[df.isna().any(axis=1)]
#Drop null vales
df = df.dropna()


#Correcting data types
df['Quantity Ordered'] = df['Quantity Ordered'].astype('int64')
df['Price Each'] = df['Price Each'].astype('float')
df['Order Date'] = pd.to_datetime(df['Order Date'])

#Adding new feature
def feature_extraction(data):

    # funtction to get the city in the data
    def get_city(address):
        return address.split(',')[1]

    # funtction to get the state in the data
    def get_state(address):
        return address.split(',')[2].split(' ')[1]

    # let's get the year data in order date column
    data['Year'] = data['Order Date'].dt.year

    # let's get the month data in order date column
    data['Month'] = data['Order Date'].dt.month

    # let's get the houe data in order date column
    data['Hour'] = data['Order Date'].dt.hour

    # let's get the minute data in order date column
    data['Minute'] = data['Order Date'].dt.minute

    # let's make the sales column by multiplying the quantity ordered colum with price each column
    data['Sales'] = data['Quantity Ordered'] * data['Price Each']

    # let's get the cities data in order date column
    data['Cities'] = data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")

    return data

df = feature_extraction(df)
df.sample(3)

#analysis

df.describe()
#Select only for year 2019
df = df[df['Year']==2019]

df.info()

total_year_order = df.shape[0]
total_product_sold = df['Quantity Ordered'].sum()
total_year_sales = df['Sales'].sum()

print(f'Total orders in 2019 : {total_year_order:,} orders')
print(f'Total products sold in 2019 : {total_product_sold:,} items')
print(f'Total sales in 2019 : {total_year_sales:,} USD')

#univariate analysis
fig, ax = plt.subplots(3,2, figsize=(12,10))
sns.histplot(data=df,x='Quantity Ordered',kde=True,ax=ax[0,0])
sns.histplot(data=df,x='Price Each',kde=True,ax=ax[1,0],bins=50)
sns.histplot(data=df,x='Sales',kde=True,ax=ax[2,0],bins=50)

ax[0,0].set_title('Quantity Ordered')
ax[1,0].set_title('Price Each')
ax[2,0].set_title('Sales')

sns.boxplot(data=df,x='Quantity Ordered',ax=ax[0,1])
sns.boxplot(data=df,x='Price Each',ax=ax[1,1])
sns.boxplot(data=df,x='Sales',ax=ax[2,1])

ax[0,1].set_title('Quantity Ordered')
ax[1,1].set_title('Price Each')
ax[2,1].set_title('Sales')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))

df['Cities'].value_counts().plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
# sns.countplot(df['Cities'])
plt.title('Cities orders distribution',weight='bold',fontsize=20,pad=20)
plt.ylabel('Orders',fontsize=12)
plt.xlabel('City',fontsize=12)
plt.show()

plt.figure(figsize=(10,6))
df['Month'].value_counts().sort_index().plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Month orders distribution',weight='bold',fontsize=20,pad=20)
plt.ylabel('Orders',fontsize=12)
plt.xlabel('Month',fontsize=12)
plt.show()

#multivariate analysis
plt.figure(figsize=(6,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f',cmap=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.show()

df_month = df.groupby('Month')['Sales'].sum()
plt.figure(figsize=(10,5))
df_month.plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Monthly Sales',weight='bold',fontsize=20,pad=20)
plt.ylabel('Sales (in million)')
plt.show()

df_city = df.groupby('Cities')['Sales'].sum()
plt.figure(figsize=(10,5))
df_city.sort_values(ascending=False).plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Cities Sales',weight='bold',fontsize=20)
plt.ylabel('Sales (in million)')

plt.show()

df_hour = df.groupby('Hour')['Quantity Ordered'].count()
plt.figure(figsize=(10,5))
plt.plot(df_hour.index,df_hour.values)
plt.title('Hour Sales',weight='bold',fontsize=20)
plt.grid(True)
plt.xticks(ticks=df_hour.index)
plt.ylabel('Sales (in million)')
plt.xlabel('Hour')


plt.show()

df_product = df.groupby('Product')['Quantity Ordered'].sum()
df_product = df_product.sort_values(ascending=False)

plt.figure(figsize=(10,6))
df_product.plot(kind='bar',color=['#0892a5','#2e9b9b','#50a290','#6fa985','#8dad7f','#a9b17e','#c4b383','#dbb68f'])
plt.title('Product Quantity Orders',weight='bold',fontsize=20)

plt.ylabel('Quantity (pcs)')

plt.show()

#market basket analysis
from itertools import combinations
from collections import Counter

# drop it using duplicated() funct
data = df[df['Order ID'].duplicated(keep=False)]
# create a new column
data['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
# # let's make a new variable
data = data[['Order ID', 'Grouped']].drop_duplicates()
# # create a new variable for Counter
count = Counter()
# # make a for loop
for row in data['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))
# # and make another for loop
for key, value in count.most_common(10):
    print(key, value)

#purchase probability
def proba_prod(product):
    product_size = df.shape[0]
    product_size1 = df[df.Product == product]
    product_size_ = product_size1.shape[0]
    prob_year = round(product_size_/product_size*100,2)

    product_month = []
    product_month1 = []
    prob_month = []
    for i in range(1,13):
        prod_size = df[df['Month']==i].shape[0]
        product_month.append(prod_size)
        prod_size1 = product_size1[product_size1['Month']==i].shape[0]
        product_month1.append(prod_size1)
    for a,b in zip(product_month1, product_month):
        prob = round(a/b,3)
        prob_month.append(prob)
    return np.array(prob_month),prob_year

fig, ax = plt.subplots(1,3,figsize=(15,5),sharey=True)
prod1 = 'USB-C Charging Cable'
ax[0].plot(range(1,13),(proba_prod(prod1)[0]*100),label='USB-C Charging Cable',color='r')
prod2 = 'Lightning Charging Cable'
ax[0].plot(range(1,13),(proba_prod(prod2)[0]*100),label='Lightning Charging Cable',color='b')
# ax[0].set_ylim(0,15)
ax[0].set_title(f'{prod1} & {prod2} \n Monthly Probability',weight='bold',fontsize=12,pad=10)
ax[0].grid()
ax[0].set_xticks(range(1,13))
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Probability (%)')
ax[0].legend()



prod1 = 'Google Phone'
ax[1].plot(range(1,13),(proba_prod(prod1)[0]*100),label='Google Phone',color='r')
prod2 = 'iPhone'
ax[1].plot(range(1,13),(proba_prod(prod2)[0]*100),label='iPhone',color='b')
# ax[1].set_ylim(0,6)
ax[1].set_title(f'{prod1} & {prod2} \n Monthly Probability',weight='bold',fontsize=12,pad=10)
ax[1].grid(True)
ax[1].set_xticks(range(1,13))
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Probability (%)')
ax[1].legend()

prod1 = 'Wired Headphones'
ax[2].plot(range(1,13),(proba_prod(prod1)[0]*100),label='Wired Headphones',color='r')
prod2 = 'Apple Airpods Headphones'
ax[2].plot(range(1,13),(proba_prod(prod2)[0]*100),label='Apple Airpods Headphones',color='b')
prod3 = 'Bose SoundSport Headphones'
ax[2].plot(range(1,13),(proba_prod(prod3)[0]*100),label='Bose SoundSport Headphones',color='g')
# ax[2].set_ylim(0,12)
ax[2].set_title(f'{prod1} & {prod2} \n {prod3} Monthly Probability',weight='bold',fontsize=12,pad=10)
ax[2].grid(True)
ax[2].set_xticks(range(1,13))
ax[2].set_xlabel('Month')
ax[2].set_ylabel('Probability (%)')
ax[2].legend()

plt.tight_layout()
plt.show()

prods =['USB-C Charging Cable', 'Lightning Charging Cable','Google Phone','iPhone',
        'Wired Headphones','Apple Airpods Headphones','Bose SoundSport Headphones']
for i in prods:
    print(f'Probability in year {i} : {proba_prod(i)[1]}')

print(f'Total orders in 2019 : {total_year_order:,} orders')
print(f'Total products sold in 2019 : {total_product_sold:,} items')
print(f'Total sales in 2019 : {total_year_sales:,} USD')
