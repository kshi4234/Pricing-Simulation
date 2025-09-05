import pandas as pd
import numpy as np

from plotnine import *
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from pygam import GAM, ExpectileGAM, s, l, f
import statsmodels.api as sm

import warnings
warnings.simplefilter('ignore')

def explore():
# load dataset
    file_ = './financial_sample2.xlsx'
    df = pd.read_excel(file_)

    # print(f'Data Shape: {df.shape}')

    # View data head
    # print(df.sample(3))

    df.fillna('No Discount', inplace=True)

    # print(df.describe(include='all').T) # WOW! Remember this

    fig = px.scatter(
        data_frame=df,
        x='price', y='Quantity',
        color= 'discount',
        facet_col='Product',
        facet_col_wrap=3,
        facet_col_spacing=0.1,
        facet_row_spacing=0.1,
        trendline='lowess',
        opacity=0.5,
        title='Price vs Quantity',
        width=1800, height=1000,
        color_discrete_sequence=px.colors.qualitative.G10
    ).update_traces(
        marker= dict(size=7),
        hoverlabel= dict(font= dict(size=10))
    ).update_xaxes(
        title_text = 'Price',
        tickfont= dict(size=10)
    ).update_yaxes(
        title_text = 'Quantity Sold',
        tickfont= dict(size=10)
    )

    # fig.show()
    # One hot encode the discount and product columns, as they are categorical
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded,
                                columns=['discount', 'Product'],
                                drop_first=False)
    # Cast the discount values as integers instead of booleans
    discount_types = df_encoded.columns[df_encoded.columns.str.startswith('discount')].tolist()
    df_encoded[discount_types] = df_encoded[discount_types].astype(int)
    # Cast the product values as integers instead of booleans
    product_types = df_encoded.columns[df_encoded.columns.str.startswith('Product')].tolist()
    df_encoded[product_types] = df_encoded[product_types].astype(int)
    
    # # Drop features that we don't believe are predictors
    # df_encoded.drop(['Date',  'mth_num', 'mth_name', 'yr', 'Segment'], axis=1)
    
    # Get the predictors and response columns
    X = df_encoded[['price'] + discount_types + product_types]
    y = df_encoded['Quantity']
    # Add a column of 1's that will be multiplied with a parameter; this will be our intercept term
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

explore()
