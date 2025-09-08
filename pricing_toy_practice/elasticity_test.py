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

    # # View data head
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
    
    # print(model.summary())
    
    param_df = pd.DataFrame(model.params).T
    
    ratio = (param_df['const'] + param_df['discount_High']) / (param_df['const'] + param_df['discount_No Discount'])
    # print(ratio)
    
    """
    Authors of this analysis wanted to do a split sample analysis, equivalent to a fully interacted multiple regression.
    This essentially varies slope and intercept among all different discount types.
    
    To do this, I will need to split out all of the discount types, and then for each discount type further split
    into each product type. However, for the sake of this analysis, I will only do so for the "No Discount" discount type.
    Otherwise I would get like 42 different graphs, which would be annoying.
    """
    data_filtered = df.query('discount == "No Discount"')
    
    products = data_filtered['Product'].unique()
    
    # Empty dataframe to store all results, as we have expectileGAM for all products
    all_gam_results = pd.DataFrame()
    # For-loop in order to run splits. Iterate over each product
    for product in products:
        product_data = data_filtered.query(f'Product == "{product}"')
        X = product_data[['price']]
        y = product_data[['Quantity']]

        expectiles = [0.025, 0.5, 0.975]
        # Empty dictionary to store the results from running GAM regression on each expectile (len(expectiles) number)
        gam_results = {}
        for expectile in expectiles:
            gam = ExpectileGAM(s(0), expectile=expectile)   # s(0) indicates that we are fitting just a single spline on the 0th feature, which for X is only price
            gam.fit(X, y)
            gam_results[f'pred_{expectile}'] = gam.predict(X)   # Run prediciton on the training data to get expectile predictions
            print(expectile, "|", product, "|", gam.deviance_residuals(X,y).mean()) # Print out the average squared deviation (MSE) of the fit
            # quit()
        print("-----------\n")
        gam_preds = pd.DataFrame(gam_results).set_index(X.index)    # Make the dataframe so corresponding predictions are aligned with data indices
        df_gam_preds = pd.concat([product_data[['price', 'Product', 'Quantity']], gam_preds], axis=1)   # Concatenate with product data along the column dimension, with matching indices
        """
        Concatenate along the rows as we loop through the products, which will get us the full dataset under discount_type,
        with ExpectileGAM predictions for each product at all expectiles
        """
        all_gam_results = pd.concat([all_gam_results, df_gam_preds], axis=0)    
    # Create the plot
    p = (ggplot(
                data = all_gam_results,
                mapping = aes(x='price', y='Quantity', color='Product', group= 'Product') ) +
                geom_ribbon(aes(ymax= 'pred_0.975', ymin= 'pred_0.025'), 
                            fill='#d3d3d3', color= '#FF000000', alpha=0.7, show_legend=False) +
                geom_point(alpha=0.75) + 
                geom_line(aes(y='pred_0.5'), color='blue') +
                facet_wrap('Product', scales='free') + 
                labs(title='GAM Price vs Quantity') +
                theme(figure_size=(12,6)
            ))
    p.show()
    
    # TODO: Implement price optimization
    

explore()
