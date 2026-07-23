import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\HP\Downloads\Nassau Candy Distributor.csv")
#print(df.head())
# print(df.shape)
# print(df.info())
print(df.columns)
# print(df.describe())
# print(df.isnull().sum())
#print(df.duplicated().sum())

#print(df[['Order Date','Ship Date']].head())

df['Order Date'] = pd.to_datetime(df['Order Date'],format='mixed',dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'],format='mixed',dayfirst=True)

# print(df.dtypes)

#print(df[['Order Date','Ship Date']].head())







df['Profit Margin'] = (df['Gross Profit'] / df['Sales']).round(2)
# print(df[['Sales','Gross Profit','Profit Margin']].head())

# print(df.info())


num_cols = df.select_dtypes(include = ['int64','float64']).columns
cat_cols = df.select_dtypes(include = ['object']).columns

# print('Numerical cols: \n', num_cols)
# print('Categorical cols: \n',cat_cols)

# print(df['Product Name'].unique())





factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",

    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",

    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",

    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory"
}





df["Factory"] = df["Product Name"].map(factory_map)
print(df["Factory"].isnull().sum())

print("========================\nProduct Name and Factory Mapping\n========================")
print(df[["Product Name", "Factory"]].head(15))
print("========================\n")





factory_coordinates = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}




df["Factory Latitude"] = df["Factory"].map(lambda x: factory_coordinates[x][0])
df["Factory Longitude"] = df["Factory"].map(lambda x: factory_coordinates[x][1])

print("========================")
print("Factory Coordinates")
print("========================")

print(df[[
    "Factory",
    "Factory Latitude",
    "Factory Longitude"
]].drop_duplicates())








product_summary = (
    df.groupby("Product Name")
      .agg(
          Total_Sales=("Sales", "sum"),
          Total_Profit=("Gross Profit", "sum"),
          Total_Units=("Units", "sum")
      )
)

product_summary = product_summary.sort_values(
    by="Total_Sales",
    ascending=False
)

print("========================\nProduct Summary\n========================")
print(product_summary)


product_summary.to_csv(
    "product_summary.csv",
    index=True
)








region_summary = (
    df.groupby("Region")
      .agg(
          Total_Sales=("Sales", "sum"),
          Total_Profit=("Gross Profit", "sum"),
          Total_Orders=("Order ID", "count")
      )
)

region_summary = region_summary.sort_values(
    by="Total_Sales",
    ascending=False
)

print("========================\nRegion Summary\n========================")
print(region_summary)


region_summary.to_csv(
    "region_summary.csv",
    index=True
)








factory_product = pd.crosstab(
    df["Factory"],
    df["Product Name"]
)

print("========================\nFactory-Product Crosstab\n========================")
print(factory_product)








factory_kpi = (
    df.groupby("Factory")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Total_Orders=("Order ID", "count"),
        Avg_Profit_Margin=("Profit Margin", "mean"),
        Total_Units=("Units", "sum")
    )
)


factory_kpi = factory_kpi.sort_values(
    by="Total_Sales",
    ascending=False
)

factory_kpi["Avg_Profit_Margin"] = (
    factory_kpi["Avg_Profit_Margin"]
    .round(2)
)

print("========================\nFactory KPIs\n========================")
print(factory_kpi)


factory_kpi.to_csv(
    "factory_kpi.csv",
    index=True
)









scaler = MinMaxScaler()

factory_kpi[["Sales Score"]] = scaler.fit_transform(factory_kpi[["Total_Sales"]])

factory_kpi[["Profit Score"]] = scaler.fit_transform(factory_kpi[["Total_Profit"]])

factory_kpi[["Margin Score"]] = scaler.fit_transform(factory_kpi[["Avg_Profit_Margin"]])


# ==========================================
# Factory Score Formula
# 40% Sales
# 40% Profit
# 20% Profit Margin
# ==========================================

factory_kpi["Factory Score"] = (
    0.4 * factory_kpi["Sales Score"] +
    0.4 * factory_kpi["Profit Score"] +
    0.2 * factory_kpi["Margin Score"]
)

print("========================\nFactory Scores\n========================")
print(factory_kpi.sort_values("Factory Score", ascending=False))









factory_ranking = factory_kpi.sort_values(
    "Factory Score",
    ascending=False
)

print("========================\nFactory Ranking\n========================")
print(factory_ranking)





best_factory = factory_ranking.index[0]

print("========================\nBest Factory Recommendation\n========================")    
print("Recommended Factory:", best_factory)







# ==========================================
# SALES BY PRODUCT
# ==========================================

sales_product = (
    df.groupby("Product Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

print("========================")
print("Sales by Product")
print("========================")
print(sales_product)




# ==========================================
# SALES BY PRODUCT GRAPH
# ==========================================

plt.figure(figsize=(10,8))

sns.barplot(
    x=sales_product.index,
    y=sales_product.values,
    hue=sales_product.index,      # Gives each bar a different color
    palette="Set3",
    legend=False
)

plt.title("Total Sales by Product", fontsize=18, fontweight="bold")
plt.xlabel("Product Name", fontsize=13)
plt.ylabel("Total Sales ($)", fontsize=13)

plt.xticks(rotation=45, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()









# ==========================================
# PROFIT BY PRODUCT
# ==========================================

profit_product = (
    df.groupby("Product Name")["Gross Profit"]
      .sum()
      .sort_values(ascending=False)
)

print("========================")
print("Profit by Product")
print("========================")
print(profit_product)




# ==========================================
# PROFIT BY PRODUCT GRAPH
# ==========================================

plt.figure(figsize=(10,8))

sns.barplot(
    x=profit_product.index,
    y=profit_product.values,
    hue=profit_product.index,
    palette="Set2",
    legend=False
)

plt.title(
    "Total Profit by Product",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Product Name", fontsize=13)
plt.ylabel("Gross Profit ($)", fontsize=13)

plt.xticks(rotation=45, ha="right")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()









sales_factory = (
    df.groupby("Factory")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

print("========================")
print("Sales by Factory")
print("========================")
print(sales_factory)





# ==========================================
# SALES BY FACTORY GRAPH
# ==========================================

plt.figure(figsize=(10,8))

sns.barplot(
    x=sales_factory.index,
    y=sales_factory.values,
    hue=sales_factory.index,
    palette="Pastel1",
    legend=False
)

plt.title(
    "Total Sales by Factory",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Factory", fontsize=13)
plt.ylabel("Total Sales ($)", fontsize=13)

plt.xticks(rotation=20)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()








profit_factory = (
    df.groupby("Factory")["Gross Profit"]
      .sum()
      .sort_values(ascending=False)
)

print("========================")
print("Profit by Factory")
print("========================")
print(profit_factory)







# ==========================================
# PROFIT BY FACTORY GRAPH
# ==========================================

plt.figure(figsize=(10,6))

sns.barplot(
    x=profit_factory.index,
    y=profit_factory.values,
    hue=profit_factory.index,
    palette="Dark2",
    legend=False
)

plt.title(
    "Total Profit by Factory",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Factory", fontsize=13)
plt.ylabel("Gross Profit ($)", fontsize=13)

plt.xticks(rotation=20)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()









sales_region = (
    df.groupby("Region")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

print("========================")
print("Sales by Region")
print("========================")
print(sales_region)






# ==========================================
# SALES BY REGION DONUT CHART
# ==========================================

plt.figure(figsize=(8,8))

colors = sns.color_palette("Set2", len(sales_region))

plt.pie(
    sales_region.values,
    labels=sales_region.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    wedgeprops={"width":0.4},
    textprops={"fontsize":11}
)

plt.title(
    "Sales Contribution by Region",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()







profit_region = (
    df.groupby("Region")["Gross Profit"]
      .sum()
      .sort_values(ascending=False)
)

print("========================")
print("Profit by Region")
print("========================")
print(profit_region)





# ==========================================
# PROFIT BY REGION PIE CHART
# ==========================================

plt.figure(figsize=(8,8))

colors = sns.color_palette("Pastel1", len(profit_region))

plt.pie(
    profit_region.values,
    labels=profit_region.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    textprops={"fontsize":11}
)

plt.title(
    "Profit Contribution by Region",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()






ship_mode = (
    df["Ship Mode"]
    .value_counts()
)

print("========================")
print("Ship Mode Distribution")
print("========================")
print(ship_mode)






# ==========================================
# SHIP MODE COUNT PLOT
# ==========================================

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    x="Ship Mode",
    hue="Ship Mode",
    palette="Set2",
    legend=False
)

plt.title(
    "Orders by Ship Mode",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Ship Mode", fontsize=13)
plt.ylabel("Number of Orders", fontsize=13)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()






division_summary = (
    df.groupby("Division")
      .agg(
          Total_Sales=("Sales","sum"),
          Total_Profit=("Gross Profit","sum"),
          Total_Units=("Units","sum"),
          Total_Orders=("Order ID","count")
      )
)

division_summary = division_summary.sort_values(
    by="Total_Sales",
    ascending=False
)

print("========================")
print("Division Summary")
print("========================")
print(division_summary)





# ==========================================
# SALES & PROFIT BY DIVISION
# ==========================================

plt.figure(figsize=(10,6))

division_plot = division_summary.reset_index()

division_plot = division_plot.melt(
    id_vars="Division",
    value_vars=["Total_Sales","Total_Profit"],
    var_name="Metric",
    value_name="Value"
)

sns.barplot(
    data=division_plot,
    x="Division",
    y="Value",
    hue="Metric",
    palette="Set2"
)

plt.title(
    "Sales and Profit by Division",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Division",fontsize=13)
plt.ylabel("Amount ($)",fontsize=13)

plt.grid(axis="y",linestyle="--",alpha=0.4)

plt.tight_layout()

plt.show()







# ==========================================
# CREATE ORDER MONTH
# ==========================================

df["Order Month"] = df["Order Date"].dt.month


# ==========================================
# MONTHLY SALES SUMMARY
# ==========================================

monthly_sales = (
    df.groupby("Order Month")
      .agg(
          Total_Sales=("Sales","sum"),
          Total_Profit=("Gross Profit","sum"),
          Total_Orders=("Order ID","count")
      )
)

print("========================")
print("Monthly Sales Summary")
print("========================")
print(monthly_sales)




# ==========================================
# MONTHLY SALES TREND
# ==========================================

plt.figure(figsize=(12,6))

sns.lineplot(
    data=monthly_sales,
    x=monthly_sales.index,
    y="Total_Sales",
    marker="o",
    linewidth=3,
    color="royalblue"
)

plt.title(
    "Monthly Sales Trend",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Month",fontsize=13)
plt.ylabel("Total Sales ($)",fontsize=13)

plt.xticks(range(1,13))

plt.grid(True,linestyle="--",alpha=0.4)

plt.tight_layout()

plt.show()





# ==========================================
# MONTHLY PROFIT TREND
# ==========================================

plt.figure(figsize=(12,6))

sns.lineplot(
    data=monthly_sales,
    x=monthly_sales.index,
    y="Total_Profit",
    marker="s",
    linewidth=3,
    color="darkgreen"
)

plt.title(
    "Monthly Profit Trend",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Month",fontsize=13)
plt.ylabel("Total Profit ($)",fontsize=13)

plt.xticks(range(1,13))

plt.grid(True,linestyle="--",alpha=0.4)

plt.tight_layout()

plt.show()





# ==========================================
# SELECT NUMERICAL FEATURES
# ==========================================

numeric_df = df.select_dtypes(include=["int64", "float64"])

print("========================")
print("Numerical Columns")
print("========================")
print(numeric_df.columns)




# ==========================================
# CORRELATION MATRIX
# ==========================================

correlation = numeric_df.corr()

print("========================")
print("Correlation Matrix")
print("========================")
print(correlation)




# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10,8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=1,
    square=True
)

plt.title(
    "Correlation Heatmap",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()






# ==========================================
# SALES VS GROSS PROFIT
# ==========================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="Sales",
    y="Gross Profit",
    hue="Division",
    palette="Set2",
    s=80
)

plt.title(
    "Sales vs Gross Profit",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Sales ($)",fontsize=13)
plt.ylabel("Gross Profit ($)",fontsize=13)

plt.grid(True,linestyle="--",alpha=0.4)

plt.tight_layout()

plt.show()






# ==========================================
# REGRESSION LINE
# ==========================================

plt.figure(figsize=(10,6))

sns.regplot(
    data=df,
    x="Sales",
    y="Gross Profit",
    scatter_kws={"alpha":0.4},
    line_kws={"color":"red"}
)

plt.title(
    "Relationship between Sales and Gross Profit",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Sales ($)")
plt.ylabel("Gross Profit ($)")

plt.grid(True,linestyle="--",alpha=0.4)

plt.tight_layout()

plt.show()







# ==========================================
# SALES OUTLIER DETECTION
# ==========================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["Sales"],
    color="skyblue"
)

plt.title(
    "Sales Outlier Detection",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Sales ($)",fontsize=12)

plt.grid(alpha=0.3)

plt.show()





# ==========================================
# GROSS PROFIT OUTLIER DETECTION
# ==========================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["Gross Profit"],
    color="lightgreen"
)

plt.title(
    "Gross Profit Outlier Detection",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Gross Profit ($)",fontsize=12)

plt.grid(alpha=0.3)

plt.show()






# ==========================================
# UNITS OUTLIER DETECTION
# ==========================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["Units"],
    color="orange"
)

plt.title(
    "Units Outlier Detection",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Units",fontsize=12)

plt.grid(alpha=0.3)

plt.show()






# ==========================================
# PROFIT MARGIN OUTLIER DETECTION
# ==========================================

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["Profit Margin"],
    color="violet"
)

plt.title(
    "Profit Margin Outlier Detection",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Profit Margin",fontsize=12)

plt.grid(alpha=0.3)

plt.show()






# ==========================================
# MULTIPLE NUMERICAL FEATURES
# ==========================================

plt.figure(figsize=(14,7))

columns = [
    "Sales",
    "Gross Profit",
    "Cost",
    "Units",
    "Profit Margin"
]

sns.boxplot(
    data=df[columns],
    palette="Set2"
)

plt.title(
    "Outlier Detection of Numerical Features",
    fontsize=18,
    fontweight="bold"
)

plt.xticks(rotation=20)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()






# ==========================================
# FIND SALES OUTLIERS
# ==========================================

Q1 = df["Sales"].quantile(0.25)

Q3 = df["Sales"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

sales_outliers = df[
    (df["Sales"] < lower) |
    (df["Sales"] > upper)
]

print("========================")
print("Sales Outliers")
print("========================")

print(sales_outliers)

print()

print("Total Sales Outliers :", len(sales_outliers))




print(df.dtypes)





# ==========================================
# CREATE A COPY OF DATASET
# ==========================================

df_ml = df.copy()

print(df_ml.head())



# ==========================================
# REMOVE USELESS COLUMNS
# ==========================================

df_ml.drop(
    columns=[
        "Row ID",
        "Order ID",
        "Customer ID",
        "Product ID",
        "Postal Code",
        "Order Date",
        "Ship Date"
    ],
    inplace=True
)

print(df_ml.columns)
print(df_ml.dtypes)


cat_cols = df_ml.select_dtypes(include="object").columns

print(cat_cols)



# ==========================================
# CONVERT CATEGORICAL DATA
# ==========================================

df_ml = pd.get_dummies(
    df_ml,
    columns=cat_cols,
    drop_first=True,
    dtype=int
)

print(df_ml.head())
print(df_ml.info())
print(df_ml.select_dtypes(include="object").columns)





# ==========================================
# CREATE SHIPPING EFFICIENCY SCORE
# ==========================================

df_ml["Shipping Efficiency"] = (
    0.40 * df_ml["Gross Profit"] +
    0.30 * df_ml["Sales"] -
    0.20 * df_ml["Cost"] +
    0.10 * df_ml["Units"]
)

print(df_ml[[
    "Sales",
    "Gross Profit",
    "Cost",
    "Units",
    "Shipping Efficiency"
]].head())




# ==========================================
# DEFINE FEATURES & TARGET
# ==========================================

X = df_ml.drop(
    columns=["Shipping Efficiency"]
)

y = df_ml["Shipping Efficiency"]

print(X.head())

print()

print(y.head())



print("X Shape :", X.shape)

print("y Shape :", y.shape)



print(X.isnull().sum().sum())

print(y.isnull().sum())



final_dataset = X.copy()

final_dataset["Shipping Efficiency"] = y

final_dataset.to_csv(
    "ml_dataset.csv",
    index=False
)




# ==========================================
# FEATURES (X) AND TARGET (y)
# ==========================================

print(df_ml.columns)

# Features (Independent Variables)
X = df_ml.drop(columns=["Sales"])

# Target (Dependent Variable)
y = df_ml["Sales"]

print("="*50)
print("X Shape :", X.shape)
print("y Shape :", y.shape)

print("\n==========================================")
print("First 5 Rows of X")
print("==========================================")
print(X.head())

print("\n==========================================")
print("First 5 Values of y")
print("==========================================")
print(y.head())






# ==========================================
# SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)



# ==========================================
# CHECK SHAPES
# ==========================================

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)

print()

print("Training Target :", y_train.shape)
print("Testing Target  :", y_test.shape)





print(X_train.head())

print()

print(y_train.head())






# ==========================================
# STANDARDIZE THE DATA
# ==========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit scaler on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

print("="*60)
print("STANDARDIZED DATA")
print("="*60)

print("Training Data Shape :", X_train_scaled.shape)
print("Testing Data Shape  :", X_test_scaled.shape)

# Display first 5 rows of scaled data
scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

print(scaled_df.head())



# ==========================================
# IMPORT LINEAR REGRESSION
# ==========================================

from sklearn.linear_model import LinearRegression

# Create Model
lr_model = LinearRegression()



# ==========================================
# TRAIN THE MODEL
# ==========================================

lr_model.fit(
    X_train_scaled,
    y_train
)



# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = lr_model.predict(
    X_test_scaled
)

print("="*60)
print("FIRST 10 PREDICTIONS")
print("="*60)

print(np.round(y_pred[:10], 2))



# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

comparison = pd.DataFrame({

    "Actual": y_test,

    "Predicted": np.round(y_pred, 2)

})

# Prediction Error
comparison["Error"] = (
    comparison["Actual"] -
    comparison["Predicted"]
)

print("="*60)
print("ACTUAL VS PREDICTED")
print("="*60)

print(comparison.head(10))



# ==========================================
# SAVE PREDICTIONS
# ==========================================

comparison.to_csv(
    "linear_regression_predictions.csv",
    index=False
)










# ==========================================
# IMPORT EVALUATION METRICS
# ==========================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# CALCULATE MODEL PERFORMANCE
# ==========================================

# Mean Absolute Error (MAE)
mae = mean_absolute_error(
    y_test,
    y_pred
)

# Mean Squared Error (MSE)
mse = mean_squared_error(
    y_test,
    y_pred
)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R2 Score
r2 = r2_score(
    y_test,
    y_pred
)

# ==========================================
# DISPLAY PERFORMANCE METRICS
# ==========================================

print("=" * 60)
print("LINEAR REGRESSION PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")






# ==========================================
# ==========================================
# ROOT MEAN SQUARED ERROR (RMSE)
# ==========================================

rmse = np.sqrt(mse)

# ==========================================
# R2 SCORE
# ==========================================

r2 = r2_score(
    y_test,
    y_pred
)

# ==========================================
# FINAL MODEL PERFORMANCE
# ==========================================

print("=" * 60)
print("LINEAR REGRESSION PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error (MAE)      : {mae:.2f}")
print(f"Mean Squared Error (MSE)       : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"R2 Score                       : {r2:.4f}")



# ==========================================
# STORE MODEL RESULTS
# ==========================================

model_results = pd.DataFrame({

    "Model": ["Linear Regression"],

    "MAE": [round(mae, 2)],

    "MSE": [round(mse, 2)],

    "RMSE": [round(rmse, 2)],

    "R2 Score": [round(r2, 4)]

})

print("=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(model_results)



# ==========================================
# SAVE MODEL RESULTS
# ==========================================

model_results.to_csv(
    "model_results.csv",
    index=False
)

print("\nModel results saved successfully.")





# ==========================================
# IMPORT RANDOM FOREST REGRESSOR
# ==========================================

from sklearn.ensemble import RandomForestRegressor

# ==========================================
# CREATE RANDOM FOREST MODEL
# ==========================================

rf_model = RandomForestRegressor(

    n_estimators=100,

    random_state=42,

    n_jobs=-1

)


# ==========================================
# TRAIN THE MODEL
# ==========================================

rf_model.fit(
    X_train,
    y_train
)


# ==========================================
# MAKE PREDICTIONS
# ==========================================

rf_pred = rf_model.predict(
    X_test
)

print("=" * 60)
print("FIRST 10 RANDOM FOREST PREDICTIONS")
print("=" * 60)

print(np.round(rf_pred[:10], 2))


# ==========================================
# RANDOM FOREST EVALUATION
# ==========================================

rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

rf_mse = mean_squared_error(
    y_test,
    rf_pred
)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(
    y_test,
    rf_pred)






# ==========================================
# RANDOM FOREST PERFORMANCE
# ==========================================

print("=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error (MAE)      : {rf_mae:.2f}")
print(f"Mean Squared Error (MSE)       : {rf_mse:.2f}")
print(f"Root Mean Squared Error (RMSE) : {rf_rmse:.2f}")
print(f"R2 Score                       : {rf_r2:.4f}")


# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

rf_comparison = pd.DataFrame({

    "Actual": y_test,

    "Predicted": np.round(rf_pred, 2)

})

rf_comparison["Error"] = (
    rf_comparison["Actual"] -
    rf_comparison["Predicted"]
)

print("=" * 60)
print("RANDOM FOREST PREDICTIONS")
print("=" * 60)

print(rf_comparison.head(10))


# ==========================================
# SAVE PREDICTIONS
# ==========================================

rf_comparison.to_csv(
    "random_forest_predictions.csv",
    index=False
)

print("\nRandom Forest predictions saved successfully.")







# ==========================================
# ADD RANDOM FOREST RESULTS
# ==========================================

new_result = pd.DataFrame({

    "Model": ["Random Forest"],

    "MAE": [round(rf_mae, 2)],

    "MSE": [round(rf_mse, 2)],

    "RMSE": [round(rf_rmse, 2)],

    "R2 Score": [round(rf_r2, 4)]

})

# Add Random Forest result to comparison table
model_results = pd.concat(
    [model_results, new_result],
    ignore_index=True
)

print("=" * 60)
print("MODEL RESULTS AFTER ADDING RANDOM FOREST")
print("=" * 60)

print(model_results)






# ==========================================
# IMPORT GRADIENT BOOSTING REGRESSOR
# ==========================================

from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# CREATE GRADIENT BOOSTING MODEL
# ==========================================

gb_model = GradientBoostingRegressor(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42

)

# ==========================================
# TRAIN THE MODEL
# ==========================================

gb_model.fit(
    X_train,
    y_train
)

# ==========================================
# MAKE PREDICTIONS
# ==========================================

gb_pred = gb_model.predict(
    X_test
)

print("=" * 60)
print("FIRST 10 GRADIENT BOOSTING PREDICTIONS")
print("=" * 60)

print(np.round(gb_pred[:10], 2))


# ==========================================
# MODEL EVALUATION
# ==========================================

gb_mae = mean_absolute_error(
    y_test,
    gb_pred
)

gb_mse = mean_squared_error(
    y_test,
    gb_pred
)

gb_rmse = np.sqrt(gb_mse)

gb_r2 = r2_score(
    y_test,
    gb_pred
)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

print("=" * 60)
print("GRADIENT BOOSTING PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error (MAE)      : {gb_mae:.2f}")
print(f"Mean Squared Error (MSE)       : {gb_mse:.2f}")
print(f"Root Mean Squared Error (RMSE) : {gb_rmse:.2f}")
print(f"R2 Score                       : {gb_r2:.4f}")


# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

gb_comparison = pd.DataFrame({

    "Actual": y_test,

    "Predicted": np.round(gb_pred, 2)

})

gb_comparison["Error"] = (
    gb_comparison["Actual"] -
    gb_comparison["Predicted"]
)

print("=" * 60)
print("GRADIENT BOOSTING PREDICTIONS")
print("=" * 60)

print(gb_comparison.head(10))


# ==========================================
# SAVE PREDICTIONS
# ==========================================

gb_comparison.to_csv(
    "gradient_boosting_predictions.csv",
    index=False
)

print("\nGradient Boosting predictions saved successfully.")


# ==========================================
# ADD RESULT TO MODEL COMPARISON TABLE
# ==========================================

new_result = pd.DataFrame({

    "Model": ["Gradient Boosting"],

    "MAE": [round(gb_mae, 2)],

    "MSE": [round(gb_mse, 2)],

    "RMSE": [round(gb_rmse, 2)],

    "R2 Score": [round(gb_r2, 4)]

})

model_results = pd.concat(
    [model_results, new_result],
    ignore_index=True
)

print("=" * 60)
print("UPDATED MODEL RESULTS")
print("=" * 60)

print(model_results)


# ==========================================
# SAVE ALL MODEL RESULTS
# ==========================================

model_results.to_csv(
    "all_model_results.csv",
    index=False
)

model_results.to_csv(
    "Model_Comparison.csv",
    index=False
)

print("\nModel comparison saved successfully.")



# ==========================================
# FIND BEST MODEL
# ==========================================

best_model = model_results.loc[
    model_results["R2 Score"].idxmax()
]

print("=" * 60)
print("BEST MACHINE LEARNING MODEL")
print("=" * 60)

print(f"Model Name : {best_model['Model']}")
print(f"MAE        : {best_model['MAE']:.2f}")
print(f"MSE        : {best_model['MSE']:.2f}")
print(f"RMSE       : {best_model['RMSE']:.2f}")
print(f"R2 Score   : {best_model['R2 Score']:.4f}")



# ==========================================
# FINAL RECOMMENDATION
# ==========================================

print("\n" + "=" * 60)
print("FINAL RECOMMENDATION")
print("=" * 60)

print(f"Best Model : {best_model['Model']}")
print(f"Best R2 Score : {best_model['R2 Score']:.4f}")



# ==========================================
# SAVE BEST MODEL
# ==========================================

best_model.to_frame().T.to_csv(
    "best_model.csv",
    index=False
)

print("\nBest model saved successfully.")



# ==========================================
# R2 SCORE COMPARISON
# ==========================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=model_results,
    x="Model",
    y="R2 Score",
    hue="Model",
    palette="viridis",
    legend=False
)

plt.title(
    "R² Score Comparison of Machine Learning Models",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Models",
    fontsize=13
)

plt.ylabel(
    "R² Score",
    fontsize=13
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()






# ==========================================
# RMSE COMPARISON
# ==========================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=model_results,
    x="Model",
    y="RMSE",
    hue="Model",
    palette="Set2",
    legend=False
)

plt.title(
    "RMSE Comparison of Machine Learning Models",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Models",
    fontsize=13
)

plt.ylabel(
    "RMSE",
    fontsize=13
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()



# ==========================================
# MAE COMPARISON
# ==========================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=model_results,
    x="Model",
    y="MAE",
    hue="Model",
    palette="Paired",
    legend=False
)

plt.title(
    "MAE Comparison of Machine Learning Models",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Models",
    fontsize=13
)

plt.ylabel(
    "MAE",
    fontsize=13
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()



# ==========================================
# OVERALL MODEL PERFORMANCE
# ==========================================

comparison_graph = model_results.set_index("Model")

plt.figure(figsize=(11,6))

comparison_graph.plot(
    kind="bar",
    colormap="viridis",
    edgecolor="black"
)

plt.title(
    "Overall Machine Learning Model Performance",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Models",
    fontsize=13
)

plt.ylabel(
    "Evaluation Score",
    fontsize=13
)

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()





# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})

print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance.head())



# ==========================================
# SORT FEATURES
# ==========================================

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nTop 20 Important Features\n")

print(feature_importance.head(20))



# ==========================================
# SAVE FEATURE IMPORTANCE
# ==========================================

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)



# ==========================================
# TOP 15 FEATURES
# ==========================================

top15 = feature_importance.head(15)

print(top15)



# ==========================================
# TOP 15 IMPORTANT FEATURES GRAPH
# ==========================================

plt.figure(figsize=(12,8))

sns.barplot(

    data=top15,

    x="Importance",

    y="Feature",

    hue="Feature",

    palette="viridis",

    legend=False

)

plt.title(

    "Top 15 Most Important Features",

    fontsize=18,

    fontweight="bold"

)

plt.xlabel(
    "Importance Score",
    fontsize=13
)

plt.ylabel(
    "Feature Name",
    fontsize=13
)

plt.grid(

    axis="x",

    linestyle="--",

    alpha=0.4

)

plt.tight_layout()

plt.show()






# ==========================================
# TOP 10 FEATURE CONTRIBUTION (PIE CHART)
# ==========================================

top10 = feature_importance.head(10)

plt.figure(figsize=(10,10))

plt.pie(

    top10["Importance"],

    labels=top10["Feature"],

    autopct="%1.1f%%",

    startangle=90,

    explode=[0.08] + [0]*9,

    shadow=True

)

plt.title(

    "Top 10 Feature Contribution",

    fontsize=18,

    fontweight="bold"

)

plt.tight_layout()

plt.show()



# ==========================================
# SAVE TOP 10 FEATURES
# ==========================================

top10.to_csv(
    "top10_features.csv",
    index=False
)



# ==========================================
# MOST IMPORTANT FEATURE
# ==========================================

print("="*60)

print("MOST IMPORTANT FEATURE")

print("="*60)

print(f"Feature Name : {feature_importance.iloc[0]['Feature']}")

print(f"Importance   : {feature_importance.iloc[0]['Importance']:.4f}")



# ==========================================
# TOP 5 BUSINESS INSIGHTS
# ==========================================

print("="*60)

print("TOP 5 BUSINESS INSIGHTS")

print("="*60)

for i in range(5):

    print(

        f"{i+1}. {feature_importance.iloc[i]['Feature']}"

        f"  --> Importance Score : "

        f"{feature_importance.iloc[i]['Importance']:.4f}"

    )

    












feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)









top15 = feature_importance.head(15)

plt.figure(figsize=(12,7))

sns.barplot(
    data=top15,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)

plt.title("Top 15 Important Features")

plt.tight_layout()

plt.show()









top10 = feature_importance.head(10)

plt.figure(figsize=(8,8))

plt.pie(
    top10["Importance"],
    labels=top10["Feature"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Top 10 Feature Contribution")

plt.show()








print("="*60)
print("BUSINESS INSIGHTS")
print("="*60)

for i in range(5):
    print(
        f"{i+1}.",
        feature_importance.iloc[i]["Feature"],
        "- Importance:",
        round(feature_importance.iloc[i]["Importance"],4)
    )









    import joblib

joblib.dump(
    rf_model,
    "best_factory_model.pkl"
)

print("Best model saved successfully.")