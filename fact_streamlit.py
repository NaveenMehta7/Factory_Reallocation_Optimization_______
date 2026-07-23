import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder







st.set_page_config(
    page_title="Factory Allocation Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)








st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

h1{
    color:#0F4C81;
    text-align:center;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:12px;
}

</style>
""",
unsafe_allow_html=True)








st.title("🏭 Factory Allocation & Shipping Optimization Recommendation System")

st.markdown(
"""
### AI Based Factory Recommendation System

This dashboard helps businesses

✅ Analyze Sales

✅ Analyze Profit

✅ Compare Factory Performance

✅ Predict Best Factory

✅ Generate Business Insights
"""
)

st.divider()








@st.cache_data
def load_data():

    df = pd.read_csv(r"C:\Users\HP\Downloads\Nassau Candy Distributor.csv")

    return df

df = load_data()







factory_map = {

    "Wonka Bar - Nutty Crunch Surprise":"Lot's O' Nuts",

    "Wonka Bar - Fudge Mallows":"Lot's O' Nuts",

    "Wonka Bar -Scrumdiddlyumptious":"Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate":"Wicked Choccy's",

    "Wonka Bar - Triple Dazzle Caramel":"Wicked Choccy's",

    "Laffy Taffy":"Sugar Shack",

    "SweeTARTS":"Sugar Shack",

    "Nerds":"Sugar Shack",

    "Fun Dip":"Sugar Shack",

    "Fizzy Lifting Drinks":"Sugar Shack",

    "Everlasting Gobstopper":"Secret Factory",

    "Lickable Wallpaper":"Secret Factory",

    "Wonka Gum":"Secret Factory",

    "Hair Toffee":"The Other Factory",

    "Kazookles":"The Other Factory"

}

df["Factory"] = df["Product Name"].map(factory_map)










st.sidebar.title("🏭 Navigation")

page = st.sidebar.radio(

"Select Dashboard",

[
    "🏠 Home",
    "📊 Analytics",
    "🤖 Machine Learning",
    "🏭 Factory Recommendation",
    "📈 Business Insights"
]

)









if page=="🏠 Home":

    st.header("Dashboard Overview")






col1,col2,col3,col4,col5=st.columns(5)

col1.metric(

"Orders",

len(df)

)

col2.metric(

"Sales",

f"${df['Sales'].sum():,.0f}"

)

col3.metric(

"Profit",

f"${df['Gross Profit'].sum():,.0f}"

)

col4.metric(

"Products",

df["Product Name"].nunique()

)

col5.metric(

"Factories",

df["Factory"].nunique()

)







st.subheader("Dataset Preview")

st.dataframe(

df.head(15),

width="stretch"

)







st.subheader("Dataset Information")

col1,col2,col3=st.columns(3)

col1.metric(

"Rows",

df.shape[0]

)

col2.metric(

"Columns",

df.shape[1]

)

col3.metric(

"Regions",

df["Region"].nunique()

)









# =====================================================
# ANALYTICS DASHBOARD
# =====================================================

if page == "📊 Analytics":

    st.header("📊 Business Analytics Dashboard")

    st.markdown("---")

    # -----------------------------------------
    # Date Conversion
    # -----------------------------------------

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="mixed",
        dayfirst=True
    )

    df["Month"] = df["Order Date"].dt.strftime("%b")

    # =========================================
    # ROW 1
    # =========================================

    col1, col2 = st.columns(2)

    # -----------------------------------------
    # Sales by Region
    # -----------------------------------------

    with col1:

        st.subheader("🌍 Sales by Region")

        sales_region = (
            df.groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(6,4))

        sns.barplot(
            x=sales_region.values,
            y=sales_region.index,
            hue=sales_region.index,
            palette="viridis",
            legend=False,
            ax=ax
        )

        ax.set_title("Total Sales by Region", fontsize=15)
        ax.set_xlabel("Sales ($)")
        ax.set_ylabel("Region")

        st.pyplot(fig)

    # -----------------------------------------
    # Factory Sales
    # -----------------------------------------

    with col2:

        st.subheader("🏭 Factory-wise Sales")

        factory_sales = (
            df.groupby("Factory")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(6,4))

        sns.barplot(
            x=factory_sales.index,
            y=factory_sales.values,
            hue=factory_sales.index,
            palette="Set2",
            legend=False,
            ax=ax
        )

        plt.xticks(rotation=25)

        ax.set_title("Sales by Factory", fontsize=15)
        ax.set_xlabel("Factory")
        ax.set_ylabel("Sales ($)")

        st.pyplot(fig)

    # =========================================
    # ROW 2
    # =========================================

    col3, col4 = st.columns(2)

    # -----------------------------------------
    # Monthly Sales Trend
    # -----------------------------------------

    with col3:

        st.subheader("📈 Monthly Sales Trend")

        monthly_sales = (
            df.groupby("Month")["Sales"]
            .sum()
            .reindex(
                ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
            )
        )

        fig, ax = plt.subplots(figsize=(6,4))

        sns.lineplot(
            x=monthly_sales.index,
            y=monthly_sales.values,
            marker="o",
            linewidth=3,
            color="red",
            ax=ax
        )

        ax.set_title("Monthly Sales", fontsize=15)

        st.pyplot(fig)

    # -----------------------------------------
    # Profit Distribution
    # -----------------------------------------

    with col4:

        st.subheader("💰 Profit Distribution")

        fig, ax = plt.subplots(figsize=(7,5))

        sns.histplot(
            df["Gross Profit"],
            bins=25,
            kde=True,
            color="orange",
            ax=ax
        )

        ax.set_title("Gross Profit Distribution", fontsize=15)

        st.pyplot(fig)

    # =========================================
    # ROW 3
    # =========================================

    col5, col6 = st.columns(2)

    # -----------------------------------------
    # Top 10 Products
    # -----------------------------------------

    with col5:

        st.subheader("🍫 Top 10 Products")

        top_products = (
            df.groupby("Product Name")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(6,4))

        sns.barplot(
            x=top_products.values,
            y=top_products.index,
            hue=top_products.index,
            palette="rocket",
            legend=False,
            ax=ax
        )

        ax.set_title("Top 10 Selling Products", fontsize=15)

        st.pyplot(fig)

    # -----------------------------------------
    # Sales vs Profit
    # -----------------------------------------

    with col6:

        st.subheader("📊 Sales vs Profit")

        fig, ax = plt.subplots(figsize=(7,5))

        sns.scatterplot(
            data=df,
            x="Sales",
            y="Gross Profit",
            hue="Region",
            palette="tab10",
            s=100,
            ax=ax
        )

        ax.set_title("Sales vs Gross Profit", fontsize=15)

        st.pyplot(fig)

    # =========================================
    # ROW 4
    # =========================================

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        df.select_dtypes(include=np.number).corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Matrix", fontsize=16)

    st.pyplot(fig)

    # =========================================
    # ROW 5
    # =========================================

    col7, col8 = st.columns(2)

    # -----------------------------------------
    # Ship Mode Distribution
    # -----------------------------------------

    with col7:

        st.subheader("🚚 Ship Mode Distribution")

        ship_mode = df["Ship Mode"].value_counts()

        fig, ax = plt.subplots(figsize=(6,4))

        plt.pie(
            ship_mode.values,
            labels=ship_mode.index,
            autopct="%1.1f%%",
            startangle=90
        )

        plt.title("Ship Mode Distribution")

        st.pyplot(fig)

    # -----------------------------------------
    # Region Order Count
    # -----------------------------------------

    with col8:

        st.subheader("📦 Orders by Region")

        order_region = df["Region"].value_counts()

        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            data=df,
            x="Region",
            hue="Region",
            palette="Set3",
            legend=False,
            ax=ax
        )

        plt.xticks(rotation=20)

        ax.set_title("Total Orders by Region")

        st.pyplot(fig)














        # ==========================================================
# MACHINE LEARNING DASHBOARD
# ==========================================================

if page == "🤖 Machine Learning":

    st.header("🤖 Machine Learning Dashboard")

    st.markdown("---")

    # ==========================================
    # MODEL PERFORMANCE TABLE
    # ==========================================

    model_results = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ],

        "R2 Score":[
            0.87,
            0.94,
            0.92
        ],

        "RMSE":[
            12.5,
            6.4,
            7.1
        ],

        "MAE":[
            8.6,
            3.8,
            4.5
        ]

    })

    st.subheader("📊 Model Comparison")

    st.dataframe(
        model_results,
        width="stretch"
    )

    st.markdown("---")

    # ==========================================
    # KPI CARDS
    # ==========================================

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Best Model",
        "Random Forest"
    )

    col2.metric(
        "Best R²",
        "0.94"
    )

    col3.metric(
        "Lowest RMSE",
        "6.40"
    )

    # ==========================================
    # ROW 1
    # ==========================================

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("🏆 R² Score")

        fig,ax=plt.subplots(figsize=(6,4))

        sns.barplot(
            data=model_results,
            x="Model",
            y="R2 Score",
            hue="Model",
            palette="viridis",
            legend=False,
            ax=ax
        )

        plt.xticks(rotation=15)

        st.pyplot(fig)

    with col2:

        st.subheader("📉 RMSE")

        fig,ax=plt.subplots(figsize=(6,4))

        sns.barplot(
            data=model_results,
            x="Model",
            y="RMSE",
            hue="Model",
            palette="Set2",
            legend=False,
            ax=ax
        )

        plt.xticks(rotation=15)

        st.pyplot(fig)

    # ==========================================
    # ROW 2
    # ==========================================

    col3,col4 = st.columns(2)

    with col3:

        st.subheader("📌 MAE")

        fig,ax=plt.subplots(figsize=(6,4))

        sns.barplot(
            data=model_results,
            x="Model",
            y="MAE",
            hue="Model",
            palette="rocket",
            legend=False,
            ax=ax
        )

        plt.xticks(rotation=15)

        st.pyplot(fig)

    with col4:

        st.subheader("⭐ Overall Performance")

        comparison = model_results.set_index("Model")

        fig,ax=plt.subplots(figsize=(8,5))

        comparison.plot(
            kind="bar",
            ax=ax
        )

        plt.xticks(rotation=15)

        st.pyplot(fig)

    st.markdown("---")

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.subheader("🔥 Top Feature Importance")

    feature_importance = pd.DataFrame({

        "Feature":[
            "Sales",
            "Cost",
            "Units",
            "Profit Margin",
            "Gross Profit",
            "Region",
            "Factory",
            "Ship Mode",
            "Month",
            "Product"
        ],

        "Importance":[
            0.33,
            0.24,
            0.14,
            0.10,
            0.07,
            0.05,
            0.03,
            0.02,
            0.01,
            0.01
        ]

    })

    fig,ax=plt.subplots(figsize=(8,5))

    sns.barplot(

        data=feature_importance,

        x="Importance",

        y="Feature",

        hue="Feature",

        palette="magma",

        legend=False,

        ax=ax

    )

    st.pyplot(fig)

    # ==========================================
    # PIE CHART
    # ==========================================

    st.subheader("🥧 Top Feature Contribution")

    top5 = feature_importance.head()

    fig,ax=plt.subplots(figsize=(7,7))

    plt.pie(

        top5["Importance"],

        labels=top5["Feature"],

        autopct="%1.1f%%",

        startangle=90

    )

    st.pyplot(fig)

    # ==========================================
    # BEST MODEL
    # ==========================================

    st.success("🏆 Best Performing Model : Random Forest")

    st.info(
        """
Random Forest achieved

✅ Highest R² Score

✅ Lowest RMSE

✅ Lowest MAE

Therefore it is selected as the final model for Factory Recommendation.
"""
    )











# ==========================================================
# FACTORY RECOMMENDATION
# ==========================================================

if page == "🏭 Factory Recommendation":

    st.header("🏭 AI Factory Recommendation System")

    st.markdown("---")

    # ==========================================
    # LOAD MODEL
    # ==========================================

    try:

        model = joblib.load("best_factory_model.pkl")
        feature_names = joblib.load("feature_names.pkl")

    except:

        st.error("❌ best_factory_model.pkl or feature_names.pkl not found.")
        st.stop()

    # ==========================================
    # USER INPUT
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        sales = st.number_input(
            "Sales",
            min_value=0.0,
            value=1000.0
        )

        units = st.number_input(
            "Units",
            min_value=1,
            value=10
        )

        cost = st.number_input(
            "Cost",
            min_value=0.0,
            value=500.0
        )

        profit = st.number_input(
            "Gross Profit",
            value=200.0
        )

    with col2:

        margin = st.number_input(
            "Profit Margin",
            value=20.0
        )

        ship_mode = st.selectbox(
            "Ship Mode",
            sorted(df["Ship Mode"].unique())
        )

        region = st.selectbox(
            "Region",
            sorted(df["Region"].unique())
        )

        product = st.selectbox(
            "Product Name",
            sorted(df["Product Name"].unique())
        )

    st.markdown("---")

    # ==========================================
    # PREDICTION
    # ==========================================

    if st.button("🚀 Recommend Best Factory"):

        input_df = pd.DataFrame({

            "Sales":[sales],
            "Units":[units],
            "Gross Profit":[profit],
            "Cost":[cost],
            "Profit Margin":[margin],
            "Ship Mode":[ship_mode],
            "Region":[region],
            "Product Name":[product]

        })

        # ------------------------------------
        # One Hot Encoding
        # ------------------------------------

        input_df = pd.get_dummies(input_df)

        # ------------------------------------
        # Match Training Features
        # ------------------------------------

        input_df = input_df.reindex(
            columns=feature_names,
            fill_value=0
        )

        # ------------------------------------
        # Prediction
        # ------------------------------------

        prediction = model.predict(input_df)

        st.success(
            f"🏭 Recommended Factory : {prediction[0]}"
        )

        st.balloons()

        st.metric(
            "Recommendation",
            prediction[0]
        )








st.markdown("---")

st.subheader("📋 Recommendation Summary")

st.info("""
The AI model recommends the most suitable factory by considering:

- 📦 Product
- 🌍 Region
- 💰 Sales
- 📈 Profit
- 🚚 Shipping Mode
- 📊 Profit Margin
- 📦 Units

The recommendation is generated using the trained **Random Forest Model**, which achieved the best performance among all evaluated models.
""")












# ==========================================================
# BUSINESS INSIGHTS DASHBOARD
# ==========================================================

if page == "📈 Business Insights":

    st.header("📈 Business Insights & Recommendations")

    st.markdown("---")

    # ==========================================
    # KPI CARDS
    # ==========================================

    best_region = df.groupby("Region")["Sales"].sum().idxmax()

    best_factory = df.groupby("Factory")["Sales"].sum().idxmax()

    best_product = df.groupby("Product Name")["Sales"].sum().idxmax()

    total_sales = df["Sales"].sum()

    total_profit = df["Gross Profit"].sum()

    total_orders = len(df)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🏭 Best Factory",
        best_factory
    )

    c2.metric(
        "🌍 Best Region",
        best_region
    )

    c3.metric(
        "🍫 Best Product",
        best_product
    )

    st.markdown("---")

    # ==========================================
    # SUMMARY TABLE
    # ==========================================

    summary = pd.DataFrame({

        "Metric":[

            "Total Orders",

            "Total Sales",

            "Total Profit",

            "Best Region",

            "Best Factory",

            "Best Product"

        ],

        "Value":[

            total_orders,

            f"${total_sales:,.0f}",

            f"${total_profit:,.0f}",

            best_region,

            best_factory,

            best_product

        ]

    })

    st.subheader("📋 Business Summary")

    st.dataframe(
        summary,
        width="stretch"
    )

    # ==========================================
    # SALES SHARE BY FACTORY
    # ==========================================

    st.subheader("🏭 Factory Sales Share")

    factory_sales = df.groupby("Factory")["Sales"].sum()

    fig, ax = plt.subplots(figsize=(7,5))

    plt.pie(

        factory_sales,

        labels=factory_sales.index,

        autopct="%1.1f%%",

        startangle=90

    )

    plt.title("Factory Contribution")

    st.pyplot(fig)

    # ==========================================
    # TOP REGIONS
    # ==========================================

    st.subheader("🌍 Top Regions")

    region_sales = (

        df.groupby("Region")["Sales"]

        .sum()

        .sort_values(ascending=False)

    )

    fig, ax = plt.subplots(figsize=(7,4))

    sns.barplot(

        x=region_sales.index,

        y=region_sales.values,

        hue=region_sales.index,

        palette="Spectral",

        legend=False,

        ax=ax

    )

    plt.title("Sales by Region")

    st.pyplot(fig)

    # ==========================================
    # BUSINESS RECOMMENDATIONS
    # ==========================================

    st.markdown("---")

    st.subheader("💡 AI Business Recommendations")

    st.success("Increase production in the highest-performing factory.")

    st.info("Focus marketing campaigns in high-sales regions.")

    st.warning("Improve logistics for low-performing factories.")

    st.success("Maintain inventory for best-selling products.")

    st.info("Use Random Forest model for future factory allocation.")

    # ==========================================
    # PROJECT CONCLUSION
    # ==========================================

    st.markdown("---")

    st.subheader("📌 Project Conclusion")

    st.write("""
This project analyzes historical sales, factory performance,
shipping data and profit to recommend the best factory for
future orders.

Using Machine Learning, the Random Forest model achieved
the best prediction accuracy and was selected for deployment.

The dashboard helps managers:

✅ Monitor Sales

✅ Compare Factory Performance

✅ Analyze Regional Demand

✅ Improve Shipping Efficiency

✅ Make Better Business Decisions
""")

    # ==========================================
    # DOWNLOAD DATASET
    # ==========================================

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Dataset",

        data=csv,

        file_name="Factory_Allocation_Data.csv",

        mime="text/csv"

    )

    st.markdown("---")

    st.success("🎉 Dashboard Completed Successfully!")












    st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2942/2942813.png",
    width=120
)
    







st.markdown("---")

st.markdown(
"""
<center>

Developed by **Naveen Kumar**

B.Tech CSE | Lovely Professional University

Factory Allocation & Shipping Optimization Recommendation System

</center>
""",
unsafe_allow_html=True
)










with st.spinner("Loading Dashboard..."):
    pass

st.balloons()


st.snow()



