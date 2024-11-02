import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set the theme for seaborn plots
sns.set_theme()

# Display a title
st.title("Linear Regression on California Housing Prices")

# Load the California Housing dataset
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["PRICE"] = dataset.target

# View Dataset and Description Section
with st.expander("View Dataset and Description"):
    # Display dataset when checkbox is ON
    if st.checkbox("View dataset in table data format"):
        st.dataframe(df)

    # Show each column description when checkbox is ON.
    if st.checkbox("Show each column name and its description"):
        st.markdown(
            """
            ### Column name and its Description
            - **MedInc**: Median income in block group
            - **HouseAge**: Median house age in block group
            - **AveRooms**: Average number of rooms per household
            - **AveBedrms**: Average number of bedrooms per household
            - **Population**: Population in block group
            - **AveOccup**: Average household size
            - **Latitude**: Block group latitude
            - **Longitude**: Block group longitude
            """
        )

# Outlier removal option
with st.expander("Outlier Removal"):
    remove_outliers = st.checkbox("Remove outliers from target variable (PRICE)")

    if remove_outliers:
        Q1 = df["PRICE"].quantile(0.25)
        Q3 = df["PRICE"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_size = df.shape[0]
        df = df[(df["PRICE"] >= lower_bound) & (df["PRICE"] <= upper_bound)]
        removed_size = initial_size - df.shape[0]
        st.write(f"Removed {removed_size} outliers from target variable.")

# Plot relationship between target and explanatory variables
if st.checkbox("Plot the relation between target and explanatory variables"):
    checked_variable = st.selectbox("Select one explanatory variable:", df.columns[:-1])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x=df[checked_variable], y=df["PRICE"], alpha=0.5)
    plt.xlabel(checked_variable)
    plt.ylabel("PRICE")
    st.pyplot(fig)

# Preprocessing Section
with st.expander("Data Preprocessing"):
    # Select the variables to exclude
    Features_NonUsed = st.multiselect(
        "Select the variables you will NOT use", df.columns[:-1]
    )
    df = df.drop(columns=Features_NonUsed)

    # Logarithmic transformation
    columns = st.columns(2)
    bool_log = columns[0].radio("Apply logarithmic transformation?", ("No", "Yes"))
    df_log = df.copy()
    if bool_log == "Yes":
        Log_Features = columns[1].multiselect(
            "Select features for logarithmic transformation", df.columns[:-1]
        )
        try:
            df_log[Log_Features] = np.log1p(df_log[Log_Features])
        except Exception as e:
            st.error(f"Log transformation error: {e}")

    # Standardization
    bool_std = columns[0].radio("Apply standardization?", ("No", "Yes"))
    df_std = df_log.copy()
    if bool_std == "Yes":
        Std_Features_NotUsed = columns[1].multiselect(
            "Select features to exclude from standardization", df.columns[:-1]
        )
        Std_Features_chosen = [
            name for name in df.columns[:-1] if name not in Std_Features_NotUsed
        ]
        scaler = StandardScaler()
        df_std[Std_Features_chosen] = scaler.fit_transform(df_std[Std_Features_chosen])

# Split the dataset into training and validation datasets
st.subheader("Split the dataset")
split_columns = st.columns(2)
test_size = split_columns[0].slider("Validation data size (ratio):", 0.1, 0.5, 0.2)
random_seed = split_columns[1].number_input(
    "Random seed:", min_value=0, step=1, value=42
)

X_train, X_val, Y_train, Y_val = train_test_split(
    df_std.drop(columns=["PRICE"]),
    df_std["PRICE"],
    test_size=test_size,
    random_state=random_seed,
)

# Model Training
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred_train = regressor.predict(X_train)
Y_pred_val = regressor.predict(X_val)

# Inverse transformation if logarithmic transformation was performed
if bool_log == "Yes" and "PRICE" in Log_Features:
    Y_pred_train, Y_pred_val = np.expm1(Y_pred_train), np.expm1(Y_pred_val)
    Y_train, Y_val = np.expm1(Y_train), np.expm1(Y_val)

# Model Evaluation Section
with st.expander("Model Evaluation"):
    # Model accuracy
    R2 = r2_score(Y_val, Y_pred_val)
    MAE = mean_absolute_error(Y_val, Y_pred_val)
    MSE = mean_squared_error(Y_val, Y_pred_val)

    st.write(f"R² value: {R2:.2f}")
    st.write(f"Mean Absolute Error (MAE): {MAE:.2f}")
    st.write(f"Mean Squared Error (MSE): {MSE:.2f}")

    # Plot the results
    plot_columns = st.columns(2)
    show_train = plot_columns[0].radio(
        "Plot the training dataset result:", ("Yes", "No")
    )
    show_val = plot_columns[1].radio(
        "Plot the validation dataset result:", ("Yes", "No")
    )

    # Dynamic axis range
    range_columns = st.columns(2)
    x_min = range_columns[0].number_input("x_min:", value=0)
    x_max = range_columns[1].number_input(
        "x_max:",
        value=int(max(Y_train.max(), Y_val.max(), Y_pred_train.max(), Y_pred_val.max()))
        + 5,
    )

    y_min = range_columns[0].number_input("y_min:", value=0)
    y_max = range_columns[1].number_input(
        "y_max:",
        value=int(max(Y_train.max(), Y_val.max(), Y_pred_train.max(), Y_pred_val.max()))
        + 5,
    )

    # Plot the results
    fig = plt.figure(figsize=(5, 5))
    if show_train == "Yes":
        plt.scatter(Y_train, Y_pred_train, color="r", label="Training data", alpha=0.5)
    if show_val == "Yes":
        plt.scatter(Y_val, Y_pred_val, color="b", label="Validation data", alpha=0.5)
    plt.plot([x_min, x_max], [y_min, y_max], "k--", lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    st.pyplot(fig)
