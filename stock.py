import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


@st.cache(persist= True)
def read_data():
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    return df


def main():
    st.title("Stock Data Analysis and Prediction")

    df = read_data()
    df['Date']=pd.to_datetime(df.Date)
    if df is None:
        st.write(df)
        st.warning("Please upload a CSV file.")
        return

    st.sidebar.title("Data Analysis and Regression Models")

    # Data Analysis Options
    if st.sidebar.checkbox("Show Missing Values"):
        st.subheader("Missing Values")
        msno.bar(df)

    if st.sidebar.checkbox("Show Data Information"):
        st.subheader("Data Information")
        st.write(df)
        st.write(df.info())

    if st.sidebar.checkbox("Show Start Date, End Date, and Duration"):
        st.subheader("Date Information")
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        duration = end_date - start_date
        st.write("Start date:", start_date)
        st.write("End date:", end_date)
        st.write("Duration:", duration)

    # Data Preprocessing Options
    if st.sidebar.checkbox("Calculate Moving Averages and Standard Deviations"):
        st.subheader("Moving Averages and Standard Deviations")

        data_open_close = df[['Open', 'Close']]
        data_open_close['Average_open'] = data_open_close.Open.rolling(window=10).mean()
        data_open_close['Average_close'] = data_open_close.Close.rolling(window=10).mean()
        data_open_close['STD_open'] = data_open_close.Open.rolling(window=10).std()
        data_open_close['STD_close'] = data_open_close.Close.rolling(window=10).std()

        fig = px.line(data_open_close, x=data_open_close.index,
                      y=['Open', 'Close', 'Average_open', 'Average_close', 'STD_open', 'STD_close'],
                      labels={'Date': 'Date', 'value': 'Price'}, width=800, height=500)
        fig.update_layout(title='Overview of Open and Close Prices',
                          font_size=15, legend_title_text='Legend')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

    # Data Visualization Options
    if st.sidebar.checkbox("Show Stock Price by Month"):
        st.subheader("Stock Price by Month")
        monthwise = df.groupby(df['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
        monthwise = monthwise.reindex(pd.date_range(start=start_date, end=end_date, freq='M').strftime('%B'))
        fig = px.bar(monthwise)
        fig.update_layout(barmode='group', title='Monthwise Comparison between Stock Open and Close Prices',
                          font_size=15, xaxis_title='Months', yaxis_title='Price')
        fig.update_layout({'plot_bgcolor': 'white'})
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Show High and Low Prices by Month"):
        st.subheader("High and Low Prices by Month")
        monthwise1 = df.groupby(df['Date'].dt.strftime('%B'))['Low'].min()
        monthwise2 = df.groupby(df['Date'].dt.strftime('%B'))['High'].max()
        monthwise1 = monthwise1.reindex(pd.date_range(start=start_date, end=end_date, freq='M').strftime('%B'))
        monthwise2 = monthwise2.reindex(pd.date_range(start=start_date, end=end_date, freq='M').strftime('%B'))
        monthwise1 = pd.DataFrame(monthwise1)
        monthwise2 = pd.DataFrame(monthwise2)
        monthwise2['Low'] = monthwise1['Low']

        fig = px.bar(monthwise2)
        fig.update_layout(barmode='group', title='Monthwise Comparison between Stock High and Low Prices',
                          font_size=15, xaxis_title='Months', yaxis_title='Price')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout({'plot_bgcolor': 'white'})
        st.plotly_chart(fig)

    # Regression Models Options
    st.sidebar.title("Regression Models")

    regression_model = st.sidebar.selectbox("Select a Regression Model",
                                            ['Linear Regression', 'Ridge Regression', 'Support Vector Regression'])

    if regression_model == 'Linear Regression':
        st.subheader("Linear Regression Model")

        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        grid_search_checkbox = st.sidebar.checkbox("Perform GridSearchCV", key="grid_search_checkbox")
        if grid_search_checkbox:
            param_grid = {'fit_intercept': [True, False]}
            model = LinearRegression()
            grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            st.sidebar.write("Best Parameters:", grid_search.best_params_)
            
        else:
            best_model = LinearRegression()

        best_model.fit(X_train, y_train)

        predictions = best_model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.write('Mean Squared Error:', mse)
        st.write('Mean Absolute Error:', mae)
        st.write('R-squared:', r2)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(y_test)), y_test.values, label='Actual')
        plt.plot(np.arange(len(y_test)), predictions, label='Predicted')
        plt.xlabel('Data Index')
        plt.ylabel('Stock Price')
        plt.title('Comparison of Actual and Predicted Values')
        plt.legend()
        st.pyplot(plt)

    elif regression_model == 'Ridge Regression':
        st.subheader("Ridge Regression Model")

        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {'alpha': [0.1, 1, 10]}
        model = Ridge()

        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        predictions = best_model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        st.write('Mean Squared Error:', mse)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(y_test)), y_test.values, label='Actual')
        plt.plot(np.arange(len(y_test)), predictions, label='Predicted (Ridge Regression)')
        plt.xlabel('Data Index')
        plt.ylabel('Stock Price')
        plt.title('Comparison of Actual and Predicted Values')
        plt.legend()
        st.pyplot(plt)

    elif regression_model == 'Support Vector Regression':
        st.subheader("Support Vector Regression (SVR) Model")

        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        st.write('Mean Squared Error:', mse)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(y_test)), y_test.values, label='Actual')
        plt.plot(np.arange(len(y_test)), predictions, label='Predicted (SVR)')
        plt.xlabel('Data Index')
        plt.ylabel('Stock Price')
        plt.title('Comparison of Actual and Predicted Values')
        plt.legend()
        st.pyplot(plt)


if __name__ == '__main__':
    main()
