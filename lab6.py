import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn import tree
import matplotlib.pyplot as plt


# загрузка данных
@st.cache_data
def load_data():
    data = pd.read_csv('~/Downloads/Admission_Predict.csv')
    return data

# выбор модели для обучения
def select_model():
    models = ['Linear Regression', 'Decision Tree']
    model = st.selectbox('Select a model', models, key='my_selectbox')
    if model == 'Linear Regression':
        return LinearRegression()
    elif model == 'Decision Tree':
        return DecisionTreeRegressor(max_depth = 6, min_samples_leaf = 3, min_samples_split = 2)


# обучение модели
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# просмотр результатов обучения
def show_results(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    st.write('R2 Score:', r2_score(y_test, y_pred))
    st.write('Median absolute error:', median_absolute_error(y_test, y_pred))

def show_plot(model,X_test):
    y_pred = model.predict(X_test)
    if isinstance(model, DecisionTreeRegressor):
        fig, ax = plt.subplots()
        tree.plot_tree(model, ax=ax)
        st.pyplot(fig)
    elif isinstance(model, LinearRegression):
        plt.plot(X_test, y_pred, color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Regression')
        st.pyplot()

def main():
    st.title('Machine Learning Model Training')
    data = load_data()
    st.write(data.head())
    X = data.drop(['Chance of Admit '], axis=1)
    y = data['Chance of Admit ']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=45)

    model = select_model()
    trained_model = train_model(model, X_train, y_train)
    show_results(trained_model, X_test, y_test)
    show_plot(trained_model, X_test)

if __name__ == '__main__':
   main()
