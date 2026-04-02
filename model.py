import streamlit as st 
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor  
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

from analysis import generate_insights, suggest_improvements


# ------------------ CONFIG ------------------
st.set_page_config('ML & AI Insights', layout='wide')

st.title('📊 Auto ML + AI Insight App')
st.subheader(':green[Learn data, train models, and get AI insights]')

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader('Upload CSV file 📁', type=['csv'])

if file:
    df = pd.read_csv(file)
    
    st.write('### Data Preview')
    st.dataframe(df.head())

    # ------------------ TARGET ------------------
    target = st.selectbox("Select Target Column 🎯", df.columns)

    if target:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        # ------------------ PREPROCESSING ------------------
        num_cols = X.select_dtypes(include=['int64','float64']).columns.to_list()
        cat_cols = X.select_dtypes(include=['object', 'str']).columns.to_list()

        # Fill missing values
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        X[cat_cols] = X[cat_cols].fillna('Missing')

        # One-hot encoding
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # ------------------ DETECT PROBLEM TYPE ------------------
        if df[target].dtype == 'object' or len(np.unique(y)) < 15:
            problem_type = 'Classification'
        else:
            problem_type = 'Regression'

        st.write(f"## Problem Type: {problem_type}")

        # Split the data
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()

        for i in xtrain.columns:
            xtrain[i] = scaler.fit_transform(xtrain[[i]])
            xtest[i] = scaler.transform(xtest[[i]])
            
        # Models
        results=[]
        if problem_type == 'Regression':
            models={'Linear Regression' : LinearRegression(),
                    'Random Forest':RandomForestRegressor(),
                    'Gradient Boosting': GradientBoostingRegressor()}
            for name,model in models.items():
                model.fit(xtrain,ytrain)
                ypred=model.predict(xtest)
                
                results.append({'Model Name':name,
                                'R2 Score': round(r2_score(ytest,ypred),3),
                                'RMSE': round(np.sqrt(mean_squared_error(ytest,ypred)),3)})
                
                
        else:
            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            }

            for name, model in models.items():
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)

                results.append({
                    'Model Name': name,
                    'Accuracy': round(accuracy_score(ytest, ypred), 3),
                    'Precision': round(precision_score(ytest, ypred, average='weighted'), 3),
                    'Recall': round(recall_score(ytest, ypred, average='weighted'), 3),
                    'F1 Score': round(f1_score(ytest, ypred, average='weighted'), 3)
                })
                
        results_df = pd.DataFrame(results)

        st.write("### :red[🎯 Model Results]")
        st.dataframe(results_df)

        if problem_type == 'Regression':
            st.bar_chart(results_df.set_index('Model Name')['R2 Score'])
            st.bar_chart(results_df.set_index('Model Name')['RMSE'])

        else:
            st.bar_chart(results_df.set_index('Model Name')['F1 Score'])
            
        # AI Insights
        if st.button(':blue[Generate ]'):
            summary = generate_insights(results_df)
            st.write("### :green[AI Insights]")
            st.write(summary)   
        if st.button(':orange[Suggest Improvements]'):
            improvements = suggest_improvements(results_df)
            st.write("### :orange[AI Suggestions]")
            st.write(improvements)
                
        # Download
        
        csv=results_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV here',csv,'model_results.csv')
        