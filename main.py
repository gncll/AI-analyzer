import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

openai.api_key = st.secrets["openai"]["api_key"]

# Page config
st.set_page_config(page_title="Advanced CSV Explorer", layout="wide")
st.title("LearnAIWithMe Data Explorer")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔍 Data Exploration", "📈 Visualizations", "🤖 AI Assistant", "🧠 Model Building"])
    
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Show Data Types"):
                st.write("Data Types:")
                st.write(df.dtypes)
                
            if st.button("📏 Show Shape"):
                st.write(f"DataFrame shape: {df.shape}")
                
            if st.button("🔚 Show Last Rows"):
                st.write("Last 5 rows:")
                st.dataframe(df.tail())
                
        with col2:
            if st.button("📊 Show Summary Statistics"):
                st.write("Summary Statistics:")
                st.dataframe(df.describe())
                
            if st.button("❓ Show Missing Values"):
                st.write("Missing Values:")
                missing = df.isnull().sum()
                st.dataframe(missing.to_frame(name="Missing Count"))
                
            if st.button("🔄 Show Correlation Matrix"):
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.write("Correlation Matrix:")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr = numeric_df.corr()
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("No numeric columns available for correlation.")
    
    with tab2:
        st.subheader("Column Analysis")
        
        # Column selection
        selected_column = st.selectbox("Select a column to analyze:", df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔢 Show Value Counts"):
                st.write(f"Value counts for '{selected_column}':")
                counts = df[selected_column].value_counts()
                st.dataframe(counts)
                
            if st.button("🌟 Show Unique Values"):
                st.write(f"Unique values in '{selected_column}':")
                uniques = df[selected_column].unique()
                st.write(uniques)
                
        with col2:
            if st.button("📊 Show Histogram"):
                if df[selected_column].dtype in [np.float64, np.int64]:
                    st.write(f"Histogram of '{selected_column}':")
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Selected column is not numeric. Please select a numeric column for histogram.")
            
            if st.button("📦 Show Boxplot"):
                if df[selected_column].dtype in [np.float64, np.int64]:
                    st.write(f"Boxplot of '{selected_column}':")
                    fig, ax = plt.subplots()
                    sns.boxplot(y=df[selected_column].dropna(), ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Selected column is not numeric. Please select a numeric column for boxplot.")
    
    with tab3:
        st.subheader("Advanced Visualizations")
        
        viz_type = st.selectbox("Select visualization type:", 
                               ["Histogram", "Box Plot", "Scatter Plot", "Pair Plot", "Count Plot"])
        
        if viz_type == "Histogram":
            num_col = st.selectbox("Select column for histogram:", 
                                  df.select_dtypes(include=[np.number]).columns)
            bins = st.slider("Number of bins:", 5, 100, 20)
            
            fig, ax = plt.subplots()
            sns.histplot(df[num_col].dropna(), bins=bins, kde=True, ax=ax)
            st.pyplot(fig)
            
        elif viz_type == "Box Plot":
            num_col = st.selectbox("Select numeric column:", 
                                  df.select_dtypes(include=[np.number]).columns)
            cat_col_option = st.checkbox("Group by category")
            
            if cat_col_option:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    cat_col = st.selectbox("Select category column:", cat_cols)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning("No categorical columns available for grouping.")
            else:
                fig, ax = plt.subplots()
                sns.boxplot(y=df[num_col], ax=ax)
                st.pyplot(fig)
                
        elif viz_type == "Scatter Plot":
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                x_col = st.selectbox("Select X column:", num_cols, index=0)
                y_col = st.selectbox("Select Y column:", num_cols, index=min(1, len(num_cols)-1))
                
                fig, ax = plt.subplots()
                sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for scatter plot.")
                
        elif viz_type == "Pair Plot":
            num_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(num_cols) > 1:
                selected_cols = st.multiselect("Select columns for pair plot:", 
                                              num_cols, 
                                              default=list(num_cols[:min(4, len(num_cols))]))
                
                if len(selected_cols) >= 2:
                    fig = sns.pairplot(df[selected_cols])
                    st.pyplot(fig)
                else:
                    st.warning("Please select at least 2 columns.")
            else:
                st.warning("Need at least 2 numeric columns for pair plot.")
                
        elif viz_type == "Count Plot":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) > 0:
                cat_col = st.selectbox("Select categorical column:", cat_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get top categories if there are too many
                value_counts = df[cat_col].value_counts()
                if len(value_counts) > 15:
                    st.warning(f"Column has {len(value_counts)} unique values. Showing top 15.")
                    top_cats = value_counts.index[:15]
                    sns.countplot(y=cat_col, data=df[df[cat_col].isin(top_cats)], order=top_cats, ax=ax)
                else:
                    sns.countplot(y=cat_col, data=df, order=value_counts.index, ax=ax)
                    
                st.pyplot(fig)
            else:
                st.warning("No categorical columns available for count plot.")
    
    with tab4:
        st.subheader("AI Data Assistant")
        
        # Show first 5 rows automatically
        st.write("Here's a preview of your data:")
        st.dataframe(df.head())
        
        st.write("Ask questions about your data in natural language:")
        
        # Create agent with the API key
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                api_key=api_key
            ),
            df,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            **{"allow_dangerous_code": True}
        )
        
        # Ask prompt
        prompt = st.text_input("💬 What would you like to know about your data?")
        if prompt:
            with st.spinner("Thinking..."):
                response = agent.invoke(prompt)
            st.success("✅ Answer:")
            st.markdown(f"{response['output']}")
            
            # Option to save the query and response
            if st.button("📥 Save this insight"):
                st.session_state.setdefault('insights', []).append((prompt, response['output']))
                st.success("Insight saved!")
        
        # Display saved insights
        if 'insights' in st.session_state and st.session_state['insights']:
            st.subheader("Saved Insights")
            for i, (q, a) in enumerate(st.session_state['insights']):
                with st.expander(f"Insight {i+1}: {q}"):
                    st.markdown(a)
    
    with tab5:
        st.subheader("Machine Learning Models")
        
        # Show data types before proceeding
        st.write("Current data types in your dataset:")
        dtypes_df = pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes.values})
        st.dataframe(dtypes_df)
        
        # Show sample of data 
        st.write("Preview of first 5 rows:")
        st.dataframe(df.head())
        
        # Select target variable
        target_col = st.selectbox("Select target variable (what you want to predict):", df.columns)
        
        # Check target data type and give a warning if needed
        if df[target_col].dtype == 'object':
            st.warning(f"⚠️ The target variable '{target_col}' contains text data. It may need conversion for modeling.")
        
        # Determine if regression or classification
        unique_values = df[target_col].nunique()
        problem_type = "classification" if unique_values < 10 else "regression"
        
        st.write(f"Based on the target variable, this appears to be a **{problem_type}** problem.")
        st.write(f"Number of unique values in target: {unique_values}")
        
        # Select features
        st.write("Select features to include in the model:")
        feature_cols = st.multiselect("Choose features:", 
                                     [col for col in df.columns if col != target_col], 
                                     default=[col for col in df.columns if col != target_col])
        
        if len(feature_cols) > 0:
            # ML settings
            with st.expander("Model Settings"):
                test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
                random_state = st.number_input("Random State", 0, 100, 42)
                
                if problem_type == "classification":
                    model_type = st.selectbox("Select model type:", 
                                             ["Logistic Regression", "Random Forest", "Decision Tree", 
                                              "Support Vector Machine", "Gradient Boosting", "Neural Network"])
                else:
                    model_type = st.selectbox("Select model type:", 
                                             ["Linear Regression", "Random Forest", "Decision Tree", 
                                              "Support Vector Machine", "Gradient Boosting", "Neural Network"])
            
            # Build and evaluate model
            if st.button("🚀 Build Model"):
                try:
                    with st.spinner("Training model..."):
                        st.write("Preparing data...")
                        
                        import sklearn
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler, OneHotEncoder
                        from sklearn.compose import ColumnTransformer
                        from sklearn.pipeline import Pipeline
                        from sklearn.impute import SimpleImputer
                        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                                   mean_squared_error, mean_absolute_error, r2_score)
                        
                        # Pre-process data
                        st.write("Pre-processing data...")
                        X = df[feature_cols].copy()
                        y = df[target_col].copy()
                        
                        # Data type handling strategy
                        st.subheader("Feature Type Classification")
                        st.write("Classifying features by data type for proper encoding:")
                        
                        # Create dictionaries to store feature types
                        numeric_features = []
                        categorical_features = []
                        datetime_features = []
                        
                        # Identify data types
                        for col in X.columns:
                            # Check if column is already numeric
                            if pd.api.types.is_numeric_dtype(X[col]):
                                numeric_features.append(col)
                                st.write(f"✅ '{col}' - Numeric feature (will be scaled)")
                            
                            # Check if it can be converted to numeric without creating NaNs
                            elif pd.to_numeric(X[col], errors='coerce').notna().all():
                                X[col] = pd.to_numeric(X[col])
                                numeric_features.append(col)
                                st.write(f"🔄 '{col}' - Converted to numeric feature")
                            
                            # Check if it's a datetime
                            elif pd.to_datetime(X[col], errors='coerce').notna().all():
                                X[col] = pd.to_datetime(X[col])
                                # Extract useful datetime features
                                X[f"{col}_year"] = X[col].dt.year
                                X[f"{col}_month"] = X[col].dt.month
                                X[f"{col}_day"] = X[col].dt.day
                                numeric_features.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
                                # Drop the original datetime column
                                X = X.drop(columns=[col])
                                st.write(f"📅 '{col}' - Converted to datetime features (year, month, day)")
                                datetime_features.append(col)
                            
                            # Otherwise, treat as categorical
                            else:
                                # Check cardinality (number of unique values)
                                n_unique = X[col].nunique()
                                if n_unique < 10 or (n_unique < 100 and n_unique / len(X) < 0.05):
                                    categorical_features.append(col)
                                    # Fill missing values with 'Unknown'
                                    X[col] = X[col].fillna('Unknown')
                                    st.write(f"📊 '{col}' - Categorical feature with {n_unique} unique values (will be one-hot encoded)")
                                else:
                                    st.warning(f"⚠️ '{col}' - High cardinality categorical feature with {n_unique} unique values")
                                    st.write("This may lead to too many features after one-hot encoding. Consider:")
                                    st.write("1. Grouping less frequent categories")
                                    st.write("2. Using another encoding method")
                                    categorical_features.append(col)
                                    # Fill missing values with 'Unknown'
                                    X[col] = X[col].fillna('Unknown')
                        
                        # Summary of feature types
                        st.write(f"✓ Total features: {len(numeric_features) + len(categorical_features)}")
                        st.write(f"✓ Numeric features: {len(numeric_features)}")
                        st.write(f"✓ Categorical features: {len(categorical_features)}")
                        
                        # Handle missing values
                        st.subheader("Missing Value Strategy")
                        
                        # For numeric features
                        for col in numeric_features:
                            na_count = X[col].isna().sum()
                            if na_count > 0:
                                st.write(f"Filling {na_count} missing values in '{col}' with median")
                                X[col] = X[col].fillna(X[col].median())
                        
                        # For categorical features
                        for col in categorical_features:
                            na_count = X[col].isna().sum()
                            if na_count > 0:
                                st.write(f"Filling {na_count} missing values in '{col}' with 'Unknown'")
                                X[col] = X[col].fillna('Unknown')
                        
                        # Handle target variable
                        if problem_type == "classification":
                            # For classification target (categorical)
                            if y.dtype == 'object':
                                st.write(f"Target variable '{target_col}' is categorical with {y.nunique()} unique values")
                                y = y.fillna('Unknown')  # Handle missing values
                            else:
                                st.write(f"Target variable '{target_col}' is numeric but used for classification")
                                # Keep it as is for classification
                            
                        else:  # Regression
                            # For regression target (must be numeric)
                            if y.dtype == 'object':
                                try:
                                    y = pd.to_numeric(y, errors='coerce')
                                    na_count = y.isna().sum()
                                    if na_count > 0:
                                        st.warning(f"⚠️ Created {na_count} missing values in target during conversion")
                                        # For regression, we'll use the mean for missing targets
                                        y = y.fillna(y.mean())
                                except:
                                    st.error("❌ Target variable cannot be converted to numeric for regression.")
                                    st.stop()
                            
                            # Check if target still has missing values
                            na_count = y.isna().sum()
                            if na_count > 0:
                                st.warning(f"⚠️ Filling {na_count} missing values in target with mean")
                                y = y.fillna(y.mean())
                        
                        # Check if we still have enough data
                        if len(X) < 10:
                            st.error("❌ Not enough data points left after preprocessing.")
                            st.stop()
                        
                        # Log info about the data
                        st.write(f"Final dataset shape: {X.shape}")
                        st.write(f"Numeric features: {', '.join(numeric_features)}")
                        st.write(f"Categorical features: {', '.join(categorical_features)}")
                        if datetime_features:
                            st.write(f"Datetime features (extracted): {', '.join(datetime_features)}")
                        
                        # Store feature types for preprocessing pipeline
                        X_feature_types = {
                            'numeric': numeric_features,
                            'categorical': categorical_features
                        }
                        
                        # Log info about the data
                        st.write(f"Final dataset shape: {X.shape}")
                        st.write(f"Features: {', '.join(X.columns.tolist())}")
                        
                        # Split data
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            st.write(f"Training set: {X_train.shape[0]} samples")
                            st.write(f"Test set: {X_test.shape[0]} samples")
                        except Exception as e:
                            st.error(f"❌ Error splitting data: {str(e)}")
                            st.stop()
                                
                        # Create preprocessing pipelines based on our feature classification
                        numerical_transformer = Pipeline(steps=[
                            ('scaler', StandardScaler())
                        ])
                        
                        categorical_transformer = Pipeline(steps=[
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                        ])
                        
                        # Use feature types identified during preprocessing
                        numerical_cols = X_feature_types['numeric']
                        categorical_cols = X_feature_types['categorical']
                        
                        # Combine preprocessing with checks for empty feature lists
                        transformers = []
                        
                        if numerical_cols:
                            transformers.append(('num', numerical_transformer, numerical_cols))
                            
                        if categorical_cols:
                            transformers.append(('cat', categorical_transformer, categorical_cols))
                            
                        if not transformers:
                            st.error("❌ No usable features left after preprocessing.")
                            st.stop()
                            
                        preprocessor = ColumnTransformer(transformers=transformers)
                        
                        # Select the estimator based on user choice
                        if problem_type == "classification":
                            if model_type == "Logistic Regression":
                                from sklearn.linear_model import LogisticRegression
                                estimator = LogisticRegression()
                            elif model_type == "Random Forest":
                                from sklearn.ensemble import RandomForestClassifier
                                estimator = RandomForestClassifier()
                            elif model_type == "Decision Tree":
                                from sklearn.tree import DecisionTreeClassifier
                                estimator = DecisionTreeClassifier()
                            elif model_type == "Support Vector Machine":
                                from sklearn.svm import SVC
                                estimator = SVC(probability=True)
                            elif model_type == "Gradient Boosting":
                                from sklearn.ensemble import GradientBoostingClassifier
                                estimator = GradientBoostingClassifier()
                            elif model_type == "Neural Network":
                                from sklearn.neural_network import MLPClassifier
                                estimator = MLPClassifier(max_iter=1000)
                        else:  # Regression
                            if model_type == "Linear Regression":
                                from sklearn.linear_model import LinearRegression
                                estimator = LinearRegression()
                            elif model_type == "Random Forest":
                                from sklearn.ensemble import RandomForestRegressor
                                estimator = RandomForestRegressor()
                            elif model_type == "Decision Tree":
                                from sklearn.tree import DecisionTreeRegressor
                                estimator = DecisionTreeRegressor()
                            elif model_type == "Support Vector Machine":
                                from sklearn.svm import SVR
                                estimator = SVR()
                            elif model_type == "Gradient Boosting":
                                from sklearn.ensemble import GradientBoostingRegressor
                                estimator = GradientBoostingRegressor()
                            elif model_type == "Neural Network":
                                from sklearn.neural_network import MLPRegressor
                                estimator = MLPRegressor(max_iter=1000)
                        
                        # Create pipeline
                        model = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('model', estimator)
                        ])
                        
                        # Train model
                        st.write("Training model...")
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Evaluate model
                        st.subheader("Model Evaluation")
                        
                        if problem_type == "classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.metric("Accuracy", f"{accuracy:.4f}")
                                
                                # Try to calculate precision and recall
                                try:
                                    precision = precision_score(y_test, y_pred, average='weighted')
                                    st.metric("Precision", f"{precision:.4f}")
                                except:
                                    st.metric("Precision", "N/A")
                            
                            with metrics_col2:
                                try:
                                    recall = recall_score(y_test, y_pred, average='weighted')
                                    st.metric("Recall", f"{recall:.4f}")
                                    
                                    f1 = f1_score(y_test, y_pred, average='weighted')
                                    st.metric("F1 Score", f"{f1:.4f}")
                                except:
                                    st.metric("Recall", "N/A")
                                    st.metric("F1 Score", "N/A")
                            
                            # Confusion Matrix
                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(y_test, y_pred)
                            
                            st.write("Confusion Matrix:")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            plt.xlabel('Predicted Labels')
                            plt.ylabel('True Labels')
                            st.pyplot(fig)
                            
                        else:  # Regression
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.metric("MSE", f"{mse:.4f}")
                                st.metric("RMSE", f"{rmse:.4f}")
                                
                            with metrics_col2:
                                st.metric("MAE", f"{mae:.4f}")
                                st.metric("R² Score", f"{r2:.4f}")
                            
                            # Actual vs Predicted Plot
                            st.write("Actual vs Predicted Values:")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                            plt.xlabel('Actual Values')
                            plt.ylabel('Predicted Values')
                            
                            # Add the identity line (perfect predictions)
                            min_val = min(y_test.min(), y_pred.min())
                            max_val = max(y_test.max(), y_pred.max())
                            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                            
                            st.pyplot(fig)
                        
                        # Feature importance (if available)
                        if hasattr(model[-1], 'feature_importances_'):
                            st.subheader("Feature Importance")
                            
                            # Get feature names after preprocessing
                            if hasattr(model[0], 'get_feature_names_out'):
                                try:
                                    feature_names = model[0].get_feature_names_out()
                                except:
                                    feature_names = [f"feature_{i}" for i in range(model[-1].feature_importances_.shape[0])]
                            else:
                                feature_names = [f"feature_{i}" for i in range(model[-1].feature_importances_.shape[0])]
                            
                            # Create DataFrame of feature importances
                            importances = pd.DataFrame({
                                'feature': feature_names[:len(model[-1].feature_importances_)],
                                'importance': model[-1].feature_importances_
                            })
                            
                            # Sort by importance
                            importances = importances.sort_values('importance', ascending=False)
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='importance', y='feature', data=importances.head(15), ax=ax)
                            plt.title('Feature Importance')
                            st.pyplot(fig)
                        
                        # Allow model download
                        import pickle
                        import base64
                        
                        # Save model
                        model_filename = f"{model_type.lower().replace(' ', '_')}_model.pkl"
                        with open(model_filename, 'wb') as file:
                            pickle.dump(model, file)
                        
                        # Function to create download link
                        def get_binary_file_downloader_html(bin_file, file_label='File'):
                            with open(bin_file, 'rb') as f:
                                data = f.read()
                            bin_str = base64.b64encode(data).decode()
                            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Download {file_label}</a>'
                            return href
                        
                        st.markdown(get_binary_file_downloader_html(model_filename, 'Trained Model'), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Error building model: {str(e)}")
                    st.error("Stack Trace:")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
    
    # Show example of what the app can do
    st.subheader("🌟 Features")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📊 Data Overview**")
        st.markdown("- View data summary")
        st.markdown("- Check data types")
        st.markdown("- Find missing values")
        st.markdown("- See correlation matrix")
    
    with col2:
        st.markdown("**🔍 Data Exploration**")
        st.markdown("- Analyze individual columns")
        st.markdown("- View value distributions")
        st.markdown("- Check unique values")
        st.markdown("- Examine outliers")
    
    with col3:
        st.markdown("**🤖 AI Assistant**")
        st.markdown("- Ask questions in plain English")
        st.markdown("- Get insights about your data")
        st.markdown("- Perform complex analysis")
        st.markdown("- Save insights for later")
        
    with col4:
        st.markdown("**🧠 Model Building**")
        st.markdown("- Train ML models")
        st.markdown("- Evaluate performance")
        st.markdown("- View feature importance")
        st.markdown("- Download trained models")
