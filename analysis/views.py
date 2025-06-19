import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server-side plotting

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from django.shortcuts import render
from django.http import HttpResponseBadRequest
from .forms import UploadFileForm

# ------------------------------------------------------------------
#  Store DataFrames in memory (keyed by Django session key)
# ------------------------------------------------------------------
class FrameStore:
    _frames = {}

    @classmethod
    def get(cls, key):
        return cls._frames.get(key)

    @classmethod
    def set(cls, key, df):
        cls._frames[key] = df

    @classmethod
    def clear(cls, key):
        cls._frames.pop(key, None)

# ------------------------------------------------------------------
#  Cleaning helpers
# ------------------------------------------------------------------
def clean_df(df, action, fill_method=None):
    if action == 'dropna':
        return df.dropna()
    if action == 'drop_duplicates':
        return df.drop_duplicates()
    if action == 'fillna':
        if fill_method == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        if fill_method == 'median':
            return df.fillna(df.median(numeric_only=True))
        if fill_method == 'mode':
            return df.fillna(df.mode().iloc[0])
    return df

# ------------------------------------------------------------------
#  Plot helpers – returns base64-encoded PNG for <img src="...">
# ------------------------------------------------------------------
def make_plot(df, kind, x, y=None, color=None, linewidth=2, bins=30, xlabel=None, ylabel=None):
    plt.clf()  # clear previous figure

    if kind == 'box':
        if y:
            sns.boxplot(data=df, x=x, y=y, color=color)
        else:
            sns.boxplot(data=df, x=x, color=color)
    elif kind == 'hist':
        df[x].plot(kind='hist', bins=bins, color=color)
    elif kind == 'scatter':
        # Use 2D scatter plot for x and y columns with user specified color and linewidth
        if x in df.columns and y in df.columns:
            sns.scatterplot(data=df, x=x, y=y, color=color, linewidth=linewidth)
        else:
            raise ValueError('x and y columns required for scatter plot')
    elif kind == 'heatmap':
        # Correlation heatmap of numeric columns x and y if present
        cols = []
        if x and x in df.columns:
            cols.append(x)
        if y and y in df.columns:
            cols.append(y)
        if len(cols) < 2:
            raise ValueError('Not enough columns for heatmap')
        corr = df[cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=linewidth)
        plt.title('Correlation Heatmap')
    else:
        raise ValueError('Unsupported chart')

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{encoded}'

def main_view(request):
    context = {}
    session_key = request.session.session_key or request.session.save() or request.session.session_key

    if request.method == 'POST':
        if 'file' in request.FILES:
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                csv = request.FILES['file']
                if not csv.name.endswith('.csv'):
                    return HttpResponseBadRequest('Only CSV files are supported.')
                try:
                    df = pd.read_csv(csv)
                    FrameStore.set(session_key, df)
                    null_counts = df.isnull().sum()
                    duplicated_rows_count = df.duplicated().sum()
                    total_null_count = null_counts.sum()
                    context['total_null_count'] = total_null_count
                    context['duplicated_rows_count'] = duplicated_rows_count
                    context['total_shape'] = f"Total shape of the DataFrame: {df.shape}"
                except Exception as e:
                    context['error'] = f"Error reading CSV file: {str(e)}"
        else:
            df = FrameStore.get(session_key)
            if df is None:
                context['error'] = 'Please upload a CSV file first.'

    else:
        df = FrameStore.get(session_key)
        if df is not None:
            null_counts = df.isnull().sum()
            duplicated_rows_count = df.duplicated().sum()
            total_null_count = null_counts.sum()
            context['total_null_count'] = total_null_count
            context['duplicated_rows_count'] = duplicated_rows_count
            context['total_shape'] = f"Total shape of the DataFrame: {df.shape}"

    context['upload_form'] = UploadFileForm()
    return render(request, 'main.html', context)

def cleaning_view(request):
    context = {}
    session_key = request.session.session_key or request.session.save() or request.session.session_key
    df = FrameStore.get(session_key)
    if df is None:
        context['error'] = 'Please upload a CSV file first.'
        return render(request, 'cleaning.html', context)

    original_null_count = 0  # Initialize to avoid unbound error

    if request.method == 'POST':
        action = request.POST.get('action')
        fill_method = request.POST.get('fill_method')
        try:
            if action == 'drop_duplicates':
                original_shape = df.shape
                context['original_shape'] = f"Original shape before removing duplicates: {original_shape}"
            if action == 'dropna':
                original_null_count = df.isnull().sum().sum()
            if action == 'value_count':
                value_counts_sum = {col: df[col].value_counts().sum() for col in df.columns}
                context['value_counts_sum'] = value_counts_sum
            if action != 'value_count':
                df = clean_df(df, action, fill_method)
                FrameStore.set(session_key, df)
                if action == 'fillna' and fill_method == 'mean':
                    if original_null_count > 0:
                        context['fillna_mean_message'] = "Values have been filled with null values."
                    else:
                        context['fillna_mean_message'] = "No null value found."
            if action == 'drop_duplicates':
                new_shape = df.shape
                context['new_size'] = f"Shape after removing duplicates: {new_shape}"
            if action == 'dropna':
                new_null_count = df.isnull().sum().sum()
                new_shape_after_dropna = df.shape
                context['new_shape_after_dropna'] = f"Shape after dropping null values: {new_shape_after_dropna}"
                removed_nulls = original_null_count - new_null_count
                if removed_nulls > 0:
                    context['nulls_removed'] = f"Removed {removed_nulls} null values."
                else:
                    context['no_null_values'] = "No null values in the data."
        except Exception as e:
            context['error'] = f"Error cleaning data: {str(e)}"
            return render(request, 'cleaning.html', context)

    context['cols'] = list(df.columns)
    context['preview_html'] = df.head().to_html(classes='table table-sm')
    return render(request, 'cleaning.html', context)

def visualization_view(request):
    context = {}
    session_key = request.session.session_key or request.session.save() or request.session.session_key
    df = FrameStore.get(session_key)

    if df is None:
        context['error'] = 'Please upload a CSV file first.'
        return render(request, 'visualization.html', context)

    context['columns'] = list(df.columns)  # Always pass columns for dropdowns

    if request.method == 'POST':
        kind = request.POST.get('chart_type')
        x = request.POST.get('x_col')
        y = request.POST.get('y_col') or None
        color = request.POST.get('color') or None
        xlabel = request.POST.get('xlabel') or None
        ylabel = request.POST.get('ylabel') or None

        # Keep form state
        context['selected_chart_type'] = kind
        context['selected_x'] = x
        context['selected_y'] = y
        context['selected_color'] = color
        context['linewidth'] = request.POST.get('linewidth')
        context['bins'] = request.POST.get('bins')
        context['xlabel'] = xlabel
        context['ylabel'] = ylabel

        try:
            linewidth = float(context['linewidth'] or 2)
        except Exception as e:
            context['error'] = f"Invalid linewidth value: {str(e)}"
            return render(request, 'visualization.html', context)

        try:
            bins = int(context['bins'] or 30)
        except Exception as e:
            context['error'] = f"Invalid bins value: {str(e)}"
            return render(request, 'visualization.html', context)

        try:
            context['chart_uri'] = make_plot(df, kind, x, y, color, linewidth, bins, xlabel, ylabel)
        except Exception as e:
            context['error'] = f"Error creating plot: {str(e)}"
            return render(request, 'visualization.html', context)

    return render(request, 'visualization.html', context)

import pandas as pd
from django.shortcuts import render, redirect
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_squared_error

def ml_algo(request):
    import logging
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    logger = logging.getLogger(__name__)

    session_key = request.session.session_key or request.session.save() or request.session.session_key
    df = FrameStore.get(session_key)
    if df is None:
        error = "No data found. Please upload a CSV file first."
        return render(request, 'ml_algo.html', {'columns': [], 'result': None, 'error': error})

    columns = df.columns.tolist()
    result = None
    error = None

    if request.method == 'POST':
        task = request.POST.get("task")
        algo = request.POST.get("algorithm")
        features = request.POST.getlist("features")
        target = request.POST.get("target")

        if not features or not target:
            error = "Please select feature columns and target column."
            return render(request, 'ml_algo.html', {'columns': columns, 'result': None, 'error': error})

        # Limit dataset size to 10,000 rows for performance
        if len(df) > 10000:
            df_sampled = df.sample(n=10000, random_state=42)
        else:
            df_sampled = df

        X = df_sampled[features]
        y = df_sampled[target]

        # Validate data types for classification and regression
        try:
            if task == "classification":
                # Target should be categorical or discrete
                if not pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_categorical_dtype(y):
                    error = "Target column must be categorical or numeric for classification."
                    return render(request, 'ml_algo.html', {'columns': columns, 'result': None, 'error': error})
            elif task == "regression":
                # Target should be numeric
                if not pd.api.types.is_numeric_dtype(y):
                    error = "Target column must be numeric for regression."
                    return render(request, 'ml_algo.html', {'columns': columns, 'result': None, 'error': error})
        except Exception as e:
            logger.error(f"Data type validation error: {str(e)}")
            error = f"Data type validation error: {str(e)}"
            return render(request, 'ml_algo.html', {'columns': columns, 'result': None, 'error': error})

        try:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if task == "regression":
                if algo == "linear_regression":
                    model = LinearRegression()
                elif algo == "polynomial_regression":
                    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                else:
                    error = "Unsupported regression algorithm selected."
                    return render(request, 'ml_algo.html', {'columns': columns, 'result': None, 'error': error})

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                from sklearn.metrics import r2_score
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                result = f"Mean Squared Error: {mse}, R² Score: {r2}"

            elif task == "classification":
                if algo == "logistic_regression":
                    model = LogisticRegression(max_iter=1000)
                elif algo == "random_forest":
                    model = RandomForestClassifier()
                elif algo == "svm":
                    # Add StandardScaler in pipeline for SVM
                    model = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
                elif algo == "gradient_boost":
                    model = GradientBoostingClassifier()
                else:
                    error = "Unsupported classification algorithm selected."
                    return render(request, 'ml_algo.html', {'columns': columns, 'result': None, 'error': error})

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                result = f"Accuracy: {accuracy_score(y_test, predictions)}"

        except Exception as e:
            logger.error(f"Model training or prediction error: {str(e)}")
            error = f"Error during model training or prediction: {str(e)}"

    return render(request, 'ml_algo.html', {'columns': columns, 'result': result, 'error': error})
