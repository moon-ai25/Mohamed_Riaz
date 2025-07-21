import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os



def extract_data(file_path):
    print("Extracting data...")
    return pd.read_csv(file_path)



def transform_data(df):
    print("Transforming data...")

    X = df.copy()
    y = None  

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X_transformed = preprocessor.fit_transform(X)


    X_transformed_df = pd.DataFrame(X_transformed)

    return X_transformed_df, y


def load_data(X, y=None, output_path="processed-data.csv"):
    print("Loading data...")
    df = X if y is None else pd.concat([X, y.reset_index(drop=True)], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def run_pipeline(input_csv):
    df = extract_data(input_csv)
    X_transformed, y = transform_data(df)
    load_data(X_transformed, y)


if __name__ == "__main__":
    INPUT_CSV = "data.csv"  
    if not os.path.exists(INPUT_CSV):
        print(f"File '{INPUT_CSV}' not found. Please add your dataset.")
    else:
        run_pipeline(INPUT_CSV)
