import numpy  as np  
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, MultiLabelBinarizer, StandardScaler

def standarizate(df, scaler = None):
    """
    Standardizes numerical features in a DataFrame using scikit-learn's StandardScaler.

    Fits a scaler to the data (if not provided) and returns the standardized DataFrame
    along with the scaler object for consistent transformations on new data.
    """
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df)
    stand = scaler.transform(df)

    df_stand = pd.DataFrame(stand, columns = df.columns)
    df_stand.index = df.index

    return df_stand, scaler
    
def feature_multilabelbin(df, feature, mlb = None):
    """
    Encodes a categorical feature with comma-separated labels into a binary matrix.

    Uses scikit-learn's MultiLabelBinarizer to transform strings like "a,b,c" into
    one-hot encoded columns (e.g., columns "a", "b", "c" with values 0/1).
    """
    df_amenities = pd.DataFrame({'list' : df[feature].str.lower().str.split(',').tolist()})
    if mlb is None:
        mlb = MultiLabelBinarizer()
        mlb.fit(df_amenities['list']) 
    encoded = mlb.transform(df_amenities['list'])

    # Convert to DataFrame with proper column names
    df_result = pd.DataFrame(encoded, columns=mlb.classes_)
    df_result.index = df.index
    return df_result, mlb

def features_onehot(df, encoder = None):
    """
    Performs one-hot encoding on categorical features in a DataFrame.

    Uses scikit-learn's OneHotEncoder to convert categorical variables into binary columns,
    automatically dropping the first category to avoid dummy variable trap.
    """
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy trap
        encoder.fit(df)
    encoded = encoder.transform(df)
    feature_names = encoder.get_feature_names_out(input_features=df.columns)
    df_result = pd.DataFrame(encoded, columns=feature_names)
    df_result.index = df.index

    return df_result, encoder
    
def features_target(df, df_target = None, encoder = None):
    """
    Applies target encoding to categorical features using category means.

    Uses category-mean encoding (with smoothing) for categorical variables based on
    the target variable values. Useful for high-cardinality categorical features.
    """
    features_TE = list(map(lambda x: x + ' TE', df.columns))
    if encoder is None:
        encoder = TargetEncoder(target_type="continuous", smooth = "auto")
        encoder.fit(df, df_target)
    encoded = encoder.transform(df)
    df_result = pd.DataFrame(encoded, columns=features_TE)
    df_result.index = df.index

    return df_result, encoder
    
def imputation(df):
    """
    Performs custom missing value imputation for Airbnb-style listing data.
    """
    df_imp = df.copy()
    # Remove samples without info about Price
    df_imp = df_imp[~(df_imp['Price'].isna())]
    # impute na values in Bedrooms
    df_imp['Bedrooms'] = df_imp['Bedrooms'].fillna(0)
    # impute na values in Bathrooms
    df_imp['Bathrooms'] = df_imp['Bathrooms'].fillna(0)
    # impute Beds with value from Bedrooms
    df_imp['Beds'] = df_imp['Beds'].fillna(df_imp['Bedrooms'])
    # impute na values in Amenities
    df_imp['Amenities'] = df_imp['Amenities'].fillna('No services')
    # impute zero values in Guests Included by Beds
    df_imp['Guests Included'] = df_imp['Guests Included'].where(df_imp['Guests Included'] != 0, df_imp['Beds'])
    return df_imp

def data_proc(train, test, features):
    """
    Orchestrates complete feature processing pipeline for train/test datasets.
    """
    train = imputation(train)
    test = imputation(test)
    train1, mlb_amenities = feature_multilabelbin(train, 'Amenities')
    test1, _ = feature_multilabelbin(test, 'Amenities', mlb = mlb_amenities)
    train2, scaler = standarizate(train[features['standarizate']])
    test2, _ = standarizate(test[features['standarizate']], scaler = scaler)
    train3, encoder_onehot = features_onehot(train[features['onehotencode']])
    test3, _ = features_onehot(test[features['onehotencode']], encoder = encoder_onehot)
    train4, encoder_target = features_target(train[features['targetencode']], train[features['target']])
    test4, _ = features_target(test[features['targetencode']], encoder = encoder_target)
    X_train = pd.concat([train1, train2, train3, train4], axis = 1).drop(features['to_drop'], axis = 1)
    X_test = pd.concat([test1, test2, test3, test4], axis = 1).drop(features['to_drop'], axis = 1)
    y_train = train[features['target']]
    y_test = test[features['target']]
    return X_train, X_test, y_train, y_test