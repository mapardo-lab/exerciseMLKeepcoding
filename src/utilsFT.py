from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class MultiLabelBinarizerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for MultiLabelBinarizer to work with ColumnTransformer
    """
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X, y=None):
        X_series = self._safe_squeeze(X)
        self.mlb.fit(X_series)
        return self
    
    def transform(self, X):
        X_series = self._safe_squeeze(X)
        return self.mlb.transform(X_series)
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_

    def _safe_squeeze(self, X):
        """
        Safely convert one column dataframe to pd.Series with squeeze function
        Only squeezes if X is a one-column DataFrame, otherwise raises error.
        """
        # Check if X is a DataFrame
        if not hasattr(X, 'iloc') or not hasattr(X, 'shape'):
            raise ValueError(
                f"Input must be a DataFrame. Got {type(X)} instead. "
                "Make sure this transformer is used in ColumnTransformer with proper column selection."
            )
        
        # Check if DataFrame has exactly one column
        if X.shape[1] != 1:
            raise ValueError(
                f"Expected 1 column, but got {X.shape[1]} columns. "
                "This transformer should only be applied to a single column. "
                "Check your ColumnTransformer configuration."
            )
        
        # Extract the single column
        return X.squeeze()


class TransformMetadata:
    """Factory to create different transformation configurations"""
    
    @staticmethod
    def get_transformation(name):
        """Get different transformation configurations"""
        
        # Target Encoding
        categorical_features = ['Property Type', 'Cancellation Policy']
        categorical_pipeline = Pipeline(steps=[
            ('target_enc', TargetEncoder(target_type="continuous", smooth = "auto")),
            ('scaler', StandardScaler())
        ])
        # Standard Scaler
        numeric_features = ['Accommodates','Bathrooms']
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
        ])
        # Amenities
        amenities_features = ['AmenitiesStr']
        amenities_pipeline = Pipeline(steps=[
            ('multilabelbin', MultiLabelBinarizerWrapper())
        ])

        transformations = {
            'transform01': ColumnTransformer(
                transformers=[
                    ('num', numeric_pipeline, numeric_features),
                    ('cat', categorical_pipeline, categorical_features),
                    ('ame', MultiLabelBinarizerWrapper(), amenities_features)
                ],
                remainder="drop"
            )
        }

        if name not in transformations:
            raise ValueError(f"Unknown transformation: {name}. "
                           f"Available: {list(transformations.keys())}")
        
        return transformations[name]
