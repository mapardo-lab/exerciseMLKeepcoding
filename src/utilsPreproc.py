import pandas as pd

def preprocess_features01(df): 
    """
    Preprocess Amenities feature and select the most relevant ones
    """
    df_out = df.copy()
    # amenities feature
    amenities_sel = ['air conditioning', 'family/kid friendly', 'tv', 'carbon monoxide detector', 
                    'dryer', 'lock on bedroom door', 'cable tv', 'iron', 'elevator in building', 'internet', 'kitchen']
    df_out['AmenitiesStr'] = df_out['Amenities'].str.lower().str.split(',').tolist()
    df_out['AmenitiesStr'] = df_out['AmenitiesStr'].apply(lambda x: preprocess_amenities(x, amenities_sel))
    return df_out

def preprocess_amenities(amenities, selection):
    """
    Select amenities from Amenities feature
    """
    result = []
    if isinstance(amenities, list):
        for amenity in amenities:
            if amenity in selection:
                result.append(amenity)
        return result
    else:
        return result

def preprocess_target01(df): 
    """
    Remove samples with missing Price data and set Price as target
    """
    df_out = df[~(df['Price'].isna())].copy()
    df_out['target'] = df_out['Price']
    return df_out
