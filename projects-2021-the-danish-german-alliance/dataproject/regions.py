def only_keep_regions(df):
    """ Only keeps geographical areas at the smalles aggregation levels
    Args:
        df (pd.DataFrame): pandas dataframe with digits smaller than 4 in the column "geo"

    Returns:
        df (pd.DataFrame): pandas dataframe with digits equal to 4 in the columns "geo"

    """ 

    df = df[df['geo'].apply(lambda x: len(str(x)) == 4)]
    
    return df