from .session_data_manager import session_data_manager

def choose_dataset(session_id, use_default_dataset=True, file=None):
    """
    Stores a dataset for a given session ID.

    Parameters
    ----------
    session_id : str
        The session ID
    use_default_dataset : bool, optional
        Whether to use the default dataset (default is True)
    file : str or file-like, optional
        The file to load as the dataset

    Returns
    -------
    A dictionary with the following fields:
        'fields': a list of the field names in the dataset
        'nonCtsFields': a list of the non-continuous fields in the dataset
    """
    session_data_manager.add_dataset(session_id, use_default_dataset, file)

    dataset = session_data_manager.get_session_data(session_id)['dataset']

    # Identify and return the field names
    fields = dataset.columns.values.tolist()
    nonCtsFields = list(dataset.dtypes[(dataset.dtypes != "int64") & (dataset.dtypes != "float64")].index)

    return {
        'fields': fields,
        'nonCtsFields': nonCtsFields
    }
