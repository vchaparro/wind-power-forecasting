import numpy as np


def get_cape(Y_true, Y_pred):
    """
    CAPE (Cumulated Absolute Percentage Error) function used by CNR for the evaluation of predictions

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """

    # CAPE function
    cape = 100 * np.sum(np.abs(Y_pred - Y_true)) / np.sum(Y_true)

    return cape
