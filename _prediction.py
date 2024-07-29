import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uns_predict_output(img, estimator):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()
    flat_pixels = flat_pixels.filled(0)

    # predict and replace mask
    result_cla = estimator.predict(flat_pixels)+1
    result_cla = np.ma.masked_array(
        data=result_cla, mask=flat_pixels_mask.any(axis=1)
    )

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result_cla = result_cla.reshape((1, window.height, window.width))

    return result_cla

def uns_predict_multioutput(img, estimator):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result with the
        dimensions in the order of (target, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    mask2d = img.mask.any(axis=0)

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)

    # predict probabilities
    result = estimator.predict(flat_pixels)+1

    # reshape class probabilities back to 3D array [class, rows, cols]
    result = result.reshape((window.height, window.width, result.shape[1]))

    # reshape band into rasterio format [band, row, col]
    result = result.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result.shape[0],
        axis=0
    )

    # convert proba to masked array
    result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

    return result

def predict_output(img, estimator):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()
    flat_pixels = flat_pixels.filled(0)

    # predict and replace mask
    flat_pixels = np.nan_to_num(flat_pixels)
    result_cla = estimator.predict(flat_pixels)
    result_cla = np.ma.masked_array(
        data=result_cla, mask=flat_pixels_mask.any(axis=1)
    )

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result_cla = result_cla.reshape((1, window.height, window.width))

    return result_cla

def dpl_predict_output(img, estimator, shape):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()
    flat_pixels = flat_pixels.filled(0)
    flat_pixels = np.reshape(flat_pixels, (flat_pixels.shape[0], shape[0], shape[1]))
    #print(flat_pixels.shape)

    # predict and replace mask
    result_cla = estimator.predict(flat_pixels)
    result_cla = np.argmax(result_cla, axis=1) + 1
    result_cla = np.ma.masked_array(
        data=result_cla, mask=flat_pixels_mask.any(axis=1)
    )

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result_cla = result_cla.reshape((1, window.height, window.width))

    return result_cla

def torch_predict_output(img, estimator, shape):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()
    flat_pixels = flat_pixels.filled(0)
    flat_pixels = np.reshape(flat_pixels, (flat_pixels.shape[0], shape[0], shape[1]))
    #print(flat_pixels.shape)

    # predict and replace mask
    Test_dataloader = torch.utils.data.DataLoader(flat_pixels, batch_size=16)  # Adjust batch size as needed

    result_cla = []
    all_targets = []

    # Loop through batches for evaluation
    for batch_x in tqdm(Test_dataloader, desc="Predicting Bits"):
        batch_x = batch_x.to(device)  # Move batch to device
        batch_y_pred = estimator(batch_x)
        batch_y_pred = torch.argmax(batch_y_pred, dim=1) + 1
        batch_y_pred = batch_y_pred.cpu().detach().numpy()  # Transfer back to CPU

        # Process and append predictions and targets
        #batch_y_pred = batch_y_pred.reshape(-1, batch_y_pred.shape[-1])
        result_cla.extend(batch_y_pred)
    result_cla = np.array(result_cla)
    result_cla = np.ma.masked_array(
        data=result_cla, mask=flat_pixels_mask.any(axis=1)
    )

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result_cla = result_cla.reshape((1, window.height, window.width))

    return result_cla

def torch_regression_output_scaler(img, estimator, shape, scaler):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()
    flat_pixels = flat_pixels.filled(0)
    flat_pixels = np.reshape(flat_pixels, (flat_pixels.shape[0], shape[0], shape[1]))
    #print(flat_pixels.shape)

    # predict and replace mask
    #flat_pixels = torch.from_numpy(flat_pixels)
    Test_dataloader = torch.utils.data.DataLoader(flat_pixels, batch_size=16)  # Adjust batch size as needed

    result_cla = []
    all_targets = []

    # Loop through batches for evaluation
    for batch_x in tqdm(Test_dataloader, desc="Predicting Bits"):
        batch_x = batch_x.to(device)  # Move batch to device
        batch_y_pred = estimator(batch_x)
        batch_y_pred = batch_y_pred.cpu().detach().numpy()  # Transfer back to CPU

        # Process and append predictions and targets
        #batch_y_pred = batch_y_pred.reshape(-1, batch_y_pred.shape[-1])
        if scaler is None:
            result_cla.extend(batch_y_pred)
        else:
            result_cla.extend(scaler.inverse_transform(batch_y_pred))
    result_cla = np.array(result_cla)
    print("result_cla shape", result_cla.shape)
    print("flat_pixels_mask shape", flat_pixels_mask.shape)
    result_cla = np.ma.masked_array(
        data=result_cla, mask=flat_pixels_mask.any(axis=1)
    )

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result_cla = result_cla.reshape((1, window.height, window.width))

    return result_cla

def mlp_predict_output(img, estimator):
    """Prediction function for classification or regression response.

    Parameters
    ----
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        2d numpy array representing a single band raster containing the
        classification or regression result.
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))

    # create mask for NaN values and replace with number
    flat_pixels_mask = flat_pixels.mask.copy()
    flat_pixels = flat_pixels.filled(0)
    #print(flat_pixels.shape)

    # predict and replace mask
    result_cla = estimator.predict(flat_pixels)
    result_cla = np.argmax(result_cla, axis=1) + 1
    result_cla = np.ma.masked_array(
        data=result_cla, mask=flat_pixels_mask.any(axis=1)
    )

    # reshape the prediction from a 1D into 3D array [band, row, col]
    result_cla = result_cla.reshape((1, window.height, window.width))

    return result_cla


def predict_prob(img, estimator):
    """Class probabilities function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        Multi band raster as a 3d numpy array containing the probabilities
        associated with each class. ndarray dimensions are in the order of
        (class, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands (transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    mask2d = img.mask.any(axis=0)

    # then resample into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)

    # predict probabilities
    result_proba = estimator.predict_proba(flat_pixels)

    # reshape class probabilities back to 3D array [class, rows, cols]
    result_proba = result_proba.reshape(
        (window.height, window.width, result_proba.shape[1])
    )

    # reshape band into rasterio format [band, row, col]
    result_proba = result_proba.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result_proba.shape[0],
        axis=0
    )

    # convert proba to masked array
    result_proba = np.ma.masked_array(
        result_proba,
        mask=mask3d,
        fill_value=np.nan
    )

    return result_proba


def predict_multioutput(img, estimator):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result with the
        dimensions in the order of (target, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    mask2d = img.mask.any(axis=0)

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)

    # predict probabilities
    flat_pixels = np.nan_to_num(flat_pixels)
    result = estimator.predict(flat_pixels)

    # reshape class probabilities back to 3D array [class, rows, cols]
    result = result.reshape((window.height, window.width, result.shape[1]))

    # reshape band into rasterio format [band, row, col]
    result = result.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result.shape[0],
        axis=0
    )

    # convert proba to masked array
    result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

    return result

def dpl_predict_multioutput(img, estimator, shape):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result with the
        dimensions in the order of (target, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    mask2d = img.mask.any(axis=0)

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)
    #print(flat_pixels.shape)
    flat_pixels = np.reshape(flat_pixels, (flat_pixels.shape[0], shape[0], shape[1]))
    #print(flat_pixels.shape)

    # predict probabilities
    result = estimator.predict(flat_pixels)
    result = np.argmax(result, axis=1) + 1

    # reshape class probabilities back to 3D array [class, rows, cols]
    result = result.reshape((window.height, window.width, result.shape[1]))

    # reshape band into rasterio format [band, row, col]
    result = result.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result.shape[0],
        axis=0
    )

    # convert proba to masked array
    result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

    return result


def torch_predict_multioutput(img, estimator, shape):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result with the
        dimensions in the order of (target, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    mask2d = img.mask.any(axis=0)

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)
    #print(flat_pixels.shape)
    flat_pixels = np.reshape(flat_pixels, (flat_pixels.shape[0], shape[0], shape[1]))
    #print(flat_pixels.shape)

    # predict probabilities
    flat_pixels = torch.from_numpy(flat_pixels)
    result = estimator(flat_pixels.to(device))
    result = torch.argmax(result, dim=1) + 1
    result = result.cpu()
    result = result.numpy()
    result = scaler.inverse_transform(result)

    # reshape class probabilities back to 3D array [class, rows, cols]
    result = result.reshape((window.height, window.width, result.shape[1]))

    # reshape band into rasterio format [band, row, col]
    result = result.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result.shape[0],
        axis=0
    )

    # convert proba to masked array
    result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

    return result

def torch_regression_multioutput_scaler(img, estimator, shape, scaler):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result with the
        dimensions in the order of (target, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    mask2d = img.mask.any(axis=0)

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)
    #print(flat_pixels.shape)
    flat_pixels = np.reshape(flat_pixels, (flat_pixels.shape[0], shape[0], shape[1]))
    #print(flat_pixels.shape)

    # predict probabilities
    flat_pixels = torch.from_numpy(flat_pixels)
    Test_dataloader = torch.utils.data.DataLoader(flat_pixels, batch_size=16)  # Adjust batch size as needed

    result = []
    all_targets = []

    # Loop through batches for evaluation
    for batch_x in tqdm(Test_dataloader, desc="Predicting Bits"):
        batch_x = batch_x.to(device)  # Move batch to device
        batch_y_pred = estimator(batch_x)
        batch_y_pred = batch_y_pred.cpu().detach().numpy()  # Transfer back to CPU

        # Process and append predictions and targets
        #batch_y_pred = batch_y_pred.reshape(-1, batch_y_pred.shape[-1])
        if scaler is None:
            result.extend(batch_y_pred)
        else:
            result.extend(scaler.inverse_transform(batch_y_pred))
    result = np.array(result)
    result = estimator(flat_pixels.to(device))
    result = result.cpu().detach().numpy()

    # reshape class probabilities back to 3D array [class, rows, cols]
    result = result.reshape((window.height, window.width, result.shape[1]))
    print("Multioutput result shape: ", result.shape)

    # reshape band into rasterio format [band, row, col]
    result = result.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result.shape[0],
        axis=0
    )

    # convert proba to masked array
    result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

    return result

def mlp_predict_multioutput(img, estimator):
    """Multi-target prediction function.

    Parameters
    ----------
    img : tuple (window, numpy.ndarray)
        A window object, and a 3d ndarray of raster data with the dimensions in
        order of (band, rows, columns).

    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Returns
    -------
    numpy.ndarray
        3d numpy array representing the multi-target prediction result with the
        dimensions in the order of (target, row, column).
    """
    window, img = img

    # reorder into rows, cols, bands(transpose)
    n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]
    mask2d = img.mask.any(axis=0)

    # reshape into 2D array (rows=sample_n, cols=band_values)
    n_samples = rows * cols
    flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features))
    flat_pixels = flat_pixels.filled(0)
    #print(flat_pixels.shape)
    #print(flat_pixels.shape)

    # predict probabilities
    result = estimator.predict(flat_pixels)
    result = np.argmax(result, axis=1) + 1

    # reshape class probabilities back to 3D array [class, rows, cols]
    result = result.reshape((window.height, window.width, result.shape[1]))

    # reshape band into rasterio format [band, row, col]
    result = result.transpose(2, 0, 1)

    # repeat mask for n_bands
    mask3d = np.repeat(
        a=mask2d[np.newaxis, :, :],
        repeats=result.shape[0],
        axis=0
    )

    # convert proba to masked array
    result = np.ma.masked_array(result, mask=mask3d, fill_value=np.nan)

    return result
