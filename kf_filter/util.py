"""
Utility functions for the kf_filter module.

Author: Richard Zhuang
Date: Apr 14, 2025
"""
import numpy as np
import xarray as xr
from kf_filter.consts import (
    g,
    radius_earth,
    beta,
    wave_func,
    wave_types,
    wave_args
)

from functools import reduce
import operator

def harmonic_func(n, period=365, num_fs=4):
    """
    Construct a harmonic function for regression.

    Parameters
    ----------
    n : int
        The sampling dimension obtained from the original data.

    period : float
        The period of the regression function.

    num_fs : int
        The number of frequency bands to use.

    Returns
    -------
    func : np.ndarray
        The matrix to regress on to the original timeseries.
    """
    func = np.zeros((num_fs*2+1, n), dtype=float)
    time = np.arange(0, n) * 2 * np.pi / period
    func[0, :] = np.ones(n)
    for i in range(num_fs):
        func[2*i+1, :] = np.sin(i * time)
        func[(i+1)*2, :] = np.cos(i * time)
    return func

def split_hann_taper(series_length, fraction):
    """
    Parameters
    ----------
    series_length : int
        The length of an array-like object.

    fraction : float
        The fraction of data points to be tapered off at each end
        of the array.

    Returns
    -------
    taper_weights : np.array
        A series of weights of length `series_length`.

    Implements `split cosine bell` taper of length `series_length`
    where only fraction of points are tapered (combined on both ends).

    Notes
    -----
    This returns a function that tapers to zero on the ends. 
    To taper to the mean of a series X:
    XTAPER = (X - X.mean())*series_taper + X.mean()
    """
    npts = int(np.rint(fraction * series_length))  # total size of taper
    taper = np.hanning(npts)
    series_taper = np.ones(series_length)
    series_taper[0 : npts // 2 + 1] = taper[0 : npts // 2 + 1]
    series_taper[-npts // 2 + 1 :] = taper[npts // 2 + 1 :]
    return series_taper

def _combine_plus_minus(logical_plus : list, 
                        logical_minus : list) -> xr.DataArray:
    """
    Combine masks from positive frequency domain
    with that from negative frequency domain.

    Parameters
    ----------
    logical_plus : list
        List of logical conditions for positive frequencies.
    logical_minus : list
        List of logical conditions for negative frequencies.

    Returns
    -------
    mask : xr.DataArray
        Combined mask.
    """
    omega_plus = reduce(lambda x, y: xr.apply_ufunc(xr.ufuncs.logical_and, x, y), logical_plus)
    omega_minus = reduce(lambda x, y: xr.apply_ufunc(xr.ufuncs.logical_and, x, y), logical_minus)

    return xr.ufuncs.logical_or(omega_plus, omega_minus)

def _nondim_k_omega(wavenumber : xr.DataArray | np.ndarray,
                    frequency : xr.DataArray | np.ndarray,
                    h : float) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Non-dimensionalize k and \omega based on the equivalent depth.

    Parameters
    ----------
    h : float
        Equivalent depth in meters.

    Returns
    k_nondim, omega_nondim : tuple[xr.DataArray, xr.DataArray]

    Notes
    -----
    c = sqrt(g * h)
    """
    wavenumber, frequency = _wrap_to_xarray(wavenumber, frequency)
    
    c = np.sqrt(g * h)

    # First convert to wavenumber per Earth radius (m^1)
    # before we nondimensionalize it according to Vallis (2012)
    k_nondim = wavenumber / radius_earth * np.sqrt(c / beta)

    # Here we convert linear frequency from np.fftfreq to 
    # angular frequency
    omega_nondim = frequency * 2 * np.pi / (24 * 3600) / np.sqrt(beta * c)

    # return nondimensionalized k, \omega
    return k_nondim, omega_nondim

def _wrap_to_xarray(wavenumber : np.ndarray,
                    frequency : np.ndarray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Wrap the wavenumber and frequency to xarray DataArray.
    """
    # Wrap the input to xarray DataArray if it is not already
    if not isinstance(wavenumber, xr.DataArray):
        wavenumber = xr.DataArray(wavenumber, coords=dict(wavenumber=wavenumber))
    if not isinstance(frequency, xr.DataArray):
        frequency = xr.DataArray(frequency, coords=dict(frequency=frequency))

    return wavenumber, frequency

def kf_mask(wavenumber : xr.DataArray | np.ndarray,
            frequency : xr.DataArray | np.ndarray,
            fmin: float | None = None, 
            fmax: float | None = None, 
            kmin: int | None = None, 
            kmax: int | None = None,
            return_individual: bool = False) -> xr.DataArray | tuple[list, list]:
    r"""
    A wavenumber-frequency filter for a combination of min/max frequency
    and min/max wavenumber.

    Parameters
    ----------
    fmin, fmax : float or None
        Minimum and maximum frequency for filtering

    kmin, kmax : int or None
        Minimum and maximum frequency for filtering

    return_individual : bool
        Whether or not to return logical_plus and logical_minus separately.

    Returns
    -------
    mask : xr.DataArray

    Notes
    -----
    In order to have the right results, *I think* we need to select
    frequency with [-fmax, -fmin] & [-kmax, -kmin] \union 
    [fmin, fmax] & [kmin, kmax].
    """
    wavenumber, frequency = _wrap_to_xarray(wavenumber, frequency)

    # do separately for positive and negative omega
    logical_plus = [(frequency > 0)]  # bounding box for positive omega
    logical_minus = [(frequency < 0)]  # bounding box for negative omega

    # need to do separately for positive frequency and negative frequency
    if fmin is not None:
        assert fmin > 0, 'Frequency "fmin" must be greater than 0.'
        logical_plus.append((frequency > fmin))
        logical_minus.append((frequency < -fmin))

    if fmax is not None:
        assert fmax > 0, 'Frequency "fmax" must be greater than 0.'
        logical_plus.append((frequency < fmax))
        logical_minus.append((frequency > -fmax))

    if kmin is not None:
        logical_plus.append((wavenumber > kmin))
        logical_minus.append((wavenumber < -kmin))

    if kmax is not None:
        logical_plus.append((wavenumber < kmax))
        logical_minus.append((wavenumber > -kmax))

    # Check if fmin and fmax are provided
    if (fmin is not None) and (fmax is not None):
        assert fmin < fmax, '"fmin" should be smaller than "fmax".'

    if (kmin is not None) and (kmax is not None):
        assert kmin < kmax, 'Wavenumber "kmin" should be smaller than "kmax".'

    # Filter both positive and negative frequencies (plus and minus)
    if return_individual:
        return logical_plus, logical_minus
    else:
        return _combine_plus_minus(logical_plus, logical_minus)
    
def wave_mask(wavenumber : xr.DataArray | np.ndarray,
              frequency : xr.DataArray | np.ndarray,
              wave_type : str,
              **kwargs) -> xr.DataArray:
        r"""Generic wave filtering.
        
        Parameters
        ----------
        wave_type : str
            One of 'kelvin', 'er', 'ig', 'eig', 'mrg'.

        Notes
        -----
        For TD-type wave, we do not attempt to nondimensionalize
        omega and k. 
        """
        wavenumber, frequency = _wrap_to_xarray(wavenumber, frequency)

        args = wave_args[wave_type]
        args.update(kwargs)

        logical_plus, logical_minus = kf_mask(wavenumber, 
                                              frequency,
                                              args['fmin'],
                                              args['fmax'],
                                              args['kmin'],
                                              args['kmax'], 
                                              return_individual=True)

        if wave_type not in wave_types:
            raise ValueError(f'Unsupported wave_type "{wave_type}".')

        # Select the dispersion relation function from class attributes
        func = wave_func[wave_type]

        hmin = args.get('hmin', None)
        hmax = args.get('hmax', None)
        n = args.get('n', 1)
        
        if hmin is not None:
            k, omega = _nondim_k_omega(wavenumber, frequency, hmin)
            logical_plus.append(func(operator.gt, omega, k, n))

            # For IG, EIG, MRG, need to use gt because of Omega^2
            if wave_type in ['ig', 'eig', 'mrg']:
                logical_minus.append(func(operator.gt, omega, k, n))
            # For ER and KW, use lt because of Omega
            else:
                logical_minus.append(func(operator.lt, omega, k, n))

        if hmax is not None:
            k, omega = _nondim_k_omega(wavenumber, frequency, hmax)
            logical_plus.append(func(operator.lt, omega, k, n))

            if wave_type in ['ig', 'eig', 'mrg']:
                logical_minus.append(func(operator.lt, omega, k, n))
            # For ER and KW, use gt because of Omega
            else:
                logical_minus.append(func(operator.gt, omega, k, n))

        return _combine_plus_minus(logical_plus, logical_minus)

def td_mask(wavenumber : xr.DataArray | np.ndarray,
            frequency : xr.DataArray | np.ndarray,
            **kwargs) -> xr.DataArray:
    r"""Returns a mask for tropical depression (TD).

    Parameters
    ----------
    wavenumber : xr.DataArray | np.ndarray
    frequency : xr.DataArray | np.ndarray

    fmin, fmax : float or None
        Minimum and maximum frequency for filtering

    kmin, kmax : int or None
        Minimum and maximum frequency for filtering

    filter_dict : dict or None
        A dictionary containing the filter parameters in the form of
        {
            'upper': (a, b),
            'lower': (a, b),
            'right': (a, b),
            'left': (a, b),
        }
        where a and b are the slope and intercept of each boundary.
        If None, default values are used.

    Returns
    -------
    mask : xr.DataArray
    """
    wavenumber, frequency = _wrap_to_xarray(wavenumber, frequency)

    # Update td_args with the [fmin, fmax, kmin, kmax] values
    td_args = wave_args['td']
    td_args.update(kwargs)

    logical_plus, logical_minus = kf_mask(wavenumber,
                                          frequency,
                                          td_args['fmin'], 
                                          td_args['fmax'],
                                          td_args['kmin'],
                                          td_args['kmax'], 
                                          return_individual=True)

    filter_dict = td_args.get('filter_dict', None)

    upper = filter_dict.get('upper')
    lower = filter_dict.get('lower')
    right = filter_dict.get('right')
    left = filter_dict.get('left')

    upper_a, upper_b = (None, None) if upper is None else upper
    lower_a, lower_b = (None, None) if lower is None else lower
    right_a, right_b = (None, None) if right is None else right
    left_a, left_b = (None, None) if left is None else left

    if upper is not None:
        logical_plus.append((upper_a * frequency + wavenumber < upper_b))
        logical_minus.append((upper_a * frequency + wavenumber + upper_b > 0))

    if lower is not None:
        logical_plus.append((lower_a * frequency + wavenumber > lower_b))
        logical_minus.append((lower_a * frequency + wavenumber + lower_b < 0))

    # Test parallelogram construction
    if right is not None:
        logical_plus.append((right_a * frequency + wavenumber < right_b))
        logical_minus.append((right_a * frequency + wavenumber + right_b > 0))

    if left is not None:
        logical_plus.append((left_a * frequency + wavenumber > left_b))
        logical_minus.append((left_a * frequency + wavenumber + left_b < 0))

    return _combine_plus_minus(logical_plus, logical_minus)
