"""
Filtering on the wavenumber-frequency domain based on WK99.

Author: Richard Zhuang
Date: Apr 8, 2025
"""

import numpy as np
import xarray as xr

from scipy.signal import detrend
from functools import reduce
import operator
from typing import Callable

import matplotlib.pyplot as plt
import seaborn as sns

from kf_filter.consts import (
    wave_title
)

from kf_filter.util import (
    split_hann_taper,
    _combine_plus_minus,
    kf_mask,
    wave_mask
)

sns.set_theme(style="ticks")


class KF():
    """
    A class to filter equatorial waves in the wavenumber-frequency domain.
    This class implements the wavenumber-frequency filter based on
    WK99.

    Notes
    -----
    This class is in part inspired by the two following Github projects:

    1. https://github.com/tmiyachi/mcclimate/blob/master/kf_filter.py
    2. https://github.com/brianpm/wavenumber_frequency
    """

    def __init__(self, 
                 data : xr.DataArray, 
                 sampling_per_day : int = 1, 
                 time_taper_frac : float = 0.1, 
                 remove_annual : bool = False) -> None:
        """
        Initialize an object to filter equatorial waves.

        Parameters
        ----------
        data : xr.DataArray
            The data to be filtered. Must contain dimensions corresponding to
            time, latitude, and longitude. The latitude dimension can be
            'lat' or 'latitude', and the longitude dimension can be 'lon' or 'longitude'.
            
            There is no assumption on the shape of the data.

        sampling_per_day : int
            The number of samples per day. Default is 1.

        time_taper_frac : float
            The fraction of the time series to taper. Default is 0.1.

        remove_annual : bool
            Whether to remove the annual cycle from the data. Default is False.

        Notes
        -----
        `remove_annual` is not implemented yet.
        """
        self._preprocess(data)

        # Detrend the data and add back the mean
        x_mean = self.data.mean(dim='time')
        x_detrended = detrend(self.data.values, axis=0, type='linear')
        x_detrended = xr.DataArray(x_detrended, dims=self.data.dims, coords=self.data.coords)
        x_detrended += x_mean

        taper = split_hann_taper(self.data.sizes['time'], time_taper_frac)
        x_taper = x_detrended * taper[:, np.newaxis, np.newaxis]

        x_fft = np.fft.fft2(x_taper, axes=(0, 2)) / (self.lon_size * self.time_size)

        data_fft = xr.DataArray(
            x_fft,
            dims=('frequency', 'lat', 'wavenumber'),
            coords={
                'frequency': np.fft.fftfreq(self.time_size, 
                                            1 / sampling_per_day),
                'lat': x_taper['lat'],
                'wavenumber': np.fft.fftfreq(self.lon_size, 1 / self.lon_size),
            },
        )

        data_fft = data_fft.transpose('lat', 'wavenumber', 'frequency')
        self.kf = data_fft

        self._reorder_fft()

    def get_fft(self) -> xr.DataArray:
        """Retrieve the fft data following CCEW convention."""
        return self.kf_reordered
    
    def get_wavenumber(self) -> xr.DataArray:
        """Retrieve the wavenumber data following CCEW convention."""
        return self.kf_reordered.wavenumber
    
    def get_frequency(self) -> xr.DataArray:
        """Retrieve the frequency data following CCEW convention."""
        return self.kf_reordered.frequency
    
    def _reorder_fft(self) -> None:
        r"""Reorder fft from NumPy convention to CCEW convention.
        
        Notes
        -----
        NumPy convention:
        e^{i (-k * x - \omega t)}

        CCEW convention:
        e^{i (k * x - \omega t)}
        
        So we need to flip the sign for wavenumber.

        Original ordering
        -----------------
        0 : pos high wavenumber : neg high wavenumber : 0
        0 : pos high frequency : neg high frequency : 0
        """
        kf = self.kf

        # Use fftshift to reorder frequency
        kf_shifted = xr.DataArray(
            np.fft.fftshift(kf, axes=(1, 2)),
            dims=['lat', 'wavenumber', 'frequency'],
            coords={'lat': kf.lat,
                    'wavenumber': np.fft.fftshift(kf.wavenumber),
                    'frequency': np.fft.fftshift(kf.frequency), 
                    }
        )

        # multiply wavenumber by -1 to flip the sign
        self.kf_reordered = kf_shifted.assign_coords(wavenumber=-1 * kf_shifted['wavenumber']).sortby('wavenumber')

    def _preprocess(self, da : xr.DataArray) -> None:
        """
        Preprocess pipeline.
        """
        # Rename 'latitude' and 'longitude' if they exist
        rename_dict = {}
        if 'latitude' in da.dims or 'latitude' in da.coords:
            rename_dict['latitude'] = 'lat'
        if 'longitude' in da.dims or 'longitude' in da.coords:
            rename_dict['longitude'] = 'lon'
        da = da.rename(rename_dict)

        # Check for required coordinates
        required_coords = ['time', 'lat', 'lon']
        missing = [coord for coord in required_coords if coord not in da.dims and coord not in da.coords]
        if missing:
            raise ValueError(f"Missing required dimension(s) or coordinate(s): {missing}")
        
        current_dims = list(da.dims)
        desired_order = [dim for dim in ['time', 'lat', 'lon'] if dim in current_dims]
        
        if current_dims != desired_order:
            da = da.transpose(*desired_order)

        self.data = da
        self.lon_size = da.sizes['lon']
        self.time_size = da.sizes['time']

    @staticmethod
    def remove_annual(self):
        """
        Remove annual cycle.

        TODO
        """
        pass

    def kf_filter(self, mask : xr.DataArray) -> xr.DataArray:
        """
        Filter on the wavenumber-frequency domain based on `mask`.
        
        Parameters
        ----------
        mask : xr.DataArray
            Expect input (..., frequency, wavenumber).
            Need to transpose before applying mask.

        Returns
        -------
        TODO
        """
        kf_reordered = self.kf_reordered

        # Apply the mask to the kf_reordered data
        kf_filtered = kf_reordered * mask.values.transpose()[np.newaxis, ...]

        # Revert back to NumPy convention
        kf_filtered_reordered = kf_filtered.assign_coords(wavenumber=-1 * kf_filtered['wavenumber']).sortby('wavenumber')

        # Perform ifftshift to revert back to the original ordering
        kf_filtered_reordered = xr.DataArray(
            np.fft.ifftshift(kf_filtered_reordered, axes=(1, 2)),
            dims=['lat', 'wavenumber', 'frequency'],
            coords={
                'lat': kf_filtered_reordered.lat,
                'wavenumber': np.fft.ifftshift(kf_filtered_reordered.wavenumber),
                'frequency': np.fft.ifftshift(kf_filtered_reordered.frequency),
            }
        )
        # Perform inverse FFT to get back to the original data
        data_filtered = np.fft.ifft2(kf_filtered_reordered, axes=(1, 2)).real

        da_filtered = xr.DataArray(
            data_filtered,
            dims=('lat', 'lon', 'time'),
            coords={
                'lat': self.data.lat,
                'lon': self.data.lon,
                'time': self.data.time
            }
        ) * self.lon_size * self.time_size

        # back to the original time, lat, lon
        self.data_filtered = da_filtered.transpose('time', 'lat', 'lon')

        return self.data_filtered
    
    def _save_mask(self, 
                   mask : xr.DataArray, 
                   wave_type : str) -> None:
        """Save the mask to the object.
        
        Parameters
        ----------
        mask : xr.DataArray
            The mask to be saved.
        wave_type : str
            The type of wave to be saved.
        """
        if wave_type == 'kelvin':
            self.kelvin_mask = mask
        elif wave_type == 'er':
            self.er_mask = mask
        elif wave_type == 'ig':
            self.ig_mask = mask
        elif wave_type == 'eig':
            self.eig_mask = mask
        elif wave_type == 'mrg':
            self.mrg_mask = mask
        elif wave_type == 'td':
            self.td_mask = mask
        elif wave_type == 'mjo':
            self.mjo_mask = mask
        else:
            raise ValueError(f'Unsupported wave type "{wave_type}".')
    
    def wave_filter(self,
                    wave_type : str,
                    fmin : float | None=0.05, 
                    fmax : float | None=0.4, 
                    kmin : int | None=None, 
                    kmax : int | None=14, 
                    hmin : int | None=8,
                    hmax : int | None=90,
                    n : int = 1) -> xr.DataArray:
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
        mask = wave_mask(self.kf_reordered.wavenumber,
                         self.kf_reordered.frequency,
                         wave_type,
                         fmin, fmax, kmin, kmax, hmin, hmax, n)
        self._save_mask(mask, wave_type)

        return self.kf_filter(mask)

    def kelvin_filter(self, 
                      fmin : float | None=0.05, 
                      fmax : float | None=0.4, 
                      kmin : int | None=None, 
                      kmax : int | None=14, 
                      hmin : int | None=8,
                      hmax : int | None=90) -> xr.DataArray:
        r"""
        Filter Kelvin wave.

        Parameters
        ----------
        fmin, fmax : float or None
            Minimum and maximum frequency (cycles per day).

        kmin, kmax : int or None
            Minimum and maximum wavenumber.

        hmin, hmax : int or None
            Minimum and maximum equivalent depth.

        Notes
        -----
        Nondimensionalized \omega and k for KW:
        \hat{\omega} = \hat{k}
        """
        return self.wave_filter('kelvin', fmin, fmax, kmin, kmax, hmin, hmax)

    def er_filter(self, 
                  fmin : float | None=None, 
                  fmax : float | None=None, 
                  kmin : int | None=-10, 
                  kmax : int | None=-1, 
                  hmin : int | None=8, 
                  hmax : int | None=90, 
                  n : int=1) -> xr.DataArray:
        r"""
        Filter Equatorial Rossby wave.

        Parameters
        ----------
        fmin, fmax : float or None
            Minimum and maximum frequency (cycles per day).

        kmin, kmax : int or None
            Minimum and maximum wavenumber.

        hmin, hmax : int or None
            Minimum and maximum equivalent depth.

        n : int

        Notes
        -----
        Nondimensionalized k and \omega for ER:

        \hat{omega} = \hat{k} / {2n + 1 + \hat{k} ** 2}
        """
        return self.wave_filter('er', fmin, fmax, kmin, kmax, hmin, hmax, n)

    def ig_filter(self,
                  fmin : float | None=None, 
                  fmax : float | None=None, 
                  kmin : int | None=-15, 
                  kmax : int | None=-1, 
                  hmin : int | None=12, 
                  hmax : int | None=90, 
                  n : int=1) -> xr.DataArray:
        r"""
        Filter westward-propagating inertia-gravity waves.

        Parameters
        ----------
        fmin, fmax : float or None
            Minimum and maximum frequency (cycles per day).

        kmin, kmax : int or None
            Minimum and maximum wavenumber.

        hmin, hmax : int or None
            Minimum and maximum equivalent depth.

        n : int

        Notes
        -----
        Nondimensionalized k and \omega for IG:

        \hat{omega}^2 = \hat{k}^2 + 2 * n + 1
        """
        return self.wave_filter('ig', fmin, fmax, kmin, kmax, hmin, hmax, n)

    def eig_filter(self,
                   fmin : float | None=None, 
                   fmax : float | None=0.55, 
                   kmin : int | None=0, 
                   kmax : int | None=15, 
                   hmin : int | None=12, 
                   hmax : int | None=50) -> xr.DataArray:
        r"""
        Filter eastward-propagating inertia-gravity waves.

        Parameters
        ----------
        fmin, fmax : float or None
            Minimum and maximum frequency (cycles per day).

        kmin, kmax : int or None
            Minimum and maximum wavenumber.

        hmin, hmax : int or None
            Minimum and maximum equivalent depth (m).

        Notes
        -----
        Nondimensionalized k and \omega for IG:

        \hat{omega}^2 = \hat{k}^2 + 2 * n + 1
        """
        return self.wave_filter('eig', fmin, fmax, kmin, kmax, hmin, hmax)

    def mrg_filter(self,
                   fmin : float | None=None, 
                   fmax : float | None=None, 
                   kmin : int | None=-10, 
                   kmax : int | None=-1, 
                   hmin : int | None=8, 
                   hmax : int | None=90):
        """
        Filter mixed-Rossby gravity waves.

        Parameters
        ----------
        fmin, fmax : float or None
            Minimum and maximum frequency.

        kmin, kmax : int or None
            Minimum and maximum wavenumber.

        hmin, hmax : int or None
            Minimum and maximum equivalent depth.

        Notes
        -----
        No additional constraint on MRG waves such as wavenumber/frequency
        cutoff is applied.
        """
        return self.wave_filter('mrg', fmin, fmax, kmin, kmax, hmin, hmax)

    def td_filter(self, 
                  fmin : float | None=None, 
                  fmax : float | None=None, 
                  kmin : int | None=-20, 
                  kmax : int | None=-6,
                  filter_func : tuple[Callable, Callable] | None=None):
        """
        Filter tropical depressions.

        Parameters
        ----------
        fmin, fmax : float or None
            Minimum and maximum frequency (cycles per day).

        kmin, kmax : int or None
            Minimum and maximum wavenumber.
        """
        wavenumber, frequency = self.kf_reordered.wavenumber, self.kf_reordered.frequency

        logical_plus, logical_minus = kf_mask(wavenumber,
                                              frequency,
                                              fmin, fmax, 
                                              kmin, kmax,
                                              return_individual=True)

        logical_plus.append((84 * frequency + wavenumber - 22 < 0))
        logical_minus.append((84 * frequency + wavenumber + 22 > 0))

        logical_plus.append((210 * frequency + 2.5 * wavenumber - 13 > 0))
        logical_minus.append((210 * frequency + 2.5 * wavenumber + 13 < 0))

        mask = _combine_plus_minus(logical_plus, logical_minus)

        self._save_mask(mask, 'td')
        return self.kf_filter(mask)
    
    def mjo_filter(self,
                   fmin : float | None=1/96, 
                   fmax : float | None=1/20, 
                   kmin : int | None=0, 
                   kmax : int | None=10) -> xr.DataArray:
        """
        Filter the Madden-Julian Oscillation (MJO).

        Parameters
        ----------
        fmin, fmax : float or None
            Minimum and maximum frequency (cycles per day).

        kmin, kmax : int or None
            Minimum and maximum wavenumber.

        Notes
        -----
        Here, we filter the MJO as a band of wavenumbers between 1 and 10
        and a band of frequencies between 1/96 and 1/20 cycles per day
        as in Mayta and Adames (2023).
        """
        mask = kf_mask(self.kf_reordered.wavenumber,
                          self.kf_reordered.frequency,
                          fmin, fmax, kmin, kmax)
        self._save_mask(mask, 'mjo')
        return self.kf_filter(mask)

    def _visualize_aux(self, 
                       wave_type : str, 
                       mask : xr.DataArray, 
                       hide_negative : bool=True) -> None:
        """Auxiliary function to visualize the filter."""
        plt.contour(self.kf_reordered.wavenumber,
                    self.kf_reordered.frequency,
                    mask,
                    levels=[0.5],
                    colors='black')
        plt.xlabel('Wavenumber')
        plt.ylabel('Frequency (cpd)')
        if hide_negative:
            plt.ylim([0, self.kf_reordered.frequency.max()])
        # plt.grid()
        plt.title(f'{wave_title[wave_type]} Filter')
        plt.show()

    def get_mask(self, wave_type):
        """
        Get the mask for a specific wave type.

        Parameters
        ----------
        wave_type : str
            Must be one of 'er', 'kelvin', 'ig', 'eig', 'mrg', 'td', 'mjo'.

        Returns
        -------
        mask : xr.DataArray
            The mask for the specified wave type.
        """
        if wave_type == 'er':
            if not hasattr(self, 'er_mask'):
                self.er_filter()
            return self.er_mask
        elif wave_type == 'kelvin':
            if not hasattr(self, 'kelvin_mask'):
                self.kelvin_filter()
            return self.kelvin_mask
        elif wave_type == 'ig':
            if not hasattr(self, 'ig_mask'):
                self.ig_filter()
            return self.ig_mask
        elif wave_type == 'eig':
            if not hasattr(self, 'eig_mask'):
                self.eig_filter()
            return self.eig_mask
        elif wave_type == 'mrg':
            if not hasattr(self, 'mrg_mask'):
                self.mrg_filter()
            return self.mrg_mask
        elif wave_type == 'td':
            if not hasattr(self, 'td_mask'):
                self.td_filter()
            return self.td_mask
        elif wave_type == 'mjo':
            if not hasattr(self, 'mjo_mask'):
                self.mjo_filter()
            return self.mjo_mask
        else:
            raise ValueError(f'Unsupported wave type "{wave_type}".')

    def visualize_filter(self, 
                         wave_type : str,
                         hide_negative : bool=True) -> None:
        """
        Visualize the equatorial wave filter.

        Parameters
        ----------
        wave_type : str
            Must be one of 'er', 'kelvin', 'ig', 'eig', 'mrg', 'td'.
        """
        mask = self.get_mask(wave_type)
        self._visualize_aux(wave_type, mask, hide_negative)
            