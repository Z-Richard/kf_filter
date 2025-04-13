# KF Filter

A Python package for filtering equatorial waves in the wavenumber-frequency domain.

![Example Filter Visualization](example/figures/kelvin_filtered_OLR.png)

## Features

- Implements wavenumber-frequency filtering for equatorial waves.
- Supports filtering for MJO and TD as well.
- Provides visualization tools for filter masks.

## Installation

The easiest way to install is through pip.

```bash
pip install kf-filter
```

## Usage

```python
import xarray as xr
from kf_filter import KF

# Load your data as an xarray DataArray
data = xr.open_dataset("your_data.nc")["variable_name"]

# Initialize the KF filter
kf_filter = KF(data, sampling_per_day=2, time_taper_frac=0.1)

# Apply a Kelvin wave filter
filtered_data = kf_filter.kelvin_filter()

# Visualize the filter
kf_filter.visualize_filter("kelvin")
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
