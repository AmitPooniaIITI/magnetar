# Study of Emission Behavior and Pulse Profile Characteristics of a Magnetar

## Overview

This project is based on my M.Sc. thesis, titled *"Study of Emission Behavior and Pulse Profile Characteristics of a Magnetar"*, submitted to the **Indian Institute of Technology Indore**. The project focuses on analyzing the temporal and spectral characteristics of the magnetar **4U 0142+61** using observations from **AstroSat** and **NICER**.

The repository includes Python scripts for light curve generation, pulsation frequency estimation, and pulse profile fitting for timing analysis.

## Dataset

The data used in this study were obtained from:

- **AstroSat LAXPC and SXT observations** (January 2016)
- **NICER observations** (July 2017 - January 2022)

These observations were processed to determine the timing solution, pulse profiles, and emission characteristics of the magnetar.

## Features

- **Light Curve Generation**: Extracts photon arrival times and constructs light curves.
- **Z-Squared Test**: Determines the pulsation frequency using the Z²n statistic.
- **Pulse Profile Fitting**: Fits sinusoidal models to estimate pulse profiles.
- **Energy-Resolved Pulse Profiles**: Analyzes pulse profiles in different energy bands.
- **Photon Count Rate Calculation**: Computes the photon count rate and pulse fraction.

## Installation

### Prerequisites
Ensure you have Python 3 installed, along with the following dependencies:
```bash
pip install numpy matplotlib pandas astropy scipy tqdm
```

## Usage

### Light Curve Generation
Run the following command to generate a histogram of photon arrival times:
```bash
python light_curve.py
```

### Pulsation Frequency Estimation
To determine the pulsation frequency using the Z²n test, execute:
```bash
python z_test.py
```

### Pulse Profile Fitting
For sinusoidal function fitting to the pulse profile, run:
```bash
python profile_fitting.py
```

### Energy-Resolved Pulse Profiles
To generate pulse profiles in different energy bands, use:
```bash
python ene_res_pf.py
```

### Pulse Fraction Calculation
Compute the pulse fraction and count rate with:
```bash
python pf.py
```

## File Descriptions

| File | Description |
|------|-------------|
| `light_curve.py` | Generates a light curve from event data. |
| `z_test.py` | Computes the pulsation frequency using the Z-squared test. |
| `profile_fitting.py` | Fits pulse profiles using sinusoidal models. |
| `ene_res_pf.py` | Generates energy-resolved pulse profiles. |
| `pf.py` | Calculates the pulse fraction and count rate. |

## Results

- The pulsation frequency of 4U 0142+61 was determined to be **0.1150849 Hz**.
- Pulse profiles exhibit energy-dependent variations, suggesting complex emission mechanisms.
- Timing analysis provides insights into the magnetar's magnetic field evolution.

## Future Work

- Further spectral analysis using NICER data.
- Extending the analysis to other magnetars.

## License

This project is licensed under the MIT License.

## Author

[Amit Poonia]

