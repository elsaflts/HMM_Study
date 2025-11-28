# Kalman Filter for Rayleigh Fading Channels

This project explores the application of the Kalman filter to estimate the state of a Rayleigh fading communication channel. The repository includes implementations of the Kalman filter, smoother, and analysis of their performance under different noise conditions.

## Repository Structure

*   `notebooks/`: Contains Jupyter notebooks with detailed explanations, simulations, and visualizations.
    *   `kalman_rayleigh.ipynb`: Implements the Kalman filter and smoother for a Rayleigh fading channel model, assuming the transmitted bits are known. It includes analysis on the impact of process noise on filter performance.
    *   `change_point.ipynb`: (Content not available)
*   `src/`: Contains Python scripts.
    *   `kalman_filter.py`: A Python script for simulating a Rayleigh fading channel and applying a Kalman filter.
*   `data/`: (Empty)
*   `figures/`: (Empty)

## Key Concepts

The project is based on the following state-space model:

*   **State Evolution (AR(1) process for fading):**
    $$W_{k+1} = \rho W_k + U_k, \quad U_k \sim \mathcal{N}(0, \sigma_U^2)$$

*   **Observation Model:**
    $$Y_k = W_k C_k + V_k, \quad V_k \sim \mathcal{N}(0, \sigma_V^2)$$

Where:
*   $W_k$: The hidden state representing the fading coefficient at time $k$.
*   $C_k$: The known transmitted bit sequence (either -1 or +1).
*   $Y_k$: The observed signal at time $k$.
*   $U_k$ and $V_k$: Process and observation noise, respectively.
*   $\rho, \sigma_U, \sigma_V$: Model parameters.

## Implementations

The notebooks and scripts cover the following algorithms:

1.  **Kalman Filter (Prediction and Filtering):** Estimates the current state $W_k$ given all observations up to time $k$.
2.  **Kalman Smoother (Disturbance Smoother):** Improves the filtered estimates by using all available data (both past and future observations).

## How to Run

To run the simulations and see the results, you can execute the Jupyter notebooks in the `notebooks/` directory. You will need to have Python installed with libraries such as `numpy` and `matplotlib`.

```bash
# Example of how to run the notebook
jupyter notebook notebooks/kalman_rayleigh.ipynb
```

## Summary of Results

The analysis in `kalman_rayleigh.ipynb` demonstrates the trade-offs in tuning the Kalman filter, particularly the process noise variance $\sigma_u^2$:

*   **Small $\sigma_u^2$**: The filter is more resistant to noise but can be slow to adapt to real changes in the state (lagging).
*   **Large $\sigma_u^2$**: The filter tracks changes more aggressively but is more susceptible to being influenced by observation noise.

The project also shows that the **smoother** provides significantly better estimates than the filter alone, as it leverages more information to estimate the state at any given time step.
