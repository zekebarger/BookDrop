import numpy as np
import pandas as pd
import re

def scale_prices(pr):
    """ 
    Scale prices by the 5th and 95th percentiles.
    This is similar to min-max scaling.
  
    Parameters: 
    pr : numpy array of prices
  
    Returns: 
    scaled numpy array of prices
    """
    # Calculate 5th and 95th percentiles
    prctiles = np.quantile(pr,[.05, .95])
    # If this range is very small, use a reasonable guess instead
    median_to_date = np.median(pr)
    if prctiles[1] - prctiles[0] < .02 * median_to_date:
        prctiles[0] = .5 * median_to_date
        prctiles[1] = 1.5 * median_to_date

    # Scale the prices
    pr_scaled = (pr - prctiles[0]) / (prctiles[1] - prctiles[0])
    # Set upper and lower bounds
    pr_scaled[pr_scaled > 2] = 2
    pr_scaled[pr_scaled < -1] = -1
    
    return pr_scaled


def compute_features(pr, drop_frac):
    """ 
    Compute informative features of an item's price history.
  
    Parameters: 
    pr (numpy array): an array of prices
    drop_frac (int): fraction of the price considered a 'drop'
                     E.g., 0.1 for a 10% price drop
  
    Returns: 
    numpy array of features
    """
    
    # Number of timepoints considered 'recent'
    recent = 120
    
    # Current price, in dollars
    current_price = pr[-1]

    # Prices scaled by historical high and low
    prices_scaled = scale_prices(pr)

    # Current scaled price
    current_scaled = prices_scaled[-1]

    # Find current price scaled by recent prices, rather than all-time
    recent_prices_scaled = scale_prices(pr[-1 * recent:])
    # Get the current value
    current_scaled_recent = recent_prices_scaled[-1]

    # Median scaled price, recently
    median_scaled = np.median(prices_scaled[-1 * recent:])

    # Standard deviation of recent scaled prices
    std_recent = np.std(prices_scaled[-1 * recent:])

    # Probability that recent prices were equal to current price
    p_equal = np.sum(pr[-1 * recent:] == current_price) / recent

    # Probability of any price change, all-time
    p_change = np.sum(np.abs(np.diff(pr)) > 0) / np.size(pr)

    # Probability of any price change, recently
    p_change_recent = np.sum(np.abs(np.diff(pr[-1 * recent:])) > 0) / recent

    # Probability that price has been below the threshold, all-time
    p_below = np.sum(pr < (1 - drop_frac) * current_price) / np.size(pr)

    # Probability that price has been below the threshold, recently
    p_below_recent = np.sum(
        pr[-1 * recent:] < (1 - drop_frac) * current_price) / recent

    # Time elapsed since the last time the price dropped below
    # the threshold, divided by total time elapsed
    # First, find indices of timepoints when the price was below threshold
    belowpricesidx = np.argwhere(pr[:-1] < (1 - drop_frac) * current_price)
    # If none were found, the value is 1
    if np.size(belowpricesidx) == 0:
        time_since_drop = 1
    else:
        time_since_drop = belowpricesidx[-1].astype(int)[0] / np.size(pr)

    # Take the log of some skewed features
    time_since_drop = np.log(1.05 + -1 * time_since_drop)
    p_below_recent = np.log(.05 + p_below_recent)
    p_change = np.log(.05 + p_change)
    p_change_recent = np.log(.05 + p_change_recent)
    std_recent = np.log(.05 + std_recent)

    # Return the feature vector
    return np.array([
        p_equal, p_change_recent, p_below_recent, median_scaled,
        current_scaled_recent, std_recent, current_scaled, time_since_drop,
        p_change, p_below
    ])
    
def impute_nan(x):
    """ 
    Impute nan values in a series of prices using most recent non-nan values.
  
    Parameters: 
    x (numpy array): A series of prices.
  
    Returns: 
    Array with nan values imputed.
    """
    # Convert result of isnan to a string for regex searching
    nans = ''.join([str(int(np.isnan(i))) for i in x])
    # Find groups of nans
    matches = [m.span() for m in re.finditer('0[^0]+', nans)]
    # Replace each with most recent non-nan value
    for match in matches:
        x[match[0] + 1:match[1]] = x[match[0]]
    return x