import streamlit as st
import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import tree, ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn import calibration
import pickle
from PIL import Image, ImageOps
import skimage
from skimage import io
import requests
from io import BytesIO
import difflib
import pytesseract
import re
import datetime
import time
import functools
import random
import string
from bisect import bisect_left

# We need to specify where tesseract OCR is installed.
# When running locally, use something like this:
#pytesseract.pytesseract.tesseract_cmd = r'D:\Software\Tesseract-OCR\tesseract.exe'
# Otherwise, use this:
pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

def main():
    # Run this so that models are cached
    two_week_model, one_month_model = load_models()

    # Show some information in the sidebar
    st.sidebar.info(
        "Created by Zeke Barger, "
        "Insight Data Science Fellow.\n\n"
        "For more information, see the project [GitHub]("
        "http://code.book-drop.site) and "
        "[Slides](http://slides.book-drop.site).\n\n"
        "To get started, try dragging and dropping "
        "these links into the search bar:\n\n"
        "[*1984*, by George Orwell](https://www.amazon.com/"
        "1984-Signet-Classics-George-Orwell/dp/0451524934/ref=sr_1_1?dchild=1)\n\n"
        "[*The Handmaid's Tale*, by Margaret Atwood](https://www.amazon.com/dp/038"
        "549081X?tag=camelproducts-20&linkCode=ogi&th=1&psc=1&language=en_US)\n\n"
        "[*East of Eden*, by John Steinbeck](https://www.amazon.com/dp/0140186395"
        "?tag=camelproducts-20&linkCode=ogi&th=1&psc=1&language=en_US)"
    )

    # Create the main components of the interface
    st.title('BookDrop')
    st.subheader(
        "Enter a book's Amazon URL to find out whether the "
        "price is likely to drop by at least 10% in the next..."
    )
    timeframe = st.radio("",('two weeks', 'month'))
    product_url = st.text_input('', value='', key='asinstr', type='default')
    
    # Extract the ASIN from the product URL
    asin = get_asin(product_url)

    # If an ASIN was found, try to collect the price history
    if asin is not None:
        features, history = collect_data(asin)

        # If there was some proble, display an error message
        if features is None:
            st.subheader('Sorry, that product was not found or was released too recently.')
            
        else:    
            # Use the relevant model to make a prediction
            # Note: the decision thresholds are no longer 0.5 because
            # the models have been calibrated
            if timeframe == 'two weeks':
                drop_probability = two_week_model.predict_proba([features])[0][1]
                drop_predicted = drop_probability > 0.12247253
            else:
                drop_probability = one_month_model.predict_proba([features])[0][1]
                drop_predicted = drop_probability > 0.19920479
        
            # Display information about the prediction
            if drop_predicted:
                st.subheader(
                    "Don't buy! The price has a "+str(np.round(100*drop_probability))+ \
                    "% chance of dropping in the next "+timeframe+"."
                )
                st.write('Set your price tracker here:')
                st.write('https://camelcamelcamel.com/product/'+asin+"?active=price_amazon#watch")
            else:
                st.subheader(
                    "Go for it! The price only has a "+str(np.round(100*drop_probability))+ \
                    "% chance of dropping in the next "+timeframe+"."
                )

            # Select just the last year (at most) of the price history to plot
            start_idx = np.max(np.array([(-1*len(history.index)), -365*2]))
            df_to_plot = history.iloc[start_idx:].copy()
            # Plot the price history
            fig = px.line(df_to_plot, x='date', y = 'price',
                title='Price history, last '+str(int(-1*round(start_idx/2/30.5)))+' months')
            fig.update_yaxes(tickprefix="$",tickformat='.2f')
            fig.update_layout(xaxis_title="Date",yaxis_title="Price",)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # If something was entered, but no ASIN was found
        if len(product_url) > 0:
            st.write('Please enter a valid URL.')

# The functions get_session_id and fancy_cache were copied from 
# https://gist.github.com/treuille/f988f78c4610c78322d089eb77f74598
# and allow one to cache for a limited period of time only.
# This is useful because camelcamelcamel updates about every 6 hours.
def get_session_id():
    # Copied from tvst's great gist:
    # https://gist.github.com/tvst/6ef6287b2f3363265d51531c62a84f51
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    session = None
    session_infos = Server.get_current()._session_infos.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            # Streamlit < 0.54.0
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
        ):
            session = session_info.session

    if session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            'Are you doing something fancy with threads?')

    return id(session)

def fancy_cache(func=None, ttl=None, unique_to_session=False, **cache_kwargs):
    """A fancier cache decorator which allows items to expire after a certain time
    as well as promises the cache values are unique to each session.
    Parameters
    ----------
    func : Callable
        If not None, the function to be cached.
    ttl : Optional[int]
        If not None, specifies the maximum number of seconds that this item will
        remain in the cache.
    unique_to_session : boolean
        If so, then hash values are unique to that session. Otherwise, use the default
        behavior which is to make the cache global across sessions.
    **cache_kwargs
        You can pass any other arguments which you might to @st.cache
    """
    # Support passing the params via function decorator, e.g.
    # @fancy_cache(ttl=10)
    if func is None:
        return lambda f: fancy_cache(
            func=f,
            ttl=ttl,
            unique_to_session=unique_to_session,
            **cache_kwargs
        )

    # This will behave like func by adds two dummy variables.
    dummy_func = st.cache(
        func = lambda ttl_token, session_token, *func_args, **func_kwargs: \
            func(*func_args, **func_kwargs),
        **cache_kwargs)

    # This will behave like func but with fancy caching.
    @functools.wraps(func)
    def fancy_cached_func(*func_args, **func_kwargs):
        # Create a token which changes every ttl seconds.
        ttl_token = None
        if ttl is not None:
            ttl_token = int(time.time() / ttl)

        # Create a token which is unique to each session.
        session_token = None
        if unique_to_session:
            session_token = get_session_id()

        # Call the dummy func
        return dummy_func(ttl_token, session_token, *func_args, **func_kwargs)
    return fancy_cached_func


# Only keep this cached for 3 hours, max
@fancy_cache(ttl=60*60*3,show_spinner=False,suppress_st_warning=True)
def collect_data(asin):
    """ 
    Get the price history of an item from Amazon, supplementing
    this with 3rd-party data if needed.
  
    Parameters: 
    asin (str): Amazon Standard Identification Number for an item.
  
    Returns: 
    features: numpy array of features of the price history that can 
        be used as inputs to a trained random forest classifier
    history: A pandas dataframe of the item's price at 12-hours
        time resolution, collected from camelcamelcamel.com
    """
    # Set the size of the image to be downloaded. Larger sizes
    # will product more accurate results at the cost of 
    # increased processing time.
    image_width = 6555
    image_height = 1013

    # get the URL to the camelcamelcamel page
    url_amazon = 'https://charts.camelcamelcamel.com/us/' + asin + \
    '/amazon.png?force=0&zero=1&w='+str(image_width)+'&h='+str(image_height)+\
    '&desired=false&legend=0&ilt=1&tp=all&fo=0&lang=en'

    url_3rdparty = 'https://charts.camelcamelcamel.com/us/' + asin + \
    '/new.png?force=0&zero=1&w='+str(image_width)+'&h='+str(image_height)+\
    '&desired=false&legend=0&ilt=1&tp=all&fo=0&lang=en'

    # Extract price data for times when the item was sold by Amazon
    dates_amazon, prices_amazon = scrape_prices(url_amazon, image_width, image_height)
    
    # If the price data are insufficient, collect data from times 
    # when the item was sold by a 3rd party through Amazon
    if dates_amazon is None or \
       np.size(dates_amazon) < 60 or \
       np.all(np.isnan(prices_amazon[-60:])) or \
       np.sum(np.isnan(prices_amazon))/np.size(dates_amazon) > .1:
       
        # Extract price data for times when the item was sold by a 3rd party
        dates_3rdparty, prices_3rdparty = scrape_prices(url_3rdparty, image_width, image_height)

        # If we have both types of price data
        if dates_amazon is not None and dates_3rdparty is not None:
            # Find the union of the two date ranges
            dates = np.union1d(dates_amazon, dates_3rdparty)
            prices_padded = np.ones((np.size(dates), 2)) * np.nan
            # Align the price data in time
            idx_a = int(np.where(dates == dates_amazon[0])[0])
            idx_3 = int(np.where(dates == dates_3rdparty[0])[0])
            prices_padded[idx_a:(idx_a + np.size(prices_amazon)), 0] = prices_amazon
            prices_padded[idx_3:(idx_3 + np.size(prices_3rdparty)), 1] = prices_3rdparty
            # Fill in gaps in Amazon prices with 3rd party prices
            prices = np.copy(prices_padded[:, 0])
            nan_a = np.isnan(prices)
            prices[nan_a] = prices_padded[nan_a, 1]
            # Sometimes the date ranges are different, so make sure
            # we keep the most recent Amazon prices just in case.
            # This solution will be problematic if there are no Amazon prices
            # available recently and the 3rd party price changed significantly
            # in the last 36 hours.
            prices[-3:]=prices_padded[-3:, 0]

        else:
            # If we only have Amazon prices, use them 
            if dates_amazon is not None:
                dates = dates_amazon
                prices = prices_amazon
            else:
                # If we only have 3rd party prices, use them
                if dates_3rdparty is not None:
                    dates = dates_3rdparty
                    prices = prices_3rdparty
                else:
                    # No data were found
                    return None, None
    else:
        # Since the Amazon data had no problems, use them
        dates = dates_amazon
        prices = prices_amazon

    # Impute nans
    prices = impute_nan(prices)
    # Convert the price history to a dataframe
    history = pd.DataFrame({'date': dates, 'price': prices})
    # Extract features for the classifier
    features = compute_features(history)
    
    return features, history


def scrape_prices(url, image_width, image_height):
    """ 
    Extract dates and prices from a camelcamelcamel item URL.
  
    Parameters: 
    url (str): camelcamelcamel URL for a product.
    image_width (int): width of the image to be used, in pixels
    image_height (int): height of the image to be used, in pixels
  
    Returns: 
    dates: numpy array of dates at 12-hour intervals
    prices: a numpy array of prices
    """
    
    ################
    # Collect data #
    ################
    
    # Show a message indicating progress
    progress_string = st.text('Collecting data...')
    
    # Define colors of elements of the plot (RGB)
    # Plotted lines
    plot_colors = np.array([[194, 68, 68], [119, 195, 107], [51, 51, 102]])
    # Gray axis lines
    gray = np.array([215, 215, 214])
    # Black axis lines
    black = np.array([75, 75, 75])

    # Download the image
    response = requests.get(url)
    image_temp = Image.open(BytesIO(response.content))

    # Convert image to float
    im = np.array(image_temp)

    # Get masks for each plot color
    masks = list()
    for i in range(3):
        masks.append(np.all(im == plot_colors[i], axis=-1))

    # Check if there image is empty (camel has no data)
    if not np.any(masks[1]):
        return None, None

    ######################
    # Find x and y scale #
    ######################

    progress_string.text('Aligning data...')
    
    # Find the y axis upper limit
    # Crop a portion of the image containing the top of the grid
    top_line_crop = im[:,
                       round(image_width * .5) - 5:round(image_width * .5) +
                       6, :]
    # Get the position of the line
    line_y_value = find_line(top_line_crop, gray)
    
    # If it wasn't found, quit
    # Checks of this nature are rarely needed, as long
    # as camel keeps their plotting code the same
    if line_y_value is None:
        return None, None
    else:
        line_y_value = int(line_y_value)

    # Find x axis limits
    # Crop the left-most and right-most vertical lines in the grid
    left_line_crop = np.transpose(
        im[round(image_height * .5) - 8:round(image_height * .5) +
           9, :round(image_width * .1), :],
        axes=[1, 0, 2])
    right_line_crop = np.transpose(im[round(image_height * .5) -
                                      8:round(image_height * .5) + 9,
                                      round(image_width * .7):, :],
                                   axes=[1, 0, 2])
    lo_x_value = find_line(left_line_crop, black)
    hi_x_value = find_line(right_line_crop[::-1, :, :], gray)
    if lo_x_value is None or hi_x_value is None:
        return None, None
    else:
        lo_x_value = int(lo_x_value)
        hi_x_value = int(hi_x_value)

    # Find price corresponding to the y axis upper limit
    # First, crop the price text
    upper_price_crop = im[line_y_value - 8:line_y_value + 10,
                          0:lo_x_value - 9, :]
    upper_price_crop = Image.fromarray(upper_price_crop)
    # Resize and apply OCR
    upper_price_crop = upper_price_crop.resize(
        (upper_price_crop.width * 12, upper_price_crop.height * 12))
    upper_price_string = pytesseract.image_to_string(upper_price_crop)
    upper_price = float(upper_price_string[1:].replace(',', ''))

    # Store y position of price limits
    # The position and price of the lower limit are constant
    limit_y_positions = np.array([line_y_value, image_height - 49])

    # Calculate dollars per pixel
    dollarspp = upper_price / (np.max(limit_y_positions) -
                               np.min(limit_y_positions))

    # Crop year text from bottom of image so that we
    # can find the date of the first timepoint
    year_crop = im[-14:, 0:round(image_width / 3), :]
    year_crop = Image.fromarray(year_crop)
    # Resize and apply OCR
    year_crop = year_crop.resize((year_crop.width * 3, year_crop.height * 3))
    year_string = pytesseract.image_to_string(year_crop)
    year_string = year_string[:4]

    # Crop month and day from bottom left corner
    date_crop = im[-44:-14, (lo_x_value - 40):(lo_x_value + 6), :]
    # Convert to image
    date_crop = Image.fromarray(date_crop)
    # Invert, so that rotation works
    date_crop = ImageOps.invert(date_crop)
    # Pad the image
    date_crop_padded = Image.new(
        'RGB', (round(date_crop.width * 1.5), round(date_crop.height * 1.5)),
        (0, 0, 0))
    date_crop_padded.paste(date_crop, box=(0, round(date_crop.height * .5)))
    # Resize
    date_crop_padded = date_crop_padded.resize(
        (date_crop_padded.width * 7, date_crop_padded.height * 7),
        resample=Image.LANCZOS)
    # Rotate and invert
    date_crop_padded = ImageOps.invert(date_crop_padded.rotate(-45))
    # Apply OCR
    date_string = pytesseract.image_to_string(date_crop_padded)
    # Find closest match to a month
    start_month = difflib.get_close_matches(date_string, [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
        'Nov', 'Dec'
    ],
                                            n=1,
                                            cutoff=0.2)
    # Quit if no month was found
    if np.size(start_month) < 1:
        return None, None

    start_month = start_month[0]

    # Get the day of the first timepoint
    # Try to fix mixups between 'o' and 0
    if date_string[-1] == 'o':
        date_string = date_string[:-1] + '0'
    # Remove all whitespace
    date_string_stripped = "".join(date_string.split())
    # Take last 2 digits if the second-to-last is reasonable
    if date_string_stripped[-2].isdigit() and 0 < int(
            date_string_stripped[-2]) < 4:
        start_day = date_string_stripped[-2:]
    else:
        start_day = '0' + date_string_stripped[-1]

    # Store x axis locations of time limits
    limit_x_positions = [lo_x_value, image_width - hi_x_value]

    # Check if our date is valid
    try:
        start_time = datetime.datetime.strptime(
            start_month + start_day + year_string, '%b%d%Y')
    except ValueError:
        return None, None

    # Get current time
    end_time = datetime.datetime.now()

    # Calculate days per pixel
    time_delta = end_time - start_time
    dayspp = time_delta.days / int(1 + np.diff(limit_x_positions))

    # Get number of observations
    num_obs = int(np.diff(limit_x_positions))

    # Preallocate prices as nan
    prices = np.ones(num_obs) * np.nan
    
    ##################
    # Extract prices #
    ##################
    
    progress_string.text('Extracting prices...')

    # Find y-axis value of blue pixels in each time step - 
    # these are the prices we're looking for
    y = [[i for i, x in enumerate(q) if x] for q in np.transpose(
        masks[2][:, limit_x_positions[0]:limit_x_positions[1]])]

    # Adjust values if necessary, then convert to prices
    # Missing data are set to nan
    for i in range(num_obs):
        # Check if the bottom of the blue line is covered by a red or green line
        if np.size(y[i]) == 1:
            if masks[0][int(y[i][0]) + 1, limit_x_positions[0] +
                        i] or masks[1][int(y[i][0]) + 1,
                                       limit_x_positions[0] + i, ]:
                y[i][0] += 1

        # Check if the blue line is covered by both red and green lines
        if np.size(y[i]) == 0:
            red_idx = [
                q for q, x in enumerate(masks[0][:, limit_x_positions[0] + i])
                if x
            ]
            grn_idx = [
                q for q, x in enumerate(masks[1][:, limit_x_positions[0] + i])
                if x
            ]
            if np.size(red_idx) == 1 and np.size(grn_idx) == 1 and np.abs(
                    int(red_idx[0]) - int(grn_idx[0])) == 1:
                y[i] = grn_idx
            else:
                y[i] = np.nan

        prices[i] = dollarspp * (image_height - np.max(y[i]) - 50)

    # Adjust periods with no data
    # First, find nans and convert to a str for regex searching
    nans = ''.join([str(int(np.isnan(i))) for i in prices])
    # Ensure the beginnings of empty periods are correct
    matches = [m.span() for m in re.finditer('000110011', nans)]
    for match in matches:
        prices[match[0] + 3:match[0] + 5] = prices[match[0] + 5]
    # Then remove empty periods
    nans = ''.join([str(int(np.isnan(i))) for i in prices])
    matches = [m.span() for m in re.finditer('1100', nans)]
    for match in matches:
        prices[match[0] + 2:match[0] + 4] = np.nan


    ###################
    # Resample prices #
    ###################

    progress_string.text('Resampling prices...')

    # Resample to 2x daily observations at 6:00 and 18:00
    # First, get the dates of our observations
    dates = pd.date_range(start_time, end_time,
                          periods=num_obs).to_pydatetime()
    # Initialize new dates and prices at the desired interval
    dates_2x_daily = pd.date_range(datetime.datetime(start_time.year,
                                                     start_time.month,
                                                     start_time.day, 6),
                                   datetime.datetime(end_time.year,
                                                     end_time.month,
                                                     end_time.day, 18),
                                   freq='12H').to_pydatetime()
    prices_2x_daily = np.ones(np.size(dates_2x_daily)) * np.nan
    
    # Find price at the closest date to each timepoint
    for i in range(np.size(dates_2x_daily)):
        prices_2x_daily[i] = prices[take_closest_date(dates-dates_2x_daily[i])]

    # Make sure most recent price is correct
    prices_2x_daily[-1] = prices[-1]
    # Round prices to 2 decimal places
    prices_2x_daily = np.around(prices_2x_daily, 2)

    # Clear the message
    progress_string.empty()
    
    return dates_2x_daily, prices_2x_daily

# This function is a modified version of the one posted by 
# Lauritz V. Thaulow on stackoverflow at
# https://stackoverflow.com/a/12141511
def take_closest_date(myList):
    """
    Assumes myList is sorted. Returns index of closest value to myNumber.

    If two numbers are equally close, return the index of the smallest number.
    """
    pos = bisect_left(myList, datetime.timedelta(0))
    if pos == 0:
        return 0
    if pos == len(myList):
        return len(myList)-1
    before = myList[pos - 1]
    after = myList[pos]
    if abs(after) < abs(before):
       return pos
    else:
       return pos - 1


def find_line(img, c):
    """ 
    Find the position of a line that bisects an image horizontally. 
    If there are multiple lines, only the first is returned.
  
    Parameters: 
    img (float): numpy array containing the image to search.
    c (float): 3-element numpy array equal to the line's color.
  
    Returns: 
    int: Location of the line, in pixels from the top of the image.
  
    """
    # Colors in the image should be withing this range of the target
    color_tolerance = 15
    # This fraction of the image should contain matching colors
    # to be considered a line
    match_threshold = .75
    # Get the width of the image, in pixels
    img_width = np.size(img, 1)
    # Find all pixels within color tolerance
    img[(img < c - color_tolerance) | (img > c + color_tolerance)] = 0
    img[img > 0] = 1
    # Only take pixels where all channels are within color tolerance
    mask2d = np.all(img, axis=-1)
    # Sum across columns
    sums = np.sum(mask2d, axis=1)
    # Find rows with sufficient matches
    matches = np.argwhere(sums > match_threshold * img_width)
    if np.size(matches) < 1:
        return
    else:
        return matches[0]


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


def compute_features(df):
    """ 
    Compute informative features of an item's price history.
  
    Parameters: 
    df (DataFrame): a dataframe of dates and prices
  
    Returns: 
    numpy array of features
    """
    
    # Number of timepoints considered 'recent'
    recent = 120
    # Fraction of the price considered a 'drop'
    # E.g., 0.1 for a 10% price drop
    drop_frac = .1

    # Drop missing values
    df.dropna(inplace=True)
    # Convert to array
    pr = np.array(df['price'])
    
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

def get_asin(url):
    """ 
    Extract the ASIN from an Amazon product URL.
  
    Parameters: 
    url (str): Amazon URL for a product.
  
    Returns: 
    str: the ASIN
    """
    match = re.search(r"/dp/([^/?]+)",url)
    if match:
        return match.group(1)
    else:
        return None

# Load the random forest models
@st.cache(allow_output_mutation=True,show_spinner=False)
def load_models():
    two_week_model = pickle.load(open('RFmodel28.sav', 'rb'))
    one_month_model = pickle.load(open('RFmodel60.sav', 'rb'))
    return two_week_model, one_month_model
	

if __name__ == "__main__":
    main()
