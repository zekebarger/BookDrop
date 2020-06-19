import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import datetime
from sklearn import tree, ensemble
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image, ImageOps
import requests
from io import BytesIO
import difflib
import pytesseract
import re

import skimage
from skimage import io

import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server
import time
import functools
import random
import string
from sklearn import calibration


#pytesseract.pytesseract.tesseract_cmd = r'D:\Software\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'


# these next two functions were copied from 
# https://gist.github.com/treuille/f988f78c4610c78322d089eb77f74598
# and allow one to cache for a limited period of time
# Copied from tvst's great gist:
# https://gist.github.com/tvst/6ef6287b2f3363265d51531c62a84f51
def get_session_id():
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


def get_prices(asin,prog):
    image_width = 6555
    image_height = 1013

    # get the URL to the data
    url_a = 'https://charts.camelcamelcamel.com/us/' + asin + \
    '/amazon.png?force=0&zero=1&w='+str(image_width)+'&h='+str(image_height)+\
    '&desired=false&legend=0&ilt=1&tp=all&fo=0&lang=en'

    url_3 = 'https://charts.camelcamelcamel.com/us/' + asin + \
    '/new.png?force=0&zero=1&w='+str(image_width)+'&h='+str(image_height)+\
    '&desired=false&legend=0&ilt=1&tp=all&fo=0&lang=en'


    da, pa, prog = url2price(url_a, image_width, image_height, prog)
    
    # only check 3rd-party prices if amazon data is missing
    if da is None or np.size(da) < 60 or np.all(np.isnan(pa[-60:])) or np.sum(np.isnan(pa))/np.size(da) > .1:
        d3, p3, prog = url2price(url_3, image_width, image_height, prog)

        if da is not None and d3 is not None:
            dates = np.union1d(da, d3)
            prices_padded = np.ones((np.size(dates), 2)) * np.nan
            idx_a = int(np.where(dates == da[0])[0])
            idx_3 = int(np.where(dates == d3[0])[0])
            prices_padded[idx_a:(idx_a + np.size(pa)), 0] = pa
            prices_padded[idx_3:(idx_3 + np.size(p3)), 1] = p3

            prices = np.copy(prices_padded[:, 0])
            nan_a = np.isnan(prices)
            prices[nan_a] = prices_padded[nan_a, 1]
            # except the last entry for safety
            prices[-3:]=prices_padded[-3:, 0]

        else:
            if da is not None:
                dates = da
                prices = pa
            else:
                if d3 is not None:
                    dates = d3
                    prices = p3
                else:
                    return
    else:
        if da is not None:
            dates = da
            prices = pa
    # impute nans and convert to dataframe
    prices = impute_nan(prices)
    df = pd.DataFrame({'date': dates, 'price': prices})

    return df


def url2price(url, image_width, image_height, prog, verbose=False):
    #st.write('fetching image')
    # define colors of the lines in the plot
    # red, green, blue

    plot_colors = np.array([[194, 68, 68], [119, 195, 107], [51, 51, 102]])
    gray = np.array([215, 215, 214])
    black = np.array([75, 75, 75])

    # download the image
    response = requests.get(url)
    image_temp = Image.open(BytesIO(response.content))

    # convert image to float
    im = np.array(image_temp)

    # get masks for each plot color
    masks = list()
    for i in range(3):
        masks.append(np.all(im == plot_colors[i], axis=-1))

    # check if there's no data
    if not np.any(masks[1]):
        return None, None, prog

    prog += .125
    prog_bar.progress(prog)

    #st.write('aligning image')
    # find the y axis upper limit
    # off by 1?
    top_line_crop = im[:,
                       round(image_width * .5) - 5:round(image_width * .5) +
                       6, :]
    # off by 1?
    line_y_value = find_line(top_line_crop, gray)
    # if it wasn't found, quit
    if line_y_value is None:
        if verbose:
            print('could not find y axis')
        return None, None, prog
    else:
        line_y_value = int(line_y_value)

    # find x axis limits
    # off by 1?
    left_line_crop = np.transpose(
        im[round(image_height * .5) - 8:round(image_height * .5) +
           9, :round(image_width * .1), :],
        axes=[1, 0, 2])
    # off by 1?
    right_line_crop = np.transpose(im[round(image_height * .5) -
                                      8:round(image_height * .5) + 9,
                                      round(image_width * .7):, :],
                                   axes=[1, 0, 2])
    # off by 1?
    lo_x_value = find_line(left_line_crop, black)
    # off by 1?
    hi_x_value = find_line(right_line_crop[::-1, :, :], gray)
    if lo_x_value is None or hi_x_value is None:
        if verbose:
            print('could not find x axis')
        return None, None, prog
    else:
        lo_x_value = int(lo_x_value)
        hi_x_value = int(hi_x_value)

    # find value of y axis upper limit
    # first, crop the price out
    # off by 1?
    upper_price_crop = im[line_y_value - 8:line_y_value + 10,
                          0:lo_x_value - 9, :]
    upper_price_crop = Image.fromarray(upper_price_crop)
    upper_price_crop = upper_price_crop.resize(
        (upper_price_crop.width * 12, upper_price_crop.height * 12))
    upper_price_string = pytesseract.image_to_string(upper_price_crop)
    upper_price = float(upper_price_string[1:].replace(',', ''))
    if verbose:
        print('price guess')
        print(upper_price)

    # store y position of price limits
    # off by 1?
    limit_y_positions = np.array([line_y_value, image_height - 49])

    # calculate dollars per pixel
    dollarspp = upper_price / (np.max(limit_y_positions) -
                               np.min(limit_y_positions))

    # crop year from bottom of image
    year_crop = im[-14:, 0:round(image_width / 3), :]
    year_crop = Image.fromarray(year_crop)
    year_crop = year_crop.resize((year_crop.width * 3, year_crop.height * 3))
    year_string = pytesseract.image_to_string(year_crop)
    year_string = year_string[:4]

    # crop month and day from bottom left corner
    # initial crop
    date_crop = im[-44:-14, (lo_x_value - 40):(lo_x_value + 6), :]
    # convert to image
    date_crop = Image.fromarray(date_crop)
    # invert, so that rotation works
    date_crop = ImageOps.invert(date_crop)
    # pad
    date_crop_padded = Image.new(
        'RGB', (round(date_crop.width * 1.5), round(date_crop.height * 1.5)),
        (0, 0, 0))
    date_crop_padded.paste(date_crop, box=(0, round(date_crop.height * .5)))
    # resize
    date_crop_padded = date_crop_padded.resize(
        (date_crop_padded.width * 7, date_crop_padded.height * 7),
        resample=Image.LANCZOS)
    # rotate and invert
    date_crop_padded = ImageOps.invert(date_crop_padded.rotate(-45))
    date_string = pytesseract.image_to_string(date_crop_padded)
    if verbose:
        print('date string')
        print(date_string)
    # find closest match to month
    start_month = difflib.get_close_matches(date_string, [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
        'Nov', 'Dec'
    ],
                                            n=1,
                                            cutoff=0.2)
    if np.size(start_month) < 1:
        if verbose:
            print('could not identify month')
        return None, None, prog

    start_month = start_month[0]
    if verbose:
        print('month guess')
        print(start_month)

    # get day
    # try to fix 0-o mixups
    if date_string[-1] == 'o':
        date_string = date_string[:-1] + '0'
    # remove all whitespace
    date_string_stripped = "".join(date_string.split())
    # take last 2 digits if the second-to-last is reasonable
    if date_string_stripped[-2].isdigit() and 0 < int(
            date_string_stripped[-2]) < 4:
        start_day = date_string_stripped[-2:]
    else:
        start_day = '0' + date_string_stripped[-1]
    if verbose:
        print('day guess')
        print(start_day)

    # x axis locations of time limits
    # again... pretty sure one of these is off by 1.
    limit_x_positions = [lo_x_value, image_width - hi_x_value]

    # check if our date is valid
    try:
        start_time = datetime.datetime.strptime(
            start_month + start_day + year_string, '%b%d%Y')
    except ValueError:
        if verbose:
            disp('invalid date')
        return None, None, prog

    # get current time
    end_time = datetime.datetime.now()

    # calculate days per pixel
    time_delta = end_time - start_time
    dayspp = time_delta.days / int(1 + np.diff(limit_x_positions))

    # get number of observations
    num_obs = int(np.diff(limit_x_positions))
    #     print('number of observations:')
    #     print(num_obs)

    # preallocate prices as nan
    prices = np.ones(num_obs) * np.nan
    
    prog += .125
    prog_bar.progress(prog)
    
    #st.write('scanning image')

    # find y-axis value of blue pixels in each time step
    y = [[i for i, x in enumerate(q) if x] for q in np.transpose(
        masks[2][:, limit_x_positions[0]:limit_x_positions[1]])]

    # adjust values if necessary, then convert to prices
    for i in range(num_obs):
        # check if the bottom of the blue line is covered by a red or green line
        if np.size(y[i]) == 1:
            if masks[0][int(y[i][0]) + 1, limit_x_positions[0] +
                        i] or masks[1][int(y[i][0]) + 1,
                                       limit_x_positions[0] + i, ]:
                y[i][0] += 1

        # check if the blue line is covered by both red and green lines
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

    # adjust periods with no data
    # first, find nans and convert to a str for regex searching
    nans = ''.join([str(int(np.isnan(i))) for i in prices])
    # ensure the beginnings of empty periods are correct
    matches = [m.span() for m in re.finditer('000110011', nans)]
    for match in matches:
        prices[match[0] + 3:match[0] + 5] = prices[match[0] + 5]
    # then remove empty periods
    nans = ''.join([str(int(np.isnan(i))) for i in prices])
    matches = [m.span() for m in re.finditer('1100', nans)]
    for match in matches:
        prices[match[0] + 2:match[0] + 4] = np.nan

    # resample to 2x daily observations at 6:00 and 18:00
    # date and time of each observation
    dates = pd.date_range(start_time, end_time,
                          periods=num_obs).to_pydatetime()

    dates_2x_daily = pd.date_range(datetime.datetime(start_time.year,
                                                     start_time.month,
                                                     start_time.day, 6),
                                   datetime.datetime(end_time.year,
                                                     end_time.month,
                                                     end_time.day, 18),
                                   freq='12H').to_pydatetime()

    
    prog += .125
    prog_bar.progress(prog)
    
    #st.write('resampling timepoints')
    # find closest price to each timepoint
    prices_2x_daily = np.ones(np.size(dates_2x_daily)) * np.nan
    # very slow...
    for i in range(np.size(dates_2x_daily)):
        old_date = np.argmin(np.abs(dates - dates_2x_daily[i]))
        prices_2x_daily[i] = prices[old_date]

    # make sure most recent price is correct
    prices_2x_daily[-1] = prices[-1]
    prices_2x_daily = np.around(prices_2x_daily, 2)

    prog += .125
    prog_bar.progress(prog)
    #st.write('last 4 p2xd')
    #st.write(prices_2x_daily[-4:])

    return dates_2x_daily, prices_2x_daily, prog


def find_line(img, c):
    color_tolerance = 15
    match_threshold = .75
    im_width = np.size(img, 1)
    # find pixels within tolerance
    img[img < c - color_tolerance] = 0
    img[img > c + color_tolerance] = 0
    img[img > 0] = 1
    mask2d = np.all(img, axis=-1)
    sums = np.sum(mask2d, axis=1)
    matches = np.argwhere(sums > match_threshold * im_width)
    if np.size(matches) < 1:
        return
    else:
        return matches[0]


def impute_nan(x):
    nans = ''.join([str(int(np.isnan(i))) for i in x])
    matches = [m.span() for m in re.finditer('0[^0]+', nans)]
    for match in matches:
        x[match[0] + 1:match[1]] = x[match[0]]
    return x


def scale_prices(pr):
    # calculate 5th and 95th percentiles
    prctiles = np.quantile(pr,[.05, .95])
    # if this range is very small, use a reasonable guess instead
    median_to_date = np.median(pr)
    if prctiles[1] - prctiles[0] < .02 * median_to_date:
        prctiles[0] = .5 * median_to_date
        prctiles[1] = 1.5 * median_to_date

    # scale the prices
    pr_scaled = (pr - prctiles[0]) / (prctiles[1] - prctiles[0])
    # set upper and lower bounds
    pr_scaled[pr_scaled > 2] = 2
    pr_scaled[pr_scaled < -1] = -1
    
    return pr_scaled


def df2feats(df):
    past = 120
    drop_frac = .1
    X = list([])
    # drop missing values
    df.dropna(inplace=True)
    # convert to array
    pr = np.array(df['price'])
    
    # current price, in dollars
    current_p = pr[-1]

    # prices scaled by historical high and low
    prices_scaled = scale_prices(pr)

    # current scaled price
    current_scaled = prices_scaled[-1]

    # find current price scaled by recent prices, rather than all-time
    recent_prices_scaled = scale_prices(pr[-1 * past:])
    # get the current value
    current_scaled_recent = recent_prices_scaled[-1]

    # median scaled price, recently
    median_scaled = np.median(prices_scaled[-1 * past:])

    # standard deviation of recent scaled prices
    std_recent = np.std(prices_scaled[-1 * past:])

    # probability that recent prices were equal to current price
    p_equal = np.sum(pr[-1 * past:] == current_p) / past

    # probability of any price change, all-time
    p_change = np.sum(np.abs(np.diff(pr)) > 0) / np.size(pr)

    # probability of any price change, recently
    p_change_recent = np.sum(np.abs(np.diff(pr[-1 * past:])) > 0) / past

    # probability that price has been below the threshold, all-time
    p_below = np.sum(pr < (1 - drop_frac) * current_p) / np.size(pr)

    # probability that price has been below the threshold, recently
    p_below_recent = np.sum(
        pr[-1 * past:] < (1 - drop_frac) * current_p) / past

    # time elapsed since the last time the price dropped below
    # the threshold, divided by total time elapsed
    # first, find indices of timepoints when the price was below threshold
    belowpricesidx = np.argwhere(pr[:-1] < (1 - drop_frac) * current_p)
    # if none were found, the value is 1
    if np.size(belowpricesidx) == 0:
        time_since_drop = 1
    else:
        time_since_drop = belowpricesidx[-1].astype(int)[0] / np.size(pr)

    # take the log of some skewed features
    time_since_drop = np.log(1.05 + -1 * time_since_drop)
    p_below_recent = np.log(.05 + p_below_recent)
    p_change = np.log(.05 + p_change)
    p_change_recent = np.log(.05 + p_change_recent)
    std_recent = np.log(.05 + std_recent)

    # return the feature vector
    return np.array([
        p_equal, p_change_recent, p_below_recent, median_scaled,
        current_scaled_recent, std_recent, current_scaled, time_since_drop,
        p_change, p_below
    ])



# load models
@st.cache(allow_output_mutation=True,show_spinner=False)
def load_models():
    #st.write('loading models')
    rfmodel28 = pickle.load(open('RFmodel28.sav', 'rb'))
    rfmodel60 = pickle.load(open('RFmodel60.sav', 'rb'))
    return rfmodel28, rfmodel60

#@st.cache(show_spinner=False)
# only keep this cached for 3 hours, max
@fancy_cache(ttl=60*60*3,show_spinner=False)
def predict_future(asin,prog):
    df = get_prices(asin,prog)
    if df is None:
        return None, None
    x = df2feats(df)

    return x, df
	

# run this so that models are cached
rfmodel28, rfmodel60 = load_models()


st.sidebar.info(

    "[GitHub]("
    "https://github.com/zekebarger/BookDrop)\n\n"
    "[Slides]("
    "https://docs.google.com/presentation/d/1b0i7GQlaIl-ASu68PvatwZ6ZqItuLcHAttV_xMjt5TI/edit?usp=sharing)"
)

st.title('BookDrop')

st.subheader('Enter a book\'s ASIN (Amazon Standard Identification Number) to find out whether the price is likely to drop by at least 10% in the next...')
timeframe = st.radio("",('two weeks', 'month'))

isbn = st.text_input('', value='', key='isbnstr', type='default').strip() #,max_chars=10

if len(isbn)==10:
    prog_bar = st.progress(0)
    prog = 0
    x, df = predict_future(isbn,prog)
    prog_bar.empty()
    
    if x is None:
        st.write('Sorry, that ASIN was not found or is invalid.')
        
    else:    
        if timeframe == 'two weeks':
            prob = rfmodel28.predict_proba([x])[0][1]
            yp = prob > 0.12247253
        else:
            prob = rfmodel60.predict_proba([x])[0][1]
            yp = prob > 0.19920479
    
        if yp:
            st.subheader('Don\'t buy! The price has a '+str(np.round(100*prob))+'% chance of dropping in the next '+timeframe+'.')
            st.write('Set your price tracker here:')
            st.write('https://camelcamelcamel.com/product/'+isbn+"?active=price_amazon#watch")
        else:
            st.subheader('Go for it! The price only has a '+str(np.round(100*prob))+'% chance of dropping in the next '+timeframe+'.')

        start_idx = np.max(np.array([(-1*len(df.index)), -365*2]))
        df2plot = df.iloc[start_idx:].copy()
    
        fig = px.line(df2plot, x='date', y = 'price',title='Price history, last '+str(int(-1*round(start_idx/2/30.5)))+' months')
    
        fig.update_yaxes(tickprefix="$")
        fig.update_yaxes(tickformat='.2f')
        fig.update_layout(xaxis_title="Date",yaxis_title="Price",)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.write('Examples:')
    st.write("0451524934 (*1984*, by George Orwell)")
    st.write("038549081X (*The Handmaid's Tale*, by Margaret Atwoot)")
    st.write("0140186395 (*East of Eden*, by John Steinbeck)")