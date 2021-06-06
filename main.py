from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# import uvicorn
import pandas as pd
pd.set_option('chained_assignment', None)
import mysql.connector as cnx
import datetime as dt
from datetime import timedelta
import numpy as np
import copy
import json
from pytz import timezone
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get('/')
def show_form(request: Request):
    return templates.TemplateResponse('form.html', context={'request': request})

@app.get("/api")
def callAPI(request: Request):
    print('POSTED')
    format = "%Y-%m-%d %H:%M"
    now_utc = dt.datetime.strptime(dt.datetime.now().strftime(format), format)
    now_ist = now_utc.astimezone(timezone('Asia/Kolkata'))
    print(f"System TZ {time.tzname}")
    print(f"UTC {now_utc}")
    print(f"IST {now_ist}")
    curr_datetime = dt.datetime.strptime(now_ist.strftime(format), format)
    curr_date = curr_datetime.date()
    curr_time = curr_datetime.time()

    if curr_date.weekday()==5:
        curr_date = curr_date - timedelta(days=1)
    elif curr_date.weekday()==6:
        curr_date = curr_date - timedelta(days=2)

    time_dict = {dt.time(14, 13): dt.time(14, 10), dt.time(14, 43): dt.time(14, 40), dt.time(15, 18): dt.time(15, 15), dt.time(12, 40): dt.time(12, 30)}

    try:
        cutoff_time = time_dict[curr_time]
        full_list, btst, stbt, manual_btst, manual_stbt = scanner_func(curr_date, cutoff_time)
        s1 = full_list.to_dict(orient="records")
        s2 = btst.to_dict(orient="records")
        s3 = stbt.to_dict(orient="records")
        s4 = manual_btst.to_dict(orient="records")
        s5 = manual_stbt.to_dict(orient="records")
        # time_str = f'{start_time} - {end_time}'
        rtn_data = {'full_list': s1, 'btst': s2, 'stbt': s3, 'manual_btst': s4, 'manual_stbt': s5}
        payload = json.dumps(rtn_data)
    except KeyError as e:
        rtn_data = {'full_list': [], 'btst': [], 'stbt': [], 'manual_btst': [], 'manual_stbt': []}
        payload = json.dumps(rtn_data)

    return payload



## FUNCTION DEFNS

def resampler(data, sample_interval, date_col, agg_dict, na_subset):
    sampled_df = data.resample(sample_interval, on=date_col).aggregate(agg_dict)
    sampled_df.dropna(subset=[na_subset], inplace=True)
    sampled_df.reset_index(inplace=True)
    return sampled_df

def TrueRange(data):
    data = data.copy()
    data["TR"] = np.nan
    for i in range(1,len(data)):
        h = data.loc[i,"high"]
        l = data.loc[i,"low"]
        pc = data.loc[i-1,"close"]    
        x = h-l
        y = abs(h-pc)
        z = abs(l-pc)
        TR = max(x,y,z)
        data.loc[i,"TR"] = TR
    return data

def average_true_range(data, period, drop_tr=True, smoothing="RMA"):
    data = data.copy()
    if smoothing == "RMA":
        data['atr_' + str(period) + '_' + str(smoothing)] = data['TR'].ewm(com=period - 1, min_periods=period).mean()
    elif smoothing == "SMA":
        data['atr_' + str(period) + '_' + str(smoothing)] = data['TR'].rolling(window=period).mean()
    elif smoothing == "EMA":
        data['atr_' + str(period) + '_' + str(smoothing)] = data['TR'].ewm(span=period, adjust=False).mean()
    if drop_tr:
        data.drop(['TR'], inplace=True, axis=1)
    data = data.round(decimals=2)
    return data


# V1 FINAL - 31/05/2021 11:00 PM (with instrument_scan)

def scanner_func(trading_date, cutoff_time):

    try:
        stocks_db = cnx.connect(host="164.52.207.158", user="stock", password="stockdata@data", database='stock_production')
        stock_query = f'select instrument_id, ins_date, open, high, low, close, volume from instrument_scan where date(ins_date) between "{trading_date - dt.timedelta(days=50)}" and "{trading_date}";'
        stock_df = pd.read_sql(stock_query,stocks_db, parse_dates=['ins_date'])

        oi_query = f'select id, ins_date, oi from future_all where date(ins_date) between "{trading_date - dt.timedelta(days=5)}" and "{trading_date}";'
        oi_df = pd.read_sql(oi_query,stocks_db, parse_dates=['ins_date'])

        sl_query = 'select id, tradingsymbol from instruments where f_n_o=1 and tradingsymbol not like "%NIFTY%";'
        sl_df = pd.read_sql(sl_query,stocks_db)

        stocks_db.close() 
    except Exception as e:
        stocks_db.close()
        print(str(e))

    stock_df.drop_duplicates(subset=['instrument_id', 'ins_date'], inplace=True)
    stock_df.reset_index(inplace=True, drop=True)

    stock_df.drop(stock_df[(stock_df['ins_date'].dt.time<dt.time(9, 15))].index, inplace = True)
    stock_df.reset_index(inplace=True, drop=True)
    stock_df.drop(stock_df[(stock_df['ins_date'].dt.time>dt.time(15, 29))].index, inplace = True)
    stock_df.reset_index(inplace=True, drop=True)

    oi_df.drop(oi_df[(oi_df['ins_date'].dt.time<dt.time(9, 15))].index, inplace = True)
    oi_df.reset_index(inplace=True, drop=True)
    oi_df.drop(oi_df[(oi_df['ins_date'].dt.time>dt.time(15, 29))].index, inplace = True)
    oi_df.reset_index(inplace=True, drop=True)

    # will prevent against empty df for that day
    instru_ids = stock_df[stock_df['ins_date'].dt.date==trading_date]['instrument_id'].unique()
    sl_df = sl_df[sl_df['id'].isin(instru_ids)]
    stock_dict = dict(sl_df.values)

    scanner_list = []

    for id, name in stock_dict.items():

        scrip_df = stock_df[stock_df['instrument_id']==id].reset_index(drop=True)
        scrip_oi = oi_df[oi_df['id']==id].reset_index(drop=True)

        ######## OI
        try:
            yest_close_oi = scrip_oi.iloc[scrip_oi[scrip_oi['ins_date'].dt.date==trading_date].head(1).index-1]['oi'].to_list()[0]
            tdy_coff_oi = scrip_oi[(scrip_oi['ins_date'].dt.date==trading_date) & (scrip_oi['ins_date'].dt.time>=cutoff_time)]['oi'].to_list()[0]
            oic_yest_coff = round(((tdy_coff_oi - yest_close_oi)/yest_close_oi)*100, 3)
        except IndexError as e:
            oic_yest_coff = np.nan

        ######## VOL
        vol_df = scrip_df.copy()
        vol_df.drop(vol_df[(vol_df['ins_date'].dt.time>cutoff_time)].index, inplace=True)

        agg_dict = {'instrument_id': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        vol_resample = resampler(vol_df, '1D', 'ins_date', agg_dict, 'close')
        vol_resample.drop(vol_resample[vol_resample['volume']==0].index, inplace=True)
        vol_resample.reset_index(inplace=True, drop=True)
        vol_resample['10Davg'] = vol_resample['volume'].rolling(window=10).mean()

        tdy_vol = vol_resample[vol_resample['ins_date'].dt.date==trading_date]['volume'].to_list()[0]
        last_10d_vol = vol_resample.iloc[vol_resample[vol_resample['ins_date'].dt.date==trading_date].index-1]['10Davg'].to_list()[0]
        vc_tdy_10day = round(tdy_vol/last_10d_vol, 3)    

        ######## CANDLE 80
        cutoff_df = vol_resample[(vol_resample['ins_date'].dt.date==trading_date)]
        cutoff_high = cutoff_df['high'].to_list()[0]
        cutoff_low = cutoff_df['low'].to_list()[0]
        cutoff_close = cutoff_df['close'].to_list()[0]

        candle_80_type = 'Green' if cutoff_close>=cutoff_low+(0.8*(cutoff_high-cutoff_low)) else ('Red' if cutoff_close<=cutoff_high-(0.8*(cutoff_high-cutoff_low)) else 'NA')

        if candle_80_type=='Green':
            candle_80_val = (((cutoff_close - (cutoff_low+(0.8*(cutoff_high-cutoff_low)))) + (0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            candle_80_val = round(candle_80_val, 2)
        elif candle_80_type=='Red':
            candle_80_val = ((((cutoff_high-(0.8*(cutoff_high-cutoff_low))) - cutoff_close) + (0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            candle_80_val = round(candle_80_val, 2)
        else:
            v1 = (((cutoff_close - (cutoff_low+(0.8*(cutoff_high-cutoff_low)))) + (0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            v2 = ((((cutoff_high-(0.8*(cutoff_high-cutoff_low))) - cutoff_close) + (0.8*(cutoff_high-cutoff_low)))*100)/(cutoff_high-cutoff_low)
            candle_80_val = (round(v1, 2), round(v2, 2))

        ######## PC 1PM-COFF
        try:
            tdy_100_open = scrip_df[(scrip_df['ins_date'].dt.date==trading_date) & (scrip_df['ins_date'].dt.time>=dt.time(13, 0))]['open'].to_list()[0]
            pc_100_coff = round(((cutoff_close - tdy_100_open)/tdy_100_open)*100, 3)
        except IndexError as e:
            pc_100_coff = np.nan

        ######## ATR
        agg_dict = {'instrument_id': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        scrip_resample = resampler(scrip_df, '1D', 'ins_date', agg_dict, 'close')

        scrip_tr = TrueRange(scrip_resample)
        scrip_atr = average_true_range(data = scrip_tr, period=14, drop_tr=True, smoothing="RMA")
        yest = scrip_atr.iloc[scrip_atr[scrip_atr['ins_date'].dt.date==trading_date].index-1]
        atr_val = yest['atr_14_RMA'].to_list()[0]
        yest_close = yest['close'].to_list()[0]
        atr_flag_btst = 'True' if cutoff_close>(yest_close+(0.5*atr_val)) else 'False'
        atr_flag_stbt = 'True' if cutoff_close<(yest_close-(0.5*atr_val)) else 'False'

        params = [id, name, cutoff_close, atr_val, atr_flag_btst, atr_flag_stbt, pc_100_coff, oic_yest_coff, vc_tdy_10day, candle_80_type, candle_80_val]
        scanner_list.append(params)
        print(id, name)
    
    full_list = pd.DataFrame(scanner_list, columns=['id', 'name', 'cutoff_close', 'atr_val', 'atr_flag_btst', 'atr_flag_stbt', 'pc_100_coff', 'oic_yest_coff', 'vc_tdy_10day', 'candle_80_type', 'candle_80_val'])
    btst = full_list[(full_list['atr_flag_btst']=='True') & (abs(full_list['oic_yest_coff'])>4) & (full_list['vc_tdy_10day']>1) & (full_list['candle_80_type']=='Green')]
    stbt = full_list[(full_list['atr_flag_stbt']=='True') & (abs(full_list['oic_yest_coff'])>4) & (full_list['vc_tdy_10day']>1) & (full_list['candle_80_type']=='Red')]
    manual_btst = full_list[(full_list['atr_flag_btst']=='True') & (full_list['vc_tdy_10day']>1) & (full_list['candle_80_type']=='Green')]
    manual_stbt = full_list[(full_list['atr_flag_stbt']=='True') & (full_list['vc_tdy_10day']>1) & (full_list['candle_80_type']=='Red')]

    return full_list, btst, stbt, manual_btst, manual_stbt