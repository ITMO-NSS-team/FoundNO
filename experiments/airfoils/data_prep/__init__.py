"""
data_prep - raw AirfRANS data preprocessing.

Stages:
    index.py        - COMMON: raw graph -> blocks -> rows (index_raw_data_w)
    prep_geofno.py  - rows -> unfolded C-grid (Geo-FNO_data)
    prep_dno.py     - rows -> zonal resampling -> diffeomorphism -> npz (dno_small)
    prep_f_r_no.py  - graph -> regular grid 128x256 (fno_data, for FNO/RNO)
"""
