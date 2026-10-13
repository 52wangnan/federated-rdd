# -*- coding: utf-8 -*-
"""Build the CHARLS 2018 (wave 4) analysis file from the raw .dta files.  No imputation: every
variable keeps its observed value or is missing.

    python charls/extract.py --raw /path/to/CHARLS
    -> data/charls2018_rd.csv          (one row per respondent in the analysis cohort)
       results/charls/extraction_log.json

Running variable: age in months at the interview = (iyear*12 + imonth) - (birth year*12 + birth month),
with the birth date recorded on the hukou / ID card (ba004_w3_1, ba004_w3_2; the date that determines
pension eligibility) and the interview year/month from Sample_Infor.  x = (age_months - 720) / 12 is
age minus 60 in years on a monthly grid; treated (eligible) if x >= 0.

Cohort: agricultural hukou (current type after any change reported in 2018), age >= 45 (540 months),
at least one of the 14 doctor-diagnosed chronic conditions (reported in 2018 or carried forward from an
earlier wave, zdiagnosed_*).

Take-up D: currently receiving the urban-rural residents' pension (new rural / urban residents /
integrated scheme): fn058_w4 == 1 and fn058_w4_b == 1 -> 1; fn058_w4 == 2 or fn058_w4_b == 2 -> 0.

Outcomes: srh (da002, 1 very good ... 5 very poor), srh_good = 1{srh <= 3}
(fair or better), cesd10 (0-30, items dc009-dc018, dc013 and dc016 reversed, all ten items required),
adl_any = any difficulty with the six basic ADLs (db010-db015, all six required); defined only for respondents
asked the ADL items (the questionnaire routes past them those with no difficulty on any of db001-db009).
adl_any_all = adl_any with 0 for respondents routed past the ADL items (same rule as extract_2015.py).

Sites: province from the 2013 PSU file (communityID; GBK-encoded names), grouped into seven
geographic regions.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REGION = {
    '北京': '华北', '天津': '华北', '河北省': '华北', '山西省': '华北', '内蒙古自治区': '华北',
    '辽宁省': '东北', '吉林省': '东北', '黑龙江省': '东北',
    '上海市': '华东', '江苏省': '华东', '浙江省': '华东', '安徽省': '华东', '福建省': '华东', '江西省': '华东', '山东省': '华东',
    '河南省': '华中', '湖北省': '华中', '湖南省': '华中',
    '广东省': '华南', '广西省': '华南',
    '重庆市': '西南', '四川省': '西南', '贵州省': '西南', '云南省': '西南',
    '陕西省': '西北', '甘肃省': '西北', '青海省': '西北', '新疆维吾尔自治区': '西北',
}
CHRONIC = ['hypertension', 'dyslipidemia', 'diabetes', 'cancer', 'lung', 'liver', 'heart', 'stroke', 'kidney',
           'stomach', 'emotional', 'memory', 'arthritis', 'asthma']


def read(raw, f, cols):
    return pd.read_stata(os.path.join(raw, '2018 CHARLS Wave 4', f), convert_categoricals=False, columns=cols)


def valid(s, lo, hi):
    s = pd.to_numeric(s, errors='coerce')
    return s.where((s >= lo) & (s <= hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='folder with the raw CHARLS files (obtain them from charls.pku.edu.cn)')
    a = ap.parse_args()
    log = {}
    DB = read(a.raw, 'Demographic_Background.dta',
              ['ID', 'householdID', 'communityID', 'ba004_w3_1', 'ba004_w3_2', 'ba005_w4', 'ba002_1', 'ba002_2', 'ba003',
               'ba000_w2_3', 'bg001_w4', 'be001', 'bd001_w2_4', 'zbc004', 'bc001_w3_1', 'bc001_w3_2', 'bc002_w3',
               'bc002_w3_1'])
    SI = read(a.raw, 'Sample_Infor.dta', ['ID', 'iyear', 'imonth'])
    HS = read(a.raw, 'Health_Status_and_Functioning.dta',
              ['ID', 'da002'] + [f'da007_{i}_' for i in range(1, 15)] + [f'zdiagnosed_{i}_' for i in range(1, 15)]
              + [f'db00{i}' for i in range(1, 10)] + [f'db0{i}' for i in range(10, 16)])
    CG = read(a.raw, 'Cognition.dta', ['ID'] + [f'dc0{i:02d}' for i in range(9, 19)])
    PE = read(a.raw, 'Pension.dta', ['ID', 'fn058_w4', 'fn058_w4_a', 'fn058_w4_b'])
    psu = pd.read_stata(os.path.join(a.raw, '2013 CHARLS Wave2', 'PSU.dta'), convert_categoricals=False)
    psu['province'] = psu['province'].map(lambda s: s.encode('latin-1').decode('gbk'))
    log['raw_rows'] = dict(demographic=len(DB), sample_info=len(SI), health=len(HS), cognition=len(CG), pension=len(PE))

    m = DB.merge(SI, on='ID', how='left').merge(HS, on='ID', how='left').merge(CG, on='ID', how='left') \
          .merge(PE, on='ID', how='left')
    m['province'] = m['communityID'].map(dict(zip(psu['communityID'], psu['province'])))
    m['region'] = m['province'].map(REGION)
    # ---------------------------------------------------------------- running variable (months)
    by, bm = valid(m['ba004_w3_1'], 1900, 2018), valid(m['ba004_w3_2'], 1, 12)
    iy, im = valid(m['iyear'], 2018, 2019), valid(m['imonth'], 1, 12)
    m['age_months'] = (iy * 12 + im) - (by * 12 + bm)
    m['x'] = (m['age_months'] - 720) / 12.0
    m['eligible'] = (m['x'] >= 0).astype(float).where(m['x'].notna())
    m['lunar_birthday'] = (m['ba003'] == 2).astype(float)
    m['dob_differs_actual'] = (m['ba005_w4'] == 2).astype(float).where(m['ba005_w4'].notna())
    # ---------------------------------------------------------------- hukou (current type)
    hk = m['zbc004'].copy()
    fix = (m['bc001_w3_1'] == 2) & m['bc001_w3_2'].notna()
    hk[fix] = m.loc[fix, 'bc001_w3_2']
    chg = m['bc002_w3'].isin([1, 3]) & m['bc002_w3_1'].notna()
    hk[chg] = m.loc[chg, 'bc002_w3_1']
    m['hukou'] = hk
    # ---------------------------------------------------------------- chronic conditions
    nchr = np.zeros(len(m))
    for i, nm in enumerate(CHRONIC, start=1):
        has = ((m[f'da007_{i}_'] == 1) | (m[f'zdiagnosed_{i}_'] == 1)).astype(float)
        m[f'chr_{nm}'] = has
        nchr += has
    m['n_chronic'] = nchr
    # ---------------------------------------------------------------- outcomes
    m['srh'] = valid(m['da002'], 1, 5)
    m['srh_good'] = (m['srh'] <= 3).astype(float).where(m['srh'].notna())
    items = pd.DataFrame({c: valid(m[c], 1, 4) for c in [f'dc0{i:02d}' for i in range(9, 19)]})
    score = items - 1
    for c in ('dc013', 'dc016'):
        score[c] = 4 - items[c]
    m['cesd10'] = score.sum(axis=1, min_count=10)
    adl = pd.DataFrame({c: valid(m[c], 1, 4) for c in [f'db0{i}' for i in range(10, 16)]})
    m['adl_any'] = (adl >= 2).any(axis=1).astype(float).where(adl.notna().all(axis=1))
    fl = pd.DataFrame({c: valid(m[c], 1, 4) for c in [f'db00{i}' for i in range(1, 10)]})
    # routed past the ADL items: no difficulty on any answered functional item, db001 and db004-db009 answered
    # (db002 / db003 are themselves skipped after 'no difficulty' running / walking 1 km); as in extract_2015.py
    skipped = (fl >= 2).sum(axis=1).eq(0) & fl[['db001'] + [f'db00{i}' for i in range(4, 10)]].notna().all(axis=1) & adl.isna().all(axis=1)
    m['adl_any_all'] = m['adl_any'].where(~skipped, 0.0)
    # ---------------------------------------------------------------- take-up
    f58, f58b = m['fn058_w4'], m['fn058_w4_b']
    D = pd.Series(np.nan, index=m.index)
    D[(f58 == 2) | ((f58 == 1) & (f58b == 2))] = 0.0
    D[(f58 == 1) & (f58b == 1)] = 1.0
    m['pension'] = D
    # ---------------------------------------------------------------- predetermined covariates
    m['female'] = (m['ba000_w2_3'] == 2).astype(float).where(m['ba000_w2_3'].isin([1, 2]))
    m['han'] = (m['bg001_w4'] == 1).astype(float).where(m['bg001_w4'].between(1, 11))
    edu = valid(m['bd001_w2_4'], 1, 11)
    m['edu_illiterate'] = (edu == 1).astype(float).where(edu.notna())
    m['edu_below_primary'] = edu.isin([2, 3]).astype(float).where(edu.notna())
    m['edu_primary'] = (edu == 4).astype(float).where(edu.notna())
    m['edu_middle_plus'] = (edu >= 5).astype(float).where(edu.notna())
    m['married'] = m['be001'].isin([1, 2]).astype(float).where(m['be001'].between(1, 6))
    # ---------------------------------------------------------------- cohort filters
    steps = [('all respondents (2018 demographic file)', np.ones(len(m), bool)),
             ('age in months observed', m['age_months'].notna()),
             ('agricultural hukou', m['hukou'] == 1),
             ('age >= 45', m['age_months'] >= 540),
             ('>= 1 chronic condition', m['n_chronic'] >= 1),
             ('province known', m['province'].notna())]
    keep = np.ones(len(m), bool)
    log['filters'] = []
    for name, cond in steps:
        keep &= np.asarray(cond, bool)
        log['filters'].append(dict(step=name, n=int(keep.sum())))
    s = m[keep].copy()
    cols = ['ID', 'householdID', 'communityID', 'province', 'region', 'age_months', 'x', 'eligible', 'lunar_birthday',
            'dob_differs_actual', 'pension', 'srh', 'srh_good', 'cesd10', 'adl_any', 'adl_any_all', 'female', 'han', 'edu_illiterate',
            'edu_below_primary', 'edu_primary', 'edu_middle_plus', 'married', 'n_chronic'] + [f'chr_{c}' for c in CHRONIC]
    s = s[cols].sort_values('ID').reset_index(drop=True)
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    s.to_csv(os.path.join(ROOT, 'data', 'charls2018_rd.csv'), index=False, encoding='utf-8')
    log['cohort_n'] = len(s)
    log['missing'] = {c: int(s[c].isna().sum()) for c in cols if s[c].isna().any()}
    log['age_range_years'] = [float(s['age_months'].min() / 12), float(s['age_months'].max() / 12)]
    log['x_is_monthly_grid'] = bool(np.allclose(s['x'] * 12, np.round(s['x'] * 12)))
    log['region_n'] = s['region'].value_counts().to_dict()
    log['province_n'] = s['province'].value_counts().to_dict()
    log['pension_by_side'] = {'x<0': float(s.loc[s.x < 0, 'pension'].mean()), 'x>=0': float(s.loc[s.x >= 0, 'pension'].mean())}
    log['lunar_share'] = float(s['lunar_birthday'].mean())
    log['adl_skipped_recoded_0'] = int((s['adl_any'].isna() & s['adl_any_all'].notna()).sum())
    os.makedirs(os.path.join(ROOT, 'results', 'charls'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'charls', 'extraction_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
