# -*- coding: utf-8 -*-
"""Build the CHARLS 2015 (wave 3) analysis file, mirroring charls/extract.py (2018).  No imputation:
every variable keeps its observed value or is missing.

    python charls/extract_2015.py [--raw "data/raw"]
    -> data/charls2015_rd.csv              (one row per respondent in the analysis cohort)
       results/charls_2015/extraction_log.json

Why 2015: in the 2018 wave the age-60 cutoff falls next to the 1959-1961 famine birth cohorts.  In 2015
(interviews July-August 2015, 60 births in January 2016) age 60 corresponds to the 1955 birth cohort and
the famine cohorts are aged about 53.5-56.5 (x = -6.4 ... -3.4).

Running variable: as in 2018, age in months at the interview = (iyear*12 + imonth) - (birth year*12 + birth
month), birth date on the hukou / ID card (ba004_w3_1, ba004_w3_2; month 0 = missing), interview year/month
from Sample_Infor; x = (age_months - 720) / 12.

Cohort: agricultural hukou, age >= 45, >= 1 of the 14 doctor-diagnosed chronic conditions, province known.

Variables that the 2015 files hold only as changes relative to the last interview are rebuilt from the last
observed report (2013, else 2011), as the Harmonized CHARLS does:
  hukou       last-interview type (2013 bc001, else 2011 bc001) if confirmed (bc001_w3_1 = 1); the corrected
              last-interview type bc001_w3_2 where asked; the new type bc002_w3_1 if the type changed or the
              respondent is new.  Disputed without a corrected type -> missing.  Agrees with Harmonized
              CHARLS r3hukou for 99.9% of respondents where both exist.
  chronic i   preloaded diagnosis (zda007_i_ = 1) unless disputed (da007_w2_1_i_ = 2); a disputed preloaded
              'no' (zda007_i_ missing, da007_w2_1_i_ = 2); diagnosed since the last interview
              (da007_w2_2_i_ = 1); or, for new respondents, da007_i_ = 1.
  education   bd001_w2_4 if a level is reported (new respondents, or the level changed); if 12 'no change',
              the 2013 level (bd001 / bd001_w2_4 / corrected bd001_w2_3 / preload zbd001), else 2011 bd001.
Time-invariant traits not asked in 2015 are taken from the respondent's own report in another wave:
  han         2013 ba007_w2_1 (nationality), else 2018 bg001_w4.
  lunar       calendar of the actual birthday: 2015 ba003, else 2013, 2011, 2018 ba003 (missing if never asked).

Take-up D: currently receiving a residents' pension (new rural / residents' / urban residents' scheme):
1 if fn058_w2_k_ = 2 'receive' for any scheme k; 0 if none held (fn057_w3s4) or every held scheme is only
'participate' (fn058_w2_k_ = 1); otherwise missing.

Outcomes (same definitions as 2018):
  srh       da002 (1 very good ... 5 very poor).  2015 asks one of two randomly assigned SRH versions:
            da002 (very good ... very poor; the 2018 scale, about half the respondents) or da001 (excellent
            ... poor).  srh and srh_good use da002 only, as in 2018.
  srh_good  1{srh <= 3}
  cesd10    dc009-dc018, dc013 and dc016 reversed, all ten required
  adl_any   any difficulty with the six basic ADLs (db010-db015, all six required), as in the 2018 code.
            NOTE: in both waves the ADL items are asked only of respondents reporting a difficulty with at
            least one of the nine functional items db001-db009, so adl_any is defined on that subsample.
Additional 2015 outcomes:
  srh_good_all  fair or better on whichever version was asked (da002 <= 3 or da001 <= 4); srh_form = 1 for
                the da001 version (used as a balance check of the randomisation)
  adl_any_all   adl_any, with 0 for respondents skipped past the ADL items because they reported no
                difficulty with any of db001-db009 (questionnaire routing, all respondents)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from extract import REGION, CHRONIC, valid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WAVE_DIR = {2011: '2011 CHARLS Wave1', 2013: '2013 CHARLS Wave2', 2015: '2015 CHARLS Wave 3', 2018: '2018 CHARLS Wave 4'}


def read(raw, wave, f, cols):
    return pd.read_stata(os.path.join(raw, WAVE_DIR[wave], f), convert_categoricals=False, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    a = ap.parse_args()
    log = {}
    DB = read(a.raw, 2015, 'Demographic_Background.dta',
              ['ID', 'householdID', 'communityID', 'ba004_w3_1', 'ba004_w3_2', 'ba003', 'ba000_w2_3', 'be001', 'xrtype',
               'bc001_w3_1', 'bc001_w3_2', 'bc002_w3', 'bc002_w3_1', 'bd001_w2_4'])
    SI = read(a.raw, 2015, 'Sample_Infor.dta', ['ID', 'iyear', 'imonth'])
    HS = read(a.raw, 2015, 'Health_Status_and_Functioning.dta',
              ['ID', 'da001', 'da002'] + [f'{v}_{i}_' for v in ('da007', 'zda007', 'da007_w2_1', 'da007_w2_2') for i in range(1, 15)]
              + [f'db00{i}' for i in range(1, 10)] + [f'db0{i}' for i in range(10, 16)] + [f'dc0{i:02d}' for i in range(9, 19)])
    PE = read(a.raw, 2015, 'Work_Retirement_and_Pension.dta',
              ['ID'] + [f'fn057_w3s{k}' for k in range(1, 5)] + [f'fn058_w2_{k}_' for k in range(1, 4)])
    # earlier / later waves: last observed hukou and education, time-invariant traits
    D13 = read(a.raw, 2013, 'Demographic_Background.dta',
               ['ID', 'bc001', 'ba003', 'ba007_w2_1', 'bd001', 'zbd001', 'bd001_w2_1', 'bd001_w2_3', 'bd001_w2_4'])
    D11 = read(a.raw, 2011, 'demographic_background.dta', ['ID', 'householdID', 'bc001', 'bd001', 'ba003'])
    D11['ID'] = D11['householdID'] + '0' + D11['ID'].str[-2:]      # 2011 IDs -> 2013+ format (2013 release note)
    D18 = read(a.raw, 2018, 'Demographic_Background.dta', ['ID', 'ba003', 'bg001_w4'])
    psu = pd.read_stata(os.path.join(a.raw, WAVE_DIR[2013], 'PSU.dta'), convert_categoricals=False)
    psu['province'] = psu['province'].map(lambda s: s.encode('latin-1').decode('gbk'))
    log['raw_rows'] = dict(demographic=len(DB), sample_info=len(SI), health=len(HS), pension=len(PE),
                           demographic_2013=len(D13), demographic_2011=len(D11), demographic_2018=len(D18))

    lvl13 = valid(D13['bd001'], 1, 11)                                  # new respondents in 2013
    base13 = D13['zbd001'].where(D13['bd001_w2_1'] != 2, D13['bd001_w2_3'])
    lvl13 = lvl13.fillna(valid(D13['bd001_w2_4'], 1, 11)).fillna(valid(base13, 1, 11).where(D13['bd001_w2_4'] == 12))
    D13 = D13.assign(edu13=lvl13)
    pre = lambda d, w, cols: d[['ID'] + cols].rename(columns={c: f'{c}_{w}' for c in cols})
    m = DB.merge(SI, on='ID', how='left').merge(HS, on='ID', how='left').merge(PE, on='ID', how='left') \
          .merge(pre(D13, 13, ['bc001', 'ba003', 'ba007_w2_1', 'edu13']), on='ID', how='left') \
          .merge(pre(D11, 11, ['bc001', 'bd001', 'ba003']), on='ID', how='left') \
          .merge(pre(D18, 18, ['ba003', 'bg001_w4']), on='ID', how='left')
    m = m.copy()                                                         # de-fragment after the merges
    log['linked'] = dict(in_2013=int(m['bc001_13'].notna().sum()), in_2011_only=int((m['bc001_13'].isna() & m['bc001_11'].notna()).sum()))
    m['province'] = m['communityID'].map(dict(zip(psu['communityID'], psu['province'])))
    m['region'] = m['province'].map(REGION)
    # ---------------------------------------------------------------- running variable (months)
    by, bm = valid(m['ba004_w3_1'], 1900, 2016), valid(m['ba004_w3_2'], 1, 12)
    iy, im = valid(pd.to_numeric(m['iyear'], errors='coerce'), 2015, 2016), valid(pd.to_numeric(m['imonth'], errors='coerce'), 1, 12)
    m['birth_year'] = by
    m['interview_ym'] = iy * 100 + im
    m['age_months'] = (iy * 12 + im) - (by * 12 + bm)
    m['x'] = (m['age_months'] - 720) / 12.0
    m['eligible'] = (m['x'] >= 0).astype(float).where(m['x'].notna())
    m['famine_cohort'] = by.between(1959, 1961).astype(float).where(by.notna())
    cal = m['ba003']
    for w in (13, 11, 18):
        cal = cal.where(cal.isin([1, 2]), m[f'ba003_{w}'])
    m['lunar_birthday'] = (cal == 2).astype(float).where(cal.isin([1, 2]))
    # ---------------------------------------------------------------- hukou (current type)
    hk = m['bc001_13'].fillna(m['bc001_11'])
    hk = hk.where(m['bc001_w3_1'] != 2)                                 # disputed: only the corrected value counts
    fix = m['bc001_w3_2'].notna()
    hk[fix] = m.loc[fix, 'bc001_w3_2']
    chg = m['bc002_w3_1'].notna() & ~m['bc002_w3'].isin([2, 4])         # type changed, or new respondent
    hk[chg] = m.loc[chg, 'bc002_w3_1']
    m['hukou'] = hk
    # ---------------------------------------------------------------- chronic conditions
    nchr = np.zeros(len(m))
    for i, nm in enumerate(CHRONIC, start=1):
        z, ver, new, base = m[f'zda007_{i}_'], m[f'da007_w2_1_{i}_'], m[f'da007_w2_2_{i}_'], m[f'da007_{i}_']
        has = (((z == 1) & (ver != 2)) | ((z != 1) & (ver == 2)) | (new == 1) | (base == 1)).astype(float)
        m[f'chr_{nm}'] = has
        nchr += has
    m['n_chronic'] = nchr
    # ---------------------------------------------------------------- outcomes
    m['srh'] = valid(m['da002'], 1, 5)
    m['srh_good'] = (m['srh'] <= 3).astype(float).where(m['srh'].notna())
    srh_a = valid(m['da001'], 1, 5)
    m['srh_form'] = pd.Series(np.nan, index=m.index)
    m.loc[srh_a.notna(), 'srh_form'] = 1.0
    m.loc[m['srh'].notna(), 'srh_form'] = 0.0
    m['srh_good_all'] = m['srh_good'].fillna((srh_a <= 4).astype(float).where(srh_a.notna()))
    items = pd.DataFrame({c: valid(m[c], 1, 4) for c in [f'dc0{i:02d}' for i in range(9, 19)]})
    score = items - 1
    for c in ('dc013', 'dc016'):
        score[c] = 4 - items[c]
    m['cesd10'] = score.sum(axis=1, min_count=10)
    adl = pd.DataFrame({c: valid(m[c], 1, 4) for c in [f'db0{i}' for i in range(10, 16)]})
    m['adl_any'] = (adl >= 2).any(axis=1).astype(float).where(adl.notna().all(axis=1))
    fl = pd.DataFrame({c: valid(m[c], 1, 4) for c in [f'db00{i}' for i in range(1, 10)]})
    # skipped past the ADL items: no difficulty on any answered functional item, db001 and db004-db009 answered
    # (db002 / db003 are themselves skipped after 'no difficulty' running / walking 1 km)
    skipped = (fl >= 2).sum(axis=1).eq(0) & fl[['db001'] + [f'db00{i}' for i in range(4, 10)]].notna().all(axis=1) & adl.isna().all(axis=1)
    m['adl_any_all'] = m['adl_any'].where(~skipped, 0.0)
    # ---------------------------------------------------------------- take-up
    s58 = m[[f'fn058_w2_{k}_' for k in range(1, 4)]]
    held = pd.concat([m[f'fn057_w3s{k}'].notna() | s58.iloc[:, k - 1].notna() for k in range(1, 4)], axis=1)
    held.columns = s58.columns
    recv = (s58 == 2).any(axis=1)
    part_only = held.any(axis=1) & ((s58 == 1) | ~held).all(axis=1)
    none = (m['fn057_w3s4'] == 4) & ~held.any(axis=1)
    D = pd.Series(np.nan, index=m.index)
    D[none | part_only] = 0.0
    D[recv] = 1.0
    m['pension'] = D
    # ---------------------------------------------------------------- predetermined covariates
    m['female'] = (m['ba000_w2_3'] == 2).astype(float).where(m['ba000_w2_3'].isin([1, 2]))
    eth = m['ba007_w2_1_13'].where(m['ba007_w2_1_13'].between(1, 10))
    m['han'] = (eth == 1).astype(float).where(eth.notna())
    m['han'] = m['han'].fillna((m['bg001_w4_18'] == 1).astype(float).where(m['bg001_w4_18'].between(1, 11)))
    edu = valid(m['bd001_w2_4'], 1, 11)
    nochg = m['bd001_w2_4'] == 12
    edu = edu.fillna(m['edu13_13'].where(nochg)).fillna(valid(m['bd001_11'], 1, 11).where(nochg))
    m['edu_illiterate'] = (edu == 1).astype(float).where(edu.notna())
    m['edu_below_primary'] = edu.isin([2, 3]).astype(float).where(edu.notna())
    m['edu_primary'] = (edu == 4).astype(float).where(edu.notna())
    m['edu_middle_plus'] = (edu >= 5).astype(float).where(edu.notna())
    m['married'] = m['be001'].isin([1, 2]).astype(float).where(m['be001'].between(1, 7))   # 7 = cohabiting (2015 only)
    # ---------------------------------------------------------------- cohort filters
    steps = [('all respondents (2015 demographic file)', np.ones(len(m), bool)),
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
    log['hukou_missing_after_age_filter'] = int((m['age_months'].notna() & m['hukou'].isna()).sum())
    s = m[keep].copy()
    cols = ['ID', 'householdID', 'communityID', 'province', 'region', 'birth_year', 'interview_ym', 'age_months', 'x',
            'eligible', 'famine_cohort', 'lunar_birthday', 'pension', 'srh', 'srh_good', 'srh_form', 'srh_good_all',
            'cesd10', 'adl_any', 'adl_any_all', 'female', 'han', 'edu_illiterate', 'edu_below_primary', 'edu_primary',
            'edu_middle_plus', 'married', 'n_chronic'] + [f'chr_{c}' for c in CHRONIC]
    s = s[cols].sort_values('ID').reset_index(drop=True)
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    s.to_csv(os.path.join(ROOT, 'data', 'charls2015_rd.csv'), index=False, encoding='utf-8')
    log['cohort_n'] = len(s)
    log['missing'] = {c: int(s[c].isna().sum()) for c in cols if s[c].isna().any()}
    log['age_range_years'] = [float(s['age_months'].min() / 12), float(s['age_months'].max() / 12)]
    log['x_is_monthly_grid'] = bool(np.allclose(s['x'] * 12, np.round(s['x'] * 12)))
    log['interview_ym_n'] = {str(int(k)): int(v) for k, v in s['interview_ym'].value_counts().sort_index().items()}
    log['region_n'] = s['region'].value_counts().to_dict()
    log['province_n'] = s['province'].value_counts().to_dict()
    log['pension_by_side'] = {'x<0': float(s.loc[s.x < 0, 'pension'].mean()), 'x>=0': float(s.loc[s.x >= 0, 'pension'].mean())}
    log['lunar_share'] = float(s['lunar_birthday'].mean())
    log['srh_form_da001_share'] = float(s['srh_form'].mean())
    log['adl_skipped_recoded_0'] = int((s['adl_any'].isna() & s['adl_any_all'].notna()).sum())
    f = s[s['famine_cohort'] == 1]
    log['famine_cohort_x_range'] = [float(f['x'].min()), float(f['x'].max())]
    os.makedirs(os.path.join(ROOT, 'results', 'charls_2015'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'charls_2015', 'extraction_log.json'), 'w', encoding='utf-8') as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
