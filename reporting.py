# -*- coding: utf-8 -*-
"""Small helpers that write a DataFrame as CSV, Markdown and LaTeX (no extra dependencies)."""
import os

import numpy as np
import pandas as pd


def _fmt(v, digits):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ''
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f'{v:.{digits}f}'


def md_table(df, digits=4):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(_fmt(r[c], digits) for c in cols) + ' |')
    return '\n'.join(lines) + '\n'


def tex_table(df, digits=4, caption=None):
    esc = lambda s: str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')
    cols = list(df.columns)
    out = [r'\begin{table}[htbp]\centering\small']
    if caption:
        out.append(r'\caption{' + esc(caption) + '}')
    out.append(r'\begin{tabular}{' + 'l' * len(cols) + '}')
    out.append(r'\hline')
    out.append(' & '.join(esc(c) for c in cols) + r' \\')
    out.append(r'\hline')
    for _, r in df.iterrows():
        out.append(' & '.join(esc(_fmt(r[c], digits)) for c in cols) + r' \\')
    out.append(r'\hline')
    out.append(r'\end{tabular}')
    out.append(r'\end{table}')
    return '\n'.join(out) + '\n'


def save_table(df, name, outdir, digits=4, caption=None):
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, name + '.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(outdir, name + '.md'), 'w', encoding='utf-8') as f:
        if caption:
            f.write(f'**{caption}**\n\n')
        f.write(md_table(df, digits))
    with open(os.path.join(outdir, name + '.tex'), 'w', encoding='utf-8') as f:
        f.write(tex_table(df, digits, caption))


def pm(est, se, digits=4):
    """'estimate (MCSE)' string."""
    if not np.isfinite(est):
        return ''
    return f'{est:.{digits}f} ({se:.{digits}f})'
