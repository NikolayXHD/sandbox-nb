# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import itertools

# indicator_1_fld = 'dln_exp_3d'
# min_indicator_1 = +0  # 0.005
# max_indicator_1 = +1

# indicator_2_fld = 'dln_exp_no_vol_24d'
# min_indicator_2 = -1
# max_indicator_2 = -0.006

indicator_1_fld = 'dln_exp_4h'
min_indicator_1 = -1  # -0.04
max_indicator_1 = +1  # 0.020

indicator_2_fld = 'dln_exp_no_vol_24d'
min_indicator_2 = -1
max_indicator_2 = -0.008

profit_fld = 'profit_in_currency'

for delay in (7, 30, 180):
    print(delay)
    for dt_from, dt_to in itertools.chain(DATE_RANGES, [(None, None)]):
        dt_from_str = str(dt_from.date()) if dt_from is not None else '***'
        dt_to_str = str(dt_to.date()) if dt_to is not None else '***'

        df_all = delay_to_df[delay]
        print(f'    {dt_from_str} -- {dt_to_str}')
        time_mask = df_all['t'].between(
            (dt_from or datetime(1900, 1, 1)).timestamp(),
            (dt_to or datetime(2100, 1, 1)).timestamp() - 1,
        )
        df = df_all[time_mask]

        mask = (
            df[indicator_1_fld].between(min_indicator_1, max_indicator_1)
        ) & (
            df[indicator_2_fld].between(min_indicator_2, max_indicator_2)
        )
        
        df_positive = df[mask]
        df_negative = df[~mask]
        
        positive_average = np.average(df_positive[profit_fld].values) if len(df_positive) else np.nan
        positive_average_w = np.average(df_positive[profit_fld].values, weights=df_positive['w']) if len(df_positive) else np.nan
        neutral_average = np.average(df[profit_fld].values)
        neutral_average_w = np.average(df[profit_fld].values, weights=df['w'])
        negative_average = np.average(df_negative[profit_fld].values) if len(df_negative) else np.nan
        negative_average_w = np.average(df_negative[profit_fld].values, weights=df_negative['w']) if len(df_negative) else np.nan
        print('            ~w     w      freq')
        print(
            f'     mask:  {positive_average:+4_.2f}  {positive_average_w:+4_.2f}  '
            f'{mask.sum() / len(df):.4f}'
        )
        print(f'    ~mask:  {negative_average:+4_.2f}  {negative_average_w:+4_.2f}')
        print(f'      all:  {neutral_average:+4_.2f}  {neutral_average_w:+4_.2f}')
        print()

# %%
import itertools

indicator_1_fld = 'dln_exp_3d'
indicator_2_fld = 'dln_exp_no_vol_24d'
profit_fld = 'profit_in_currency'

slope_bin = 4 / 8
bin_x = 0.025
bin_y = 0.01
min_x = -0.1
min_y = -0.03 + bin_y * 0.85

for delay in (7, 30, 180):
    print(delay)
    for dt_from, dt_to in itertools.chain(DATE_RANGES, [(None, None)]):
        dt_from_str = str(dt_from.date()) if dt_from is not None else '***'
        dt_to_str = str(dt_to.date()) if dt_to is not None else '***'

        df_all = delay_to_df[delay]
        print(f'    {dt_from_str} -- {dt_to_str}')
        time_mask = df_all['t'].between(
            (dt_from or datetime(1900, 1, 1)).timestamp(),
            (dt_to or datetime(2100, 1, 1)).timestamp() - 1,
        )
        df = df_all[time_mask]

        x = df[indicator_1_fld]
        y = df[indicator_2_fld]

        score = np.maximum(0, slope_bin * (x - min_x) / bin_x - (y - min_y) / bin_y)
        positive_average = np.average(df[profit_fld].values, weights=score)
        positive_average_w = np.average(df[profit_fld].values, weights=score*df['w'])
        neutral_average = np.average(df[profit_fld].values)
        neutral_average_w = np.average(df[profit_fld].values, weights=df['w'])

        print('            ~w     w      freq')
        print(
            f'    score:  {positive_average:+4_.2f}  {positive_average_w:+4_.2f}  '
            f'{(score > 0).sum() / len(df):.4f}'
        )
        print(f'      all:  {neutral_average:+4_.2f}  {neutral_average_w:+4_.2f}')
        print()
