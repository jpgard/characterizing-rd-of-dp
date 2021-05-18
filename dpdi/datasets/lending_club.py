import pandas as pd
import os


def replace_with_dummy(df, col, baseline_value):
    dummies = pd.get_dummies(df[col])
    dummies.drop(columns=baseline_value, inplace=True)
    dummies.columns = ['_'.join([col, x]) for x in dummies.columns]
    df.drop(columns=col, inplace=True)
    df = pd.concat((df, dummies), axis=1)
    return df


def load_lending_club_dataset(root_dir="../data/lending-club", row_read_limit=None,
                              missing_threshold=0.7):
    lending = pd.read_csv(os.path.join(root_dir, "accepted_2007_to_2018Q4.csv"),
                          low_memory=False,
                          nrows=row_read_limit)
    # lending = pd.concat((accepted, rejected), axis=0)
    census = pd.read_csv(os.path.join(root_dir, "census-data.csv"))
    lending.rename(columns={"fico_range_high": "target"}, inplace=True)
    # drop other info about fico score, and noncausal loan info
    lending.drop(columns=['fico_range_low'], axis=1, inplace=True)
    lending.drop('loan_status', axis=1, inplace=True)
    # Apply threshold for missingness
    frac_missing = lending.isnull().sum(axis=0) / len(lending)
    cols_to_drop = frac_missing[frac_missing >= missing_threshold].index.tolist()
    print("[INFO] dropping {} columns with >= {} missing obs".format(
        len(cols_to_drop), missing_threshold))
    lending.drop(columns=cols_to_drop, inplace=True)
    # drop some dificult to use coluns. might want to get some data out of emp_title
    # using e.g. wordvectors
    lending.drop(['id', 'emp_title', 'title', 'url'], axis=1, inplace=True)
    # drop grade as sub_grade is more detailed?
    lending.drop('grade', axis=1, inplace=True)

    # Drop columns which are not informative, have excessive cardinality, or which
    # are highly correlated with other columns
    lending.drop(columns=['last_pymnt_d', 'out_prncp', 'out_prncp_inv', 'recoveries',
                          'funded_amnt',  # nearly redundant w/amt
                          'open_il_12m', 'open_il_24m',  # nearly redundant w/open_act_il
                          'open_rv_24m',  # nearly redundant w/open_rv_12m
                          'inq_fi', 'total_cu_tl',  # nearly redundant w/inq_last_12m
                          'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
                          'mo_sin_rcnt_tl',
                          'collection_recovery_fee',
                          'next_pymnt_d',
                          'last_pymnt_amnt', 'total_rec_prncp', 'total_rec_int',
                          'total_rec_late_fee', 'total_pymnt', 'total_pymnt_inv',
                          'last_fico_range_high', 'last_fico_range_low',
                          'last_credit_pull_d',
                          'debt_settlement_flag',
                          'sub_grade', 'addr_state',
                          'disbursement_method'  # No variance in this column
                          ],
                 inplace=True)
    lending['issue_d'] = pd.to_datetime(lending['issue_d'].fillna('2016-01-01')).apply(
        lambda x: int(x.strftime('%Y')))
    lending.rename(columns={'issue_d': 'issue_yr'}, inplace=True)
    earliest_cr_line_in_months = \
        pd.to_datetime(lending['earliest_cr_line'].fillna('2001-08-01')) \
            .apply(lambda x: int(x.strftime('%m')) + 12 * int(x.strftime('%Y')))
    lending['age_earliest_cr_line'] = (2020 * 12) - earliest_cr_line_in_months
    lending.drop(columns=['earliest_cr_line'], inplace=True)
    # Convert employment length to numeric
    lending['emp_length'] = lending['emp_length'] \
        .replace(['< 1 year', '10+ years'], ['0', '10']) \
        .str.extract('(\d+)').astype(float)
    # Convert the remaining string columns to indicators
    lending['term_60_months'] = pd.get_dummies(lending['term']).iloc[:, 1]
    lending.drop(columns='term', inplace=True)
    lending = replace_with_dummy(lending, 'home_ownership', 'ANY')
    lending = replace_with_dummy(lending, 'verification_status', 'Not Verified')
    lending = replace_with_dummy(lending, 'pymnt_plan', 'n')
    lending = replace_with_dummy(lending, 'purpose', 'other')
    lending = replace_with_dummy(lending, 'initial_list_status', 'w')
    lending = replace_with_dummy(lending, 'application_type', 'Individual')
    lending = replace_with_dummy(lending, 'hardship_flag', 'N')
    lending['szip'] = lending['zip_code'].str.extract('(\d+)')
    lending.drop(columns='zip_code', inplace=True)
    lending = lending.join(census[["szip", "szip_majority"]], how="inner",
                           lsuffix="_census").drop(columns=["szip_census", "szip"])
    lending.rename(columns={"szip_majority": "sensitive"}, inplace=True)
    return lending
