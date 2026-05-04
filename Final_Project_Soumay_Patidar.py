# DATS 6103 - Final Individual Project
# Soumay Patidar | G49824793
# George Washington University, Spring 2026
#
# Topic: Democratization or Diversion?
#        The Effect of Community Colleges on Educational Attainment
#
# Dataset: communitycollege.csv (High School and Beyond, Rouse 1995)
# Outcome: educ86  - years of education completed by 1986
# Treatment: twoyr - 1 = started at two-year college, 0 = started at four-year college
#
# Three SMART Questions:
# Q1 (OLS): Does the community college effect differ for Hispanic vs non-Hispanic students?
# Q2 (PSM): Does the effect differ for low-income vs higher-income students?
# Q3 (Random Forest): Which background factor matters most for predicting years of education?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

# 
# SECTION 1: Load Data
# 
df = pd.read_csv('communitycollege.csv')

print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum().sum(), "total missing values")

confounders = [
    'bytest', 'fincome', 'female', 'black', 'hispanic',
    'ownhome', 'perwhite', 'urban',
    'dadvoc', 'dadsome', 'dadcoll',
    'momvoc', 'momsome', 'momcoll'
]

# 
# SECTION 2: Exploratory Data Analysis
#
print("Group Sizes")
print(df['twoyr'].value_counts().rename({0: 'Four-year college', 1: 'Two-year college'}))

mean_two = df[df['twoyr'] == 1]['educ86'].mean()
mean_four = df[df['twoyr'] == 0]['educ86'].mean()
naive_effect = mean_two - mean_four

print("\n Naive Comparison")
print(f"Mean educ86 (two-year starters):  {mean_two:.4f}")
print(f"Mean educ86 (four-year starters): {mean_four:.4f}")
print(f"Raw gap:                          {naive_effect:.4f} years")

# Confounder means by group
print("\n Confounder Means by Group")
key_vars = ['bytest', 'fincome', 'hispanic', 'black', 'female', 'ownhome', 'perwhite', 'urban']
group_means = df.groupby('twoyr')[key_vars].mean().T
group_means.columns = ['Four-year (0)', 'Two-year (1)']
group_means['Difference'] = group_means['Two-year (1)'] - group_means['Four-year (0)']
print(group_means.round(3))

# Figure 1: Outcome distribution
fig, ax = plt.subplots(figsize=(8, 5))
df[df['twoyr'] == 0]['educ86'].hist(bins=12, alpha=0.5, color='steelblue', label='Four-year college', ax=ax)
df[df['twoyr'] == 1]['educ86'].hist(bins=12, alpha=0.5, color='coral', label='Two-year college', ax=ax)
ax.axvline(mean_four, color='steelblue', linestyle='--', label=f'Four-yr mean = {mean_four:.2f}')
ax.axvline(mean_two, color='coral', linestyle='--', label=f'Two-yr mean = {mean_two:.2f}')
ax.set_xlabel('Years of Education (educ86)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Educational Attainment by College Type')
ax.legend()
plt.tight_layout()
plt.show()

# Figure 2: Correlation of confounders with outcome
correlations = {col: df[col].corr(df['educ86']) for col in confounders}
corr_series = pd.Series(correlations).sort_values(ascending=False)

print("\n Correlation with educ86 ")
print(corr_series.round(4))

fig, ax = plt.subplots(figsize=(8, 6))
ax.hlines(range(len(corr_series)), 0, corr_series.values, color='steelblue', linewidth=2)
ax.plot(corr_series.values, range(len(corr_series)), 'o', color='steelblue', markersize=8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(range(len(corr_series)))
ax.set_yticklabels(corr_series.index)
ax.set_xlabel('Correlation with educ86')
ax.set_title('Correlation of Confounders with Educational Attainment')
plt.tight_layout()
plt.show()

#
# SECTION 3: Propensity Score Estimation (needed for Q2/PSM)
# 

X = df[confounders]
T = df['twoyr']
Y = df['educ86']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, T)
ps = lr.predict_proba(X_scaled)[:, 1]
df['ps'] = ps

print("\n Propensity Score Summary ")
print(f"Overall mean PS:      {ps.mean():.4f}")
print(f"PS range:             {ps.min():.4f} to {ps.max():.4f}")
print(f"Mean PS (two-year):   {ps[T == 1].mean():.4f}")
print(f"Mean PS (four-year):  {ps[T == 0].mean():.4f}")

# Figure 3: Propensity score overlap
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ps[T == 0], bins=30, alpha=0.5, color='steelblue', label='Four-year college')
ax.hist(ps[T == 1], bins=30, alpha=0.5, color='coral', label='Two-year college')
ax.set_xlabel('Propensity Score')
ax.set_ylabel('Count')
ax.set_title('Propensity Score Distribution - Overlap Check')
ax.legend()
plt.tight_layout()
plt.show()

#
# SECTION 4: Q1 - OLS Regression
#             Does the effect differ for Hispanic vs Non-Hispanic students?
# 

print("Q1: OLS Regression - Hispanic vs Non-Hispanic Students")


# Overall OLS
X_ols = df[['twoyr'] + confounders]
ols_overall = LinearRegression()
ols_overall.fit(X_ols, Y)
overall_effect = ols_overall.coef_[0]

print(f"\nOverall OLS coefficient on twoyr: {overall_effect:.4f}")
print(f"Interpretation: Starting at a two-year college is associated")
print(f"with {abs(overall_effect):.2f} fewer years of education, holding all else equal.")

# OLS with Hispanic interaction
df['twoyr_hisp'] = df['twoyr'] * df['hispanic']
X_hisp = df[['twoyr', 'hispanic', 'twoyr_hisp'] + confounders]
ols_hisp = LinearRegression()
ols_hisp.fit(X_hisp, Y)

non_hisp_penalty = ols_hisp.coef_[0]
hisp_penalty = ols_hisp.coef_[0] + ols_hisp.coef_[2]

print(f"\nOLS with Hispanic interaction:")
print(f"OLS effect for Non-Hispanic students: {non_hisp_penalty:.4f} years")
print(f"OLS effect for Hispanic students:     {hisp_penalty:.4f} years")
print(f"Difference:                        {hisp_penalty - non_hisp_penalty:.4f} years")
print(f"\nQ1 Result: Hispanic students show an effect of {abs(hisp_penalty):.2f} years vs")
print(f"{abs(non_hisp_penalty):.2f} years for non-Hispanic students.")
print(f"The difference is {abs(hisp_penalty - non_hisp_penalty):.2f} years.")

# Figure 4: OLS effect comparison by ethnicity
fig, ax = plt.subplots(figsize=(6, 4))
groups = ['Non-Hispanic', 'Hispanic', 'Overall']
effects = [non_hisp_penalty, hisp_penalty, overall_effect]
colors = ['steelblue', 'coral', 'gray']
bars = ax.barh(groups, effects, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('OLS Treatment Effect (years)')
ax.set_title('Q1: Community College Effect by Ethnicity (OLS)')
for i, v in enumerate(effects):
    ax.text(v - 0.02, i, f'{v:.2f}', va='center', ha='right', color='white', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.show()

#
# SECTION 5: Q2 - Propensity Score Matching
#             Does the effect differ for low-income vs higher-income students?
#
print("Q2: Propensity Score Matching - Low vs Higher Income Students")


median_income = 18000
print(f"\nIncome split point (median): ${median_income:,}")
print(f"Low-income group:   fincome <= ${median_income:,}")
print(f"Higher-income group: fincome > ${median_income:,}")

def run_psm(data, label):
    treated = data[data['twoyr'] == 1].copy()
    control = data[data['twoyr'] == 0].copy()

    if len(treated) < 5 or len(control) < 5:
        print(f"  {label}: not enough data")
        return None

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['ps']])
    distances, indices = nn.kneighbors(treated[['ps']])
    matched_control = control.iloc[indices.flatten()].copy()

    effect = treated['educ86'].mean() - matched_control['educ86'].mean()
    return effect

# Overall PSM
psm_all = run_psm(df, 'All students')

# PSM by income group
low_income = df[df['fincome'] <= median_income].copy()
high_income = df[df['fincome'] > median_income].copy()

psm_low = run_psm(low_income, 'Low-income')
psm_high = run_psm(high_income, 'Higher-income')

print(f"\nPSM Results:")
print(f"All students:            {psm_all:.4f} years")
print(f"Low-income students:     {psm_low:.4f} years")
print(f"Higher-income students:  {psm_high:.4f} years")
print(f"Difference:              {psm_low - psm_high:.4f} years")
print(f"\nAnswer to Q2: Low-income students lose {abs(psm_low):.3f} years vs")
print(f"{abs(psm_high):.3f} years for higher-income students.")
print(f"The difference between the two groups is {abs(psm_low - psm_high):.3f} years.")

# SMD check before and after matching
print("\n Balance Check: SMD Before and After PSM")
treated_all = df[df['twoyr'] == 1]
control_all = df[df['twoyr'] == 0]
nn_all = NearestNeighbors(n_neighbors=1)
nn_all.fit(control_all[['ps']])
_, idx_all = nn_all.kneighbors(treated_all[['ps']])
matched_all = control_all.iloc[idx_all.flatten()]

check_vars = ['bytest', 'fincome', 'hispanic', 'perwhite', 'black']
print(f"{'Variable':<12} {'SMD Before':>12} {'SMD After':>12}")
for col in check_vars:
    t_m = treated_all[col].mean()
    c_m = control_all[col].mean()
    mc_m = matched_all[col].mean()
    std_pool = np.sqrt((treated_all[col].std()**2 + control_all[col].std()**2) / 2)
    smd_b = abs(t_m - c_m) / std_pool
    smd_a = abs(t_m - mc_m) / std_pool
    print(f"{col:<12} {smd_b:>12.3f} {smd_a:>12.3f}")

# Figure 5: PSM income comparison
fig, ax = plt.subplots(figsize=(6, 4))
groups = ['All students', 'Low-income', 'Higher-income']
effects = [psm_all, psm_low, psm_high]
colors = ['gray', 'coral', 'steelblue']
ax.barh(groups, effects, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('PSM Treatment Effect (years)')
ax.set_title('Q2: Community College Effect by Income Group (PSM)')
for i, v in enumerate(effects):
    ax.text(v - 0.02, i, f'{v:.3f}', va='center', ha='right', color='white', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.show()

# 
# SECTION 6: Q3 - Random Forest Feature Importance
#             Which background factor matters most?
#


print("Q3: Random Forest - Which Factor Matters Most?")

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=42)
rf.fit(X, Y)

importances = pd.Series(rf.feature_importances_, index=confounders).sort_values(ascending=False)
importances_pct = (importances * 100).round(1)

print("\nFeature Importances:")
for feat, imp in importances_pct.items():
    print(f"  {feat:<15} {imp:.1f}%")

print(f"\nAnswer to Q3: Test score (bytest) is the #1 factor at {importances_pct['bytest']:.1f}%.")
print(f"This is followed by perwhite ({importances_pct['perwhite']:.1f}%) and fincome ({importances_pct['fincome']:.1f}%).")

# PSM by test score group to check if high scorers are more protected
median_test = df['bytest'].median()
print(f"\nMedian test score: {median_test:.2f}")

low_test = df[df['bytest'] <= median_test].copy()
high_test = df[df['bytest'] > median_test].copy()

psm_low_test = run_psm(low_test, 'Below median test score')
psm_high_test = run_psm(high_test, 'Above median test score')

print(f"\nPenalty by test score group:")
print(f"Below median test score: {psm_low_test:.4f} years")
print(f"Above median test score: {psm_high_test:.4f} years")
print(f"\nBoth groups show a negative effect. Test score has the highest")
print(f"feature importance score in predicting educ86.")
print(f"")

# Figure 6: Feature importance vertical bar chart
fig, ax = plt.subplots(figsize=(9, 5))
colors_rf = ['coral' if feat == 'bytest' else 'steelblue' for feat in importances.index]
ax.bar(importances.index, importances_pct.values, color=colors_rf, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Variable')
ax.set_ylabel('Feature Importance (%)')
ax.set_title('Q3: Random Forest Feature Importance for Predicting educ86')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(importances_pct.values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8)
plt.tight_layout()
plt.show()

# 
# SECTION 7: Final Summary
# 

print("FINAL SUMMARY OF ALL RESULTS")


print(f"\n{'Method':<35} {'Effect (years)':>15}")
print(f"{'Naive comparison':<35} {naive_effect:>15.4f}")
print(f"{'OLS overall':<35} {overall_effect:>15.4f}")
print(f"{'OLS Hispanic students':<35} {hisp_penalty:>15.4f}")
print(f"{'OLS Non-Hispanic students':<35} {non_hisp_penalty:>15.4f}")
print(f"{'PSM all students':<35} {psm_all:>15.4f}")
print(f"{'PSM low-income students':<35} {psm_low:>15.4f}")
print(f"{'PSM higher-income students':<35} {psm_high:>15.4f}")
print(f"{'RF top factor: bytest importance':<35} {importances_pct['bytest']:>14.1f}%")

print("\nKey Takeaways:")
print("1. Starting at a community college reduces education by about 0.88 to 1.03 years.")
print("2. Hispanic students show an OLS effect of -0.76 yrs vs -1.10 yrs for non-Hispanic students (Q1).")
print("3. PSM effect: higher-income students show -1.07 yrs vs -0.85 yrs for low-income students (Q2).")
print("4. Test score is the most important predictor of educational attainment at 57.9% (Q3 answered).")

# Figure 7: Overall comparison vertical bar chart
fig, ax = plt.subplots(figsize=(10, 5))
methods = ['Naive', 'OLS Overall', 'OLS Hispanic', 'OLS Non-Hispanic',
           'PSM All', 'PSM Low-income', 'PSM Higher-income']
effects_all = [naive_effect, overall_effect, hisp_penalty, non_hisp_penalty,
               psm_all, psm_low, psm_high]
colors_all = ['gray', 'gray', 'coral', 'steelblue', 'gray', 'coral', 'steelblue']
ax.bar(methods, effects_all, color=colors_all, edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('Estimated Treatment Effect (years of education)')
ax.set_title('Summary: Community College Effect Across Methods and Groups')
ax.tick_params(axis='x', rotation=30)
for i, v in enumerate(effects_all):
    ax.text(i, v - 0.03, f'{v:.2f}', ha='center', va='top', fontsize=9, fontweight='bold', color='white')
plt.tight_layout()
plt.show()
