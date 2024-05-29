Markdown of code
import numpy as np
import pandas as pd

# Make random number generator
rng = np.random.default_rng()

# Safe Pandas
pd.set_option('mode.copy_on_write', True)

# Plotting
import matplotlib.pyplot as plt
# Estimated group size
n = 20
# Creating initial pain scores from ethnographic data
real_pain_scores = np.array([8,8,5,2])
mean_pain_score = np.mean(real_pain_scores) 
std_dev_pain_score = np.std(real_pain_scores)
print("mean pain score:", mean_pain_score)
print("std dev pain score:", std_dev_pain_score)
# start with one estimated group size 
n = 20

# create a dataframe for our simulated patient data
patient_df = pd.DataFrame()
patient_df['group'] = np.repeat(['Experimental', 'Control'], [n, n])
pre_mean = 5.75
pre_std = 2.48
initial_pain = rng.normal(pre_mean, pre_std, size=n * 2)
patient_df['initial pain score'] = np.clip(np.round(initial_pain), 1, 10)
patient_df
# Now we simulate the post-pain scores for each group assuming that there will be between a +1/-1 difference in pain scores for each patient
within_mean = -1
within_std = 1
within_difference = rng.normal(within_mean, within_std, size=n * 2)
patient_df['post pain score'] = np.clip(np.round(initial_pain + within_difference), 1, 10)
patient_df
### Estimating VR effect size ###
The anonymous ethnographic data provides four patients pain scores before and after trialing a 7 minute 'self care' immersive VR therapy. I'll compute the differences between the pain scores before and after the VR session for each patient, calculate the mean and standard deviation of these differences and use this to calculate Cohen's d (the mean difference divided by the standard deviation of the differences) to determine a reasonable estimate for effect size. 
# Assuming that the VR treatment will have a mean effect of -1 on pain scores and that the standard deviation of the effect is 1
vr_mean = -1  # Estimated average effect of VR
vr_std = 1  # STD
vr_effect = np.zeros(n * 2)
vr_effect[:n] = rng.normal(vr_mean, vr_std, size=n)
vr_effect
# Now we can update the VR effect to the post-pain scores 
patient_df['post pain score'] = np.clip(np.round(initial_pain + within_difference + vr_effect), 1, 10)
patient_df
# what we're interested in is the difference in pain scores (per patient) post the VR treatment vs the control treatment
patient_df['difference'] = patient_df['initial pain score'] - patient_df['post pain score']
patient_df
### Permutation testing ###
# calculate the mean difference in pain scores and display by group
diffs = patient_df.groupby('group')['difference'].mean()
diffs
# calculate the actual difference in pain scores between the two groups, we estimated that the VR treatment would have a mean effect of -1 on pain scores
actual_diff = diffs['Experimental'] - diffs['Control']
actual_diff
differences = np.array(patient_df['difference'])

# Simulate one trial
shuffled = rng.permutation(differences)
fake_diff = np.mean(shuffled[:n]) - np.mean(shuffled[n:])
fake_diff
import scipy.stats as sps
# t-test version of the permutation test.  Permutation will give you similar p-value.
sps.ttest_ind(differences[:n], differences[n:])
