import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('formant_test_output_bw.csv', header=None, delimiter=' ')
clean_data=data[data[4] !=0]
kid=clean_data[clean_data[0]=='kid']
cooed=clean_data[clean_data[0]=='cooed']
could=clean_data[clean_data[0]=='could']
cod=clean_data[clean_data[0]=='cod']
cud=clean_data[clean_data[0]=='cud']
keyed=clean_data[clean_data[0]=='keyed']
idx=1
mean_absolute_error(kid[idx], kid[idx+3])
np.mean(kid[idx]-kid[idx+3])
idx=2
mean_absolute_error(kid[idx], kid[idx+3])
np.mean(kid[idx]-kid[idx+3])
idx=3
mean_absolute_error(kid[idx], kid[idx+3])
np.mean(kid[idx]-kid[idx+3])
idx=1
mean_absolute_error(keyed[idx], keyed[idx+3])
np.mean(keyed[idx]-keyed[idx+3])
idx=2
mean_absolute_error(keyed[idx], keyed[idx+3])
np.mean(keyed[idx]-keyed[idx+3])
idx=3
mean_absolute_error(keyed[idx], keyed[idx+3])
np.mean(keyed[idx]-keyed[idx+3])
idx=1
mean_absolute_error(could[idx], could[idx+3])
np.mean(could[idx]-could[idx+3])
idx=2
mean_absolute_error(could[idx], could[idx+3])
np.mean(could[idx]-could[idx+3])
idx=3
mean_absolute_error(could[idx], could[idx+3])
np.mean(could[idx]-could[idx+3])
idx=1
mean_absolute_error(cooed[idx], cooed[idx+3])
np.mean(cooed[idx]-cooed[idx+3])
idx=2
mean_absolute_error(cooed[idx], cooed[idx+3])
np.mean(cooed[idx]-cooed[idx+3])
idx=3
mean_absolute_error(cooed[idx], cooed[idx+3])
np.mean(cooed[idx]-cooed[idx+3])
idx=1
mean_absolute_error(cod[idx], cod[idx+3])
np.mean(cod[idx]-cod[idx+3])
idx=2
mean_absolute_error(cod[idx], cod[idx+3])
np.mean(cod[idx]-cod[idx+3])
idx=3
mean_absolute_error(cod[idx], cod[idx+3])
np.mean(cod[idx]-cod[idx+3])
idx=1
mean_absolute_error(cud[idx], cud[idx+3])
np.mean(cud[idx]-cud[idx+3])
idx=2
mean_absolute_error(cud[idx], cud[idx+3])
np.mean(cud[idx]-cud[idx+3])
idx=3
mean_absolute_error(cud[idx], cud[idx+3])
np.mean(cud[idx]-cud[idx+3])
