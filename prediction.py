import numpy as np
setting = 'informer_london_merged_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')
print(preds.shape)
print(trues.shape)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(trues[0,:,-1], label='GroundTruth')
plt.plot(preds[0,:,-1], label='Prediction')
plt.legend()
plt.show()

# >>>>>>>testing : informer_WTH_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# test 6989
# test shape: (218, 32, 24, 12) (218, 32, 24, 12)
# test shape: (6976, 24, 12) (6976, 24, 12)
# mse:0.31942644715309143, mae:0.3780726492404938
# >>>>>>>predicting : informer_WTH_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# pred 1