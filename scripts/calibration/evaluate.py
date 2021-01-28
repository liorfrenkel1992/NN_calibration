import numpy as np
import pandas as pd
from betacal import BetaCalibration
from os.path import join
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from cal_methods import HistogramBinning, TemperatureScaling, evaluate, cal_results


PATH = join('/mnt/dsi_vol1/users/frenkel2/data/calibration/new/NN_calibration/logits/')
files_10 = ('probs_resnet_wide32_c10_logits.p', 'probs_densenet40_c10_logits.p',
            'probs_lenet5_c10_logits.p', 'probs_resnet110_SD_c10_logits.p',
           'probs_resnet110_c10_logits.p', 'probs_resnet152_SD_SVHN_logits.p')
files_100 = ('probs_resnet_wide32_c100_logits.p', 'probs_densenet40_c100_logits.p',
             'probs_lenet5_c100_logits.p', 'probs_resnet110_SD_c100_logits.p')
files_200 = ('probs_resnet50_birds_logits.p',)
files_1k = ('probs_resnet152_imgnet_logits.p', 'probs_densenet161_imgnet_logits.p')

files = ('probs_resnet110_c10_logits.p', 'probs_resnet110_c100_logits.p', 
         'probs_densenet40_c10_logits.p', 'probs_densenet40_c100_logits.p',
        'probs_resnet_wide32_c10_logits.p', 'probs_resnet_wide32_c100_logits.p',
         'probs_resnet50_birds_logits.p', 'probs_resnet110_SD_c10_logits.p',
         'probs_resnet110_SD_c100_logits.p', 'probs_resnet152_SD_SVHN_logits.p',
        'probs_resnet152_imgnet_logits.p', 'probs_densenet161_imgnet_logits.p'  # ImageNet calibration takes rather long time.
        )
        
df_iso = cal_results(IsotonicRegression, PATH, files, {'y_min':0, 'y_max':1}, approach = "single")

df_temp_scale = cal_results(TemperatureScaling, PATH, files, approach = "all")
