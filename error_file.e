(INFO)[utils.py:340]  $AICROWD_DATASET_NAME=dsprites_full
(INFO)[utils.py:341]  $DATASET_NAME=dsprites_full
(INFO)[utils.py:342]  $DISENTANGLEMENT_LIB_DATA=/data/emarcona/disentanglement_lib/
(INFO)[base_disentangler.py:23]  Device: cuda
(INFO)[data_loader.py:535]  Datasets root: /data/emarcona/disentanglement_lib/
(INFO)[data_loader.py:536]  Dataset: dsprites_full
(INFO)[data_loader.py:290]  include_labels: ['0', '1', '2', '3', '4', '5']
/home/emarcona/GrayVAE/disentanglement-pytorch/common/data_loader.py:387: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  label_weights = np.array(label_weights)#, dtype=np.float32)
(INFO)[data_loader.py:497]  num_classes: [1, 3, 6, 40, 32, 32]
(INFO)[data_loader.py:498]  class_values: [array([0]), array([0, 1, 2]), array([0, 1, 2, 3, 4, 5]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]), array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])]
(INFO)[base_disentangler.py:84]  Number of samples: 737280
(INFO)[base_disentangler.py:85]  Number of batches per epoch: 9216
(INFO)[base_disentangler.py:86]  Number of channels: 1


  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
WARNING: Logging before flag parsing goes to stderr.
W0316 11:41:50.907560 139639434540864 __init__.py:56] Some hub symbols are not available because TensorFlow version is less than 1.14
I0316 11:42:00.004961 139639434540864 beta_vae.py:60] Generating training set.

/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 11:42:22.413280 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 11:42:22.416028 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 11:42:22.417091 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 11:42:33.593968 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 11:42:33.595171 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 11:42:33.605604 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 11:42:42.608258 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 11:42:42.823047 139639434540864 factor_vae.py:74] Generating training set.
I0316 11:42:54.514243 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 11:42:54.514561 139639434540864 factor_vae.py:85] Training set accuracy: 0.94
I0316 11:42:54.514654 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 11:43:00.356503 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 11:43:00.356715 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.94
I0316 11:43:00.365174 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9438
I0316 11:43:09.355288 139639434540864 mig.py:52] Generating training set.
I0316 11:43:09.938089 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.4192452484348729
I0316 11:43:18.931549 139639434540864 sap_score.py:61] Generating training set.
I0316 11:43:19.678450 139639434540864 sap_score.py:68] Computing score matrix.
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
I0316 11:44:10.263354 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 11:44:10.272018 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.11927999999999998
I0316 11:44:19.313527 139639434540864 dci.py:59] Generating training set.
I0316 11:45:59.792958 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.5476577727430094
I0316 11:46:08.818446 139639434540864 irs.py:58] Generating training set.
I0316 11:46:09.361222 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7124314306835775



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 11:49:25.090197 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 11:49:25.097049 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 11:49:25.097393 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 11:49:36.475507 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 11:49:36.478400 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 11:49:36.490492 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 11:49:45.472509 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 11:49:45.669544 139639434540864 factor_vae.py:74] Generating training set.
I0316 11:49:57.372969 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 11:49:57.373278 139639434540864 factor_vae.py:85] Training set accuracy: 0.94
I0316 11:49:57.373381 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 11:50:03.211139 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 11:50:03.211342 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 11:50:03.220266 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.946
I0316 11:50:12.195247 139639434540864 mig.py:52] Generating training set.
I0316 11:50:12.776794 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.45340918875454417
I0316 11:50:21.753546 139639434540864 sap_score.py:61] Generating training set.
I0316 11:50:22.485835 139639434540864 sap_score.py:68] Computing score matrix.
I0316 11:50:54.548349 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 11:50:54.556740 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12276000000000001
I0316 11:51:03.562659 139639434540864 dci.py:59] Generating training set.
I0316 11:52:41.490899 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6045083408993128
I0316 11:52:50.479727 139639434540864 irs.py:58] Generating training set.
I0316 11:52:51.035311 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7880696226247055



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 11:56:09.855813 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 11:56:09.858283 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 11:56:09.858422 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 11:56:21.014166 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 11:56:21.018104 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 11:56:21.028422 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 11:56:30.012027 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 11:56:30.210735 139639434540864 factor_vae.py:74] Generating training set.
I0316 11:56:41.891687 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 11:56:41.891990 139639434540864 factor_vae.py:85] Training set accuracy: 0.9
I0316 11:56:41.892074 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 11:56:47.737253 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 11:56:47.737461 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.9
I0316 11:56:47.745849 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.898
I0316 11:56:56.735581 139639434540864 mig.py:52] Generating training set.
I0316 11:56:57.308298 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.46865789856636064
I0316 11:57:06.296582 139639434540864 sap_score.py:61] Generating training set.
I0316 11:57:07.037255 139639434540864 sap_score.py:68] Computing score matrix.
I0316 11:57:24.094141 139639434540864 sap_score.py:81] SAP score: 0.15
I0316 11:57:24.102760 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.14900000000000002
I0316 11:57:33.104927 139639434540864 dci.py:59] Generating training set.
I0316 11:59:12.187798 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6308697593152623
I0316 11:59:21.209743 139639434540864 irs.py:58] Generating training set.
I0316 11:59:21.760334 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7662957716197002



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:02:46.581920 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:02:46.584625 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:02:46.584773 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:02:57.784731 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:02:57.786704 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:02:57.799264 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:03:06.849198 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:03:07.050105 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:03:18.787591 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:03:18.787797 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:03:18.787857 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:03:24.642975 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:03:24.643160 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:03:24.651962 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9486
I0316 12:03:33.715637 139639434540864 mig.py:52] Generating training set.
I0316 12:03:34.298295 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.42673324005755686
I0316 12:03:43.343023 139639434540864 sap_score.py:61] Generating training set.
I0316 12:03:44.090154 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:04:00.939100 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:04:00.948957 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12348
I0316 12:04:09.983194 139639434540864 dci.py:59] Generating training set.
I0316 12:05:49.089992 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6620008592387708
I0316 12:05:58.168323 139639434540864 irs.py:58] Generating training set.
I0316 12:05:58.717641 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7963196933764141



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:09:20.533918 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:09:20.546375 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:09:20.546480 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:09:31.771861 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:09:31.773118 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:09:31.788658 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:09:40.836836 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:09:41.034289 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:09:52.776428 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:09:52.776739 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:09:52.776856 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:09:58.650162 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:09:58.650381 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:09:58.661398 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9548
I0316 12:10:07.708637 139639434540864 mig.py:52] Generating training set.
I0316 12:10:08.282851 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.4473046726279894
I0316 12:10:17.362179 139639434540864 sap_score.py:61] Generating training set.
I0316 12:10:18.102439 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:10:34.168134 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:10:34.177398 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12187999999999999
I0316 12:10:43.256793 139639434540864 dci.py:59] Generating training set.
I0316 12:12:22.793983 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6633766120889761
I0316 12:12:31.856171 139639434540864 irs.py:58] Generating training set.
I0316 12:12:32.409045 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7926677468507519



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:15:53.193713 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:15:53.206235 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:15:53.206389 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:16:04.524230 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:16:04.532830 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:16:04.546614 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:16:13.713056 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:16:13.919244 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:16:25.689345 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:16:25.689668 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:16:25.689748 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:16:31.568317 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:16:31.568633 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:16:31.579055 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9542
I0316 12:16:40.798316 139639434540864 mig.py:52] Generating training set.
I0316 12:16:41.392776 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.42348089898396657
I0316 12:16:50.561907 139639434540864 sap_score.py:61] Generating training set.
I0316 12:16:51.325293 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:17:06.974189 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:17:06.985612 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12068000000000001
I0316 12:17:43.897610 139639434540864 dci.py:59] Generating training set.
I0316 12:19:24.409527 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6623188381763527
I0316 12:19:33.604728 139639434540864 irs.py:58] Generating training set.
I0316 12:19:34.172277 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7924030459487293



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:23:03.585120 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:23:03.597807 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:23:03.598071 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:23:14.858203 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:23:14.860512 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:23:14.874174 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:23:23.976290 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:23:24.183385 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:23:35.958001 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:23:35.958326 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:23:35.958407 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:23:41.836381 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:23:41.836582 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:23:41.846415 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9542
I0316 12:23:50.927189 139639434540864 mig.py:52] Generating training set.
I0316 12:23:51.529797 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.42348089898396657
I0316 12:24:00.599279 139639434540864 sap_score.py:61] Generating training set.
I0316 12:24:01.348271 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:24:17.049385 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:24:17.059260 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12068000000000001
I0316 12:24:26.125267 139639434540864 dci.py:59] Generating training set.
I0316 12:26:05.871460 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6618854125588333
I0316 12:26:14.979968 139639434540864 irs.py:58] Generating training set.
I0316 12:26:15.543042 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7924030459487293



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:29:42.757752 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:29:42.771125 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:29:42.771246 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:29:54.022059 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:29:54.023936 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:29:54.035536 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:30:03.162950 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:30:03.367743 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:30:15.112257 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:30:15.112567 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:30:15.112652 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:30:20.981026 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:30:20.981213 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:30:20.991659 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9542
I0316 12:30:30.062768 139639434540864 mig.py:52] Generating training set.
I0316 12:30:30.664991 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.42348089898396657
I0316 12:30:39.733740 139639434540864 sap_score.py:61] Generating training set.
I0316 12:30:40.481069 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:30:56.293683 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:30:56.303948 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12068000000000001
I0316 12:31:05.376863 139639434540864 dci.py:59] Generating training set.
I0316 12:32:45.321490 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.662209640129539
I0316 12:32:54.406588 139639434540864 irs.py:58] Generating training set.
I0316 12:32:54.968375 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7924030459487293



/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:36:24.427386 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:36:24.450790 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:36:24.450937 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:36:35.799494 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:36:35.809477 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:36:35.822419 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:36:44.915625 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:36:45.118134 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:36:56.863698 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:36:56.864012 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:36:56.864097 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:37:02.740710 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:37:02.740913 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:37:02.751076 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9542
I0316 12:37:11.882532 139639434540864 mig.py:52] Generating training set.
I0316 12:37:12.486351 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.42348089898396657
I0316 12:37:21.553929 139639434540864 sap_score.py:61] Generating training set.
I0316 12:37:22.306840 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:37:38.002424 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:37:38.011534 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12068000000000001
I0316 12:37:47.085098 139639434540864 dci.py:59] Generating training set.
I0316 12:39:26.943763 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6622925997184047
I0316 12:39:36.072899 139639434540864 irs.py:58] Generating training set.
I0316 12:39:36.640615 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7924030459487293


I0316 12:43:33.135446 139639434540864 beta_vae.py:65] Training sklearn model.
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:43:33.206853 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:43:33.219369 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:43:33.219972 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:43:44.528445 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:43:44.531921 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:43:44.577877 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:43:54.906764 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:43:55.119636 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:44:06.913757 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:44:06.914101 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:44:06.914267 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:44:12.795396 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:44:12.795621 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:44:12.806735 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9542
I0316 12:44:21.948965 139639434540864 mig.py:52] Generating training set.
I0316 12:44:22.551078 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.42348089898396657
I0316 12:44:31.631339 139639434540864 sap_score.py:61] Generating training set.
I0316 12:44:32.391015 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:44:48.094000 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:44:48.105239 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12068000000000001
I0316 12:44:59.581789 139639434540864 dci.py:59] Generating training set.
I0316 12:46:40.200183 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6625435343365913
I0316 12:46:50.966345 139639434540864 irs.py:58] Generating training set.
I0316 12:46:51.536088 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7924030459487293


I0316 12:50:40.480063 139639434540864 beta_vae.py:65] Training sklearn model.
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/data/emarcona/miniconda3/envs/dis/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
I0316 12:50:40.552247 139639434540864 beta_vae.py:69] Evaluate training set accuracy.
I0316 12:50:40.579474 139639434540864 beta_vae.py:72] Training set accuracy: 1
I0316 12:50:40.579609 139639434540864 beta_vae.py:74] Generating evaluation set.
I0316 12:50:52.091644 139639434540864 beta_vae.py:79] Evaluate evaluation set accuracy.
I0316 12:50:52.093821 139639434540864 beta_vae.py:81] Evaluation set accuracy: 1
I0316 12:50:52.107410 139639434540864 aicrowd_utils.py:66] Evaluation   beta_vae_sklearn=1.0
I0316 12:51:54.971482 139639434540864 factor_vae.py:61] Computing global variances to standardise.
I0316 12:51:58.436667 139639434540864 factor_vae.py:74] Generating training set.
I0316 12:52:10.397362 139639434540864 factor_vae.py:82] Evaluate training set accuracy.
I0316 12:52:10.397601 139639434540864 factor_vae.py:85] Training set accuracy: 0.95
I0316 12:52:10.397687 139639434540864 factor_vae.py:87] Generating evaluation set.
I0316 12:52:16.374308 139639434540864 factor_vae.py:93] Evaluate evaluation set accuracy.
I0316 12:52:16.374616 139639434540864 factor_vae.py:96] Evaluation set accuracy: 0.95
I0316 12:52:16.385658 139639434540864 aicrowd_utils.py:66] Evaluation   factor_vae_metric=0.9542
I0316 12:52:46.218971 139639434540864 mig.py:52] Generating training set.
I0316 12:52:46.842276 139639434540864 aicrowd_utils.py:66] Evaluation   mig=0.42348089898396657
I0316 12:52:55.934589 139639434540864 sap_score.py:61] Generating training set.
I0316 12:52:56.692570 139639434540864 sap_score.py:68] Computing score matrix.
I0316 12:53:12.422933 139639434540864 sap_score.py:81] SAP score: 0.12
I0316 12:53:12.433479 139639434540864 aicrowd_utils.py:66] Evaluation   sap_score=0.12068000000000001
I0316 12:53:21.539920 139639434540864 dci.py:59] Generating training set.
I0316 12:55:01.713234 139639434540864 aicrowd_utils.py:66] Evaluation   dci=0.6622250820912403
I0316 12:55:10.851568 139639434540864 irs.py:58] Generating training set.
I0316 12:55:11.431204 139639434540864 aicrowd_utils.py:66] Evaluation   irs=0.7924030459487293

