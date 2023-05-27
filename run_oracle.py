probs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
all_length = [16, 32, 64, 128, 256, 512, 768, 1024]

methods_ens = [
    'inception_time_{}',
    'convnet_{}',
    'resnet_{}',
    'sit_conv_patch_{}',
    'sit_linear_patch_{}',
    'sit_stem_original_{}',
    'sit_stem_relu_{}',
    'rocket_{}',
    'ada_boost_{}',
    'knn_{}',
    'decision_tree_{}',
    'random_forest_{}',
    'mlp_{}',
    'bayes_{}',
    'qda_{}',
    'svc_linear_{}']

methods_conv = [
    'inception_time_{}',
    'convnet_{}',
    'resnet_{}',]

methods_sit = [
    'sit_conv_patch_{}',
    'sit_linear_patch_{}',
    'sit_stem_original_{}',
    'sit_stem_relu_{}',]

methods_ts = ['rocket_{}']

methods_classical = [
    'ada_boost_{}',
    'knn_{}',
    'decision_tree_{}',
    'random_forest_{}',
    'mlp_{}',
    'bayes_{}',
    'qda_{}',
    'svc_linear_{}']

old_method = ['IFOREST', 'LOF', 'MP', 'NORMA', 'IFOREST1', 'HBOS', 'OCSVM', 'PCA', 'AE', 'CNN', 'LSTM', 'POLY']