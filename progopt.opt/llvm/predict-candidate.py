# -*- encoding=utf-8 -*-
# 本文件负责根据输入的feature推荐对应的优化选项序列。
# 本文件的内容看似可以合并在 run.py 中，但这是为了解决 OpenEular 系统的一个 python 多进程问题。
# 在 import 一些常用 python 库后，python 的多进程可能会只能在一个CPU核上运行，造成极低的运行效率。
# 因此，此文件存在的意义即是将这些影响多进程使用的 import 独立在一个文件中，然后在多进程中调用此文件，从而解决 import 影响多进程的问题。
# 该问题是并不是非常常见，而且似乎没有绝对可行的解决方法：
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy?lq=1
# 该网址记录的方法在 OpenEular 似乎是奏效的，但很可惜，我在重构代码之前并未找到这个解决方式
# 输入的部分可以在sys.argv中找到
# 输出为 result_file 指定的文件

import sys
import pickle
import numpy as np
import xgboost as xgb

# inputs for normalization and prediction:
#   program feature file
#   opt sequence file
#   number of opt feature file
#   opt feature files;
#   models files;
#   normalizer files;
#   number of recommend optimization sequence for each model
#   probability threshold
#   result file

program_feature_f = sys.argv[1]    # 程序特征
opt_sequence_f = sys.argv[2]   # 优化选项序列文件
file_num = int(sys.argv[3])    # 文件个数
opt_feature_files = sys.argv[4: 4+file_num]    # 优化选项特征文件（和使用的模型文件一一对应
model_files = sys.argv[4+file_num: 4+2*file_num]    # 模型文件
normalizer_files = sys.argv[4+2*file_num: 4+3*file_num]    # 归一化文件
recommend_num = int(sys.argv[4+3*file_num])    # 每个模型推荐优化序列的数量
prob = float(sys.argv[4+3*file_num+1])    # 推荐概率阈值
result_file = sys.argv[4+3*file_num+2]    # 结果文件

# print('program_feature_f = ' + str(program_feature_f))
# print('opt_sequence_f = ' + str(opt_sequence_f))
# print('file_num = ' + str(file_num))
# print('opt_feature_files = ' + str(opt_feature_files))
# print('model_files = ' + str(model_files))
# print('normalizer_files = ' + str(normalizer_files))
# print('result_file = ' + str(result_file))


def get_model(model_file):
    xgb_clf_ = xgb.XGBClassifier()
    xgb_clf_.load_model(model_file)
    return xgb_clf_


def load_normalizer(nor_f):
    scaler = pickle.load(open(nor_f, 'rb'))
    return scaler


def normalize(scaler, ml_features):
    tmp_features = ml_features[:]
    ml_features = scaler.transform(tmp_features)
    return ml_features.tolist()


def concat_opt_program_features(program_features, opt_features):
    ml_feature = [program_features[0] + opt_features[i] for i in range(len(opt_features))]
    ml_feature = np.array(ml_feature)
    return ml_feature


def get_file_content(path):
    f = open(path, 'r')
    content = f.read()
    f.close()
    return content


def get_file_lines(path):
    c = get_file_content(path)
    if c == '':
        return ''
    if c[-1] == '\n':
        return c[:-1].split('\n')
    else:
        return c.split('\n')


def predict(xgb_clf, feature):
    x = np.array(feature)
    res = xgb_clf.predict_proba(x)
    return [[res[i][1], i] for i in range(len(res))]


def recommend_sequences(xgb_clf, ml_features, opt_sequences):
    predict_probs = predict(xgb_clf, ml_features)
    predict_probs = sorted(predict_probs, reverse=True)
    predict_probs = [predict_probs[i] for i in range(recommend_num)]

    out_ff = open(result_file, 'a+')
    opt_lists = []
    for i in range(recommend_num):
        if predict_probs[i][0] >= prob:
            opt_lists.append(opt_sequences[predict_probs[i][1]])
            out_ff.write(str(predict_probs[i][0]) + ',' + opt_sequences[predict_probs[i][1]] + '\n')
            out_ff.flush()
    out_ff.close()


def main():
    program_feature = get_file_lines(program_feature_f)[0].split(',')
    program_feature = [float(_) for _ in program_feature]
    program_feature = np.array([program_feature])
    opt_features = [get_file_lines(_) for _ in opt_feature_files]
    opt_features = [[__.split(',') for __ in _] for _ in opt_features]
    opt_features = [[[float(___) for ___ in __] for __ in _] for _ in opt_features]
    opt_sequence = get_file_lines(opt_sequence_f)
    models = [get_model(_) for _ in model_files]
    normalizers = [load_normalizer(_) for _ in normalizer_files]

    program_features = [normalize(_, program_feature) for _ in normalizers]
    ml_features = [concat_opt_program_features(program_features[inx], opt_features[inx]) for inx in
                   range(len(opt_features))]

    for _ in range(len(models)):
        recommend_sequences(models[_], ml_features[_], opt_sequence)


if __name__ == '__main__':
    main()
