"""
This file is used to integrate optgen into programgen
"""
import random

import numpy as np

import common_base_funs
from main_configure_approach import MainProcessConf, OptSupportConf


def gcc_opt_neg(opt):
    if '-fno-' in opt:
        return opt[:2] + opt[5:]
    else:
        return opt[:2] + 'no-' + opt[2:]


def gen_random_opt():
    opt_feature = None
    opt_sequence = None
    if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[0]:  # gcc
        setting_seed = np.array([random.randint(0, 1) for _ in range(len(OptSupportConf.test_opt_list))])
        opt_feature = [(setting_seed ^ _).tolist() for _ in OptSupportConf.mask]
        opt_sequence = []
        for opt_idx in range(len(opt_feature)):
            opt_list = ['-O2']
            opt_list.extend([gcc_opt_neg(_) for _ in OptSupportConf.test_opt_list])
            opt_list.extend([OptSupportConf.test_opt_list[_] for _ in range(len(opt_feature[opt_idx])) if opt_feature[opt_idx][_] == 1])
            if '-funit-at-a-time' not in opt_list and '-ftoplevel-reorder' in opt_list:
                opt_list.append('-fno-toplevel-reorder')
                opt_feature[opt_idx][OptSupportConf.test_opt_list.index('-ftoplevel-reorder')] = 0
            opt_sequence.append(' '.join(opt_list))
    if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[1]:  # llvm
        setting_seed = np.array([random.randint(0, 1) for _ in range(len(OptSupportConf.test_opt_list))])
        opt_feature = [(setting_seed ^ _).tolist() for _ in OptSupportConf.mask]
        opt_sequence = [' '.join([OptSupportConf.test_opt_list[__] for __ in range(len(_)) if _[__] == 1]) for _ in opt_feature]
    assert opt_feature is not None
    assert opt_sequence is not None
    return opt_feature, opt_sequence


def map_opt_feature(opt_feature):
    mapped_opt_feature = []
    for o_idx in range(len(opt_feature)):
        tmp_feature = [0 for _ in OptSupportConf.train_opt_list]
        for f_idx in range(len(opt_feature[o_idx])):
            if OptSupportConf.mapper[f_idx] != -1:
                tmp_feature[OptSupportConf.mapper[f_idx]] = opt_feature[o_idx][f_idx]
        mapped_opt_feature.append(tmp_feature)
    return mapped_opt_feature


# hard coding, statistic from 40,000 program annotation.
def extract_annotation_feature(src):
    types = ['XXX    times read thru a pointer', 'XXX    times written thru a pointer', 'XXX average alias set size',
             'XXX backward jumps', 'XXX const bitfields defined in structs', 'XXX forward jumps',
             'XXX full-bitfields structs in the program', 'XXX max block depth', 'XXX max dereference level',
             'XXX max expression depth', 'XXX max struct depth', 'XXX non-zero bitfields defined in structs',
             'XXX number of pointers point to pointers', 'XXX number of pointers point to scalars',
             'XXX number of pointers point to structs', 'XXX percent of pointers has null in alias set',
             'XXX percentage a fresh-made variable is used', 'XXX percentage an existing variable is used',
             'XXX percentage of non-volatile access', 'XXX stmts', 'XXX structs with bitfields in the program',
             'XXX times a bitfields struct on LHS', 'XXX times a bitfields struct on RHS',
             "XXX times a bitfields struct's address is taken", 'XXX times a non-volatile is read',
             'XXX times a non-volatile is write', 'XXX times a pointer is compared with address of another variable',
             'XXX times a pointer is compared with another pointer', 'XXX times a pointer is compared with null',
             'XXX times a pointer is dereferenced on LHS', 'XXX times a pointer is dereferenced on RHS',
             'XXX times a pointer is qualified to be dereferenced', 'XXX times a single bitfield on LHS',
             'XXX times a single bitfield on RHS', 'XXX times a variable address is taken',
             'XXX times a volatile is available for access', 'XXX times a volatile is read',
             'XXX times a volatile is write', 'XXX total number of pointers', 'XXX total union variables',
             'XXX volatile bitfields defined in structs', 'XXX zero bitfields defined in structs']
    content = common_base_funs.get_file_lines(src)
    inx = content.index('/************************ statistics *************************')
    content = content[inx:]
    content = [_ for _ in content if 'XXX' in _]
    content = [content[i].split(':') for i in range(len(content))]
    tmp_type = {}
    for t in types:
        tmp_type[t] = 0
    for c in content:
        t = c[0]
        v = c[1][1:]
        v = float(v)
        tmp_type[t] = v
    return [tmp_type[key] for key in types]


def get_src_feature(src, src_feature):
    reduce_list = [5, 10, 18, 19, 21, 22, 23, 25, 28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                   48, 71, 82, 92, 97, 105, 108, 109, 110]  # for features that never change
    src_feature = common_base_funs.get_file_lines(src_feature)[0].split(',')
    src_feature = [float(_) for _ in src_feature]
    src_feature = [src_feature[_] for _ in range(len(src_feature)) if _ not in reduce_list]
    src_feature = np.array(src_feature, dtype=float)

    annotation_src_feature = extract_annotation_feature(src)
    annotation_src_feature = np.array(annotation_src_feature, dtype=float)

    src_feature = np.hstack((src_feature, annotation_src_feature))
    src_feature = src_feature.reshape(1, -1)

    src_feature = OptSupportConf.normalizer.transform(src_feature)
    src_feature = src_feature.tolist()
    src_feature = src_feature[0]

    return src_feature


def predict(xgb_clf, feature):
    x = np.array(feature)
    res = xgb_clf.predict_proba(x)
    return [[res[i][1], i] for i in range(len(res))]


def recommend_opt(src, src_feature, xgb_model):  # TODO: to add a record, let we discard options in top-level and record the top ones and their probabilities.
    opt_feature, opt_sequence = gen_random_opt()
    opt_feature = map_opt_feature(opt_feature)
    src_feature = get_src_feature(src, src_feature)

    ml_feature = [src_feature + _ for _ in opt_feature]
    ml_feature = np.array(ml_feature)

    predict_probs = predict(xgb_model, ml_feature)
    predict_probs = sorted(predict_probs, reverse=True)

    seq_prob = [[opt_sequence[predict_probs[i][1]], predict_probs[i][0]] for i in range(OptSupportConf.select_opt_num)]

    return seq_prob
