# -*- encoding=utf-8 -*-
'''
Input:
  1. a set of model
  2. a set of train gcc(train compiler)
  3. a set of feature file to normalize
  4. a mask file to find optimization setting
  5. a test compiler(dev)
Output:
  1. reports
Steps:
  # prepare - load file or get optimization flags
  # test loop
    # environment - clean the environment
    # generate program
    # generate optimization settings and features
    # process the feature
      # mapping
      # append
      # normalize
    # predict
    # original test
    # dof test
    # dol test
'''
# from copyreg import pickle
import os
import time
import random
import numpy as np
import xgboost as xgb
import pickle
from multiprocessing import Process
from sklearn.preprocessing import MinMaxScaler
############################## multi-processing configure ##############################
CORE_NUM = 60    # 并发数
############################## command configure ##############################
TIMEOUT_EACH_OPERATION = 120    # 每一步的超时时间
############################## program generate configure ##############################
CSMITH = '/data/bin/csmith_record/bin/csmith'    # Csmith路径
RANDOM_CNT = 0    # 有 RANDOM_CNT/(10+RANDOM_CNT+DEFAULT_CNT) 的概率使用随机配置
DEFAULT_CNT = 1    # 有 DEFAULT_CNT+10/(10+RANDOM_CNT+DEFAULT_CNT) 的概率使用默认配置
HICOND_CONF_DIR = 'conf/'    # hicond配置文件夹
############################## compile configure ##############################
TEST_GCC = '/data/bin/gcc-4.6.0/bin/gcc'    # 被测编译器路径
TRAIN_GCC_OPT_FILES = ['model-related/gcc-430-opt.txt',    # 训练使用的优化选项序列，用于生成映射
                       'model-related/gcc-435-opt.txt',
                       'model-related/gcc-440-opt.txt',
                       'model-related/gcc-441-opt.txt']
CSMITH_LIB = '/data/bin/csmith_record/include/csmith-2.3.0/'    # Csmith头文件路径
WORK_DIR_PREFIX = 'test_dir'    # 工作文件夹
############################## model configure ##############################
FEATURE_FILES = ['model-related/csmith-annotation-430-update.csv',    # useless，归一化模型文件（已经废弃使用，但保留允许用户重新在不同版本上输出
                 'model-related/csmith-annotation-435-update.csv',
                 'model-related/csmith-annotation-440-update.csv',
                 'model-related/csmith-annotation-441-update.csv']
NORMALIZERS = ['model-related/nor-430',    # 归一化模型
               'model-related/nor-435',
               'model-related/nor-440',
               'model-related/nor-441']
MODELS = ['model-related/GCC-4.3.0.model',    # 模型
          'model-related/GCC-4.3.5.model',
          'model-related/GCC-4.4.0.model',
          'model-related/GCC-4.4.1.model']
# MODELS、NORMALIZERS、TRAIN_GCC_OPT_FILES三者需要一一对应
############################## approach configure ##############################
OPTIMIZATION_NUM = 2000    # 每个测试程序生成的优化序列个数
TOP_N = 4    # 每个模型取用测试的优化序列个数
CUT_OOF_PROB = 0.5    # 若所有优化序列触发编译器缺陷的概率都小于 CUT_OOF_PROB，将程序废弃，测试下一个程序
############################## test configure ##############################
TMP_RES = 'tmp-res.txt'
ORI_RES = 'ori-res.txt'
COMPILE_TIMEOUT_REPORT = 'compile_timeout'
COMPILE_CRASH_REPORT = 'compile_crash'
EXEC_TIMEOUT_REPORT = 'exec_timeout'
EXEC_CRASH_REPORT = 'exec_crash'
MISCOMPILE = 'miscompile_recorder'
############################## mask configure ##############################
GCC_MASK_FILE = 'mask/GCC-4.4.0_4.6.0-mask.txt'
############################## log configure ##############################
LOG_DIR = WORK_DIR_PREFIX + '/log' + os.sep
############################## command operation ##############################
def execmd(cmd):
    import os
    pipe = os.popen(cmd)
    log('[execmd] ' + cmd)
    reval = pipe.read()
    pipe.close()
    return reval

def execmd_limit_time(cmd):
    import time
    start = time.time()
    execmd("timeout " + str(TIMEOUT_EACH_OPERATION) + " " + cmd)
    end = time.time()
    if (end - start) >= TIMEOUT_EACH_OPERATION:
        return False
    else:
        return True
############################## file operation ##############################
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

def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.close()

def rm_file(f):
    if '/' == f or '/*' == f or '*' in f:
        return
    cmd = 'rm -rf ' + f
    execmd(cmd)

def create_dir_if_not_exists(dir_name):
    if '/' == dir_name or '/*' == dir_name or '*' in dir_name:
        return
    if not os.path.exists(dir_name):
        execmd('mkdir -p ' + dir_name)
create_dir_if_not_exists(LOG_DIR)
############################## logger ##############################
def log(msg):
    file = LOG_DIR + 'log-' + str(os.getpid()) + '.txt'
    put_file_content(file, '[' + str(os.getpid()) + ']' + msg + '\n')
############################## optimization function ##############################
def filter_list(l, s):
    res = []
    for elem in l:
        if s in elem:
            res.append(elem)
    return res

def list_strip_each(l):
    res = []
    for elem in l:
        res.append(elem.strip())
    return res

def list_get_first_word_each(l):
    res = []
    for elem in l:
        res.append(elem[:elem.find(' ')])
    return res

def random_sequence(l):
    res = l[:]
    for i in range(len(res)):
        inx = random.randint(0, len(res) - 1)
        tmp = res[i]
        res[i] = res[inx]
        res[inx] = tmp
    return res

def get_O3_opts(gcc):
    opt3 = execmd(gcc + ' -O3 -Q --help=optimizers').split('\n')
    opt3 = filter_list(opt3, '[enabled]')
    opt3 = list_strip_each(opt3)
    opt3 = list_get_first_word_each(opt3)
    return opt3
TEST_GCC_OPT = get_O3_opts(TEST_GCC)
# BLACK_OPTS = get_file_lines('black-opt')
# BLACK_OPTS = [_.split(' ') for _ in BLACK_OPTS]

def get_negation_flag(flag):
    if '-fno-' not in flag:
        if '-fweb-' not in flag:
            return flag[:2] + 'no-' + flag[2:]
        else:
            return flag[6:7] + 'no-' + flag[7:]
    else:
        return flag[:2] + flag[5:]
TEST_GCC_NEGATIVE_OPT = [get_negation_flag(o) for o in TEST_GCC_OPT]

def random_level():
    return ['-O1', '-O2', '-O3', '-Os'][random.randint(0, 3)]
############################## test program function ##############################
def get_conf_type(conf_num):
    if conf_num <= 10:
        return str(conf_num)
    if conf_num > 10 and conf_num <= 10 + DEFAULT_CNT:
        return 'd'
    if conf_num > 10 + DEFAULT_CNT:
        return 'r'

def get_conf_file(conf_num):
    conf_file = 'default'
    # if conf_num <= 10:
    #     conf_file = HICOND_CONF_DIR + 'config' + str(conf_num)
    # if conf_num > 10 and conf_num <= 10 + DEFAULT_CNT:
    #     conf_file = 'default'
    # if conf_num > 10 + DEFAULT_CNT:
    #     conf_file = 'random'
    return conf_file

def generate_src(config_file, src_file, hicond_statistic_file):
    swarm_testing = ''
    # if random.randint(0, 1) == 1:
    #     swarm_testing = ' '.join(random_sequence(SWARM_TESTING_FLAGS)[:len(SWARM_TESTING_FLAGS)])
    log('[generate_src] generate program: src=' + src_file + ', config_file=' + config_file + ', feature_file=' + hicond_statistic_file)
    if config_file == 'default':
        cmd = CSMITH + ' ' + swarm_testing + ' --record-file ' + hicond_statistic_file + ' > ' + src_file
    elif config_file == 'random':
        cmd = CSMITH + ' ' + swarm_testing + ' --record-file ' + hicond_statistic_file + ' --random-random > ' + src_file
    else:
        cmd = CSMITH + ' ' + swarm_testing + ' --record-file ' + hicond_statistic_file + ' --probability-configuration ' + config_file + ' > ' + src_file
    time_in = execmd_limit_time(cmd)
    if time_in:
        return cmd
    else:
        return False

def get_seed(src):
    return get_file_lines(src)[6][14:]
############################## optimization feature function ##############################
def get_mapping(opt_file, test_compiler):
    train_opt = get_file_lines(opt_file)

    test_opt = get_O3_opts(test_compiler)

    train_flag_2_index = {train_opt[i]: i for i in range(len(train_opt))}
    test_flag_2_index = {test_opt[i]: i for i in range(len(test_opt))}
    test_index_2_train_index = [i for i in range(len(test_opt))]
    for flag in test_flag_2_index:
        if flag not in train_flag_2_index:
            test_index_2_train_index[test_flag_2_index[flag]] = -1
        else:
            test_index_2_train_index[test_flag_2_index[flag]] = train_flag_2_index[flag]
    return test_index_2_train_index


def random_opts():
    # use mask to generate the optimization feature list
    mask_seed = np.array([random.randint(0, 1) for i in range(len(TEST_GCC_OPT))])
    feature_lists = mask[:]
    for i in range(len(feature_lists)):
        feature_lists[i] ^= mask_seed
    feature_lists = feature_lists.tolist()
    # construct optimization list using mask
    opt_lists = []
    for f_idx in range(len(feature_lists)):
        opt = []
        opt.extend(TEST_GCC_NEGATIVE_OPT)
        opt.extend([TEST_GCC_OPT[__] for __ in range(len(feature_lists[f_idx])) if feature_lists[f_idx][__] == 1])
        # deal with the conflict
        if '-funit-at-a-time' not in opt and '-ftoplevel-reorder' in opt:
            opt.remove('-ftoplevel-reorder')
            feature_lists[f_idx][TEST_GCC_OPT.index('-ftoplevel-reorder')] = 0
        opt_lists.append(' '.join(opt))
    return opt_lists, feature_lists


def mapping_opt_feature(opt_features, train_opts, map):
    tmp_feature_list = opt_features[:]
    feature_lists = []
    for i in range(len(tmp_feature_list)):
        maped_feature = [0.0 for j in range(len(train_opts))]
        for j in range(len(tmp_feature_list[i])):
            if map[j] != -1:
                maped_feature[map[j]] = tmp_feature_list[i][j]
        feature_lists.append(maped_feature[:])
    return feature_lists

def get_length(compiler_path):
    return len(get_O3_opts(compiler_path))
############################## program feature function ##############################
def extract_annotatioin_feature(src):
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
    content = get_file_content(src).split('\n')
    inx = content.index('/************************ statistics *************************')
    content = content[inx:]
    content = filter_list(content, 'XXX')
    content = [content[i].split(':') for i in range(len(content))]
    tmp_type = {}
    for t in types:
        tmp_type[t] = '0'
    for c in content:
        t = c[0]
        v = c[1][1:]
        tmp_type[t] = v
    return [tmp_type[key] for key in types]

def get_program_feature(feature_file, src):
    program_feature = get_file_lines(feature_file)[-1].split(',')
    reduce_list = [5, 10, 18, 19, 21, 22, 23, 25, 28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                   48, 71, 82, 92, 97, 105, 108, 109, 110]
    reduced_program_feature = []
    for c in range(len(program_feature)):
        if c not in reduce_list:
            reduced_program_feature.append(program_feature[c])
    annotation_feature = extract_annotatioin_feature(src)
    annotation_feature = np.array(annotation_feature, dtype=float)
    program_annotation_feature = np.hstack((reduced_program_feature, annotation_feature))
    program_annotation_feature = program_annotation_feature.reshape(1, -1)
    return program_annotation_feature.reshape(1, -1)
############################## ml feature function ##############################
def concat_opt_program_features(program_features, opt_features):
    ml_feature = [program_features + opt_features[i] for i in range(len(opt_features))]
    ml_feature = np.array(ml_feature)
    return ml_feature

def normalize(scaler, ml_features):
    tmp_features = ml_features[:]
    ml_features = scaler.transform(tmp_features)
    return ml_features.tolist()[0]
############################## model function ##############################
def get_normalizer(feature_file):
    scaler = MinMaxScaler()
    content = np.loadtxt(feature_file, delimiter=',')
    scaler.fit(content)
    return scaler

# 训练归一化模型
for csv_idx in range(len(FEATURE_FILES)):
    tmp_nor = get_normalizer(FEATURE_FILES[csv_idx])
    rm_file(NORMALIZERS[csv_idx])
    tar_f = open(NORMALIZERS[csv_idx], 'wb+')
    pickle.dump(tmp_nor, tar_f)
    assert os.path.exists(NORMALIZERS[csv_idx])

def get_model(model_file):
    xgb_clf_ = xgb.XGBClassifier()
    xgb_clf_.load_model(model_file)
    return xgb_clf_

def predict(xgb_clf, feature):
    x = np.array(feature)
    res = xgb_clf.predict_proba(x)
    return [[res[i][1], i] for i in range(len(res))]

def recommend_sequences(xgb_clf, ml_features, opt_sequences):
    predict_probs = predict(xgb_clf, ml_features)
    predict_probs = sorted(predict_probs, reverse=True)
    predict_probs = [predict_probs[i] for i in range(TOP_N)]
    opt_lists = []
    for i in range(TOP_N):
        if predict_probs[i][0] >= CUT_OOF_PROB:
            opt_lists.append(opt_sequences[predict_probs[i][1]])
    return opt_lists, predict_probs
############################## compiler test function(class) ##############################
class test_process:
    work_dir = ''
    opt = ''
    report_dir = ''
    program_gen_cmd = ''

    def __init__(self, work_dir, opt, report_dir, program_gen_cmd):
        self.report_dir = report_dir
        self.work_dir = work_dir
        self.opt = opt.strip()
        self.program_gen_cmd = program_gen_cmd
        if self.report_dir[-1] == '/':
            self.report_dir = self.report_dir[:-1]
        if self.work_dir[-1] == '/':
            self.work_dir = self.work_dir[:-1]

    def compile(self):
        src = self.work_dir + '/a.c'
        out = self.work_dir + '/a.o'
        error = self.work_dir + '/error.txt'
        rm_file(out)
        rm_file(error)
        cmd = TEST_GCC + ' -I ' + CSMITH_LIB + ' ' + self.opt + ' ' + src + ' -o ' + out + ' 2>' + error
        time_in = execmd_limit_time(cmd)
        if not time_in:
            return 'timeout'
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            return 'crash'
        return 'success'

    def execute(self, final_out):
        src = self.work_dir + '/a.o'
        error = self.work_dir + '/error.txt'
        out = final_out
        rm_file(out)
        rm_file(error)
        cmd = src + ' 1>' + out + ' 2>' + error
        time_in = execmd_limit_time(cmd)
        if not time_in:
            return 'timeout'
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            return 'crash'
        return 'success'

    def diff(self):
        error = self.work_dir + '/error.txt'
        rm_file(error)
        file1 = self.work_dir + '/' + TMP_RES
        file2 = self.work_dir + '/' + ORI_RES
        cmd = 'diff ' + file1 + ' ' + file2
        diff = execmd(cmd)
        if len(diff) != 0:
            put_file_content(error, diff)
            return 'miscompile_recorder'
        return 'success'

    def gcc_opt_test(self, final_out):
        res = self.compile()
        if res != 'success':
            return 'compile_' + res
        res = self.execute(final_out)
        if res != 'success':
            return 'exec_' + res
        if '-O0' != self.opt:
            res = self.diff()
            if res != 'success':
                return res
        return 'success'

    def do_test(self):
        # original
        if self.opt == '-O0':
            res = self.gcc_opt_test(self.work_dir + '/' + ORI_RES)
            return res
        # DOF or DOL
        res = self.gcc_opt_test(self.work_dir + '/' + TMP_RES)
        if res != 'success':
            prefix = ''
            if len(self.opt) == len('-Ox'):
                prefix = 'level_'
            else:
                prefix = 'flags_'
            suffix = ''
            if res == 'compile_crash':
                suffix = COMPILE_CRASH_REPORT
            if res == 'compile_timeout':
                suffix = COMPILE_TIMEOUT_REPORT
            if res == 'exec_crash':
                suffix = EXEC_CRASH_REPORT
            if res == 'exec_timeout':
                suffix = EXEC_TIMEOUT_REPORT
            if res == 'miscompile_recorder':
                suffix = MISCOMPILE
            report_file = self.report_dir + '/' + prefix + suffix + '.txt'
            msg = get_file_content(self.work_dir + '/error.txt')
            create_dir_if_not_exists(self.report_dir)
            os.system('cp ' + self.work_dir + '/a.c ' + self.report_dir)
            put_file_content(report_file, self.opt + '\n' + msg + '\n' + self.program_gen_cmd)
        return res
############################## mask function ##############################
def random_generate(size, length):
    return [[random.randint(0, 1) for j in range(length)] for i in range(size)]

def dist(dist_record):
    dist = np.sum(dist_record, 1)
    dist = np.min(dist)
    return dist

def evaluate(mask):
    val = 0
    for i in range(len(mask)):
        dist_record = mask[:]
        mask_i_np_arr = np.array(mask[i])
        del dist_record[i]
        dist_record = np.array(dist_record)
        for j in range(len(dist_record)):
            dist_record[j] ^= mask_i_np_arr
        dit = dist(dist_record)
        val = dit if dit > val else val
    return val

def get_mask(size, length):
    mask = random_generate(size, length)
    for i in range(len(mask)):
        dist_record = mask[:]
        del dist_record[i]
        dist_record = np.array(dist_record)
        mask_i_np_arr = np.array(mask[i])
        for j in range(len(dist_record)):
            dist_record[j] ^= mask_i_np_arr
        for j in range(len(mask[i])):
            d1 = dist(dist_record)
            mask[i][j] ^= 1
            dist_record[:, j] ^= 1
            d2 = dist(dist_record)
            if d2 < d1:
                mask[i][j] ^= 1
                dist_record[:, j] ^= 1
            print('i=' + str(i) + ', d1=' + str(d1) + ', d2=' + str(d2))
    print('evaluate=' + str(evaluate(mask)))
    return mask

def read_mask():
    mask = get_file_lines(GCC_MASK_FILE)
    mask = [[int(mf) for mf in m.split(',')] for m in mask]
    return np.array(mask)

############################## multi-core function ##############################
def test_gcc_dev(process_num):
    log('[main] process-' + str(process_num) + ' start')
    # prepare
    # mask = read_mask()
    mappings = [get_mapping(opt_file, TEST_GCC) for opt_file in TRAIN_GCC_OPT_FILES]
    train_opts = [get_file_lines(opt_file) for opt_file in TRAIN_GCC_OPT_FILES]
    # normalizers = [get_normalizer(file) for file in FEATURE_FILES]
    # models = [get_model(model_file) for model_file in MODELS]
    work_dir = WORK_DIR_PREFIX + '/test' + str(process_num)
    hicond_statistic_file = WORK_DIR_PREFIX + '/hicond-record/hicond_feature-' + str(process_num) + '.txt'
    create_dir_if_not_exists(WORK_DIR_PREFIX + '/hicond-record')
    src = work_dir + '/a.c'
    total_program_cnt = 0
    discard_program_cnt = 0
    test_program_cnt = 0
    while True:
        log('***************************************************************** [next program] *****************************************************************')
        # environment
        rm_file(work_dir)
        rm_file(hicond_statistic_file)
        create_dir_if_not_exists(work_dir)
        # generate program
        conf_num = random.randint(1, 10 + DEFAULT_CNT + RANDOM_CNT)
        conf_file = get_conf_file(conf_num)
        program_gen_cmd = generate_src(conf_file, src, hicond_statistic_file)
        if not program_gen_cmd:
            log('[main] generate program failed')
            continue
        find_dir = WORK_DIR_PREFIX + '/' + get_seed(src) + '-' + get_conf_type(conf_num)
        if os.path.exists(find_dir):
            log('[main] program has been tested')
            continue

        # generate optimization setting and features
        [opt_sequences, opt_features] = random_opts()
        mapped_opt_features = [mapping_opt_feature(opt_features, train_opts[i], mappings[i]) for i in range(len(mappings))]
        program_features = get_program_feature(hicond_statistic_file, src)

        # do normalization and prediction in another file, because importing xgboost and sklearn will break multiprocessing
        # outputs from this file:
        #   1. program feature file
        #   2. optimization feature file
        #   3. optimization sequences file
        program_features_f = WORK_DIR_PREFIX + '/tmp/program_feature-' + str(os.getpid()) + '.txt'
        rm_file(program_features_f)
        program_features = program_features.tolist()[0]
        put_file_content(program_features_f, ','.join([str(_) for _ in program_features]))

        opt_feature_files = []
        for _ in range(len(mapped_opt_features)):
            opt_feature_f = WORK_DIR_PREFIX + '/tmp/opt_feature-' + str(os.getpid()) + '-' + str(_) + '.txt'
            rm_file(opt_feature_f)
            out_feature = '\n'.join([','.join([str(___) for ___ in __]) for __ in mapped_opt_features[_]])
            put_file_content(opt_feature_f, out_feature)
            opt_feature_files.append(opt_feature_f)

        opt_sequence_f = WORK_DIR_PREFIX + '/tmp/opt_sequence-' + str(os.getpid()) + '.txt'
        rm_file(opt_sequence_f)
        put_file_content(opt_sequence_f, '\n'.join(opt_sequences))

        # inputs for normalization and prediction:
        #   program feature file
        #   opt sequence file
        #   number of opt feature file
        #   opt feature files;
        #   models files;
        #   normalizer files;
        #   number of recommend optimization sequences for each model
        #   probability threshold
        #   result file
        argv = [program_features_f, opt_sequence_f, str(len(opt_feature_files))]
        argv.extend(opt_feature_files)
        argv.extend(MODELS)
        argv.extend(NORMALIZERS)  # TODO:
        argv.append(str(TOP_N))
        argv.append(str(CUT_OOF_PROB))
        argv.append(WORK_DIR_PREFIX + '/tmp/pre-result-' + str(os.getpid()) + '.txt')
        argv = ' '.join(argv)

        cmd = 'python predict-candidate.py ' + argv
        os.system(cmd)

        # This segment of code is important, I do not want to write it again...
        # program_features = [normalize(n, program_features) for n in normalizers]
        # print(program_features)
        # log('[main] program_feature=' + str(program_features))
        # ml_features = [concat_opt_program_features(program_features[inx], mapped_opt_features[inx]) for inx in range(len(mapped_opt_features))]

        # whether we should discard this program
        result_f = WORK_DIR_PREFIX + '/tmp/pre-result-' + str(os.getpid()) + '.txt'
        result = get_file_lines(result_f)
        result = [_.split(',') for _ in result]
        test_opts = [_[1] for _ in result]
        test_probs = [_[0] for _ in result]

        # get the recommend 
        log('[main] recommend probs: ' + str(test_probs))
        assert total_program_cnt == test_program_cnt + discard_program_cnt
        log('[main] [total_prog, test_prog, disc_prog]: ' + str([total_program_cnt, test_program_cnt, discard_program_cnt]))
        total_program_cnt += 1
        if len(test_opts) == 0:
            log('[main] program has been discarded')
            discard_program_cnt += 1
            continue
        test_program_cnt += 1

        # original test
        ori_test = test_process(work_dir, '-O0', find_dir, program_gen_cmd)
        res = ori_test.do_test()
        log('[main] ori test end, res=' + res)
        if res != 'success':
            log('[main] ori test failed, next program')
            continue
        # DOF test
        find_bug = False
        for opt in test_opts:
            opt = random_level() + ' ' + opt + ' -fno-strict-aliasing -fwrapv -fno-aggressive-loop-optimizations'
            dof_test = test_process(work_dir, opt, find_dir, program_gen_cmd)
            res = dof_test.do_test()
            if res != 'success':
                find_bug = True
                break
        log('[main] dof test end, res=' + res)
        if not find_bug:
            log('[main] dof test failed, next program')
            continue
        # DOL test
        for level in ['-O1 -fno-strict-aliasing -fwrapv -fno-aggressive-loop-optimizations',
                      '-O2 -fno-strict-aliasing -fwrapv -fno-aggressive-loop-optimizations',
                      '-O3 -fno-strict-aliasing -fwrapv -fno-aggressive-loop-optimizations',
                      '-Os -fno-strict-aliasing -fwrapv -fno-aggressive-loop-optimizations']:
            dol_test = test_process(work_dir, level, find_dir, program_gen_cmd)
            res = dol_test.do_test()
            if res != 'success':
                break
        log('[main] dol test end, res=' + res)


if __name__ == '__main__':
    # remove mask file support
    if not os.path.exists(GCC_MASK_FILE):
        print('Please run mask.py before running this script, or check the variable GCC_MASK_FILE to ensure the correctness of mask file\'s path!')
        exit(1)
        # mask_gcc = get_mask(OPTIMIZATION_NUM, get_length(TEST_GCC))
        # mask_gcc = [[str(mf) for mf in m] for m in mask_gcc]
        # rm_file(GCC_MASK_FILE)
        # put_file_content(GCC_MASK_FILE, '\n'.join([','.join(m) for m in mask_gcc]))
    else:
        mask = read_mask()
        if len(mask[0]) != len(TEST_GCC_OPT) or len(mask) != OPTIMIZATION_NUM:
            print('Mask pattern error! Please check the dimension of the mask!')
            exit(1)
            # mask_gcc = get_mask(OPTIMIZATION_NUM, get_length(TEST_GCC))
            # mask_gcc = [[str(mf) for mf in m] for m in mask_gcc]
            # rm_file(GCC_MASK_FILE)
            # put_file_content(GCC_MASK_FILE, '\n'.join([','.join(m) for m in mask_gcc]))
    create_dir_if_not_exists(WORK_DIR_PREFIX + '/tmp')
    create_dir_if_not_exists(LOG_DIR)
    for process_num in range(CORE_NUM):
        p = Process(target=test_gcc_dev, args=(process_num,), daemon=True)
        p.start()
    while True:
        time.sleep(30*60)
