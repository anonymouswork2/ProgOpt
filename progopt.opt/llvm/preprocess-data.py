# 本文件对应于“预处理数据”部分
import os
import random
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler


target_dir = 'collect-data/LLVM-X.X'    # 统计该文件夹中的信息
target_dir = target_dir if target_dir.endswith('/') else target_dir + os.sep
buggy_normal = 2    # 失败测试:成功测试 = 1:2
validate_train = 3    # 验证:训练 = 1:2
min_validate_buggy = 200    # 验证集最小个数
# 意味着，至少需要：
# min_validate_buggy（验证集） + validate_train * min_validate_buggy （训练集）失败测试用例
# min_validate_buggy * buggy_normal（验证集） + min_validate_buggy * buggy_normal * validate_train （训练集） 成功测试用例
# 如果运行到某个节点运行不下去，结束分配，报错
required_validate_buggy = min_validate_buggy
required_validate_normal = int(buggy_normal * required_validate_buggy)
required_train_buggy = int(required_validate_buggy * validate_train)
required_train_normal = int(required_validate_normal * validate_train)
timeout_as_bug = False  # 是否将超时作为bug数据？

output_dir = 'model-train/LLVM-4.0'    # 输出路径
output_dir = output_dir if output_dir.endswith('/') else output_dir + os.sep


def execmd(cmd):
    import os
    pipe = os.popen(cmd)
    reval = pipe.read()
    pipe.close()
    return reval


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
    content = [_ for _ in content if 'XXX' in _]
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


def create_dir_if_not_exists(dir_name):
    if '/' == dir_name or '/*' == dir_name or '*' in dir_name:
        return
    if not os.path.exists(dir_name):
        execmd('mkdir -p ' + dir_name)


def save_normalizer(feature_file, scaler_file):
    scaler = MinMaxScaler()
    content = np.loadtxt(feature_file, delimiter=',')
    scaler.fit(content)
    pickle.dump(scaler, open(scaler_file, 'wb'))


def load_normalizer(nor_f):
    scaler = pickle.load(open(nor_f, 'rb'))
    return scaler


def normalize(scaler, ml_features):
    tmp_features = ml_features[:]
    ml_features = scaler.transform(tmp_features)
    return ml_features.tolist()


def main():

    # all programs
    label_cmd = ' '.join(['find', target_dir, '-name', "'*label*'"])
    all_cases = execmd(label_cmd).split('\n')
    all_cases = [_ for _ in all_cases if len(_) > 1]
    all_program_nums = set([_.split('/')[-2] for _ in all_cases])

    # output all labels
    assert not os.path.exists(output_dir), 'output folder has already existed, handle it manually as your own risk!'
    create_dir_if_not_exists(output_dir)
    all_program_features = [list(get_program_feature(target_dir+_+os.sep+'a.csv', target_dir+_+os.sep+'a.c')[0]) for _ in all_program_nums]
    all_program_feature_file = output_dir + 'all_program_feature.csv'
    put_file_content(all_program_feature_file, '\n'.join([','.join([str(__) for __ in _]) for _ in all_program_features]))

    # train and save normalizer
    nor_file = output_dir + 'nor'
    save_normalizer(all_program_feature_file, nor_file)
    normalizer = load_normalizer(nor_file)

    # buggy programs
    cra_cmd = ' '.join(['find', target_dir, '-name', "'*cra*'"])
    mis_cmd = ' '.join(['find', target_dir, '-name', "'*mis*'"])
    cra_cases = execmd(cra_cmd).split('\n')
    cra_cases = [_ for _ in cra_cases if len(_) > 1]
    mis_cases = execmd(mis_cmd).split('\n')
    mis_cases = [_ for _ in mis_cases if len(_) > 1]
    buggy_program_nums = cra_cases + mis_cases
    if timeout_as_bug:  # deal with timeout case
        timeout_cmd = ' '.join(['find', target_dir, '-name', "'*time*'"])
        timeout_cases = execmd(timeout_cmd).split('\n')
        timeout_cases = [_ for _ in timeout_cases if len(_) > 1]
        buggy_program_nums = buggy_program_nums + timeout_cases
    buggy_program_nums = set([_.split('/')[-2] for _ in buggy_program_nums])

    # normal programs
    normal_program_nums = all_program_nums - buggy_program_nums

    # transform two main class into list
    buggy_program_nums = list(buggy_program_nums)
    random.shuffle(buggy_program_nums)
    normal_program_nums = list(normal_program_nums)
    random.shuffle(normal_program_nums)
    print(len(buggy_program_nums))
    print(len(normal_program_nums))
    print(len(all_program_nums))

    # split cases(init)
    validate_buggy_cases = []
    validate_normal_cases = []
    train_buggy_cases = []
    train_normal_cases = []

    # split cases(for train buggy cases)
    while len(train_buggy_cases) < required_train_buggy:
        buggy_num = buggy_program_nums[0]    # this statement will crash when there is no such program to use
        del buggy_program_nums[0]
        buggy_case = target_dir + buggy_num + os.sep
        # get all label and add buggy or normal ones
        label_file = buggy_case + 'label.csv'
        label = get_file_lines(label_file)[0].split(',')
        for _ in range(len(label)):
            if label[_] == '0':
                train_normal_cases.append(buggy_num + '-' + str(_))
            else:
                train_buggy_cases.append(buggy_num + '-' + str(_))
    train_buggy_cases = train_buggy_cases[:required_train_buggy]

    # split cases(for validate buggy cases)
    while len(validate_buggy_cases) < required_validate_buggy:
        buggy_num = buggy_program_nums[0]  # this statement will crash when there is no such program to use
        del buggy_program_nums[0]
        buggy_case = target_dir + buggy_num + os.sep
        # get all label and add buggy or normal ones
        label_file = buggy_case + 'label.csv'
        label = get_file_lines(label_file)[0].split(',')
        for _ in range(len(label)):
            if label[_] == '0':
                validate_normal_cases.append(buggy_num + '-' + str(_))
            else:
                validate_buggy_cases.append(buggy_num + '-' + str(_))
    validate_buggy_cases = validate_buggy_cases[:required_validate_buggy]

    # split cases(for train normal cases)
    while len(train_normal_cases) < required_train_normal:
        buggy_num = normal_program_nums[0]  # this statement will crash when there is no such program to use
        del normal_program_nums[0]
        buggy_case = target_dir + buggy_num + os.sep
        # get all label and add buggy or normal ones
        label_file = buggy_case + 'label.csv'
        label = get_file_lines(label_file)[0].split(',')
        for _ in range(len(label)):
            train_normal_cases.append(buggy_num + '-' + str(_))
    train_normal_cases = train_normal_cases[:required_train_normal]

    # split cases(for validate normal cases)
    while len(validate_normal_cases) < required_validate_normal:
        buggy_num = normal_program_nums[0]  # this statement will crash when there is no such program to use
        del normal_program_nums[0]
        buggy_case = target_dir + buggy_num + os.sep
        # get all label and add buggy or normal ones
        label_file = buggy_case + 'label.csv'
        label = get_file_lines(label_file)[0].split(',')
        for _ in range(len(label)):
            validate_normal_cases.append(buggy_num + '-' + str(_))
    validate_normal_cases = validate_normal_cases[:required_validate_normal]

    # aggregate the result(train buggy data)
    # [[program, optimization, label], ...]
    train_buggy_data = []
    for _ in train_buggy_cases:
        case_info = _.split('-')
        case_program_num = case_info[0]
        case_test_num = case_info[1]

        pf_file = target_dir + case_program_num + os.sep + 'a.csv'
        src = target_dir + case_program_num + os.sep + 'a.c'
        of_file = target_dir + case_program_num + os.sep + 'feature.csv'
        l_file = target_dir + case_program_num + os.sep + 'label.csv'

        pf = normalizer.transform(get_program_feature(pf_file, src)).tolist()[0]
        pf = [str(__) for __ in pf]
        of = get_file_lines(of_file)[int(case_test_num)].split(',')
        l = get_file_lines(l_file)[0].split(',')[int(case_test_num)]
        assert l == '1'
        train_buggy_data.append([pf + of, l])

    # aggregate the result(train normal data)
    # [[program, optimization, label], ...]
    train_normal_data = []
    for _ in train_normal_cases:
        case_info = _.split('-')
        case_program_num = case_info[0]
        case_test_num = case_info[1]

        pf_file = target_dir + case_program_num + os.sep + 'a.csv'
        src = target_dir + case_program_num + os.sep + 'a.c'
        of_file = target_dir + case_program_num + os.sep + 'feature.csv'
        l_file = target_dir + case_program_num + os.sep + 'label.csv'

        pf = normalizer.transform(get_program_feature(pf_file, src)).tolist()[0]
        pf = [str(__) for __ in pf]
        of = get_file_lines(of_file)[int(case_test_num)].split(',')
        l = get_file_lines(l_file)[0].split(',')[int(case_test_num)]
        assert l == '0'
        train_normal_data.append([pf + of, l])

    # aggregate the result(validate buggy data)
    # [[program, optimization, label], ...]
    validate_buggy_data = []
    for _ in validate_buggy_cases:
        case_info = _.split('-')
        case_program_num = case_info[0]
        case_test_num = case_info[1]

        pf_file = target_dir + case_program_num + os.sep + 'a.csv'
        src = target_dir + case_program_num + os.sep + 'a.c'
        of_file = target_dir + case_program_num + os.sep + 'feature.csv'
        l_file = target_dir + case_program_num + os.sep + 'label.csv'

        pf = normalizer.transform(get_program_feature(pf_file, src)).tolist()[0]
        pf = [str(__) for __ in pf]
        of = get_file_lines(of_file)[int(case_test_num)].split(',')
        l = get_file_lines(l_file)[0].split(',')[int(case_test_num)]
        assert l == '1'
        validate_buggy_data.append([pf + of, l])

    # aggregate the result(validate buggy data)
    # [[program, optimization, label], ...]
    validate_normal_data = []
    for _ in validate_normal_cases:
        case_info = _.split('-')
        case_program_num = case_info[0]
        case_test_num = case_info[1]

        pf_file = target_dir + case_program_num + os.sep + 'a.csv'
        src = target_dir + case_program_num + os.sep + 'a.c'
        of_file = target_dir + case_program_num + os.sep + 'feature.csv'
        l_file = target_dir + case_program_num + os.sep + 'label.csv'

        pf = normalizer.transform(get_program_feature(pf_file, src)).tolist()[0]
        pf = [str(__) for __ in pf]
        of = get_file_lines(of_file)[int(case_test_num)].split(',')
        l = get_file_lines(l_file)[0].split(',')[int(case_test_num)]
        assert l == '0'
        validate_normal_data.append([pf + of, l])

    # output all results
    validate_all = validate_normal_data + validate_buggy_data
    random.shuffle(validate_all)
    validate_f = [_[0] for _ in validate_all]
    validate_l = [_[1] for _ in validate_all]
    validate_f_f = output_dir + 'validate_f.csv'
    put_file_content(validate_f_f, '\n'.join([','.join(_) for _ in validate_f]))
    validate_l_f = output_dir + 'validate_l.csv'
    put_file_content(validate_l_f, ','.join(validate_l))

    train_all = train_normal_data + train_buggy_data
    random.shuffle(train_all)
    train_f = [_[0] for _ in train_all]
    train_l = [_[1] for _ in train_all]
    train_f_f = output_dir + 'train_f.csv'
    put_file_content(train_f_f, '\n'.join([','.join(_) for _ in train_f]))
    train_l_f = output_dir + 'train_l.csv'
    put_file_content(train_l_f, ','.join(train_l))

    pass


if __name__ == '__main__':
    main()
    pass