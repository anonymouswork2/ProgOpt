# 本文件对应于 “收集历史数据” 的部分
import os
import time
import random
from multiprocessing import Process, Queue


target_dir = 'collect-data/gcc-4.4.0/'    # 工作文件夹，同时也是历史数据存储文件夹，测试信息将会保留供后续处理
gcc = '/data/bin/gcc-4.4.0/bin/gcc'    # 历史版本编译器路径
csmith = '/data/bin/csmith_record/bin/csmith'    # Csmith路径
csmith_lib = '/data/bin/csmith_record/include/csmith-2.3.0'    # Csmith头文件路径
conf_home = ''    # useless
opt_file = 'gcc-440-opt.txt'    # 模型训练考虑的优化文件的路径，模型使用时需要复制到model-related文件夹下并在run.py中指定

opt_cnt_per_program = 10    # 每个程序生成优化选项序列的数量
max_program_cnt = 50000    # 终止条件：完成xxx数量的测试程序后停止程序
core_num = 20    # 并发数
timeout_as_bug = False  # 是否将超时作为bug

timeout_each_operation = 60    # 每个操作的超时时间

if not target_dir.endswith('/'):
    target_dir += os.sep
if not conf_home.endswith('/'):
    conf_home += os.sep

assert not os.path.exists(target_dir), target_dir + ' exists! Please confirm and delete it manually!'

# read all available optimizations
assert os.path.exists(opt_file), 'Available optimization file should exist!'
opt_file = open(opt_file)
available_opts = opt_file.readlines()
opt_file.close()
available_opts = [_ if not _.endswith('\n') else _[:-1] for _ in available_opts if len(_) > 0]


def execmd(cmd):
    import os
    pipe = os.popen(cmd)
    reval = pipe.read()
    pipe.close()
    return reval


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


def execmd_limit_time(cmd):
    import time
    start = time.time()
    execmd("timeout " + str(timeout_each_operation) + " " + cmd)
    end = time.time()
    return (end - start) < timeout_each_operation


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


def get_neg(o):
    if '-fno-' not in o:
        return o[:2] + 'no-' + o[2:]    # example -fweb => -fno-web
    else:
        return o[:2] + o[len('-fno-'):]    # example -fno-web => -fweb


available_neg_opts = [get_neg(_) for _ in available_opts]


def generate_src(src_file, hicond_statistic_file, config_file='default'):
    if config_file == 'default':
        cmd = csmith + ' --record-file ' + hicond_statistic_file + ' > ' + src_file
    elif config_file == 'random':
        cmd = csmith + ' --record-file ' + hicond_statistic_file + ' --random-random > ' + src_file
    else:
        cmd = csmith + ' --record-file ' + hicond_statistic_file + ' --probability-configuration ' + config_file + ' > ' + src_file
    rm_file(hicond_statistic_file)
    time_in = execmd_limit_time(cmd)
    return time_in and os.path.exists(hicond_statistic_file)


def compile_exe(work_dir, opt_seq, res):
    src = work_dir + 'a.c'
    out = work_dir + 'a.out'
    err = work_dir + 'err'
    cmd = ' '.join([gcc, opt_seq, '-I', csmith_lib, src, '-o', out, '2>'+err])
    log(cmd)
    rm_file(out)
    rm_file(err)
    time_in = execmd_limit_time(cmd)
    if not time_in:
        return 'compile_timeout'
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        return 'compile_crash'

    cmd = ' '.join([out, '>'+res, '2>'+err])
    log(cmd)
    rm_file(res)
    rm_file(err)
    time_in = execmd_limit_time(cmd)
    if not time_in:
        return 'exe_timeout'
    if not os.path.exists(res) or os.path.getsize(res) == 0:
        return 'exe_crash'
    return 'success'


create_dir_if_not_exists(target_dir + 'log')


def log(msg):
    file = target_dir + 'log/' + 'log-' + str(os.getpid()) + '.txt'
    put_file_content(file, '[' + str(os.getpid()) + ']' + msg + '\n')


def diff(res1, res2):
    d = execmd(' '.join(['diff', res1, res2]))
    return 'success' if len(d) == 0 else 'miscompile'


def main_each_process(todo_queue, done_queue):
    while not todo_queue.empty():

        program_cnt = todo_queue.get()
        work_dir = target_dir + str(program_cnt) + '/'
        src = work_dir + 'a.c'
        program_feature_file = work_dir + 'a.csv'

        create_dir_if_not_exists(work_dir)

        # generate program
        if not generate_src(src_file=src, hicond_statistic_file=program_feature_file):
            log('generate program failed')
            done_queue.put(program_cnt)
            continue

        # original execute
        r = compile_exe(work_dir, '-O0', work_dir + 'ori_res')
        if r != 'success':    # less interest for programs crash at -O0
            log('less interest for programs crash at -O0')
            done_queue.put(program_cnt)
            continue

        # generate random optimization sequences
        opt_features = [[str(random.randint(0, 1)) for _ in range(len(available_opts))] for __ in range(opt_cnt_per_program)]
        opt_sequences = [[available_opts[_] for _ in range(len(opt_features[__])) if opt_features[__][_] == '1']
                         for __ in range(len(opt_features))]
        opt_sequences = [['-O3'] + available_neg_opts + _ for _ in opt_sequences]
        for _ in range(len(opt_sequences)):
            if '-funit-at-a-time' not in opt_sequences[_] and '-ftoplevel-reorder' in opt_sequences[_]:
                opt_sequences[_].remove('-ftoplevel-reorder')

        opt_sequences = [' '.join(_) for _ in opt_sequences]

        # test these optimization sequences, record fault message and record label ...
        label = []
        for _ in range(len(opt_features)):
            # compile and execute
            r = compile_exe(work_dir, opt_sequences[_], work_dir + 'tmp_res')
            if r != 'success':
                if not timeout_as_bug and 'time' in r:
                    label.append('0')
                else:
                    label.append('1')
                record_file = work_dir + r + '-' + str(_) + '.txt'
                err_file = work_dir + 'err'
                record = opt_sequences[_] + '\n'
                err = get_file_content(err_file)
                put_file_content(record_file, record + err)
                continue
            # diff
            r = diff(work_dir + 'ori_res', work_dir + 'tmp_res')
            if r != 'success':
                if not timeout_as_bug and 'time' in r:
                    label.append('0')
                else:
                    label.append('1')
                record_file = work_dir + r + '-' + str(_) + '.txt'
                err_file = work_dir + 'err'
                record = opt_sequences[_] + '\n'
                err = get_file_content(err_file)
                put_file_content(record_file, record + err)
                continue
            # passing test program
            label.append('0')

        # output the result
        put_file_content(work_dir + 'label.csv', ','.join(label))    # label
        put_file_content(work_dir + 'feature.csv', '\n'.join([','.join(_) for _ in opt_features]))    # feature
        done_queue.put(program_cnt)


def main():
    todo_queue = Queue()
    for _ in range(max_program_cnt):
        todo_queue.put(_)
        pass
    done_queue = Queue()
    for _ in range(core_num):
        p = Process(target=main_each_process, args=(todo_queue, done_queue, ))
        p.daemon = True
        p.start()

    while True:
        time.sleep(60 * 5)
        if done_queue.qsize() >= max_program_cnt:
            break


if __name__ == '__main__':
    main()