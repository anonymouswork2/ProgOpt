This is the implementation of *progopt_prog*.

It contains several folders and files:

+ ```test_base_funs.py```: The implementation of compiler test process of GCC and LLVM.

+ ```hicond-conf```: Folder that contains hicond's configuration, 
  *progopt* integrated hicond's configuration into it's configuration reset step.
  When there are many timeout test programs in an iteration, 
  *progopt* will directly run to one of a hicond configuration, 
  or a default configuration.
  To accelerate to keep away from timeout test configuration region. 

+ ```opt-related```: Folder that contains optimization recommend component related data and models, 
  including optimization list that be considered into training step, 
  data that used to normalization, the recommend model of each subject compiler version.
  In all, this folder gives *progopt* all the data/model support about optimization recommend component.

+ ```a2c.py```: File that implements a2c algorithm. 

+ ```common_base_funs.py```: File that implements common basic operations, 
  such as read/write a file, 
  create a folder, 
  or execute command line.

+ ```csmith_configure.py```: File that implements write and read csmith configuration, 
  and convert csmith configuration file into environment.

+ ```environment.py```: File that implements environment modification and action evaluation.

+ ```limit-memory-run.sh```: File that restrict memory consume.

+ ```main_configure_approach.py```: File that contains all of the configurations of *progopt*.

+ ```main_configure_baseline.py```: File that contains all of the configurations of *progopt.prog*'s baselines

+ ```opt_support.py```: File that implements optimization component of *progopt*

+ ```program_feature.py```: File that implements diversity calculation of program center vectors.

+ ```run-approach.py```: Driver script that run *progopt*. 

+ ```run-baselines.py```: Driver script that run *progopt.prog* baselines.

+ ```seed.txt```: Seed file for reproduce the experiment result.

+ ```seed-gen.py```: Script that generates all of the seed file.

+ ```standardization.csv```: File that gives the data to the standardization of program feature. 

How to run *progopt*:

1. Configure ```main_configure_approach.py```.

2. Directly run ```python run-approach.py```.
