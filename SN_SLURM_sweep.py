#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:09:08 2020

@author: sr05

=========================================================
Submit sbatch jobs for SN Connectivity
analysis
=========================================================
"""

import subprocess
from os import path as op

from importlib import reload

# import study parameters
import sn_config as C
import numpy as np
reload(C)

print(__doc__)

# # wrapper to run python script via qsub. Python3
fname_wrap = op.join('/', 'home','sr05', 'my_semnet','Python2SLURM.sh')

# indices of subjects to process
# sbj_id=np.arange(0,len(C.subjects))
# sbj_id = np.array([0,1])
Cond= ['fruit','odour','milk','LD']

# job_list = [
#     # Connectivity :  Coherence
#       {'N':   'Connectivity',                  # job name
#       'Py':  'SN_functional_connectivity_bands_runs_BL',  # Python script
#       'ss':  sbj_id,                    # subject indices
#       'mem': '24G',                   # memory for qsub process
#       'dep': ''}]

# job_list = [
#     # Connectivity :  Coherence
#       {'N':   'transformation',                  # job name
#       'Py':  'test_transformation',  # Python script
#       'ss':  sbj_id,                    # subject indices
#       'mem': '24G',                   # memory for qsub process
#       'dep': ''}]


job_list = [
    # Connectivity :  Coherence
      {'N':   'transformation_50',                  # job name
      'Py':  'test_transformation',  # Python script
      'ss':  Cond,                    # subject indices
      'mem': '24G',                   # memory for qsub process
      'dep': ''}]

# directory where python scripts are
dir_py = op.join('/', 'home', 'sr05','my_semnet')

# directory for qsub output
dir_sbatch = op.join('/', 'home', 'sr05','my_semnet','sbatch_out')

# keep track of qsub Job IDs
Job_IDs = {}

for job in job_list:

    for s in job['ss']:
        # Ss = str(C.subjects[s][1:-8])  # turn into string for filenames etc.
        Ss = str(s)  # turn into string for filenames etc.

        # print(Ss)

        N = Ss + job['N']  # add number to front
        Py = op.join(dir_py, job['Py'])
        Cf = ''  # config file not necessary for FPVS
        mem = job['mem']

        # files for qsub output
        file_out = op.join(dir_sbatch,
                           job['N'] + '_' + '%s.out' % str(Ss))
        file_err = op.join(dir_sbatch,
                           job['N'] + '_' + '%s.err' % str(Ss))



        # sbatch command string to be executed
        sbatch_cmd = 'sbatch \
                        -o %s \
                        -e %s \
                        --export=pycmd="%s.py %s",subj_idx=%s, \
                        --mem=%s -t 1-00:00:00  -J %s  %s' \
                        % (file_out, file_err, Py, Cf, Ss, mem,
                            N, fname_wrap)

        # format string for display
        print_str = sbatch_cmd.replace(' ' * 25, '  ')
        print('\n%s\n' % print_str)

        # execute qsub command
        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, shell=True)

        # get linux output
        (out, err) = proc.communicate()

        # keep track of Job IDs from sbatch, for dependencies
        Job_IDs[N, Ss] = str(int(out.split()[-1]))
