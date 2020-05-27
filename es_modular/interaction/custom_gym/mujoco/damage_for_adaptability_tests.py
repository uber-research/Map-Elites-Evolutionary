# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import fileinput
import numpy as np
from shutil import copyfile

"""
functions to update the xml file during the run to implement the damages
"""


def modify_ant_xml(folder, modif, param={}):
    copyfile(folder + 'ant.xml', folder + 'ant_modified.xml')
    for i, line in enumerate(fileinput.input(folder + 'ant_modified.xml', inplace=1)):
        if modif == 'joint':
            # restrict the control range of the joints described by the joints parameter from -1,1 to -0.01,0.01
            lines = np.array(param) + 71
            if i in lines:
                sys.stdout.write(line.replace('ctrlrange="-1.0 1.0"', 'ctrlrange="-0.01 0.01"'))
            else:
                sys.stdout.write(line)
        else:
            raise NotImplementedError
