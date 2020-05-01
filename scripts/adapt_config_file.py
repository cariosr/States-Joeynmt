#!/usr/bin/env python

import argparse
import re
from os import fdopen, remove
from tempfile import mkstemp
from shutil import move, copymode

def adapt_config(file_path):
    #Create temp file
    fh, abs_path = mkstemp()

    dqn_flag = False
    with open(file_path, 'r') as old_file:
            for line in old_file:
                #print(line)
                if re.search("dqn:", line):
                    dqn_flag = True
                
                if dqn_flag:
                    if re.search("test_range", line):
                        #print("test_range_if")
                        m = re.match("(\s*test_range:\s)(\[)([0-9]*(\.)?[0-9]*)", line)
                    elif re.search("test_variable", line):
                        #print("test_variable_elif")
                        m_2 = re.match("(\s*test\_variable:\s)(\')(.*)(\')", line)

    subst = m[3]
    test_variable = m_2[3]
    test_pattern = test_variable + ": " + ".*"
    test_subst = test_variable + ": "+ subst
    
    config_pattern = "test_range: " + "\[([0-9]*(\.)?[0-9]*)(,\s?)?"
    config_subst = "test_range: " + "["

    dqn_flag = False
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if "dqn:" in line:
                    dqn_flag = True
                
                if dqn_flag:
                    if test_variable in line:
                        new_file.write(re.sub(test_pattern, test_subst, line))
                    elif "test_range" in line:
                        new_file.write(re.sub(config_pattern, config_subst, line))
                    else:
                        new_file.write(line)
                else:
                    new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get config file for testing.')
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    adapt_config(args.file_path)
