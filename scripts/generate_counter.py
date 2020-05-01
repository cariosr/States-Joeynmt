#!/usr/bin/env python

import argparse
import re
from os import remove, path

def create_counter(file_path):
    dqn_flag = False
    with open(file_path, 'r') as old_file:
            for line in old_file:
                #print(line)
                if re.search("dqn:", line):
                    dqn_flag = True
                
                if dqn_flag:
                    if re.search("test_range", line):
                        #print("test_range_if")
                        m = re.match("(\s*test_range:\s)(\[)([0-9]*(\.)?[0-9]*.*)(\])", line)
                        m_3 = re.match("(\s*test_range:\s)(\[)([0-9]*(\.)?[0-9]*)", line)
                    elif re.search("test_variable", line):
                        #print("test_variable_elif")
                        m_2 = re.match("(\s*test\_variable:\s)(\')(.*)(\')", line)

    file_name = file_path + ".sh"
    #print("file_name: {}".format(file_name))
    range_list = m[3].split()
    #print("range_list: {}".format(range_list))
    cfg_name = "log_"+m_2[3]+re.sub("\.","", m_3[3])
    #print("cfg_name: {}".format(cfg_name))

    if len(range_list) == 0:
        #print("Removing bash config file to trigger stop in bash script.")
        if path.exists(file_name):
            remove(file_name)
    else:
        #print("Adapting bash config file to produce correct output log.")
        with open(file_name, 'w') as counter_file:
            counter_file.write("cfg_name={}".format(cfg_name))
        if path.exists(file_name):
            print(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get config file for testing.')
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    create_counter(args.file_path)
