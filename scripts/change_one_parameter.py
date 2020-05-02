#Takes 5 arguments: file_path, section, parameter to change, value and string_or_not
#IMPORTANT: if the value is a list, write it without spaces so the command line parse works ([5,3.7,3], NOT [5, 3.7, 3]))

#!/usr/bin/env python

import argparse
import re
from os import fdopen, remove
from tempfile import mkstemp
from shutil import move, copymode

def adapt_config(file_path, section, parameter, value, string_or_not):
    #Create temp file
    fh, abs_path = mkstemp()

    #section_flag = False
    section = section+":"
    
    value=re.sub(",", ", ", value)

    if string_or_not:
        value = "'"+value+"'"

    section_flag = False
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if section in line:
                    section_flag = True
                
                if section_flag:
                    if re.search(parameter, line):
                        subst_str = ": "+value
                        new_file.write(re.sub(":\s.*", subst_str, line))
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
    parser = argparse.ArgumentParser(description='Get file_path, section, parameter, value and string_or_not (0 or 1)')
    for argument in ["file_path", "section", "parameter", "value"]:
        parser.add_argument(argument, type=str)
    parser.add_argument("string_or_not", type=int, default=int(1))
    args = parser.parse_args()
    adapt_config(args.file_path, args.section, args.parameter, args.value, bool(args.string_or_not))
