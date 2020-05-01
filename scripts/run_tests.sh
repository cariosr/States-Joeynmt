#Only argument to specify: configs/mini_reverse_whatever.yaml
#This configuration file has two new parameters: test_range - a python array of integers or floats, e.g. [0.3, 5, 0.7] - and test_variable - a string specifying
#which variable to vary along the test_range (e.g. 'lr'). Both need to be in the dqn section of the yaml file.
#Make sure that a joey-capable environment is in place (source /jnmt/bin/activate or whatever, plus pytorch etc. installed)
#(i.e. check that a single run of python3 -m joeynmt dqn_train $file_path > log.out will work)

if [[ "$PWD" == */scripts ]]; then
    cd ..
fi

file_path=$1
file_name="$file_path.sh"

python3 scripts/generate_counter.py $file_path

while test -f "$file_name"; do
    source $file_name
    python3 scripts/adapt_config_file.py $file_path
    python3 -m joeynmt dqn_train $file_path > "$cfg_name.out"
    python3 scripts/generate_counter.py $file_path
done