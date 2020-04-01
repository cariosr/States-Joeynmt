if [[ "$PWD" == */States-Joeynmt/scripts ]]; then
    cd ..
fi

rm -rf test/data/reverse
mkdir test/data/reverse

sample_size=$1
dev_size=$2
test_size=$3
high=$4
maxlen=$5

python3 scripts/customizable_reverse_task.py $sample_size $dev_size $test_size $high $maxlen

for l in src trg; do
    mv train.$l test/data/reverse/
    mv test.$l test/data/reverse/ 
    mv dev.$l test/data/reverse/
done