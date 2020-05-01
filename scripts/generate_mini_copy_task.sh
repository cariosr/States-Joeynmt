#Make sure the protoype config mini_prototype.yaml is in place.
#Arguments: train_size dev_size test_size max_len highest letters
#Letters is 0 if you want numbers or 1 if you want letters.

if [[ "$PWD" == */scripts ]]; then
    cd ..
fi

sample_size=$1
dev_size=$2
test_size=$3
high=$4
maxlen=$5
letters=$6

if [ $letters==0 ]; then
    letnum=numbers
else
    letnum=letters
fi


copy_f="mini_copy_${sample_size}_${dev_size}_${test_size}_${high}_${maxlen}_${letnum}"

rm -rf test/data/$copy_f
mkdir -p test/data/$copy_f

python3 scripts/customizable_copy_task.py $sample_size $dev_size $test_size $high $maxlen $letters

for l in src trg; do
    mv train.$l test/data/$copy_f/
    mv test.$l test/data/$copy_f/ 
    mv dev.$l test/data/$copy_f/
done

if [ $sample_size = 1 ]; then
    location=test/data/$copy_f

    for l in src trg; do
        cp -f $location/train.$l $location/test.$l
        cp -f $location/train.$l $location/dev.$l
    done

fi

cp configs/mini_prototype.yaml configs/$copy_f.yaml
sed -i "s/prototype/${copy_f}/g" configs/$copy_f.yaml