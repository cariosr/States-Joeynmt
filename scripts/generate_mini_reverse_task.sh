#Make sure the protoype config mini_prototype.yaml is in place.
#Arguments: train_size dev_size test_size max_len highest

if [[ "$PWD" == */scripts ]]; then
    cd ..
fi

sample_size=$1
dev_size=$2
test_size=$3
high=$4
maxlen=$5

reverse_f="mini_reverse_${sample_size}_${dev_size}_${test_size}_${high}_${maxlen}"

rm -rf test/data/$reverse_f
mkdir -p test/data/$reverse_f

python3 scripts/customizable_reverse_task.py $sample_size $dev_size $test_size $high $maxlen

for l in src trg; do
    mv train.$l test/data/$reverse_f/
    mv test.$l test/data/$reverse_f/ 
    mv dev.$l test/data/$reverse_f/
done

if [ $sample_size = 1 ]; then
    location=test/data/$reverse_f

    for l in src trg; do
        cp -f $location/train.$l $location/test.$l
        cp -f $location/train.$l $location/dev.$l
    done

fi

cp configs/mini_prototype.yaml configs/$reverse_f.yaml
sed -i "s/prototype/${reverse_f}/g" configs/$reverse_f.yaml