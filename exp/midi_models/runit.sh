training_dir=training_output
rm ./$training_dir/*
rm ./$training_dir/duration_rnn/*
rm outputs/*mid
rm outputs/*wav

runtime=$(date +%s)
mkdir -p ./$training_dir/$runtime

python duration_rnn.py
cp ./$training_dir/duration_rnn/*log ./$training_dir/$runtime
cp ./$training_dir/duration_rnn/*html ./$training_dir/$runtime
which_ckpt=$(ls ./"$training_dir"/duration_rnn/*valid*ckpt | sort | tail -n 1)
python sample_duration_rnn.py $which_ckpt $runtime
for i in outputs/*mid; do
    ./timidifyit.sh $i
done
mkdir -p ./$training_dir/$runtime
cp ./$training_dir/duration_rnn/* ./$training_dir/$runtime
mv outputs/*wav ./$training_dir/$runtime
mv outputs/*mid ./$training_dir/$runtime
