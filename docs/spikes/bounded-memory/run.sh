set -e
rustc --edition 2021 -O -o /tmp/ss spool_spike.rs
uname -m; rustc --version
for n in 1000000 4000000 16000000; do
  reps=3; [ $n = 16000000 ] && reps=2
  for r in $(seq $reps); do
    for s in direct twopass spool; do
      /tmp/ss $s $n /work/data/$s; rm -rf /work/data/$s
    done
  done
done
echo "--- multipass merge check (4M entities, 16384-record runs)"
/tmp/ss spool 4000000 /work/data/mp 16384; rm -rf /work/data/mp
echo DONE
