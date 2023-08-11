BASE="/n/data1/hms/dbmi/zitnik/lab/users/owq978/TimeSeriesCBM/experiments/scs_better"

allnums=(2 3 4 6 7 8 9)

for i in ${allnums[@]}
    do
        echo "run $i"
        cp $BASE/rexp_template.sh $BASE/rexp_$i.sh
        sed -i "s/RNUM/$i/" $BASE/rexp_$i.sh
        sbatch $BASE/rexp_$i.sh
        rm $BASE/rexp_$i.sh
    done