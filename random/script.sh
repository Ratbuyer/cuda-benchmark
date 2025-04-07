for i in 1 2 4 8 16 32
do
        for d in 27 28 29 30
        do
                make BLOCK_SIZE=$i DATA_SIZE=$d KERNEL=1
                make run
        done
done

for i in 1 2 4 8 16 32
do
        for d in 27 28 29 30
        do
                make BLOCK_SIZE=$i DATA_SIZE=$d KERNEL=2
                make run
        done
done