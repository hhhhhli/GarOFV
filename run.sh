for((i=5;i<=95;i+=5))
do
    python -B video_classify_module_zy.py --bandwidth $i --use_flow True --use_early_stop False # exp3_4
done

for((i=5;i<=95;i+=5))
do
    python -B video_classify_module_zy.py --bandwidth $i --use_flow False --use_early_stop False # exp3_2
done

for((i=5;i<=95;i+=5))
do
    python -B video_classify_module_zy.py --bandwidth $i --use_flow True --use_early_stop True # exp3_2
done

for((i=5;i<=95;i+=5))
do
    python -B video_classify_module_zy.py --bandwidth $i --use_flow False --use_early_stop True # exp3_2
done