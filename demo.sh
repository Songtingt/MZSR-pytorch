#python main.py --train --use_weighted_loss --work_dir ./experiments_m_weighted
#python pretrain.py
#python main.py --gpu 0 --inputpath /home/songtingting02/data/Set5/LRbicx4/ --gtpath /home/songtingting02/stt/MZSR-master/GT/Set5/ --savepath results/Set5 --kernelpath /home/songtingting02/stt/MZSR-master/Input/g20/kernel.mat --model 0 --num 1
python main.py --gpu 0 --inputpath /home/songtingting02/data/Set5/LRbicx2/ --gtpath /home/songtingting02/stt/MZSR-master/GT/Set5/ --savepath results/Set5 --kernelpath /home/songtingting02/stt/MZSR-master/Input/g20/kernel.mat --model 0 --num 10
