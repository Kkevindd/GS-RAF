exp_name='baseline'

voxel_size=0.01
update_init_factor=16
appearance_dim=0
ratio=1

ulimit -n 4096

./train.sh -d /data2/jkx/NeRAF/data/RAF/mix/FurnishedRoom -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --raf_data /data2/jkx/NeRAF/data/RAF/mix/FurnishedRoom & 

