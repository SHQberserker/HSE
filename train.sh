#------------------------CUB--------------------------------------
python train.py --gpu-id -1 --loss Proxy_Anchor  --model resnet50 --embedding-size 512 --batch-size 128 --lr 1e-4 --dataset CUB --warm 5 --bn-freeze 1 --lr-decay-step 10 --lr-decay-gamma 0.5 --a 2 --IPC 8

#------------------------CAR--------------------------------------
python train.py --gpu-id -1 --loss Proxy_Anchor  --model resnet50 --embedding-size 512 --batch-size 120 --lr 1e-4 --dataset CAR --warm 5 --bn-freeze 1 --lr-decay-step 10 --lr-decay-gamma 0.5 --a 0.5 --IPC 4

#------------------------SOP--------------------------------------
python train.py --gpu-id -1 --loss EPSHN  --model resnet50 --embedding-size 512 --batch-size 120 --lr 3e-4 --dataset SOP --warm 1 --bn-freeze 0 --lr-decay-step 10 --lr-decay-gamma 0.25 --a 1.0 --IPC 10

python train.py --gpu-id -1 --loss MS --model resnet50 --embedding-size 512 --batch-size 120 --lr 3e-4 --dataset SOP --warm 1 --bn-freeze 0 --lr-decay-step 10 --lr-decay-gamma 0.25 --a 0.3 --IPC 10

python train.py --gpu-id -1 --loss Proxy_Anchor --model resnet50 --embedding-size 512 --batch-size 120 --lr 3e-4 --dataset SOP --warm 1 --bn-freeze 0 --lr-decay-step 10 --lr-decay-gamma 0.25 --a 2.0 --IPC 10



