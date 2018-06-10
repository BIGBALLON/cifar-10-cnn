#
# cifar10 every scheduler 5 runs
# resnet 50 
#
# step decay
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler step_decay --count 1
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler step_decay --count 2
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler step_decay --count 3
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler step_decay --count 4
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler step_decay --count 5
# cos
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler cos --count 1
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler cos --count 2
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler cos --count 3
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler cos --count 4
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler cos --count 5
# tanh
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler tanh --count 1
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler tanh --count 2
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler tanh --count 3
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler tanh --count 4
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar10 --scheduler tanh --count 5
#
# cifar100 every scheduler 5 runs
# resnet 50 
#
# step decay
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler step_decay --count 1
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler step_decay --count 2
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler step_decay --count 3
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler step_decay --count 4
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler step_decay --count 5
# cos
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler cos --count 1
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler cos --count 2
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler cos --count 3
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler cos --count 4
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler cos --count 5
# tanh
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler tanh --count 1
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler tanh --count 2
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler tanh --count 3
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler tanh --count 4
python ResNet.py --batch_size 128 --epoch 200 --stack_n 8 --dataset cifar100 --scheduler tanh --count 5


