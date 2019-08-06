python mnist.py --nntype=Linear --nunits=512 --nhidden=3 --batch_size=100
python mnist.py --nntype=NoisyBinary --nunits=512 --nhidden=3 --batch_size=100 --sigma=0.1
python mnist.py --nntype=NoisyBinary --nunits=512 --nhidden=3 --batch_size=100 --sigma=0.3
python mnist.py --nntype=NoisyBinary --nunits=512 --nhidden=3 --batch_size=100 --sigma=0.5
python mnist.py --nntype=NoisyBinary --nunits=512 --nhidden=3 --batch_size=100 --sigma=1
python mnist.py --nntype=NoisyBinary --nunits=512 --nhidden=3 --batch_size=100 --sigma=1.5
python mnist.py --nntype=NoisyBinary --nunits=512 --nhidden=3 --batch_size=100 --sigma=2
python plot_mnist.py