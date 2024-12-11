FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip git wget

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
RUN pip3 install scikit-learn==1.5.2
RUN pip3 install matplotlib==3.9.2
RUN pip3 install pandas==2.2.3
RUN pip3 install tqdm==4.66.5

RUN git clone git@github.com:sugoma11/InconvenientWeightsUpdate.git
WORKDIR InconvenientWeightsUpdate

RUN mkdir arch
RUN wget https://media.githubusercontent.com/media/fpleoni/fashion_mnist/refs/heads/master/fashion-mnist_test.csv -O arch/fashion-mnist_test.csv
RUN wget https://media.githubusercontent.com/media/fpleoni/fashion_mnist/refs/heads/master/fashion-mnist_train.csv -O arch/fashion-mnist_train.csv

RUN mkdir results_trui
RUN mkdir results_squash

CMD ["python3", "main_triu.py"]