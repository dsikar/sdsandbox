To  prepare data:
python prepare.py --clean
To train network see train.py
To run predictions, start unity then:
python predict_client.py --model=../trained_models/nvidia1/20201107210627_nvidia1.h5
To monitor network traffic between sim and predict.py:
sudo tcpflow -i lo -c port 9091
To pipe to log file for prediction evaluation:
sudo tcpflow -i lo -c port 9091 > /tmp/tcpflow.log
