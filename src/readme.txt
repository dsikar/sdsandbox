###############################################
# Running the sim self-drive prediction engine
# with network debugging
###############################################
To collect data, start unity image:
$ sudo ./UnityHub.AppImage --no-sandbox
This will start Unity Hub. 
Open project
In project, select Scenes > menu and run (play button)
Select "Generated Track" then "Auto Drive w Rec".
Let the sim run for a few laps and stop.
Recorded images will be in sdsim/output.

To  prepare data:
python prepare.py --clean

To train network:
$ train.py --model=../trained_models/model_to_be_saved.h5

To monitor network traffic between sim and predict.py:
$ sudo tcpflow -i lo -c port 9091
To pipe to log file for prediction evaluation:
$ sudo tcpflow -i lo -c port 9091 > /tmp/tcpflow.log

To run predictions, start unity image as before and select "NN Control over Network"
then run (using relevant trained model)
$ python predict_client.py --model=../trained_models/nvidia1/20201107210627_nvidia1.h5

Stop the sim and both scripts when testing is complete. Check /tmp directory for output.

$ ls -lh /tmp/tcpflow.log 
-rw-r--r-- 1 simbox simbox 8.8M Nov 19 07:57 /tmp/tcpflow.log
