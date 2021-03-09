* YOU MUST INSTALL PYTORCH BEFORE START THIS. *

* IF NOT : 
   cd
   cd StartDL
   chmod 777 *.sh
   ./install-pytorch.sh
   -> It might take more than 40 minutes.
   ./install_torch2trt.sh

1. data_collection

save_block=0 # 0: free, 1:block


1) Take Blocked Picture

python3 data_collection2.py 1

# press 's' to take picture
(collect picture more than 40)

# Ctl-C to finish it

2) Take Free Picture

python3 data_collection2.py 0

# press 's' to take picture
(collect picture more than 40)

# Ctl-C to finish it

----------------------------------------
2. train_fca_model

python3 train_fca_model.py

----------------------------------------

3. fca_live_demo

python3 fca_live_demo.py
