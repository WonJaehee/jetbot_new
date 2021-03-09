cd
cd jetbot_new/StartDL
chmod 777 *.sh

./install-pytorch.sh
   -> It might take more than 40 minutes.
   
./install_torch2trt.sh

# To Check

python3

import torch
