# SenseDiscriminate
This repo contains the code and dataset for our paper " Discriminating soft actuators' thermal stimuli and mechanical deformation by hydrogel sensors and machine learning" 

by Shuyu Wang*, Zhaojia Sun et.al

Here, we use two hydrogel sensors for simultaneously proprioceptive, thermoceptive and mechanoceptive sensing. These stretchable sensors show high sensitivity to strain and temperature changes.  Then, we utilize a machine learning model, composed of 1D convolutional neural network and feed-forward neural network, to decode the sensing signal for various stimuli identification. We demonstrate the proposed method can accurately predict the soft actuator’s body posture changes, such as bending, twisting and stretching. Plus, the model can discern contact events with or without thermal stimuli. This data-driven method for multi-modal sensing discrimination might pave the way for future intelligent soft robots.

# Dependencies

* Python 3.7
* Pytorch
* numpy
* sklearn
* CUDA

# Usage

to train the dataset and perform cross validation: 
run
```
python train.py 
```

# Copyright and reference
If you used the dataset or code in your research please cite our paper.
