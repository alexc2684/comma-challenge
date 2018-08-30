# comma.ai Programming Challenge
by Alex Chan

The files in this directory contain:
- Training Data EDA.ipynb: A quick exploratory data analysis of the training data
- process_video.py: Preprocess each frame of the videos, and separate into training and validation set
- train_speed_model.py: Train the model on video data
- vgg.py: Contains VGG class which initializes, trains, and evaluates a VGG architecture model
- evaluate_model.py: Evaluate single input VGG model
- eval_dual_model.py: Evaluate two input VGG model

Usage
- To preprocess data, run script and specify 1 if for training data and 0 if for test
```python3 process_video.py -t <IS_TRAIN>```

- To train, run script with following parameters, specifying whether or not to use the dual input model
```python3 train_speed_model.py -e <EPOCHS> -b <BATCH SIZE> -m <MODEL NAME> -d <IS_DUAL>```

- To evaluate model, pass in data for raw frames and processed frames
```eval_dual_model.py -r <RAW FRAMES> -p <PROCESSED FRAMES> -m <MODEL>```
