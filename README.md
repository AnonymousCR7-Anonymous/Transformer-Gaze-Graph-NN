# Transformer-Gaze-Graph-NN
Paper: Efficient 2D/3D Gaze Estimation Using TGGNet: A Transformer Graph Approach 

# TGGNet (Transformer Graph Gaze Neural Network)
This repository introduces a graph-based model for gaze estimation using a Graph Neural Network (GNN) to process facial landmarks and relational edges, ensuring precise eye gaze direction estimation. Key innovations include leveraging GNN's efficiency for real-time applications and integrating Transformer architecture to enhance feature updates.

![](Media/FacialGraph.gif) 


# Landmak Extraction
## GazeCapture dataset
Please use `LandmarkExtraction_GazeCapture.py` file to extract all 478 landmarks from valid images. Please note the name of those images that the Mediapipe fails to work with, will be saved in `..._failed_images_batch_....txt` files.

## MPIIFaceGaze
Please run `LandmarkExtraction_MPIIFaceGaze.py` file to extract 478 landmarks from normalized images of MPIIFaceGaze dataset.

## Gaze360
Please run `LandmarkExtraction_Gaze360.py` to extract 478 landmarks from images of Gaze360 dataset.


# Training TGGNet
## GazeCapture
Please run `TGGNET_GazeCapture.py` code for training and testing the proposed TGGNet on Gazecapture dataset. The test scale is normal Euclidean. To convert it back to real scale (centimeter), run `RealScale_GazeCapture.py`.

## MPIIFaceGaze
Please run `MPIIFaceGaze_Normal.py` for training and testing the proposed TGGNet on MPIIFaceGaze dataset. For converting the POG to the, run `Real_world_MPIIFaceGaze.py` so the test scale is normal Euclidean in mm. 

## Gaze360
Please run `Gaze360.py` for training and testing the proposed TGGNet on Gaze360 dataset. The test scale is mean angle error in degree.

# Demo

Change the `input_video_path` to match the input video's path.

Change the `model_path` to match the pre-trained model's path (`trained_model_No_Or.pt`). 

Change the `output_video_path` to match the output video's path. 

Then run `API.py` code to have the output video visualizing gaze vectors, POG coordinates, FPS, and execution time.

## Look at the following example:

The following video shows the input video.

![](Media/Input_video.gif) 

After applying Demo to it, we will have the resulting video as follows that indicates the gaze vectors and the POG for each frame.

![](Media/Result_test5.gif) 




