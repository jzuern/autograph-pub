
# Description


We use an XMOS XUF216 
microphone array with 7 microphones in total for audio recording, where six microphones are arranged 
circularly with a 60-degree angular distance and one microphone is located at the center. 
The array is mounted horizontally for maximum angular resolution for objects moving in the horizontal plane. 

To capture the images, we use a FLIR BlackFly S RGB camera and crop the images to a resolution of 400 x 1200 pixels. 
Images are recorded at a fixed frame rate of 5 Hz, while the audio is captured at a sampling rate of 44.1 kHz for each channel. 
The microphone array and the camera are mounted on top of each other with a vertical distance of ca. 10 cm.


For our dataset, we consider two distinct scenarios: static recording platform and moving recording platform. 
In the static platform scenario, the recording setup is placed close to a street and is mounted on a static camera mount. 
In the dynamic platform scenario, the recording setup is handheld and is deliberately moved in a translational 
fashion (approx. 15 cm of positional range along each spatial axis) and rotated with a maximal deviation 
angle of 10 deg. We collected ca. 70 minutes of audio and video footage in nine distinct scenarios 
with different weather conditions ranging from clear to overcast and foggy. 
Overall, the dataset contains more than 20k images. The recording environments entail suburban, 
rural, and industrial scenes. The distance of the camera to the road varies between scenes. 

To evaluate detection metrics with our approach, we manually annotated more than 300 randomly selected images across 
all scenes with bounding boxes for moving vehicles. We also manually classified each image in the dataset whether 
it contains a moving vehicle or not. Note that static vehicles are counted as part of the background of each scene.


# Dataset Structure

The dataset consists of 9 distinct scenes where 7 scenes are recorded with a static camera and two 
scenes are recorded with a dynamic caera (sensor setup not mounted on tripod and deliberately moved).

Static
- granadaallee1
- hirtenweg1
- herrmanmitschstr4
- elsaesserstr1
- elsaesserstr2
- zinkmatttenstr1
- madisonallee1

Dynamic
- herrmanmitschstr4 handheld
- zinkmatttenstr1 handheld





The directory structure of the `train` directory is as follows:

```
/granadaallee1.wav             # The wav recording
/granadaallee1/eval/           # Anotated GT images for evaluation
              /cam0_rectified/     # The RGB images
                  /labels.txt      # Each line contains the start and end index of images where one/multiple vehicles are present.
                                   # 
                  /spec0-1.0/      # Spectrograms for each timestep of mic0 
                  /spec1-1.0/      # "" of mic0
                  ...
                  /wav0-1.0/       # waveform for each timestep of mic0
                  /wav1-1.0/       # "" of mic1
                  ...
                  
/hirtenweg1.wav             # The wav recording
/hirtenweg1/...
...
```

The directory structure of the `test` directory is as follows:

```
/granadaallee1/img_<number>_<timestamp>.jpg  # The RGB image
              /img_<number>_<timestamp>.xml  # The Bounding Box annotation
              ...
/hirtenweg1/...
```
