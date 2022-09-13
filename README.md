# Instructions (Running the TensorFlowASR Pretrained Model)
 1. Download the pretrained LibirSpeech Conformer model: https://drive.google.com/drive/folders/1BD0AK30n8hc-yR28C5FW3LqzZxtLOQfl
 2. In the "subword-conformer folder", unzip the "pretrained-subword-conformer.zip".
 3. After unzipping, enter the unzipped folder and run "setup.sh"
 4. Running "setup.sh" will clone the TensorFlowASR repository and create a folder for it.
 5. In that folder, download the requirements from "requirements.txt"
 6. Move the "tensorflow_asr" folder to "examples/demonstration" folder.
 7. Install TensorFlow and Miniconda using this tutorial: https://www.tensorflow.org/install/pip
 8. Go to the "subword-conformer" folder and edit the "config.yml" file
 9. Add a "." in front of every path with a "/"
 10. Go to the "pretrained-subword-conformer.zip" and edit the "run.sh" file
 11. Edit "conda activate tfasr" to "conda activate tf" to match the virtual environment name you created using Miniconda.
 12. Go into the virutal environment using "conda activate tf"
 13. Set the audio path of "run.sh" using "export AUDIO_PATH=pathtoaudiofile" (The audio file you want to translate)
 14. Run "run.sh" and program should output the transcipt of the audio file you set.


 Note: 
 - Edited "config.yml" and "run.sh" files are in the repository
 - The "conformer.py" in repository is from the "TensorFlowASR/examples/demonstration" folder and includes the code that was used to created a SavedModel file from the Conformer model
 - The "generate_vocab_sentencepiece.py and "generate_vocab_subwords.py" are the files that I was last working in order to run the Conformer model using the CommonVoice corpus but could not get them to output the desired outcome
 - The "audiofiles" folder contains the audio clips of the 911 calls

# Unfinished
 1. Running the Conformer model using CommonVoice corpus: https://commonvoice.mozilla.org/en/datasets
 2. Futher training the pretrained LibriSpeech Conformer model to enhance accuracy
 
