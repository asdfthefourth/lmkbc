# LM-KB Challenge

Here is our approach to the LM-KBC. The solution is based on prompt ensembling

# Installation
Install Python >3.8 and requirements.

## Execute
Requires a hugginface token.
Run the first part requires 40GB of video memory and discover the optimal ensemble sets on the train/validation data.
`python3 src/modules/main.py --model meta-llama/Llama-2-70b-hf --name lmkbc --memory_management '{"0": "40GiB"}' --batch_size 8 --discovery --multi --hftoken xxx`

Calculate the Ensemlbe
`python3 src/modules/main.py --model meta-llama/Llama-2-70b-hf --name lmkbc --memory_management '{"0": "40GiB"}' --batch_size 8 --consens --multi --hftoken xxx`


Do the final run on the test dataset.
`python3 src/modules/main.py --model meta-llama/Llama-2-70b-hf --name lmkbc --memory_management '{"0": "40GiB"}' --batch_size 8 --final --multi --hftoken xxx`
