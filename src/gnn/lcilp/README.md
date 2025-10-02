# Locality-aware subgraphs for inductive link prediction in knowledge graphs

## Dependencies
All the required packages can be installed by running
```
pip install -r requirements.txt
```
## Usage
To start training the Clust-LP model (on WN18RR v1 as an example), run the following command:
```
python train.py -d WN18RR_v1 -e grail_wn_v1
```

To test the model, run the following commands:
```
- python test_auc.py -d WN18RR_v1_ind -e exp_wn_v1
- python test_ranking.py -d WN18RR_v1_ind -e exp_wn_v1
```

## Acknowledgements
This work was supported by MEMEX project funded by the European Union's Horizon 2020 research and innovation program under grant agreement No 870743.

The code is implemented based on GraIL (https://github.com/kkteru/grail). Thanks for their code sharing.
