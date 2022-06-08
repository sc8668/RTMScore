# RTMScore

RTMScore is a a novel scoring function based on residue-atom distance likelihood potential and graph transformer, for the prediction of protein-ligand interactions. 
<div align=center>
<img src="https://github.com/sc8668/RTMScore/blob/main/121.jpg" width="1000px" height="500px">
</div> 

The proteins and ligands were first characterized as 3D residue graphs and 2D molecular graphs, respectively, followed by two groups of independent graph transformer layers to learn the node representations of proteins and ligands. Then all node features were concatenated in a pairwise manner, and input into an MDN to calculate the parameters needed for a mixture density model. Through this model, the probability distribution of the minimum distance between each residue and each ligand atom could be obtained, and aggregated into a statistical potential by summing all independent negative log-likelihood values.

### Requirements
dgl-cuda11.1==0.7.0
mdanalysis==2.0.0
pandas==1.0.3
prody==2.1.0
python==3.8.11
pytorch==1.9.0
rdkit==2021.03.5
openbabel==3.1.0
scikit-learn==0.24.2
scipy==1.6.2
seaborn==0.11.2
numpy==1.20.3
pandas==1.3.2
matplotlib==3.4.3
joblib==1.0.1

`pip install -r requirements.txt`  
# Requirements
