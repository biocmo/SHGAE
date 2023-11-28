# Prediction of miRNA–disease associations based on strengthened hypergraph convolutional autoencoder


Most existing graph neural network-based methods for predicting miRNA-disease associations
rely on initial association matrices to pass messages, but the sparsity of these matrices greatly
limits performance. To address this issue and predict potential associations between miRNAs
and diseases, we propose a method called strengthened hypergraph convolutional autoencoder
(SHGAE). SHGAE leverages multiple layers of strengthened hypergraph neural networks
(SHGNN) to obtain robust node embeddings. Within SHGNN, we design a strengthened
hypergraph convolutional network module (SHGCN) that enhances original graph associations
and reduces matrix sparsity. Additionally, SHGCN expands node receptive fields by utilizing
hyperedge features as intermediaries to obtain high-order neighbor embeddings. To improve
performance, we also incorporate attention-based fusion of self-embeddings and SHGCN embeddings.
SHGAE predicts potential miRNA-disease associations using a multilayer perceptron
as the decoder. Across multiple metrics, SHGAE outperforms other state-of-the-art methods in
five-fold cross-validation. Furthermore, we evaluate SHGAE on colon and lung neoplasms cases
to demonstrate its ability to predict potential associations. Notably, SHGAE also performs well
in the analysis of gastric neoplasms without miRNA associations.

### how to install
``` python
python 3.9
# CUDA 10.2
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 
pip install torch-sparse 
pip install torch-cluster
pip install torch-spline-conv
pip install torch-geometric 

conda install pandas

conda install openpyxl

conda install GraphRicciCurvature
```