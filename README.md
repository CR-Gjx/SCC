# Alleviating Semantics Distortion in Unsupervised Low-Level Image-to-Image Translation via Structure Consistency Constraint. 

Pytorch implementation of "[Alleviating Semantics Distortion in Unsupervised Low-Level Image-to-Image Translation via Structure Consistency Constraint.](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Alleviating_Semantics_Distortion_in_Unsupervised_Low-Level_Image-to-Image_Translation_via_Structure_CVPR_2022_paper.pdf)" (CVPR 2022).

The SCC loss aims to reduce the geometrical distortion during the image translation.

The translated images are shown as follows:

![](https://github.com/CR-Gjx/SCC/blob/main/figures/scc.jpg)
 

You can directly insert the codes of SCC into [GcGAN](https://github.com/hufu6371/GcGAN/tree/master/models), and run the codes following GcGAN scripts.

Specifically, please set  
```
batch_size = 1
```
