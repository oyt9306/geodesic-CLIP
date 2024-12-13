# geodesic-CLIP
On mitigating stability-plasticity dilemma in CLIP-guided image morphing via geodesic distillation loss

Venue: International Journal of Computer Vision (IJCV), 2024

Paper : https://link.springer.com/article/10.1007/s11263-024-02308-z

<p align="center">
    <img src = "https://github.com/oyt9306/geodesic-CLIP/assets/41467632/4a01f733-64b6-42ca-9722-14c8368e5a01" width="60%">
</p>

This repository provides the main loss code used for improved CLIP-guided image morphing.

For simplicity, we summarize our proposed method for attribution-preserved morphing as follows:
(you can easily exploit the loss term for your own code)
```py
# Reference : Simon, Christian, Piotr Koniusz, and Mehrtash Harandi. "On learning the geodesic path for incremental learning." CVPR 2021.
# bsz : batch size
# d   : latent dim (i.e., 512 for ViT-B/32)
# img_x   : Tensor, shape=[bsz, d]
#       latents for L2-normalized image feature of E_I(I_target) / |E_I(I_target)|
# dir_x   : Tensor, shape=[bsz, d]
#       latents for L2-normalized image directions of E_I(I_target) - E_I(I_source) / |E_I(I_target) - E_I(I_source)|
# dir_y   : Tensor, shape=[bsz, d]
#       latents for L2-normalized text directions of E_T(T_target) - E_T(T_source)) / |E_T(T_target) - E_T(T_source)|

from geo_loss import GeoCLIP
morphing_loss = GeoCLIP(dim=256)

# loss term 
def inter_modality_cons(dir_x, dir_y):
    # x and y denotes the normalized image and text direction, respectively.
    return 1 - morphing_loss.fit(dir_x, dir_y).mean()
    
def intra_modality_reg(img_x):
    # x denotes the normalized image feature
    return 1 - morphing_loss.fit(img_x.detach(), img_x).mean()
```

