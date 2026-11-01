"""Visualization helpers; archived Colab figures remain in the original notebook.
New figures can be generated from saved predictions without changing training.
"""
import matplotlib.pyplot as plt
import numpy as np

def depth_comparison(rgb, gt, pred, output):
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,image,title in zip(axes,[rgb,gt,pred],['RGB','GT (mm)','Prediction (mm)']):
        ax.imshow(image);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(output,dpi=160);plt.close(fig)
