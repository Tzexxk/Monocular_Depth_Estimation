"""Generate baseline figures from standardized results without refitting the constant."""
import argparse
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from c3vd_eval.protocol import load_manifest,read_depth
from c3vd_eval.metrics import METRICS
from c3vd_eval.runner import write_csv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--result-dir', type=Path, default=ROOT/'outputs/results/constant_median')
    args = parser.parse_args()
    out = args.result_dir
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    if any(figures.iterdir()) or (out/'descriptive_statistics.csv').exists() or (out/'test_depth_histogram.npy').exists():
        raise FileExistsError('Refusing to overwrite baseline figures/statistics')
    run = json.loads((out/'config.json').read_text())
    c = run['metadata']['calibration']['constant_mm']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    devhist=np.zeros(65536,np.int64); testhist=np.zeros(65536,np.int64)
    for sample in load_manifest(args.data_root,'dev'):
        raw,_,mask=read_depth(args.data_root,sample)
        devhist+=np.bincount(raw[mask],minlength=65536)
    samples=load_manifest(args.data_root,'test')
    for sample in samples:
        raw,_,mask=read_depth(args.data_root,sample)
        testhist+=np.bincount(raw[mask],minlength=65536)
    np.save(out/'test_depth_histogram.npy',testhist)
    fig,ax=plt.subplots(figsize=(8,3.3),layout='constrained')
    # Bin into 100 equal-width mm bins; show fraction, not unequal pixel counts.
    for hist,name,color in [(devhist,'Development: trans_t1_a','#087f8c'),(testhist,'Test: trans_t2_b','#ef8354')]:
        bins=np.minimum(np.arange(65536)*100//65536,99)
        weights=np.bincount(bins,weights=hist,minlength=100)/hist.sum()
        ax.plot(np.arange(100)+.5,100*weights,label=name,color=color,lw=2)
    ax.axvline(c,color='#25314c',ls='--',label=f'Frozen constant: {c:.3f} mm')
    ax.set(xlabel='Valid GT z-depth (mm)',ylabel='Valid pixels per 1 mm bin (%)',xlim=(0,100))
    ax.legend(frameon=False,fontsize=9);fig.savefig(figures/'depth_distribution.png',dpi=200);plt.close(fig)
    rows=list(csv.DictReader((out/'per_frame.csv').open()))
    fig,axes=plt.subplots(3,1,figsize=(8,6),sharex=True,layout='constrained')
    for ax,k,label in zip(axes,METRICS,['AbsRel (lower is better)','RMSE (mm; lower is better)','Delta < 1.25 (higher is better)']):
        vals=np.array([float(r[k]) for r in rows]);ax.plot([int(r['sample_id'].split('/')[-1]) for r in rows],vals,color='#087f8c',lw=1.5)
        ax.axhline(vals.mean(),ls='--',color='#ef8354',label=f'Frame mean: {vals.mean():.4f}')
        ax.set_ylabel(label);ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    axes[-1].set_xlabel('Test frame ID (temporal order)');fig.savefig(figures/'test_per_frame_metrics.png',dpi=200);plt.close(fig)
    chosen=[samples[0],samples[len(samples)//2],samples[-1]]
    fig,axes=plt.subplots(3,4,figsize=(12,8),layout='constrained')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dedede')
    errmap=plt.get_cmap('magma').copy();errmap.set_bad('#dedede')
    for i,s in enumerate(chosen):
        _,gt,mask=read_depth(args.data_root,s)
        with Image.open(args.data_root/s['rgb_path']) as im:rgb=np.array(im.convert('RGB'))
        axes[i,0].imshow(rgb);axes[i,0].set_title(s['sample_id'],fontsize=10)
        z=axes[i,1].imshow(np.ma.array(gt,mask=~mask),vmin=0,vmax=100,cmap=cmap)
        axes[i,2].imshow(np.ma.array(np.full(gt.shape,c),mask=~mask),vmin=0,vmax=100,cmap=cmap)
        e=axes[i,3].imshow(np.ma.array(np.abs(gt-c),mask=~mask),vmin=0,vmax=100,cmap=errmap)
        for ax in axes[i]:ax.axis('off')
    axes[0,1].set_title('Ground truth (mm)');axes[0,2].set_title(f'Constant baseline ({c:.3f} mm)');axes[0,3].set_title('Absolute error (mm)')
    fig.colorbar(z,ax=axes[:,1:3],shrink=.55,label='z-depth (mm)');fig.colorbar(e,ax=axes[:,3],shrink=.55,label='Absolute error (mm)')
    fig.suptitle('Fixed examples: first, middle and last test frames | gray = excluded GT pixels',fontsize=12)
    fig.savefig(figures/'test_examples.png',dpi=180);plt.close(fig)
    stats=[]
    for split in ['test']:
        rr=list(csv.DictReader((out/'per_frame.csv').open()))
        for k in METRICS:
            v=np.array([float(r[k]) for r in rr]);stats.append({'split':split,'metric':k,'frames':len(v),'mean':v.mean(),'std_across_frames':v.std(ddof=1),'median':np.median(v),'min':v.min(),'max':v.max()})
    write_csv(out/'descriptive_statistics.csv',stats)
    print('Figures and statistics complete')
if __name__=='__main__':main()
