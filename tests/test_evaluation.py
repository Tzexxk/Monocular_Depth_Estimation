"""Hand-computable tests for the shared benchmark, independent of any model."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from c3vd_eval.metrics import depth_metrics,summarize
from c3vd_eval.protocol import histogram_median

class EvaluationTests(unittest.TestCase):
    def test_hand_calculated(self):
        gt=np.array([[10.,20.,40.]])
        pred=np.array([[10.,25.,20.]])
        m=depth_metrics(pred,gt,np.ones(gt.shape,bool))
        self.assertAlmostEqual(m['absrel'],.25)
        self.assertAlmostEqual(m['rmse_mm'],np.sqrt(425/3))
        self.assertAlmostEqual(m['delta1'],1/3) # ratio exactly 1.25 must fail
    def test_perfect_and_gt_mask_only(self):
        gt=np.array([[10.,0.]])
        m=depth_metrics(np.array([[10.,np.nan]]),gt,np.array([[True,False]]))
        self.assertEqual((m['absrel'],m['rmse_mm'],m['delta1']),(0.,0.,1.))
    def test_reject_bad_predictions(self):
        gt=np.ones((2,2)); mask=np.ones((2,2),bool)
        for v in (0.,-1.,np.nan,np.inf):
            pred=gt.copy();pred[0,0]=v
            with self.assertRaises(ValueError):depth_metrics(pred,gt,mask)
    def test_bad_geometry_or_mask(self):
        gt=np.ones((2,2))
        for pred,mask in [(np.ones((1,4)),gt.astype(bool)),(gt,gt),(gt,np.zeros_like(gt,bool))]:
            with self.assertRaises(ValueError):depth_metrics(pred,gt,mask)
    def test_equal_frame_aggregation(self):
        rows=[{'absrel':0.,'rmse_mm':0.,'delta1':1.,'valid_pixels':1},
              {'absrel':2.,'rmse_mm':4.,'delta1':0.,'valid_pixels':100}]
        m=summarize(rows)
        self.assertEqual((m['absrel'],m['rmse_mm'],m['delta1']),(1.,2.,.5))
    def test_histogram_median_matches_materialized(self):
        rng=np.random.default_rng(42)
        for n in (1,2,101,1000):
            raw=rng.integers(1,65535,n,dtype=np.uint16)
            h=np.bincount(raw,minlength=65536)
            expected=np.median(raw.astype(np.float32)*np.float32(100/65535))
            self.assertEqual(histogram_median(h)[0],float(expected))
    def test_empty_median(self):
        with self.assertRaises(ValueError):histogram_median(np.zeros(65536,dtype=np.int64))
if __name__=='__main__': unittest.main()
