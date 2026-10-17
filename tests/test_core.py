import pandas as pd
from recsys.data import chronological_split
from recsys.metrics import ndcg_at_k
from recsys.mmr import mmr_rerank

def test_chronological_split_preserves_order_and_sizes():
    rows=[]
    for u in [1,2]:
        for t in range(20): rows.append((u,t+1,float(t%5+1),t))
    df=pd.DataFrame(rows,columns=['user_id','movie_id','rating','timestamp'])
    fit,val,train,test=chronological_split(df, min_ratings_per_user=5)
    assert len(fit)+len(val)==len(train)
    assert set(fit.movie_id).isdisjoint(set(val.movie_id))
    assert fit.groupby('user_id').timestamp.max().max() < val.groupby('user_id').timestamp.min().min()
    assert val.groupby('user_id').timestamp.max().max() < test.groupby('user_id').timestamp.min().min()

def test_ndcg_and_mmr_lambda_one():
    assert ndcg_at_k([1,2],{1},2)==1.0
    sim=lambda a,b: 0.0
    assert mmr_rerank([1,2,3],[.9,.8,.1],2,1.0,sim)==[1,2]
