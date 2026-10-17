"""Run the end-to-end MovieLens experiment. Test data is only evaluated after choices are frozen."""
import json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
from recsys.config import CONFIG
from recsys.data import load_movielens, chronological_split
from recsys.metrics import GenreSimilarity
from recsys.evaluator import Evaluator
from recsys.models import RandomRecommender, PopularityRecommender, SurpriseRecommender, ALSRecommender
from recsys.mmr import MMRRecommender
from recsys.sasrec import SASRecRecommender


def main():
    out=CONFIG.output_dir; out.mkdir(parents=True,exist_ok=True); CONFIG.artifact_dir.mkdir(parents=True,exist_ok=True)
    ratings,movies=load_movielens(); print(f"Loaded ratings={len(ratings):,}; users={ratings.user_id.nunique():,}; movies={movies.movie_id.nunique():,}")
    train_fit,val_data,train_final,test_data=chronological_split(ratings,CONFIG.min_ratings_per_user)
    for name,df in [("train_fit",train_fit),("validation",val_data),("train_final",train_final),("test",test_data)]:
        df.to_csv(CONFIG.artifact_dir/f"{name}.csv",index=False); print(name,len(df))
    sim=GenreSimilarity(movies); catalog=movies.movie_id.astype(int).unique()
    ev_val=Evaluator(train_fit,val_data,catalog,sim); ev_test=Evaluator(train_final,test_data,catalog,sim)
    print(f"Evaluation users: validation={len(ev_val.data)}, test={len(ev_test.data)}")
    models={
      "Random":RandomRecommender().fit(train_fit),
      "Popularity":PopularityRecommender().fit(train_fit),
      "ItemKNN":SurpriseRecommender("ItemKNN").fit(train_fit),
      "SVD":SurpriseRecommender("SVD").fit(train_fit),
      "SVD++":SurpriseRecommender("SVD++").fit(train_fit),
      "ALS":ALSRecommender(64,.05).fit(train_fit),
    }
    # SASRec is retained as a separate sequential baseline.
    try:
        models["SASRec"] = SASRecRecommender(epochs=CONFIG.sasrec_epochs).fit(train_fit)
    except RuntimeError as exc:
        print(f"SASRec skipped: {exc}")
    classic,details=ev_val.evaluate_many(models); classic.to_csv(out/"classic_validation.csv"); print("\nValidation results\n",classic)
    # Hyperparameters are selected exclusively on validation.
    grid=[]
    for factors in CONFIG.als_factors:
      for reg in CONFIG.als_regularization:
        model=ALSRecommender(factors,reg).fit(train_fit)
        metrics,_=ev_val.evaluate(model)
        grid.append({"factors":factors,"regularization":reg,**metrics})
        print(f"ALS factors={factors}, reg={reg}: NDCG@{CONFIG.k}={metrics.get(f'ndcg@{CONFIG.k}',float('nan')):.4f}")
    grid_df=pd.DataFrame(grid).sort_values(f"ndcg@{CONFIG.k}",ascending=False)
    grid_df.to_csv(out/"als_grid_validation.csv",index=False)
    best_f=int(grid_df.iloc[0].factors); best_r=float(grid_df.iloc[0].regularization)
    tuned_val=ALSRecommender(best_f,best_r).fit(train_fit)
    # MMR lambda is also selected on validation only. Use a relevance floor (>=95% of best NDCG)
    # then maximize ILD, to avoid selecting lambda=1 solely because it maximizes relevance.
    lam_rows=[]
    for lam in CONFIG.mmr_lambdas:
        mmr=MMRRecommender(tuned_val,lam,sim,k=CONFIG.k,top_n=CONFIG.mmr_top_n)
        metrics,_=ev_val.evaluate(mmr)
        lam_rows.append({"lambda":lam,**metrics})
    lam_df=pd.DataFrame(lam_rows)
    best_ndcg=lam_df[f"ndcg@{CONFIG.k}"].max()
    eligible=lam_df[lam_df[f"ndcg@{CONFIG.k}"] >= .95*best_ndcg]
    chosen=eligible.sort_values([f"ild@{CONFIG.k}",f"ndcg@{CONFIG.k}"],ascending=False).iloc[0]
    best_lambda=float(chosen["lambda"]); lam_df["selected"] = lam_df["lambda"].eq(best_lambda)
    lam_df.to_csv(out/"mmr_validation.csv",index=False)
    # Freeze selected config; refit on train_fit + validation, then test once.
    final_als=ALSRecommender(best_f,best_r).fit(train_final)
    final_mmr=MMRRecommender(final_als,best_lambda,sim,k=CONFIG.k,top_n=CONFIG.mmr_top_n)
    base_metrics,base_detail=ev_test.evaluate(final_als)
    mmr_metrics,mmr_detail=ev_test.evaluate(final_mmr)
    final_rows=[{"variant":"ALS_tuned_plain",**base_metrics},{"variant":"ALS_tuned_MMR",**mmr_metrics}]
    # Refit all classical baselines on train_final after model selection is complete.
    final_factories = {
      "Random": lambda: RandomRecommender().fit(train_final),
      "Popularity": lambda: PopularityRecommender().fit(train_final),
      "ItemKNN": lambda: SurpriseRecommender("ItemKNN").fit(train_final),
      "SVD": lambda: SurpriseRecommender("SVD").fit(train_final),
      "SVD++": lambda: SurpriseRecommender("SVD++").fit(train_final),
      "ALS": lambda: ALSRecommender(64,.05).fit(train_final),
    }
    for name, factory in final_factories.items():
        model = factory(); metrics, _ = ev_test.evaluate(model, allowed=getattr(model, "known_items", None))
        final_rows.append({"variant":name, **metrics})
    # Refit SASRec on train_final only after its validation result has been recorded.
    if "SASRec" in models:
        try:
            final_sasrec=SASRecRecommender(epochs=CONFIG.sasrec_epochs).fit(train_final)
            sas_metrics,_=ev_test.evaluate(final_sasrec, allowed=getattr(final_sasrec, "known_items", None))
            final_rows.append({"variant":"SASRec",**sas_metrics})
        except (RuntimeError, ValueError) as exc:
            print(f"Final SASRec skipped: {exc}")
    final=pd.DataFrame(final_rows).set_index("variant")
    final.to_csv(out/"final_test_results.csv")
    def bucket(n): return "1-4" if n<=4 else ("5-19" if n<20 else "20+")
    group_rows=[]
    for label,detail in [("ALS_tuned_plain",base_detail),("ALS_tuned_MMR",mmr_detail)]:
        if detail.empty: continue
        d=detail.copy(); d["history_group"]=d.history_len.map(bucket)
        metric_cols=[c for c in d if c.startswith(("recall@","ndcg@","precision@","ild@"))]
        g=d.groupby("history_group")[metric_cols].mean(); g["n_users"]=d.groupby("history_group").user_id.nunique()
        g["variant"]=label; group_rows.append(g.reset_index())
    if group_rows: pd.concat(group_rows,ignore_index=True).to_csv(out/"test_metrics_by_history.csv",index=False)
    metadata={"seed":CONFIG.seed,"rating_threshold":CONFIG.rating_threshold,"min_ratings_per_user":CONFIG.min_ratings_per_user,"k":CONFIG.k,"n_neg":CONFIG.n_neg,"max_users_eval":CONFIG.max_users_eval,"split":"per-user chronological 80/10/10; train_final=train_fit+validation","best_als_factors":best_f,"best_als_regularization":best_r,"mmr_lambda_selected_on_validation":best_lambda,"mmr_selection":"maximize ILD among lambda values with NDCG@K >= 95% of best validation NDCG","python":platform.python_version()}
    (out/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("\nFinal test results (test was not used for tuning):\n",final)
    print("Selected ALS",best_f,best_r,"MMR lambda",best_lambda)
    print("Results:",out)

if __name__=="__main__": main()
