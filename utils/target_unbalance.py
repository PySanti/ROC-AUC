def target_unbalance(df, target):
    for k,v in df[target].value_counts().items():
        print(f"{k} : {v} ({v*100/len(df)})")
