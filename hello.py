import numpy as np
p="/Users/siweiwang/Documents/stocks/TimeBridge/out/fullmarket_emb_cov.npz"
d=np.load(p, allow_pickle=True)
print("keys:", d.files)
for k in ["feat_h1","feat_h3","reward","cov","close","dates","tickers"]:
    if k in d.files:
        x=d[k]
        print(k, x.shape, x.dtype)