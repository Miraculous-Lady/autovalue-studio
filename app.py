"""
AutoValue Studio — Used Car Price Intelligence
Run: streamlit run app.py
Dataset: CarDekho-style CSV with selling_price as the target column.
"""
from datetime import date
from pathlib import Path
import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="AutoValue Studio", page_icon="🚘", layout="wide", initial_sidebar_state="expanded")

# --------------------------- visual system ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {--ink:#f6f1e9;--muted:#b6b0c4;--panel:#211e2b;--panel2:#292536;--line:#40394e;--lime:#d8ff54;--pink:#ff6f91;--violet:#ad8cff;}
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;}
/* Animate the app's own background, rather than a negative-z pseudo-element.
   This keeps Streamlit's page content reliably above the gradient in all browsers. */
.stApp {color:var(--ink);background-color:#100b18;
  background-image:
    radial-gradient(ellipse at 18% 28%,rgba(126,24,75,.58) 0%,transparent 42%),
    radial-gradient(ellipse at 78% 24%,rgba(37,49,119,.56) 0%,transparent 43%),
    radial-gradient(ellipse at 55% 78%,rgba(83,37,119,.62) 0%,transparent 46%),
    radial-gradient(ellipse at 88% 82%,rgba(185,76,126,.20) 0%,transparent 34%),
    linear-gradient(125deg,#100b18 5%,#211027 34%,#11152b 67%,#1b0d20 100%);
  background-size:180% 180%;background-position:0% 45%;background-attachment:fixed;
  animation:ambientGradient 30s ease-in-out infinite alternate;}
@keyframes ambientGradient {0%{background-position:0% 45%;}50%{background-position:100% 55%;}100%{background-position:25% 100%;}}
@media (prefers-reduced-motion:reduce){.stApp{animation:none;background-position:50% 50%;}}
[data-testid="stHeader"] {background:rgba(20,12,26,.48);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#211b2b,#17151e);border-right:1px solid #40394e;}
[data-testid="stSidebar"] * {color:#f6f1e9;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.04em;}
h1 {font-size:2.5rem!important;font-weight:700!important;}
h2 {font-size:1.65rem!important;}
p,li,label,.stMarkdown {color:#e9e2ed;}
small,.muted {color:#b6b0c4!important;}
.hero {padding:30px 32px;border:1px solid #55445d;border-radius:24px;background:linear-gradient(115deg,rgba(52,36,60,.98),rgba(37,34,48,.95) 58%,rgba(45,47,40,.95));position:relative;overflow:hidden;margin-bottom:18px;}
.hero:after {content:"";position:absolute;width:260px;height:260px;border-radius:50%;right:-70px;top:-120px;background:radial-gradient(circle,rgba(216,255,84,.22),transparent 68%);}
.eyebrow {font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--lime);font-weight:800;margin-bottom:10px;}
.hero h1 {margin:0 0 8px 0;color:#fff7ef;font-size:2.65rem!important;line-height:1.08;}
.hero p {max-width:720px;color:#d2c8da;font-size:1rem;margin-bottom:0;}
.pill {display:inline-block;padding:6px 11px;border-radius:999px;background:#34303f;border:1px solid #51475f;color:#e8e1ed;font-size:.75rem;margin:12px 6px 0 0;}
.kpi {background:linear-gradient(145deg,#292535,#211e2b);border:1px solid #42394e;border-radius:18px;padding:18px 18px 15px;min-height:118px;box-shadow:0 12px 28px #0b0a101c;}
.kpi-label {font-size:.69rem;letter-spacing:.12em;text-transform:uppercase;color:#b6b0c4;font-weight:800;}
.kpi-value {font-family:'Space Grotesk',sans-serif;color:#f8f1e9;font-size:1.75rem;font-weight:700;line-height:1.2;margin:9px 0 4px;}
.kpi-note {font-size:.75rem;color:#b6b0c4;}
.kpi-accent {color:var(--lime);}
.section-label {font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:var(--lime);font-weight:800;margin:22px 0 6px;}
.panel {background:#211e2b;border:1px solid #40394e;border-radius:18px;padding:18px 20px;margin:8px 0;}
.note {background:#292536;border-left:4px solid #d8ff54;border-radius:8px;padding:12px 15px;color:#f3ecf5;margin:10px 0;}
.step {display:flex;gap:12px;align-items:center;background:#211e2b;border:1px solid #40394e;border-radius:12px;padding:10px 12px;margin:5px 0;}
.step-no {background:#d8ff54;color:#1a191f;font-weight:900;border-radius:9px;padding:7px 10px;}
div[data-testid="stMetric"] {background:#211e2b;border:1px solid #40394e;padding:14px;border-radius:14px;}
div[data-testid="stMetricLabel"] {color:#b6b0c4;}
div[data-testid="stMetricValue"] {color:#f6f1e9;}
.stButton>button,.stDownloadButton>button {background:#AD1457;color:#ffffff;border:1px solid #C2185B;border-radius:11px;font-weight:800;min-height:2.7rem;}
.stButton>button:hover,.stDownloadButton>button:hover {background:#C2185B;color:#ffffff;border:1px solid #E91E63;}
[data-testid="stFormSubmitButton"] button {background:#AD1457;color:#ffffff;border:1px solid #C2185B;border-radius:11px;font-weight:800;}
div[data-baseweb="tab-list"] {gap:8px;}
button[data-baseweb="tab"] {background:#292536;border-radius:10px;color:#f6f1e9;}
button[data-baseweb="tab"][aria-selected="true"] {background:#AD1457;color:#ffffff;}
[data-testid="stDataFrame"] {border:1px solid #40394e;border-radius:12px;overflow:hidden;}
hr {border-color:#40394e;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="darkgrid", rc={"axes.facecolor":"#211e2b","figure.facecolor":"#211e2b","axes.edgecolor":"#51475f","grid.color":"#40394e","text.color":"#f6f1e9","axes.labelcolor":"#e9e2ed","xtick.color":"#c9c0d1","ytick.color":"#c9c0d1"})
LIME, PINK, VIOLET, TEAL = "#d8ff54", "#ff6f91", "#ad8cff", "#67e6c0"
SOURCE_URL = "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"
REF_YEAR = date.today().year
DEFAULT_PATHS = [Path(x) for x in ["car data.csv","car_data.csv","Car data.csv","CAR DETAILS FROM CAR DEKHO.csv","car details v3.csv"]]
PRETTY = {"present_price":"Showroom price","car_age":"Car age","kms_driven":"Kilometres driven","fuel_type":"Fuel type","seller_type":"Seller type","transmission":"Transmission","owner":"Previous owners","selling_price":"Selling price"}

def pretty(c): return PRETTY.get(c, str(c).replace("_"," ").title())
def money(v, lakh=True): return f"₹{v:,.2f} lakh" if lakh else f"₹{v:,.0f}"
def show_df(df, **kwargs):
    try: st.dataframe(df, width="stretch", **kwargs)
    except Exception: st.dataframe(df, use_container_width=True, **kwargs)
def heading(title, sub=None):
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
    if sub: st.caption(sub)
def kpis(items):
    cols=st.columns(len(items))
    for col,(label,value,note) in zip(cols,items):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
def observation(s): st.markdown(f'<div class="note"><b>✳ Readout</b> &nbsp; {s}</div>',unsafe_allow_html=True)
def figshow(fig): st.pyplot(fig, use_container_width=True); plt.close(fig)

ALIASES={"name":"car_name","sellingprice":"selling_price","presentprice":"present_price","km_driven":"kms_driven","fuel":"fuel_type","mileage":"mileage","engine":"engine","max_power":"max_power","torque":"torque","seats":"seats"}
@st.cache_data(show_spinner=False)
def read_csv_bytes(data): return pd.read_csv(io.BytesIO(data))
def normalize(raw):
    df=raw.copy()
    df.columns=[str(c).strip().lower().replace(" ","_").replace("-","_") for c in df.columns]
    df=df.rename(columns={c:ALIASES.get(c,c) for c in df.columns})
    if "selling_price" not in df: raise ValueError("Couldn't find a selling_price column. Please upload a CarDekho-style CSV.")
    return df
def clean_data(raw, ref_year):
    norm=normalize(raw); info={"rows_raw":len(norm),"missing_cells":int(norm.isna().sum().sum()),"duplicates":int(norm.duplicated().sum()),"dropped":[]}
    df=norm.drop_duplicates().copy()
    df["selling_price"]=pd.to_numeric(df["selling_price"].astype(str).str.replace(",","",regex=False).str.replace("₹","",regex=False),errors="coerce")
    before=len(df); df=df[df.selling_price.notna()&(df.selling_price>0)]; info["bad_target_rows"]=before-len(df)
    for c in df.columns:
        if c in ("car_name","selling_price") or df[c].dtype!=object: continue
        cleaned=df[c].astype(str).str.replace(",","",regex=False).str.extract(r"^\s*(-?\d+\.?\d*)")[0]
        num=pd.to_numeric(cleaned,errors="coerce")
        if num.notna().mean()>.9: df[c]=num
    if "year" in df:
        years=pd.to_numeric(df["year"],errors="coerce")
        if years.notna().any():
            info["year_min"],info["year_max"]=int(years.min()),int(years.max())
            df["car_age"]=ref_year-years
        df=df.drop(columns=["year"])
    X=df.drop(columns=["selling_price","car_name"],errors="ignore")
    for c in list(X.columns):
        if X[c].notna().sum()==0 or (X[c].dtype=="object" and X[c].nunique()>80):
            info["dropped"].append((c,"empty or high-cardinality text field")); X=X.drop(columns=c)
    if X.shape[1]==0: raise ValueError("No usable feature columns found.")
    return norm,df,X,df.selling_price,info

def make_pipeline(X, algorithm, trees, leaf, depth):
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    trans=[]
    if numeric: trans.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median"))]),numeric))
    if categorical: trans.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical))
    pre=ColumnTransformer(trans)
    if algorithm=="Extra Trees": reg=ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=leaf,max_depth=None if depth==0 else depth,random_state=42,n_jobs=-1)
    elif algorithm=="Gradient Boosting": reg=GradientBoostingRegressor(n_estimators=min(trees,300),learning_rate=.05,max_depth=2 if depth==0 else min(depth,5),random_state=42,loss="huber")
    else: reg=RandomForestRegressor(n_estimators=trees,min_samples_leaf=leaf,max_depth=None if depth==0 else depth,random_state=42,n_jobs=-1)
    return Pipeline([("preprocessor",pre),("regressor",reg)]),numeric,categorical

@st.cache_resource(show_spinner="Training and evaluating models…")
def train_all(X, y, algorithm, trees, leaf, depth, test_size):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42)
    model,num,cat=make_pipeline(X,algorithm,trees,leaf,depth)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    baseline=DummyRegressor(strategy="mean").fit(Xtr,ytr); basepred=baseline.predict(Xte)
    cvn=min(5,len(X))
    cv=cross_val_score(model,X,y,cv=KFold(cvn,shuffle=True,random_state=42),scoring="r2",n_jobs=None) if cvn>=2 else np.array([np.nan])
    # Original feature importance, combining one-hot categories back to their source feature.
    transformed=model.named_steps["preprocessor"].get_feature_names_out()
    raw_imp=getattr(model.named_steps["regressor"],"feature_importances_",None)
    importance={}
    if raw_imp is not None:
        for name,val in zip(transformed,raw_imp):
            source=name.split("__",1)[-1]
            for col in X.columns:
                if source==col or source.startswith(col+"_") or source.startswith(col+"_"):
                    importance[col]=importance.get(col,0)+float(val); break
        for c in X.columns: importance.setdefault(c,0.0)
    errors=np.asarray(yte)-np.asarray(pred)
    return {"model":model,"numeric":num,"categorical":cat,"X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,"preds":pred,"errors":errors,
        "mae":mean_absolute_error(yte,pred),"rmse":float(np.sqrt(mean_squared_error(yte,pred))),"r2":r2_score(yte,pred),
        "train_r2":r2_score(ytr,model.predict(Xtr)),"cv_mean":float(np.nanmean(cv)),"cv_std":float(np.nanstd(cv)),
        "baseline_mae":mean_absolute_error(yte,basepred),"importance":importance,"cv_scores":cv}

def compare_models(X,y,test_size):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42)
    numeric=X.select_dtypes(include=np.number).columns.tolist(); categorical=[c for c in X.columns if c not in numeric]
    rows=[]
    for name,reg in [
        ("Random Forest",RandomForestRegressor(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=-1)),
        ("Extra Trees",ExtraTreesRegressor(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=-1)),
        ("Gradient Boosting",GradientBoostingRegressor(n_estimators=160,learning_rate=.05,max_depth=2,loss="huber",random_state=42))]:
        trans=[]
        if numeric: trans.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median"))]),numeric))
        if categorical: trans.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical))
        pipe=Pipeline([("preprocessor",ColumnTransformer(trans)),("regressor",reg)])
        pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
        rows.append({"Model":name,"R²":r2_score(yte,p),"MAE":mean_absolute_error(yte,p),"RMSE":np.sqrt(mean_squared_error(yte,p))})
    return pd.DataFrame(rows).sort_values("R²",ascending=False)

# ------------------------------ sidebar ------------------------------
with st.sidebar:
    st.markdown('<div style="font-size:2.1rem">🚘</div><div style="font-family:Space Grotesk;font-size:1.65rem;font-weight:800">AutoValue <span style="color:#d8ff54">/ Studio</span></div><div style="font-size:.72rem;letter-spacing:.14em;color:#b6b0c4">USED-CAR INTELLIGENCE</div>',unsafe_allow_html=True)
    st.markdown("---")
    uploaded=st.file_uploader("Drop your dataset",type=["csv"],help="Upload a CarDekho-style CSV. If omitted, the app checks for a local CSV.")
    local=next((p for p in DEFAULT_PATHS if p.exists()),None)
    page=st.radio("WORKSPACE",["Overview","Explore data","Data preparation","Model lab","Prediction studio","Prediction audit","Project report"],label_visibility="visible")
    st.markdown("---")
    st.caption("PROJECT SETTINGS")
    algorithm=st.selectbox("Main model",["Random Forest","Extra Trees","Gradient Boosting"])
    trees=st.slider("Number of estimators",50,400,200,25)
    leaf=st.slider("Minimum samples per leaf",1,8,2)
    depth=st.select_slider("Tree depth",options=[0,4,6,8,12,16],value=0,format_func=lambda x:"Unlimited" if x==0 else str(x))
    test_size=0.20  # Fixed 80/20 split; keeps the sidebar simple.
    st.caption("Palette: dark rose · midnight blue · deep plum")
    st.markdown("[Dataset source ↗](%s)"%SOURCE_URL)

if uploaded is not None:
    raw=read_csv_bytes(uploaded.getvalue())
elif local is not None:
    raw=read_csv_bytes(local.read_bytes())
else:
    raw=None

if raw is None:
    st.markdown('<div class="hero"><div class="eyebrow">MACHINE LEARNING · REGRESSION · MODEL INTELLIGENCE</div><h1>Used-car pricing, with a little more personality.</h1><p>A complete ML workspace: explore your dataset, inspect data quality, compare models, evaluate prediction errors, and estimate resale prices with a transparent uncertainty band.</p><span class="pill">MODEL BENCHMARKS</span><span class="pill">CROSS-VALIDATION</span><span class="pill">FEATURE IMPORTANCE</span><span class="pill">LIVE PREDICTIONS</span></div>',unsafe_allow_html=True)
    st.markdown("### First, bring in your dataset")
    st.write("Use the upload control in the sidebar to add your CarDekho CSV. The target column must be named `selling_price` (column names such as `Selling Price` are also recognised).")
    st.info("No dataset is bundled into this app. Upload your CSV to calculate model evaluation metrics.")
    st.markdown(f"**Suggested dataset:** [Vehicle Dataset from CarDekho on Kaggle]({SOURCE_URL})")
    st.stop()

try:
    norm,cleaned,X,y,info=clean_data(raw,REF_YEAR)
    if len(X)<10: raise ValueError("At least 10 usable rows are recommended for a meaningful train/test evaluation.")
    R=train_all(X,y,algorithm,trees,leaf,depth,test_size)
except Exception as e:
    st.error(f"Couldn't prepare the dataset: {e}")
    st.stop()

lakh=float(y.max())<1000
unit="₹ lakh" if lakh else "₹"
def cash(v): return money(float(v),lakh)
test_actual=np.asarray(R["y_test"]); test_pred=np.asarray(R["preds"]); abs_err=np.abs(test_actual-test_pred)
improvement=(1-R["mae"]/R["baseline_mae"])*100 if R["baseline_mae"] else 0
quality="Excellent" if R["r2"]>=.9 else "Strong" if R["r2"]>=.75 else "Moderate" if R["r2"]>=.5 else "Needs work"
audit=pd.DataFrame({"Actual price":test_actual,"Predicted price":test_pred,"Absolute error":abs_err,"Signed error (actual − predicted)":R["errors"],"Absolute error %":abs_err/np.maximum(np.abs(test_actual),1e-9)*100},index=R["y_test"].index)
audit.insert(0,"Record index",audit.index)

# ------------------------------- overview -------------------------------
if page=="Overview":
    st.markdown(f'<div class="hero"><div class="eyebrow">AUTO INTELLIGENCE / PROJECT DASHBOARD</div><h1>Know the value.<br>Understand the model.</h1><p>From raw vehicle records to an evaluated machine-learning model. Explore the market, see what drives price, and test the prediction engine.</p><span class="pill">● DATA LOADED</span><span class="pill">{algorithm.upper()}</span><span class="pill">{len(cleaned):,} CLEAN RECORDS</span></div>',unsafe_allow_html=True)
    kpis([("TEST R²",f"{R['r2']:.3f}","Regression score on unseen cars"),("MAE",cash(R["mae"]),"Average absolute miss on unseen cars"),("RMSE",cash(R["rmse"]),"Penalises larger prediction errors"),("DATASET",f"{len(cleaned):,}",f"{X.shape[1]} input features · {len(norm):,} original rows")])
    left,right=st.columns([1.35,1],gap="large")
    with left:
        heading("Performance at a glance","Model performance is measured on a held-out test set.")
        fig,ax=plt.subplots(figsize=(7,4))
        vals=[R["r2"],R["cv_mean"]]
        ax.barh(["Test R²","5-fold CV R²"],vals,color=[LIME,VIOLET],height=.48)
        ax.axvline(0,color="#b6b0c4",lw=1); ax.set_xlabel("R² score"); ax.set_xlim(min(-.2,min(vals)-.1),max(1.05,max(vals)+.1))
        figshow(fig)
        observation(f"{quality} fit by R² classification. The model reduces MAE by {improvement:.1f}% compared with always guessing the training-set average.")
    with right:
        heading("Your model snapshot")
        st.markdown(f'<div class="panel"><div class="eyebrow">ACTIVE ESTIMATOR</div><h2>{algorithm}</h2><p>🌳 {trees} estimators<br>🧩 Minimum leaf size: {leaf}<br>🧪 5-fold CV R²: {R["cv_mean"]:.3f} ± {R["cv_std"]:.3f}</p></div>',unsafe_allow_html=True)
        st.metric("Training R²",f"{R['train_r2']:.3f}",delta=f"{R['train_r2']-R['r2']:+.3f} train-test gap")
        st.caption("A large train-test gap can indicate overfitting.")
    heading("Navigate the workspace")
    cards=st.columns(3)
    for col,title,desc in zip(cards,["Explore data","Model lab","Prediction studio"],["Distribution, correlations, category patterns and missingness.","Benchmark three algorithms and inspect error behaviour.","Enter a car profile and get a price estimate with a data-driven error band."]):
        with col:
            st.markdown(f'<div class="panel"><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)

# ------------------------------ explore data ------------------------------
elif page=="Explore data":
    st.title("Explore the market")
    st.caption("Explore before modelling: quality checks, distributions, relationships, and category differences.")
    kpis([("ROWS",f"{len(norm):,}","Original uploaded rows"),("MISSING CELLS",f"{info['missing_cells']:,}","Across original dataset"),("DUPLICATES",f"{info['duplicates']:,}","Exact duplicate rows"),("PRICE MEDIAN",cash(y.median()),"Robust centre of selling prices")])
    t1,t2,t3,t4=st.tabs(["Distributions","Relationships","Categories","Data quality"])
    with t1:
        c1,c2=st.columns(2)
        with c1:
            heading("Selling-price distribution")
            fig,ax=plt.subplots(figsize=(6,4)); sns.histplot(y,bins=28,kde=True,color=LIME,ax=ax); ax.axvline(y.median(),color=PINK,ls="--",label="Median"); ax.legend(); ax.set_xlabel(f"Selling price ({unit})"); figshow(fig)
        with c2:
            nums=cleaned.select_dtypes(include=np.number).columns.tolist()
            selected=st.selectbox("Choose a numeric variable",nums,index=nums.index("selling_price") if "selling_price" in nums else 0)
            heading(f"{pretty(selected)}")
            fig,ax=plt.subplots(figsize=(6,4)); sns.histplot(cleaned[selected].dropna(),bins=24,kde=True,color=VIOLET,ax=ax); ax.set_xlabel(pretty(selected)); figshow(fig)
        q1,q3=y.quantile(.25),y.quantile(.75); out=int(((y<q1-1.5*(q3-q1))|(y>q3+1.5*(q3-q1))).sum())
        observation(f"Mean price is {cash(y.mean())}; median is {cash(y.median())}. IQR rule flags {out:,} selling-price outliers. Outliers are not automatically errors.")
    with t2:
        nums=cleaned.select_dtypes(include=np.number).columns.tolist()
        if len(nums)>1:
            corr=cleaned[nums].corr()
            fig,ax=plt.subplots(figsize=(9,5)); sns.heatmap(corr,annot=True,fmt=".2f",cmap="rocket",center=0,ax=ax); figshow(fig)
            c=corr["selling_price"].drop("selling_price").sort_values(key=lambda s:s.abs(),ascending=False)
            show_df(pd.DataFrame({"Feature":[pretty(i) for i in c.index],"Correlation with price":c.values.round(3)}),hide_index=True)
            best=c.index[0]
            fig,ax=plt.subplots(figsize=(8,4)); sns.regplot(data=cleaned,x=best,y="selling_price",scatter_kws={"alpha":.55,"color":VIOLET},line_kws={"color":LIME},ax=ax); ax.set_xlabel(pretty(best)); ax.set_ylabel(f"Selling price ({unit})"); figshow(fig)
            observation(f"{pretty(best)} has the strongest absolute numeric correlation with price (r = {c.iloc[0]:.2f}). Correlation is association, not proof of causation.")
        else: st.info("Not enough numeric columns for a correlation view.")
    with t3:
        cats=[c for c in X.columns if X[c].dtype=="object" and X[c].nunique()<=20]
        if cats:
            cat=st.selectbox("Compare price across",cats,index=cats.index("fuel_type") if "fuel_type" in cats else 0)
            grp=cleaned.groupby(cat,dropna=False).selling_price.agg(["median","mean","count"]).sort_values("median",ascending=False)
            fig,ax=plt.subplots(figsize=(8,4)); sns.boxplot(data=cleaned,x=cat,y="selling_price",color=VIOLET,ax=ax); ax.set_ylabel(f"Selling price ({unit})"); figshow(fig)
            show_df(grp.rename(columns={"median":"Median price","mean":"Mean price","count":"Cars"}).round(2))
            observation(f"{grp.index[0]} has the highest median price in this dataset. Category sizes matter: very small groups may not be representative.")
        else: st.info("No low-cardinality categorical columns were detected.")
    with t4:
        missing=pd.DataFrame({"Column":norm.columns,"Missing":norm.isna().sum().values,"Missing %":(norm.isna().mean().values*100).round(2),"Unique":norm.nunique().values})
        show_df(missing.sort_values("Missing",ascending=False),hide_index=True)
        st.markdown("**Cleaning actions**")
        st.write(f"- Removed {info['duplicates']} duplicate row(s).")
        st.write(f"- Removed {info['bad_target_rows']} row(s) with missing or non-positive target price.")
        st.write("- Filled numeric gaps with the median and categorical gaps with the most frequent value inside the training pipeline.")
        st.write(f"- Converted manufacture year to car age using reference year {REF_YEAR}, when a year column was present.")
        if info["dropped"]: show_df(pd.DataFrame(info["dropped"],columns=["Column","Reason"]),hide_index=True)

# ------------------------------ preparation ------------------------------
elif page=="Data preparation":
    st.title("Data preparation lab")
    st.caption("See how the raw table becomes a model-ready feature matrix.")
    kpis([("RAW ROWS",f"{len(norm):,}","Before cleaning"),("MODEL ROWS",f"{len(cleaned):,}","After duplicate and target checks"),("FEATURES",f"{X.shape[1]}","Columns used as inputs"),("ENCODING","One-hot","Categorical text converted to indicators")])
    steps=[("01","Normalize column names","Consistent lowercase names and aliases."),("02","Remove duplicate records","Avoid repeated observations dominating the fit."),("03","Validate target","Only positive, known selling prices are used."),("04","Engineer car age","Manufacture year becomes age in years."),("05","Impute missing values","Median for numeric; most frequent for categories."),("06","Encode categories","One-hot encoding with unseen-category safety."),("07","Split before fitting","Test data remains held out from preprocessing."),("08","Fit model","The estimator learns patterns from training rows.")]
    for no,title,desc in steps: st.markdown(f'<div class="step"><span class="step-no">{no}</span><div><b>{title}</b><br><span style="color:#b6b0c4">{desc}</span></div></div>',unsafe_allow_html=True)
    heading("Feature dictionary")
    roles=[]
    for c in norm.columns:
        role="Target" if c=="selling_price" else "Converted to car age" if c=="year" else "Input feature" if c in X.columns else "Not used"
        roles.append(role)
    show_df(pd.DataFrame({"Column":norm.columns,"Data type":[str(norm[c].dtype) for c in norm.columns],"Unique values":[norm[c].nunique() for c in norm.columns],"Missing":[norm[c].isna().sum() for c in norm.columns],"Role":roles}),hide_index=True)
    a,b=st.columns(2)
    with a: heading("Raw preview"); show_df(raw.head(8),hide_index=True)
    with b: heading("Cleaned preview"); show_df(cleaned.head(8),hide_index=True)
    st.download_button("Download cleaned dataset",cleaned.to_csv(index=False).encode("utf-8"),"autovalue_cleaned.csv","text/csv")

# -------------------------------- model lab -------------------------------
elif page=="Model lab":
    st.title("Model lab")
    st.caption("Compare algorithms, tune the active estimator from the sidebar, and inspect where errors happen.")
    with st.spinner("Benchmarking Random Forest, Extra Trees and Gradient Boosting…"):
        bench=compare_models(X,y,test_size)
    kpis([("ACTIVE MODEL",algorithm,f"{trees} estimators · leaf {leaf}"),("TEST R²",f"{R['r2']:.3f}","Standard regression score"),("TEST MAE",cash(R["mae"]),"Average absolute prediction error"),("TEST RMSE",cash(R["rmse"]),"Penalises larger prediction errors")])
    heading("Algorithm leaderboard","Compare the models using R², mean absolute error (MAE), and root mean squared error (RMSE).")
    display=bench.copy(); display["MAE"]=display.MAE.map(cash); display["RMSE"]=display.RMSE.map(cash); display["R²"]=display["R²"].map(lambda v:f"{v:.3f}")
    show_df(display,hide_index=True)
    fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(data=bench,x="R²",y="Model",palette=[LIME,VIOLET,PINK],ax=ax); ax.set_xlim(min(-.1,bench["R²"].min()-.05),max(1,bench["R²"].max()+.05)); figshow(fig)
    best=bench.iloc[0]
    observation(f"{best['Model']} is the leader on this split by R² ({best['R²']:.3f}). Your active model is {algorithm}; use the sidebar to switch it and review the metrics.")
    heading("Prediction diagnostics")
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(test_actual,test_pred,alpha=.7,color=VIOLET,edgecolors="#17151e"); lim=[min(test_actual.min(),test_pred.min()),max(test_actual.max(),test_pred.max())]; ax.plot(lim,lim,color=LIME,ls="--"); ax.set_xlabel("Actual price"); ax.set_ylabel("Predicted price"); figshow(fig)
        st.caption("Points closer to the diagonal indicate better predictions.")
    with c2:
        fig,ax=plt.subplots(figsize=(6,4)); sns.histplot(R["errors"],bins=22,kde=True,color=PINK,ax=ax); ax.axvline(0,color=LIME,ls="--"); ax.set_xlabel("Actual − predicted"); figshow(fig)
        st.caption("A centred error distribution is preferable; long tails mean occasional large misses.")
    importance=pd.Series(R["importance"]).sort_values(ascending=True)
    if len(importance):
        heading("What drives the model")
        fig,ax=plt.subplots(figsize=(8,4)); ax.barh([pretty(c) for c in importance.index],importance.values,color=VIOLET); ax.set_xlabel("Aggregated feature importance"); figshow(fig)
        st.caption("Feature importance describes how the tree model used features, not whether they cause price changes.")

# --------------------------- prediction studio ---------------------------
elif page=="Prediction studio":
    st.title("Prediction studio")
    st.caption("Build a car profile. The estimate is generated by the trained model, not a hard-coded example.")
    user={}
    labelmap={"present_price":"Showroom price","car_age":"Manufacture year","kms_driven":"Kilometres driven","fuel_type":"Fuel type","seller_type":"Seller type","transmission":"Transmission","owner":"Previous owners"}
    with st.form("car_profile"):
        groups=[("CAR PROFILE",["present_price","car_age","fuel_type","transmission"]),("USAGE & OWNERSHIP",["kms_driven","owner"]),("SALE CONTEXT",["seller_type"])]
        for title,cols in groups:
            cols=[c for c in cols if c in X.columns]
            if cols:
                heading(title)
                grid=st.columns(2)
                for i,c in enumerate(cols):
                    with grid[i%2]:
                        s=X[c]
                        label=labelmap.get(c,pretty(c))
                        if c=="car_age":
                            med=float(pd.to_numeric(s,errors="coerce").median()); lo=float(pd.to_numeric(s,errors="coerce").min()); hi=float(pd.to_numeric(s,errors="coerce").max())
                            year=st.number_input(label,min_value=max(1950,int(REF_YEAR-hi)),max_value=REF_YEAR,value=int(np.clip(REF_YEAR-med,max(1950,int(REF_YEAR-hi)),REF_YEAR)),step=1,help="Year the car was manufactured.")
                            user[c]=REF_YEAR-year
                        elif not pd.api.types.is_numeric_dtype(s) or s.nunique()<=6:
                            opts=sorted(s.dropna().unique().tolist(),key=str)
                            mode=s.mode().iloc[0] if not s.mode().empty else opts[0]
                            user[c]=st.selectbox(label,opts,index=opts.index(mode) if mode in opts else 0)
                        else:
                            vals=pd.to_numeric(s,errors="coerce").dropna(); med=float(vals.median()); lo=float(vals.min()); hi=float(vals.max())
                            step=1000.0 if "km" in c else max((hi-lo)/100,0.1)
                            user[c]=st.number_input(label,min_value=0.0,value=med,step=float(step),help=f"Observed dataset range: {lo:,.1f}–{hi:,.1f}")
        extra=[c for c in X.columns if c not in user]
        for c in extra:
            with st.expander(f"Additional feature: {pretty(c)}"):
                s=X[c]
                if s.dtype=="object" or s.nunique()<=6:
                    opts=sorted(s.dropna().unique().tolist(),key=str); user[c]=st.selectbox(pretty(c),opts,key=f"extra_{c}")
                else:
                    vals=pd.to_numeric(s,errors="coerce").dropna(); user[c]=st.number_input(pretty(c),min_value=0.0,value=float(vals.median()),key=f"extra_{c}")
        submitted=st.form_submit_button("✦ Generate valuation",use_container_width=True)
    if submitted:
        # Build a complete prediction row without assuming text columns use dtype "object".
        # Pandas may infer text as StringDtype on another computer/version.
        row_values = {}
        for c in X.columns:
            if c in user and user[c] is not None:
                row_values[c] = user[c]
                continue
            series = X[c]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                row_values[c] = float(numeric.median()) if not numeric.empty else 0.0
            else:
                modes = series.mode(dropna=True)
                row_values[c] = modes.iloc[0] if not modes.empty else ""
        row = pd.DataFrame([row_values], columns=X.columns)
        estimate=max(0,float(R["model"].predict(row)[0]))
        # Empirical absolute residual quantiles form a transparent, approximate interval.
        lo_err=float(np.quantile(abs_err,.80)) if len(abs_err) else R["mae"]
        band=max(lo_err,R["mae"])
        low=max(0,estimate-band); high=estimate+band
        st.markdown(f'<div class="hero"><div class="eyebrow">MODEL ESTIMATE · NOT A GUARANTEED MARKET QUOTE</div><h1>{cash(estimate)}</h1><p>Estimated resale value based on the profile you entered. Approximate error band: {cash(low)} – {cash(high)}.</p><span class="pill">R² {R["r2"]:.3f}</span><span class="pill">MAE {cash(R["mae"])}</span></div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Estimated price",cash(estimate))
        if "present_price" in user and user["present_price"]>0: c2.metric("Of showroom price",f"{estimate/user['present_price']*100:.1f}%")
        else: c2.metric("Model R²",f"{R['r2']:.3f}")
        c3.metric("Approximate error band",f"±{cash(band)}")
        st.markdown("#### Your profile")
        profile=[]
        for c,v in user.items():
            shown=str(v)
            if c=="car_age": shown=str(int(REF_YEAR-v))
            elif c in ("present_price","selling_price"): shown=cash(v)
            elif c=="kms_driven": shown=f"{v:,.0f} km"
            profile.append({"Feature":pretty(c),"Entered value":shown})
        show_df(pd.DataFrame(profile),hide_index=True)
        st.download_button("Download this valuation",pd.DataFrame([{"estimated_price":estimate,"lower_estimate":low,"upper_estimate":high,**user}]).to_csv(index=False).encode("utf-8"),"autovalue_prediction.csv","text/csv")
        st.caption("The interval is based on held-out prediction errors and is a rough uncertainty guide, not a formal statistical guarantee. Condition, accident history, location and current demand may affect real prices.")

# ------------------------------ audit page -------------------------------
elif page=="Prediction audit":
    st.title("Prediction audit")
    st.caption("Inspect test-set predictions individually. Download the full audit table for your report.")
    audit=pd.DataFrame({"Actual price":test_actual,"Predicted price":test_pred,"Absolute error":abs_err,"Signed error (actual − predicted)":R["errors"],"Absolute error %":abs_err/np.maximum(np.abs(test_actual),1e-9)*100},index=R["y_test"].index)
    audit.insert(0,"Record index",audit.index)
    filter_mode=st.selectbox("Show records",["All test records","Largest errors first"])
    view=audit.copy()
    if filter_mode=="Largest errors first": view=view.sort_values("Absolute error",ascending=False)
    kpis([("TEST RECORDS",f"{len(audit):,}","Held-out rows"),("TEST R²",f"{R['r2']:.3f}","Regression score"),("TEST MAE",cash(R["mae"]),"Average absolute error"),("MEDIAN ABS ERROR",cash(np.median(abs_err)),"Median absolute miss")])
    show_df(view.round(3),hide_index=True)
    st.download_button("Download full prediction audit (CSV)",audit.to_csv(index=False).encode("utf-8"),"autovalue_prediction_audit.csv","text/csv")
    st.markdown("**How to interpret this:** This is a regression task. R² describes how well the model explains variation in selling prices, while MAE shows the average absolute difference between predicted and actual prices.")

# ------------------------------ report page ------------------------------
elif page=="Project report":
    st.title("Project report")
    st.caption("A presentation-ready summary of the project and its measured results.")
    heading("Project objective")
    st.markdown('<div class="panel">Build a supervised machine-learning regression system that estimates the resale price of a used car from its available specifications and usage history. The system cleans the data, explores patterns, trains a model, and evaluates predictions on unseen records.</div>',unsafe_allow_html=True)
    heading("Dataset and methodology")
    st.write(f"The uploaded dataset contains {len(norm):,} original records and {norm.shape[1]} columns. After removing duplicates and invalid target prices, {len(cleaned):,} records remain. The model uses {X.shape[1]} features. Numeric missing values are imputed with the median; categorical values are imputed with the most frequent category and one-hot encoded. The data is split into {1-test_size:.0%} training and {test_size:.0%} testing subsets.")
    heading("Measured results")
    kpis([("R²",f"{R['r2']:.3f}","Held-out test set"),("MAE",cash(R["mae"]),"Average absolute prediction error"),("RMSE",cash(R["rmse"]),"Penalises larger errors"),("CROSS-VALIDATION R²",f"{R['cv_mean']:.3f}","Mean score across 5 folds")])
    st.write(f"The selected {algorithm} model achieved a test R² score of {R['r2']:.3f}. Its mean absolute error was {cash(R['mae'])}, and its root mean squared error was {cash(R['rmse'])}. The 5-fold cross-validation mean R² was {R['cv_mean']:.3f} with standard deviation {R['cv_std']:.3f}. Compared with a mean-price baseline, MAE changed by {improvement:.1f}%.")
    heading("Limitations and future scope")
    st.markdown("- Model quality depends on dataset coverage and how representative the test set is.\n- Important factors such as accident history, vehicle condition, service records, region, and changing market demand may be missing.\n- R² is a regression score, not percentage accuracy.\n- Future work can add more recent data, compare more algorithms, tune parameters systematically, and monitor errors over time.")
    report=f"""AUTOVALUE STUDIO — USED CAR PRICE PREDICTION\n\nObjective\nBuild a supervised regression model to estimate used-car selling prices from available vehicle features.\n\nDataset\nOriginal records: {len(norm)}\nRecords after cleaning: {len(cleaned)}\nInput features: {X.shape[1]}\nTarget: selling_price\n\nMethod\nDuplicate and invalid target checks; year-to-age feature engineering when available; median imputation for numeric fields; most-frequent imputation and one-hot encoding for categorical fields; {1-test_size:.0%}/{test_size:.0%} train/test split; {algorithm} regression.\n\nResults\nTest R2: {R['r2']:.4f}\nMAE: {R['mae']:.4f}\nRMSE: {R['rmse']:.4f}\nTraining R2: {R['train_r2']:.4f}\n5-fold CV R2: {R['cv_mean']:.4f} +/- {R['cv_std']:.4f}\nBaseline MAE: {R['baseline_mae']:.4f}\nMAE improvement over baseline: {improvement:.2f}%\n\nLimitations\nThe model cannot account for missing factors such as condition, accident history, location, and real-time demand. R² is a regression score, not percentage accuracy.\n"""
    st.download_button("Download project results (.txt)",report,"autovalue_project_report.txt","text/plain")
    st.download_button("Download test predictions (.csv)",audit.to_csv(index=False).encode("utf-8"),"autovalue_test_predictions.csv","text/csv")
