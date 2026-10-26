"""Reproducible educational sales-data generation, cleaning, analysis, and charts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
SEED = 42


def create_sample_dataset(path=DATA_PATH, n=160, seed=SEED):
    """Create fictional educational sales rows. This is synthetic, not company data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    categories = np.array(["Electronics", "Office Supplies", "Home Goods"])
    cat = rng.choice(categories, n, p=[0.34, 0.38, 0.28])
    unit_price = np.select([cat == "Electronics", cat == "Office Supplies", cat == "Home Goods"], [120, 24, 58]).astype(float)
    unit_price *= rng.uniform(0.72, 1.35, n)
    units = rng.integers(1, 9, n)
    discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], n, p=[.25,.25,.25,.15,.10])
    revenue = np.round(unit_price * units * (1-discount), 2)
    base_margin = np.select([cat=="Electronics",cat=="Office Supplies",cat=="Home Goods"],[.25,.34,.29])
    profit = np.round(revenue * (base_margin - discount - rng.uniform(.01,.06,n)),2)
    df = pd.DataFrame({
        "order_date": rng.choice(dates, n),
        "order_id": [f"ORD-{i+1:04d}" for i in range(n)],
        "customer_id": [f"CUST-{int(i):03d}" for i in rng.integers(1,56,n)],
        "region": rng.choice(["North","South","East","West"],n),
        "segment": rng.choice(["Consumer","Small Business","Enterprise"],n,p=[.55,.30,.15]),
        "category": cat,
        "units": units,
        "unit_price": np.round(unit_price,2),
        "discount": discount,
        "revenue": revenue,
        "profit": profit,
    })
    # Demonstrate realistic quality checks: three missing region labels and one duplicate row.
    df.loc[[7, 41, 99], "region"] = np.nan
    df = pd.concat([df, df.iloc[[12]]], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path,index=False)
    return df


def load_and_clean(path=DATA_PATH):
    """Read CSV, parse dates/numerics, remove exact duplicates and incomplete core rows."""
    df = pd.read_csv(path)
    audit = {"raw_rows":len(df), "raw_missing_cells":int(df.isna().sum().sum()), "exact_duplicates":int(df.duplicated().sum())}
    df = df.drop_duplicates().copy()
    df["order_date"] = pd.to_datetime(df["order_date"],errors="coerce")
    for col in ["units","unit_price","discount","revenue","profit"]:
        df[col] = pd.to_numeric(df[col],errors="coerce")
    required=["order_date","order_id","customer_id","region","category","units","unit_price","discount","revenue","profit"]
    audit["incomplete_core_rows_removed"] = int(df[required].isna().any(axis=1).sum())
    df=df.dropna(subset=required).copy()
    df=df[(df.units>0)&(df.unit_price>=0)&(df.revenue>=0)&df.discount.between(0,1)].copy()
    df["profit_margin"] = np.where(df.revenue.ne(0),df.profit/df.revenue,np.nan)
    df["month"] = df.order_date.dt.to_period("M").dt.to_timestamp()
    audit["clean_rows"]=len(df)
    audit["date_min"]=str(df.order_date.min().date())
    audit["date_max"]=str(df.order_date.max().date())
    return df,audit


def analyze(df):
    kpis={"revenue":float(df.revenue.sum()),"profit":float(df.profit.sum()),"orders":int(df.order_id.nunique()),"customers":int(df.customer_id.nunique()),"aov":float(df.revenue.sum()/df.order_id.nunique()),"mean_discount":float(df.discount.mean()),"profit_margin":float(df.profit.sum()/df.revenue.sum())}
    numeric=df[["units","unit_price","discount","revenue","profit"]].describe().T
    by_category=df.groupby("category").agg(orders=("order_id","nunique"),revenue=("revenue","sum"),profit=("profit","sum"))
    by_category["profit_margin"]=by_category.profit/by_category.revenue
    monthly=df.groupby("month").agg(revenue=("revenue","sum"),profit=("profit","sum"))
    corr=float(df[["discount","profit_margin"]].corr(method="spearman").iloc[0,1])
    return kpis,numeric,by_category,monthly,corr


def make_charts(df, output_dir=OUTPUT_DIR):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid",palette="deep")
    outputs=[]
    fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(df.revenue,bins=20,kde=True,ax=ax,color="#2A6F97"); ax.set(title="Order Revenue Distribution",xlabel="Revenue (fictional currency units)"); fig.tight_layout(); p=output_dir/"revenue_distribution.png"; fig.savefig(p,dpi=150); plt.close(fig); outputs.append(p)
    fig,ax=plt.subplots(figsize=(8,4)); sns.boxplot(data=df,x="category",y="profit",ax=ax); ax.tick_params(axis="x",rotation=15); ax.set(title="Profit by Product Category",ylabel="Profit"); fig.tight_layout(); p=output_dir/"profit_by_category.png"; fig.savefig(p,dpi=150); plt.close(fig); outputs.append(p)
    monthly=df.groupby("month",as_index=False).agg(revenue=("revenue","sum"),profit=("profit","sum")); fig,ax=plt.subplots(figsize=(8,4)); sns.lineplot(data=monthly,x="month",y="revenue",marker="o",ax=ax,color="#2A6F97"); ax.set(title="Monthly Revenue Trend",xlabel="Month",ylabel="Revenue"); fig.tight_layout(); p=output_dir/"monthly_revenue.png"; fig.savefig(p,dpi=150); plt.close(fig); outputs.append(p)
    fig,ax=plt.subplots(figsize=(7,5)); sns.heatmap(df[["units","unit_price","discount","revenue","profit"]].corr(),annot=True,cmap="vlag",center=0,fmt=".2f",ax=ax); ax.set_title("Numeric Feature Correlations"); fig.tight_layout(); p=output_dir/"correlation_heatmap.png"; fig.savefig(p,dpi=150); plt.close(fig); outputs.append(p)
    fig,ax=plt.subplots(figsize=(8,4)); sns.scatterplot(data=df,x="discount",y="profit_margin",hue="category",alpha=.75,ax=ax); ax.set(title="Discount vs Profit Margin",xlabel="Discount fraction",ylabel="Profit margin"); fig.tight_layout(); p=output_dir/"discount_profit_scatter.png"; fig.savefig(p,dpi=150); plt.close(fig); outputs.append(p)
    return outputs


def run_all():
    if not DATA_PATH.exists(): create_sample_dataset()
    df,audit=load_and_clean(); kpis,stats_table,category,monthly,rho=analyze(df); charts=make_charts(df)
    category.to_csv(OUTPUT_DIR/"category_summary.csv")
    stats_table.to_csv(OUTPUT_DIR/"descriptive_statistics.csv")
    return df,audit,kpis,stats_table,category,monthly,rho,charts

if __name__=="__main__":
    df,audit,kpis,stats_table,category,monthly,rho,charts=run_all()
    print("Audit:",audit); print("KPIs:",kpis); print("Spearman discount-margin rho:",round(rho,3))
    print("Category summary:\n",category.round(2)); print("Charts:",[p.name for p in charts])
