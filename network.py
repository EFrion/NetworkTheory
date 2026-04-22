import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import GraphicalLasso
from sklearn.decomposition import PCA
import yfinance as yf

tickers_list=["AMAT","LRCX","BMO", "SLF", "AMT","PLD","^GSPC","XLK", "XLF", "XLRE","SHY"]

# Rename columns
column_mapping = {
    "AMAT": "Applied Materials",
    "LRCX": "Lam Research",
    "BMO": "Bank of Montreal",
    "SLF": "SunLife Insurance",
    "AMT": "American Tower",
    "PLD": "Prologis",
    "^GSPC": "SP500",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLRE": "Real Estate",
    "SHY": "Treasury"
}

# Assign types to nodes
node_types = {
    "Applied Materials":'Asset',
    "Lam Research":'Asset',
    "SP500":'Factor',
    "Bank of Montreal":'Asset',
    "SunLife Insurance":'Asset',
    "American Tower":'Asset',
    "Technology":'Factor',
    "Prologis":'Asset',
    "Financials":'Factor',
    "Real Estate":'Factor',
    "Treasury":'Factor',
}

class DataManager:
    """Handles data downloading, loading, and preprocessing."""
 
    def __init__(self, csv_path: str, tickers: list[str],
                 start: str = "2007-01-01", end: str = "2026-04-21"):
        self.csv_path = csv_path
        self.tickers = tickers
        self.start = start
        self.end = end
        self.df: pd.DataFrame = None
        self.returns: pd.DataFrame = None
 
    def load(self) -> None:
        """Load data from CSV, downloading it first if necessary."""
        if not os.path.exists(self.csv_path):
            self._download()
        self.df = pd.read_csv(self.csv_path, index_col="Date", parse_dates=True).dropna()
        self.df = self.df.rename(columns=column_mapping)
        self.returns = self.df.pct_change().dropna()
 
    def _download(self) -> None:
        """Download historical close prices from Yahoo Finance and save to CSV."""
        import yfinance as yf
        print(f"'{self.csv_path}' not found. Downloading data from Yahoo Finance...")
        prices = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )
        if prices.empty:
            raise RuntimeError("No data was downloaded from Yahoo Finance.")
        close_prices = prices.xs("Close", axis=1, level=1)
        close_prices.to_csv(self.csv_path, index=True)
        print(f"Data saved to '{self.csv_path}'.")



class AssetNetwork:
    """Base class for building and visualising asset networks."""
 
    def __init__(self, returns: pd.DataFrame, asset_columns: list[str]):
        self.returns = returns
        self.assets = returns[asset_columns]
        self.asset_columns = asset_columns
        self.adj_matrix: np.ndarray = None
        self.graph: nx.Graph = None
 
    def build(self) -> None:
        raise NotImplementedError
 
    def _to_graph(self, directed: bool = False) -> nx.Graph:
        cls = nx.DiGraph if directed else nx.Graph
        G = nx.from_numpy_array(self.adj_matrix, create_using=cls)
        mapping = {i: name for i, name in enumerate(self.asset_columns)}
        return nx.relabel_nodes(G, mapping)
 
    def visualise(self, title: str, output_path: str,
                  node_colors: list[str] | str = "lightblue",
                  directed: bool = False,
                  legend_handles: list | None = None) -> None:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(
            self.graph, pos,
            node_color=node_colors,
            node_size=500,
            with_labels=True,
            font_size=10,
            font_weight="bold",
            edge_color="gray",
            arrows=directed,
            arrowsize=15,
        )
        if legend_handles:
            plt.legend(handles=legend_handles, loc="upper right")
        plt.title(title, fontsize=14)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


class LinearRegressionNetwork(AssetNetwork):
    """Directed network built from pairwise OLS regressions."""
 
    def __init__(self, returns: pd.DataFrame, asset_columns: list[str],
                 confounder: pd.Series | None = None):
        super().__init__(returns, asset_columns)
        self.confounder = confounder
 
    def build(self) -> None:
        n = len(self.asset_columns)
        self.adj_matrix = np.zeros((n, n))
        p_threshold = 0.05 / (n * (n - 1))  # Bonferroni correction
 
        for i, asset_i in enumerate(self.asset_columns):
            for j, asset_j in enumerate(self.asset_columns):
                if i == j: # Skip self-relationships
                    continue
                if self.confounder is not None:
                    indep = pd.concat([self.assets[asset_j], self.confounder], axis=1)
                else:
                    indep = self.assets[[asset_j]]
                X = sm.add_constant(indep)
                model = sm.OLS(self.assets[asset_i], X).fit()
                p_value = model.pvalues.iloc[1]
                self.adj_matrix[i, j] = 1 if p_value < p_threshold else 0
 
        self.graph = self._to_graph(directed=True)
 
    def run(self, title: str, output_path: str) -> None:
        self.build()
        print(f"\nAdjacency matrix — {title}:\n{self.adj_matrix}")
        self.visualise(title, output_path, directed=True)



class GraphicalLassoNetwork(AssetNetwork):
    """Undirected network built from a Graphical LASSO partial-correlation matrix."""
 
    def __init__(self, returns: pd.DataFrame, asset_columns: list[str],
                 node_types: dict[str, str],
                 rho_range: tuple[float, float] = (0.3, 1.0),
                 n_rhos: int = 100,
                 cv_folds: int = 5,
                 threshold: float = 0.0):
        # Include all columns (assets + factors) in the graph
        super().__init__(returns, list(returns.columns))
        self.asset_columns_only = asset_columns
        self.node_types = node_types
        self.rho_range = rho_range
        self.n_rhos = n_rhos
        self.cv_folds = cv_folds
        self.threshold = threshold
        self.best_rho: float = None
        self.partial_corr: np.ndarray = None
 
    def build(self) -> None:
        standardized = (self.returns - self.returns.mean()) / self.returns.std()
        rhos = np.linspace(*self.rho_range, self.n_rhos)
 
        cv_model = GraphicalLassoCV(alphas=rhos, cv=self.cv_folds, tol=1e-2, max_iter=400)
        cv_model.fit(standardized)
        self.best_rho = cv_model.alpha_
        print(f"\nBest ρ (Graphical LASSO): {self.best_rho:.4f}")
 
        model = GraphicalLasso(alpha=self.best_rho, max_iter=400, tol=1e-2)
        model.fit(standardized)
 
        precision = model.precision_ # Inverse covariance
        d = np.sqrt(np.diag(precision)) # Convert precision matrix to partial correlation matrix for easier interpretation
        self.partial_corr = -precision / np.outer(d, d)
        np.fill_diagonal(self.partial_corr, 0) # Remove self-loops
        print(f"\nPartial correlation matrix:\n{self.partial_corr}")
 
        self.adj_matrix = (np.abs(self.partial_corr) > self.threshold).astype(int)
        print(f"\nAdjacency matrix:\n{self.adj_matrix}")
 
        self.graph = self._to_graph(directed=False)
        nx.set_node_attributes(self.graph, self.node_types, "type")
 
    def run(self, title: str, output_path: str) -> None:
        self.build()
        node_colors = [
            "skyblue" if self.node_types.get(n) == "Asset" else "orange"
            for n in self.graph.nodes()
        ]
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="skyblue",
                       markersize=15, label="Asset"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
                       markersize=15, label="Factor"),
        ]
        self.visualise(title, output_path,
                       node_colors=node_colors,
                       directed=False,
                       legend_handles=legend_handles)





def main():
    # Data
    data = DataManager(csv_path="data_gwp.csv", tickers=tickers_list)
    data.load()
    print(data.returns)
    
    
    # Create a dataframe that only includes assets
    assets_cols = ["Applied Materials","Lam Research","Bank of Montreal", "SunLife Insurance", "American Tower","Prologis"]

    # Simple linear regression network
    lin_net = LinearRegressionNetwork(data.returns, assets_cols)
    lin_net.run(
        title="Simple Linear Regression-Based Asset Network",
        output_path="linreg.png",
    )
 
    # Linear regression network with S&P 500 confounder
    lin_conf_net = LinearRegressionNetwork(
        data.returns, assets_cols, confounder=data.returns["SP500"]
    )
    lin_conf_net.run(
        title="Linear Regression-Based Asset Network with S&P500 as Confounder",
        output_path="linreg_confounder.png",
    )
 
    # Graphical LASSO network
    glasso_net = GraphicalLassoNetwork(
        data.returns, assets_cols, node_types=node_types
    )
    glasso_net.run(
        title="Factor-Asset Network via Graphical LASSO",
        output_path="graphical_lasso.png",
    )



if __name__ == "__main__":
    main()



