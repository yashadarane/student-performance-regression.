import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def plot_correlation(df):
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"}
    )

    plt.title("Feature Correlation Heatmap", fontsize=14, weight="bold")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred):
    plt.figure()

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)

    # Perfect prediction line
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values", weight="bold")

    plt.tight_layout()
    plt.show()


def plot_feature_relationships(df, target_col):
    features = df.drop(columns=[target_col]).columns

    for feature in features:
        plt.figure()

        sns.regplot(
            x=df[feature],
            y=df[target_col],
            scatter_kws={"alpha": 0.4},
            line_kws={"linewidth": 2}
        )

        plt.title(f"{feature} vs {target_col}", weight="bold")
        plt.xlabel(feature)
        plt.ylabel(target_col)

        plt.tight_layout()
        plt.show()


def plot_residuals(y_true, y_pred):
    """
    Residual plot with zero-line
    """
    residuals = y_true - y_pred

    plt.figure()

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)

    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot", weight="bold")

    plt.tight_layout()
    plt.show()


def plot_target_distribution(df, target_col):
    """
    Target distribution with KDE
    """
    plt.figure()

    sns.histplot(df[target_col], kde=True)

    plt.title(f"Distribution of {target_col}", weight="bold")
    plt.xlabel(target_col)

    plt.tight_layout()
    plt.show()