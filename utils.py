import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import streamlit as st

@st.cache_data(persist=True)
def load_data():
    match_df = pd.read_csv("https://raw.githubusercontent.com/Prashanthsri12/IPL/main/data/matches.csv")
    player_mapping = pd.read_csv("https://raw.githubusercontent.com/Prashanthsri12/IPL/main/data/Player_map.csv")
    team_mapping = pd.read_csv("https://raw.githubusercontent.com/Prashanthsri12/IPL/main/data/Team_map.csv")    
    delivery_df = pd.read_csv("https://raw.githubusercontent.com/Prashanthsri12/IPL/main/data/deliveries.csv")
    return match_df,player_mapping,team_mapping,delivery_df

def calculate_batsman_metrics(df):
    # Group by match_id, batter, and bowler before calculating metrics
    matchwise_stats = df.groupby(['match_id', 'batter']).agg(
        total_runs=('batsman_runs', 'sum'),
        total_balls=('ball', 'count'),
        dismissals=('dismissal', 'sum')
    ).reset_index()
    
    # Aggregate at batter-bowler level
    batsman_stats = df.groupby(['batter', 'bowler']).agg(
        total_runs=('batsman_runs', 'sum'),
        total_balls=('ball', 'sum'),
        total_innings=('match_id', 'count'),
        dismissals=('dismissal', 'sum')
    ).reset_index()
    
    # Additional Metrics
    batsman_stats['batting_average'] = batsman_stats.apply(
        lambda x: x['total_runs'] if x['dismissals'] == 0 else x['total_runs'] / x['dismissals'], axis=1
    )
    
    batsman_stats['strike_rate'] = (batsman_stats['total_runs'] / batsman_stats['total_balls']) * 100
    
    # Boundary count calculation
    boundaries = df[df['batsman_runs'].isin([4, 6])].groupby(['batter', 'bowler'])['batsman_runs'].count().reset_index()
    batsman_stats = batsman_stats.merge(boundaries, on=['batter', 'bowler'], how='left').fillna(0)
    batsman_stats.rename(columns={'batsman_runs': 'boundary_count'}, inplace=True)
    # batsman_stats['boundary_percentage'] = (batsman_stats['boundary_count'] / batsman_stats['total_runs']) * 100
    
    # Consistency Score: Percentage of innings with 30+ runs
    matchwise_stats['thirty_plus'] = matchwise_stats['total_runs'].apply(lambda x: 1 if x >= 30 else 0)
    consistency = matchwise_stats.groupby(['batter'])['thirty_plus'].sum().reset_index()
    consistency['consistency_score'] = (consistency['thirty_plus'] / matchwise_stats.groupby(['batter'])['match_id'].count().values) * 100
    consistency_bkp = matchwise_stats.copy()
    consistency = consistency.drop(columns=['thirty_plus'])
    
    # Merge consistency score back with batsman stats
    batsman_stats = batsman_stats.merge(consistency, on=['batter'], how='left')
    batsman_stats['consistency_score'] = batsman_stats['consistency_score'].fillna(0).infer_objects(copy=False)

    # Define weights for each attribute
    weights = {
        'total_runs': 0.2,
        'batting_average': 0.2,
        'strike_rate': 0.2,
        'boundary_count': 0.15,
        'consistency_score': 0.2,
        'dismissals': -0.15  # Lower dismissals are better
    }
    
    # Normalize and calculate scores
    for key in weights.keys():
        min_val = batsman_stats[key].min()
        max_val = batsman_stats[key].max()
        if weights[key] > 0:
            batsman_stats[key + '_score'] = batsman_stats[key].apply(lambda x: ((x - min_val) / (max_val - min_val)) * weights[key] if max_val > min_val else weights[key] * 0.1)
        else:  # For dismissals, lower values should have a higher score
            batsman_stats[key + '_score'] = batsman_stats[key].apply(lambda x: ((max_val - x) / (max_val - min_val)) * abs(weights[key]) if max_val > min_val else abs(weights[key]) * 0.1)
    
    # Final aggregated score
    batsman_stats['final_score'] = batsman_stats[[col + '_score' for col in weights.keys()]].sum(axis=1)
    
    return batsman_stats


def calculate_bowler_metrics(df):
    # Group by match_id, bowler, and batter before calculating metrics
    matchwise_stats = df.groupby(['match_id', 'bowler', 'batter']).agg(
        total_runs_conceded=('batsman_runs', 'sum'),
        total_wickets=('is_wicket', 'sum'),
        total_balls=('ball', 'count'),
        total_overs=('over', 'nunique'),
        dot_balls=('batsman_runs', lambda x: (x == 0).sum())
    ).reset_index()
    
    # Aggregate at bowler-batsman level
    bowler_stats = matchwise_stats.groupby(['bowler', 'batter']).agg(
        total_runs_conceded=('total_runs_conceded', 'sum'),
        total_wickets=('total_wickets', 'sum'),
        total_balls=('total_balls', 'sum'),
        total_overs=('total_overs', 'sum'),
        dot_balls=('dot_balls', 'sum'),
        total_innings=('match_id', 'count')
    ).reset_index()
    
    # Additional Metrics
    bowler_stats['economy_rate'] = (bowler_stats['total_runs_conceded'] / bowler_stats['total_overs'])
    bowler_stats['dot_ball_percentage'] = (bowler_stats['dot_balls'] / bowler_stats['total_balls']) * 100
    bowler_stats['bowling_strike_rate'] = bowler_stats.apply(
        lambda x: x['total_balls'] / x['total_wickets'] if x['total_wickets'] > 0 else x['total_balls'], axis=1
    )
    
    # Boundary count calculation (Fours and Sixes conceded)
    boundaries = df[df['batsman_runs'].isin([4, 6])].groupby(['bowler', 'batter'])['batsman_runs'].count().reset_index()
    bowler_stats = bowler_stats.merge(boundaries, on=['bowler', 'batter'], how='left').fillna(0)
    bowler_stats.rename(columns={'batsman_runs': 'boundary_count'}, inplace=True)
    
    # Define weights for each attribute
    weights = {
        'total_wickets': 0.3,
        'economy_rate': 0.3,  # Lower economy is better
        'boundary_count': 0.2,  # Lower boundary count is better
        'dot_ball_percentage': 0.2,  # Higher dot ball percentage is better
        'bowling_strike_rate': 0.2  # Lower strike rate is better
    }
    
    # Normalize and calculate scores
    for key in weights.keys():
        min_val = bowler_stats[key].min()
        max_val = bowler_stats[key].max()
        if key in ['economy_rate', 'boundary_count', 'bowling_strike_rate']:
            bowler_stats[key + '_score'] = bowler_stats[key].apply(lambda x: ((max_val - x) / (max_val - min_val)) * weights[key] if max_val > min_val else weights[key] * 0.1)
        else:
            bowler_stats[key + '_score'] = bowler_stats[key].apply(lambda x: ((x - min_val) / (max_val - min_val)) * weights[key] if max_val > min_val else weights[key] * 0.1)
    
    # Apply scaling factor to penalize low ball counts
    min_balls_threshold = 30
    bowler_stats['scaling_factor'] = bowler_stats['total_balls'].apply(lambda x: min(x, min_balls_threshold) / min_balls_threshold)
    
    # Final aggregated score
    bowler_stats['final_score'] = bowler_stats[[col + '_score' for col in weights.keys()]].sum(axis=1) * bowler_stats['scaling_factor']
    
    return bowler_stats


def selected_player_bowling_1(bowler_plot_data,selected_player):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10,6))

    # Scatterplot
    sns.scatterplot(data=bowler_plot_data, x="total_wickets", y="total_runs_conceded", ax=ax)

    # Titles and labels
    ax.set_title(f'Batsmen against {selected_player} in IPL historically')
    ax.set_xlabel(f'Number of times {selected_player} dismissed them')
    ax.set_ylabel(f'Amount of runs scored against {selected_player}')
    
    max_dismissals = bowler_plot_data["total_wickets"].max()
    ax.set_xticks(np.arange(0, max_dismissals + 1, 1))

    # Label each point
    for ind in bowler_plot_data.index:
        ax.text(bowler_plot_data['total_wickets'][ind] + 0.1, 
                bowler_plot_data['total_runs_conceded'][ind], 
                str(bowler_plot_data['batter'][ind]), 
                fontsize=9, 
                ha='left')
    plt.close(fig)
    return fig  # Returning the figure object


def selected_player_batting_1(batter_plot_data,selected_player):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10,6))

    # Scatterplot
    sns.scatterplot(data=batter_plot_data, x="dismissals", y="total_runs", ax=ax)

    # Titles and labels
    ax.set_title(f'Bowlers against {selected_player} in IPL historically')
    ax.set_xlabel(f'Number of times {selected_player} was dismissed by them')
    ax.set_ylabel(f'Amount of runs scored by {selected_player}')

    max_dismissals = batter_plot_data["dismissals"].max()
    ax.set_xticks(np.arange(0, max_dismissals + 1, 1))
    
    # Label each point
    for ind in batter_plot_data.index:
        ax.text(batter_plot_data['dismissals'][ind] + 0.1, 
                batter_plot_data['total_runs'][ind], 
                str(batter_plot_data['bowler'][ind]), 
                fontsize=9, 
                ha='left')
    plt.close(fig)
    return fig  # Returning the figure object


def create_batter_effectiveness_plots(df):

    # Aggregate data at the batter level
    df = df.groupby("batter").agg(
        total_runs=("total_runs", "sum"),
        total_balls=("total_balls", "sum"),
        dismissals=("dismissals", "sum"),
        boundary_count=("boundary_count", "sum"),
        final_score=("final_score", "mean")  # Averaging final scores for consistency
    ).reset_index()

    # Recalculate strike rate and boundary percentage
    df["strike_rate"] = (df["total_runs"] / df["total_balls"]) * 100
    df["boundary_percentage"] = (df["boundary_count"] / df["total_balls"]) * 100

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False, sharey=False)

    # Define color palette
    palette = sns.color_palette("tab10", n_colors=df["batter"].nunique())

    # scatter0 = sns.scatterplot(data=df, x="total_balls", y="total_runs", hue="batter", style="batter", markers=True, palette=palette, s=100)
    # handles, labels = scatter0.get_legend_handles_labels()

    # 1. Runs vs. Balls Faced (Scatter Plot) - Capture legend handles
    sns.scatterplot(data=df, x="total_balls", y="total_runs", hue="batter", style="batter", markers=True, palette=palette, legend=False, ax=axes[0, 0], s=100)
    axes[0, 0].set_title("Total Runs vs. Balls Faced")
    axes[0, 0].set_xlabel("Balls Faced")
    axes[0, 0].set_ylabel("Total Runs")
    

    # 2. Strike Rate vs. Total Runs (Scatter Plot)
    sns.scatterplot(data=df, x="strike_rate", y="total_runs", hue="batter", style="batter", markers=True, palette=palette, ax=axes[0, 1], legend=False, s=100)
    axes[0, 1].set_title("Strike Rate vs. Total Runs")
    axes[0, 1].set_xlabel("Strike Rate")
    axes[0, 1].set_ylabel("Total Runs")

    # 3. Boundary Percentage vs. Strike Rate (Scatter Plot)
    scatter_ = sns.scatterplot(data=df, x="strike_rate", y="boundary_percentage", hue="batter", style="batter", markers=True, palette=palette, ax=axes[0, 2], s=100)
    axes[0, 2].set_title("Boundary Percentage vs. Strike Rate")
    axes[0, 2].set_xlabel("Strike Rate")
    axes[0, 2].set_ylabel("Boundary %")
    axes[0, 2].legend(fontsize=0.01)
    handles, labels = scatter_.get_legend_handles_labels()

    # 4. Dismissals vs. Runs Scored (Scatter Plot)
    sns.scatterplot(data=df, x="dismissals", y="total_runs", hue="batter", style="batter", markers=True, palette=palette, ax=axes[1, 0], legend=False, s=100)
    axes[1, 0].set_title("Dismissals vs. Runs Scored")
    axes[1, 0].set_xlabel("Number of Dismissals")
    axes[1, 0].set_ylabel("Total Runs")

    # 5. Top 10 Batters by Strike Rate (Bar Plot)
    top_strikers = df.groupby("batter")["strike_rate"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_strikers.values, y=top_strikers.index, ax=axes[1, 1], palette="magma", legend=False)
    axes[1, 1].set_title("Top 10 Batters by Strike Rate")
    axes[1, 1].set_xlabel("Strike Rate")

    # 6. Top 10 Batters by Final Score (Bar Plot)
    top_batters = df.groupby("batter")["final_score"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_batters.values, y=top_batters.index, ax=axes[1, 2], palette="viridis", legend=False)
    axes[1, 2].set_title("Top 10 Batters by Final Score")
    axes[1, 2].set_xlabel("Final Score")

    # Create a single legend outside the plots
    # fig.legend(handles, labels, title="Batter", loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=6, frameon=False)
    fig.legend(handles, labels, title="Batter", loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=6, frameon=True, fontsize=10, title_fontsize=12, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to fit the legend at the bottom
    plt.close(fig)  # Prevents automatic display

    return fig  # Return the final improved figure


def create_bowler_effectiveness_plots(df):
    # Aggregate data at the bowler level
    df = df.groupby("bowler").agg(
        total_runs_conceded=("total_runs_conceded", "sum"),
        total_wickets=("total_wickets", "sum"),
        total_balls=("total_balls", "sum"),
        dot_balls=("dot_balls", "sum"),
        economy_rate=("economy_rate", "mean"),
        dot_ball_percentage=("dot_ball_percentage", "mean"),
        bowling_strike_rate=("bowling_strike_rate", "mean"),
        final_score=("final_score", "mean")
    ).reset_index()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False, sharey=False)
    palette = sns.color_palette("tab10", n_colors=df["bowler"].nunique())
    
    # Runs Conceded vs. Wickets Taken
    # scatter0 = sns.scatterplot(data=df, x="total_wickets", y="total_runs_conceded", hue="bowler",palette=palette, style="bowler", s=150)
    # handles, labels = scatter0.get_legend_handles_labels()
    sns.scatterplot(data=df, x="total_wickets", y="total_runs_conceded", hue="bowler", style="bowler", palette=palette, ax=axes[0, 0], legend=False, s=150)
    axes[0, 0].set_title("Runs Conceded vs. Wickets Taken")
    axes[0, 0].set_xlabel("Total Wickets")
    axes[0, 0].set_ylabel("Total Runs Conceded")
    
    # Economy Rate vs. Dot Ball Percentage
    sns.scatterplot(data=df, x="economy_rate", y="dot_ball_percentage", hue="bowler", style="bowler", palette=palette, ax=axes[0, 1], legend=False, s=150)
    axes[0, 1].set_title("Economy Rate vs. Dot Ball Percentage")
    axes[0, 1].set_xlabel("Economy Rate")
    axes[0, 1].set_ylabel("Dot Ball %")
    
    # Bowling Strike Rate vs. Wickets Taken
    scatter_ = sns.scatterplot(data=df, x="bowling_strike_rate", y="total_wickets", hue="bowler", style="bowler", palette=palette, ax=axes[0, 2], s=150)
    axes[0, 2].set_title("Bowling Strike Rate vs. Wickets Taken")
    axes[0, 2].set_xlabel("Bowling Strike Rate")
    axes[0, 2].set_ylabel("Total Wickets")
    axes[0, 2].legend(fontsize=0.01)
    handles, labels = scatter_.get_legend_handles_labels()
    
    # Dot Balls vs. Total Balls Bowled
    sns.scatterplot(data=df, x="total_balls", y="dot_balls", hue="bowler", style="bowler", palette=palette, ax=axes[1, 0], legend=False, s=150)
    axes[1, 0].set_title("Dot Balls vs. Total Balls Bowled")
    axes[1, 0].set_xlabel("Total Balls Bowled")
    axes[1, 0].set_ylabel("Dot Balls")
    
    # Top 10 Bowlers by Economy Rate
    top_economy = df.nsmallest(10, "economy_rate")
    sns.barplot(x=top_economy["economy_rate"], y=top_economy["bowler"], ax=axes[1, 1], palette="magma")
    axes[1, 1].set_title("Top 10 Bowlers by Economy Rate")
    axes[1, 1].set_xlabel("Economy Rate")
    
    # Top 10 Bowlers by Final Score
    top_bowlers = df.nlargest(10, "final_score")
    sns.barplot(x=top_bowlers["final_score"], y=top_bowlers["bowler"], ax=axes[1, 2], palette="viridis")
    axes[1, 2].set_title("Top 10 Bowlers by Final Score")
    axes[1, 2].set_xlabel("Final Score")
    
    # Create a single legend
    fig.legend(handles, labels, title="Bowler", loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=6, frameon=True, fontsize=10, title_fontsize=12, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.close(fig)
    
    return fig