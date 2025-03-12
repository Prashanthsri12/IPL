import numpy as np
import pandas as pd
from utils import *
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 

def main():
    st. set_page_config(layout="wide")
    st.title("IPL Player Effectiveness",help=None)
    st.caption("Explore in-depth player performance insights with our interactive IPL analytics dashboard. Select any two teams, and the algorithm analyzes past encounters to evaluate how players have performed against each other. Gain valuable data-driven insights to compare player effectiveness and make informed predictions for upcoming matches.")
    st.sidebar.title("Inputs section")
    st.sidebar.caption("User inputs for analysis.")
    
    match_df,player_mapping,team_mapping,delivery_df = load_data()

    delivery_df['dismissal_kind_old'] = delivery_df['dismissal_kind'] 
    delivery_df['dismissal_kind'] = [i if i in ['caught','bowled','run out','lbw','caught and bowled','stumped'] else 'NA' for i in delivery_df['dismissal_kind_old']]
    delivery_df['dismissal'] = [1 if i in ['caught','bowled','run out','lbw','caught and bowled','stumped'] else 0 for i in delivery_df['dismissal_kind_old']]

    team_mapping['team_input'] = team_mapping.apply(lambda row: f"{row['Team']} ({row['Team_Short']})", axis=1)
    team_input_names = team_mapping['team_input'].to_list()

    team_1_ = st.sidebar.selectbox('Team 1', team_input_names,index=None, key = 'team_1_')
    team_2_ = st.sidebar.selectbox('Team 2', team_input_names[::-1],index=None, key = 'team_2_')

    if team_1_ == team_2_:
        st.error("Please select two different teams for analysis!", icon="🚨")
    elif team_1_ in team_input_names and team_2_ in team_input_names: 
        st.info(f'Teams Selected : {team_1_} & {team_2_}')

        team_1 = team_mapping[team_mapping['team_input']==team_1_]['Team'].values[0]
        team_2 = team_mapping[team_mapping['team_input']==team_2_]['Team'].values[0]

        team_1_short = team_mapping[team_mapping['team_input']==team_1_]['Team_Short'].values[0]
        team_2_short = team_mapping[team_mapping['team_input']==team_2_]['Team_Short'].values[0]

        selected_players_df = player_mapping[player_mapping['Team'].isin([team_1,team_2])].reset_index(drop=True)
        player_team_1_df = player_mapping[player_mapping['Team'].isin([team_1])].reset_index(drop=True)
        player_team_2_df = player_mapping[player_mapping['Team'].isin([team_2])].reset_index(drop=True)

        player_team_1_list = player_team_1_df['new_player_names'].unique()
        player_team_2_list = player_team_2_df['new_player_names'].unique()

        team_1_batting = delivery_df[(delivery_df['batter'].isin(player_team_1_list)) & (delivery_df['bowler'].isin(player_team_2_list))].reset_index(drop=True)
        team_1_batting_metrics = calculate_batsman_metrics(team_1_batting)

        team_2_batting = delivery_df[(delivery_df['batter'].isin(player_team_2_list)) & (delivery_df['bowler'].isin(player_team_1_list))].reset_index(drop=True)
        team_2_batting_metrics = calculate_batsman_metrics(team_2_batting)

        team_1_bowling = delivery_df[delivery_df['bowler'].isin(player_team_1_list) & (delivery_df['batter'].isin(player_team_2_list))].reset_index(drop=True)
        team_1_bowling_metrics = calculate_bowler_metrics(team_1_bowling)

        team_2_bowling = delivery_df[delivery_df['bowler'].isin(player_team_2_list) & (delivery_df['batter'].isin(player_team_1_list))].reset_index(drop=True)
        team_2_bowling_metrics = calculate_bowler_metrics(team_2_bowling)

        team_1_batter_level_plots = create_batter_effectiveness_plots(team_1_batting_metrics)
        team_1_bowler_level_plots = create_bowler_effectiveness_plots(team_1_bowling_metrics)

        team_2_batter_level_plots = create_batter_effectiveness_plots(team_1_batting_metrics)
        team_2_bowler_level_plots = create_bowler_effectiveness_plots(team_1_bowling_metrics)


        with st.expander("Team 1 Overall Analysis", icon=":material/info:"):
            st.text("Team 1 Batting Metrics")
            st.pyplot(team_1_batter_level_plots)
            st.text("Team 1 Bowling Metrics")
            st.pyplot(team_1_bowler_level_plots)

        # with st.expander("Team 1 Player Level Analysis", icon=":material/info:"):
        #     team_1_selected_plater = st.selectbox('Team 1', player_team_1_list, index=None, key = 'team_1_selected_plater') #use players from team_1_bowling_metrics and team_1_batting_metrics only and check if player present before plot
        #     if team_1_selected_plater != None:
        #         team_1_selected_plater_bowler_plot_data = team_1_bowling_metrics[team_1_bowling_metrics['bowler']==team_1_selected_plater]
        #         team_1_selected_plater_batter_plot_data = team_1_batting_metrics[team_1_batting_metrics['batter']==team_1_selected_plater]
        #         selected_player_bowling_plot_1 = selected_player_bowling_1(team_1_selected_plater_bowler_plot_data,team_1_selected_plater)
        #         selected_player_batting_plot_1 = selected_player_batting_1(team_1_selected_plater_batter_plot_data,team_1_selected_plater)



        
        with st.expander("Team 2 overall Analysis", icon=":material/info:"):
            st.text("Team 2 Batting Metrics")
            st.pyplot(team_2_batter_level_plots)
            st.text("Team 2 Bowling Metrics")
            st.pyplot(team_2_bowler_level_plots)




if __name__ == '__main__':
    main()


