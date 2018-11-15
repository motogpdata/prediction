import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns

import pandas as pd
import numpy as np

from collections import defaultdict
from pprint import pprint

def read_data():

	def clean_bike_name( x ) :
		if x is np.nan:
			return x

		if 'Yamaha' in x:
			return 'Yamaha'

		if 'Honda' in x:
			return 'Honda'

		if 'Kawasaki' in x:
			return 'Kawasaki'

		if 'APR' in x:
			return 'Aprilia'

		if 'Ilmor' in x:
			return 'Ilmor'

		if 'KR211V'in x:
			return 'Roberts KR211V'

		return x

	df_all = pd.read_csv('../../data/all_data-motogp_2018.11.06.csv')
	print('Entries loaded: ', len(df_all.index))

	# remove annoying caps on surname
	df_all['rider'] = df_all['rider'].str.title()

	# correct bikes
	# print(df_all.isna().sum())
	# print( df_all[ df_all['bike'].isna() ] )
	df_all['bike'] = df_all['bike'].apply( clean_bike_name )

	# select only Race results (not practice or qualy)
	df_all = df_all[ df_all['session'].str.startswith('RAC')].reset_index(drop=True)

	# find races with RAC2, so we can remove the corresponding RAC entry
	if True:
		short_races = df_all[ df_all['session'] != 'RAC' ]

		short_races_grp = short_races.groupby(['year', 'race_id'])
		for grp_name, grp in short_races_grp:
			# print(grp_name)
			# print( df_all[(df_all['year'] == grp_name[0]) & (df_all['race_id']==grp_name[1])] )
			# print()

			# select any other entries
			df_all = df_all[ (df_all['year'] != grp_name[0]) | (df_all['race_id'] != grp_name[1]) | (df_all['session']=='RAC') ]

		# rename accordingly
		df_all['session'] = df_all['session'].map({'RAC2': 'RAC'})

	return df_all

def features_year(df, year, pred_circuit, is_prediction=False):
	print()
	print( year )

	ft_list = []
	rider_list = []

	if is_prediction:
		# for predicted year, race_id is the next race
		race_id = df.loc[(df['year'] == year), 'race_id'].max() + 1
		print(race_id)
	else:
		# race_id for that year
		race_id = df.loc[(df['year'] == year) & (df['circuit'] == pred_circuit), 'race_id'].values[0]
	# print(race_id)

	# data from races BEFORE or DURING the predicted race for a given year
	dfy =  df[ (df['year'] == year) & (df['race_id'] <= race_id) ]

	# # riders who rode on the track that year
	# riders = dfy.loc[ (dfy['circuit'] == pred_circuit), 'rider'].unique()

	if is_prediction:
		# all riders from the season
		riders = dfy['rider'].unique()
	else:
		# riders who rode on the track that year
		riders = dfy.loc[ (dfy['circuit'] == pred_circuit), 'rider'].unique()

	for rider in riders:

		# rider information for races BEFORE the predicted race
		dfyr = dfy[ (dfy['rider'] == rider)  & (dfy['race_id'] < race_id) ]

		# rider information for the predicted race
		dfyr_pred = dfy[ (dfy['rider'] == rider)  & (dfy['race_id'] == race_id) ]

		started = len(dfyr)
		finished = len(dfyr[dfyr['finished'] == True])

		ft = {}
		if started > 0 and finished > 0:

			print(rider)

			## AL PREVIOUS RACES
			# finished races / races
			ft['finished_started'] = finished / started * 100

			# points / race
			ft['points_started'] = dfyr['points'].sum() / started
			ft['points_finished'] = dfyr['points'].sum() / finished

			# podium / races
			ft['podiums_started'] = len(dfyr[dfyr['position'].isin([1,2,3])]) / started
			ft['podiums_finished'] = len(dfyr[dfyr['position'].isin([1,2,3])]) / finished

			# wins / races
			ft['wins_started'] = len(dfyr[dfyr['position'] == 1]) / started
			ft['wins_finished'] = len(dfyr[dfyr['position'] == 1]) / finished


			## Same for 5 previous races
			dfyr5 = dfyr[ dfyr['race_id'].isin(range(race_id-5, race_id)) ]
			started = len(dfyr5)
			finished = len(dfyr5[dfyr5['finished'] == True])

			if started > 0 and finished > 0:				
				ft['finished_started5']= finished / started * 100
				ft['points_started5'] = dfyr5['points'].sum() / started
				ft['points_finished5'] = dfyr5['points'].sum() / finished
				ft['podiums_started5'] = len(dfyr5[dfyr5['position'].isin([1,2,3])]) / started
				ft['podiums_finished5'] = len(dfyr5[dfyr5['position'].isin([1,2,3])]) / finished
				ft['wins_started5'] = len(dfyr5[dfyr5['position'] == 1]) / started
				ft['wins_finished5'] = len(dfyr5[dfyr5['position'] == 1]) / finished

			## Same for 3 previous races
			dfyr3 = dfyr[ dfyr['race_id'].isin(range(race_id-3, race_id)) ]
			started = len(dfyr3)
			finished = len(dfyr3[dfyr3['finished'] == True])

			if started > 0 and finished > 0:				
				ft['finished_started3']= finished / started * 100
				ft['points_started3'] = dfyr3['points'].sum() / started
				ft['points_finished3'] = dfyr3['points'].sum() / finished
				ft['podiums_started3'] = len(dfyr3[dfyr3['position'].isin([1,2,3])]) / started
				ft['podiums_finished3'] = len(dfyr3[dfyr3['position'].isin([1,2,3])]) / finished
				ft['wins_started3'] = len(dfyr3[dfyr3['position'] == 1]) / started
				ft['wins_finished3'] = len(dfyr3[dfyr3['position'] == 1]) / finished

			## result on each previous circuit
			for shift in range(1, race_id):
				ft['points_race_-'+str(shift)] = dfyr.loc[ dfyr['race_id'] == race_id-shift, 'points'].sum()# ft['pos_race_-1'] will be the position 1 race before the race we are evaluating
				try:
					ft['podium_race_-'+str(shift)] = dfyr.loc[ dfyr['race_id'] == race_id-shift, 'position'].isin([1,2,3]).values[0]
				except:
					ft['podium_race_-'+str(shift)] = False


			# and the labels we want to train for
			if not is_prediction:
				ft['pred_points'] = dfyr_pred['points'].values[0]
				ft['pred_podium'] = dfyr_pred['position'].isin([1,2,3]).values[0]
				ft['pred_win'] = (dfyr_pred['position'] == 1).all()

			ft_list.append( ft )
			rider_list.append(rider)

	return ft_list, rider_list

def main():

	pred_circuit = 'VAL'
	pred_year = 2018

	print('-- predicting {} {} results --'.format(pred_circuit, pred_year))

	# import matplotlib.font_manager
	# print( sorted(list(set(([f.name for f in matplotlib.font_manager.fontManager.ttflist])))) )
	# print( sorted(list(set(([f.name for f in matplotlib.font_manager.fontManager.afmlist])))) )

	# sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.0}) 
	sns.set_style("white")
	
	# position, points racer_number, rider, country, team, bike, speed, time_str, finished, time, time_diff, race_id, circuit, category, session, year
	df_all = read_data()
	df = df_all[ (df_all['category'] == 'MotoGP') ].copy() # make a copy to avoid warning on manipulating views

	# years with data on the circuit (previous to prediction)
	years_with_data = df.loc[ (df['circuit'] == pred_circuit) & (df['year'] < pred_year), 'year'].unique()
	print('years with data: ', years_with_data)

	ft_training = []
	for year in years_with_data[:-1]:
		fts, _ = features_year(df, year, pred_circuit) 
		ft_training.extend(fts)

	ft_testing, riders = features_year(df, pred_year, pred_circuit, is_prediction=True)

	print()
	print('Number of samples for training: ', len(ft_training))

	df_train = pd.DataFrame(ft_training).fillna(0)
	df_pred = pd.DataFrame(ft_testing).fillna(0)

	print(df_pred)

	ft_columns = [c for c in df_train.columns if not c.startswith('pred_') ]
	pred_columns = [c for c in df_train.columns if c.startswith('pred_') ]

	print('ft_columns: ', ft_columns)
	print('pred_columns: ', pred_columns)

	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

	print('Wins')
	clf.fit(df_train[ft_columns], df_train['pred_win'] )
	probs = clf.predict_proba(df_pred[ft_columns])
	print(probs)

	print('Podiums')
	clf.fit(df_train[ft_columns], df_train['pred_podium'] )
	probs = clf.predict_proba(df_pred[ft_columns])
	print(probs)

	print('Points')
	clf.fit(df_train[ft_columns], df_train['pred_points'] )
	probs = clf.predict_proba(df_pred[ft_columns])
	print(probs)

	points = pd.DataFrame(probs, columns=['>15'] + [str(i) for i in range(15, 0, -1)]) * 100
	points.insert(0, 'Rider', riders)
	points = points.set_index('Rider')

	points = points.sort_values(by=['>15'], ascending=True)
	print(points)

	ax = sns.heatmap(data=points, annot=True, yticklabels=True, linewidths=.5, vmin=0, vmax=35, cmap='Oranges', cbar=False)
	for t in ax.texts: t.set_text(t.get_text() + '%')

	ax.xaxis.tick_top() # x axis on top
	ax.xaxis.set_label_position('top')
	plt.xlabel('Probabiliy for each position')
	plt.ylabel('')
	
	plt.show()


if __name__ == "__main__":
	main()
	plt.show()
