# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:08:55 2025

@author: aleex
Fichero experimental planteado para convertir ficheros json de WHOSCORED a csv
"""


import json
import pandas as pd
import os
import numpy as np
import shutil

os.path.abspath(__file__)



def lectura_json(ruta, fichero):
    with open("{}/{}".format(ruta,fichero), "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

def get_plains(data,key):
    return data[key]


def extract_qualifiers(qualifiers):
    if not isinstance(qualifiers, list):  # Si no es una lista, devolver diccionario vacío
        return {}

    extracted = {}
    for q in qualifiers:
        if isinstance(q, dict):  # Verifica que sea un diccionario
            type_info = q.get('type', {})  # Obtiene el diccionario 'type'
            display_name = type_info.get('displayName')  # Extrae 'displayName'
            if q.get('value'):
                value = q.get('value')  # Extrae 'value'
            else:
                value=1
            if display_name:  # Solo agrega si displayName existe
                extracted[f"value_{display_name}"] = value
    return extracted

# def get_match(data,expand_cols=["referee"]):
#     d={k: v for k, v in data['matchCentreData'].items() if not isinstance(v, list) and not isinstance(v,dict)}
#     df_match=pd.DataFrame(d,index=[get_plains(data,'matchId')]).reset_index().rename({"index":"matchId"},axis=1)
#     for c in expand_cols:
#         ref = {k: v for k, v in data['matchCentreData'][c].items() if not isinstance(v, list) and not isinstance(v,dict)}
#         kid=c+"Id"
#         kname=c+"Name"
#         df_match[kid] = ref["officialId"]
#         df_match[kname] = ref["name"]
#     return df_match

def get_match(data, expand_cols=("referee",)):
    mc = data.get('matchCentreData', {})
    d = {k: v for k, v in mc.items() if not isinstance(v, (list, dict))}
    df_match = pd.DataFrame(d, index=[get_plains(data, 'matchId')]).reset_index().rename({"index": "matchId"}, axis=1)

    for c in expand_cols:
        ref = mc.get(c)
        if isinstance(ref, dict):
            df_match[f"{c}Id"] = ref.get("officialId")
            df_match[f"{c}Name"] = ref.get("name")
        else:
            # si no existe referee o no es dict, rellenamos con None
            df_match[f"{c}Id"] = None
            df_match[f"{c}Name"] = None
    return df_match

def get_teams(data,dict_cols=['scores'],expand_cols=["stats"],drop_stats=1):
    df_teams=pd.DataFrame()
    df_tmstats = pd.DataFrame()
    for hd in ['home','away']:
        tm={k: v for k, v in data['matchCentreData'][hd].items() if not isinstance(v, list)}
        teams = pd.json_normalize(tm)
        teamstats = pd.DataFrame(tm)[["teamId","stats","name"]].reset_index()
        teamstats=teamstats[teamstats['index']!='minutesWithStats']
        teams.drop([i for i in teams.columns if "." in i and "scores" not in i],axis=1,inplace=True)
        df_teams = pd.concat([df_teams,teams])
        df_tmstats = pd.concat([df_tmstats,teamstats])
    df_teams['matchId']=get_plains(data,'matchId')
    df_tmstats['matchId']=get_plains(data,'matchId')
    df_tmstats["stats"] = df_tmstats.apply(
        lambda x: np.mean(list(x["stats"].values())) 
        if isinstance(x["stats"], dict) and (x["index"].endswith("Success") or x["index"].endswith("Accuracy") or x["index"] == "ratings") 
        else (x["stats"] if isinstance(x["stats"], float) else sum(x["stats"].values())), 
        axis=1
        )
    df_tmstats = df_tmstats.pivot_table(index=["matchId","teamId","name"], columns='index', values='stats', aggfunc='first').reset_index()
    
    if drop_stats and "stats" in df_teams.columns:
        df_teams.drop("stats",inplace=True,axis=1)
    
    for df in [df_teams,df_tmstats]:
        df.rename({"name":"teamName"},inplace=True,axis=1)
    return df_teams,df_tmstats

def get_players(data,dict_cols=['subbedOutPeriod','subbedInPeriod'],expand_cols=["stats"],drop_stats=1):
    df_players=pd.DataFrame()
    for hd in ['home','away']:
        ply=pd.DataFrame(data['matchCentreData'][hd]['players'])
        
        #for attrs,name in zip(['teamId','managerName','name'],['teamId','managerName','teamName']):
        #    ply[name] = get_plains(data['matchCentreData'][hd],attrs)
        for col in dict_cols:
            if col in ply.columns:
                ply[col] = ply[col].apply(lambda x: x if isinstance(x, dict) else {})
                expanded_cols = pd.json_normalize(ply[col]).add_prefix(f"{col}_")  # Prefijo con el nombre de la columna original
                ply = ply.join(expanded_cols)
                ply.drop(col,inplace=True,axis=1)
        ply["field"] = hd
        df_players=pd.concat([df_players,ply])
    df_players['long_name'] = (df_players["shirtNo"].astype(str) + ". " + df_players["name"]).replace(".0","")

    expanded_rows = []

    for _, row in df_players[["playerId","stats","name"]].iterrows():
        stats_dict = row["stats"]
        
        # Extraer cada minuto y cada estadística
        for stat, values in stats_dict.items():
            for minute, value in values.items():
                expanded_rows.append({"playerId": row["playerId"],"name":row["name"],
                                      "minute": int(minute), "stat": stat, "value": value})
    
    # Crear DataFrame expandido
    df_expanded = pd.DataFrame(expanded_rows)
    
    # Pivotar para tener cada estadística como una columna
    df_stats = df_expanded.pivot_table(index=["playerId", "minute","name"], columns="stat", values="value").reset_index()
    cols_mean = [col for col in df_stats.columns if col.endswith("Success") or col.endswith("Accuracy") or col == "ratings"]
    cols_drop=["minute"]
    cols_sum = [col for col in df_stats.columns if col not in cols_mean + ["playerId","name"]]
    df_stats_ag = df_stats.groupby(["playerId","name"]).agg({**{col: "mean" for col in cols_mean},
                                                  **{col: "sum" for col in cols_sum}}).reset_index()
    df_stats_ag.drop(cols_drop,inplace=True,axis=1)
    if drop_stats and "stats" in df_players.columns:
        df_players.drop("stats",inplace=True,axis=1)
    for df in [df_players,df_stats_ag]:
        df['matchId']=get_plains(data,'matchId')
        df.rename({"name":"playerName"},inplace=True,axis=1)
    return df_players,df_stats_ag



def get_events(data, 
               dict_cols=['cardType','period', 'type', 'outcomeType'], 
               merge_ply=['playerId', 'shirtNo', 'playerName', 'position','isFirstEleven','long_name'],
               merge_teams=['teamId','teamName'],
               drop_qualifiers=0):
    df = {key: pd.DataFrame(value) for key, value in data['matchCentreData'].items() if isinstance(value, list)}['events']
    df['matchId']=get_plains(data,'matchId')
    
    
    for col in dict_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
            expanded_cols = pd.json_normalize(df[col]).add_prefix(f"{col}_")  # Prefijo con el nombre de la columna original
            df = df.join(expanded_cols)
    
    df = df.drop(columns=[i for i in dict_cols if i in df.columns])
    df_expanded = df.join(df['qualifiers'].apply(lambda x: pd.Series(extract_qualifiers(x))))
    
    # Eliminar la columna original 'qualifiers'
    if drop_qualifiers:
        df_expanded.drop(columns=['qualifiers'],inplace=True)
    
    
    df_pl,df_plstats = get_players(data)
    df_tm,df_tmstats = get_teams(data)
    df_plstats=pd.merge(df_plstats,df_pl[['field',"playerId"]],how='left',on="playerId")
    
    df_pl=pd.merge(df_pl,df_tm[['field',"teamId","teamName"]],how='left',on="field")
    df_plstats=pd.merge(df_plstats,df_tm[['field',"teamId","teamName"]],how='left',on="field")
    df_match=get_match(data)
    df_expanded=pd.merge(df_expanded,df_pl[merge_ply],on="playerId",how='left')
    df_expanded=pd.merge(df_expanded,df_tm[merge_teams],on="teamId",how='left')
    df_expanded["oppositionTeamName"] = np.where(df_expanded.teamName==df_tm.teamName.values[0],
                                                 df_tm.teamName.values[1],
                                                 df_tm.teamName.values[0])
    #df_expanded["refName"] = get_plains(data['matchCentreData']["referee"],"name")
    df_expanded["time_seconds"]= df_expanded.minute*60 + df_expanded.second
    
    #df_expanded = calcula_metricas(df_expanded)
    
    return {"eventData":df_expanded,"playerData":df_pl,"playerStats":df_plstats,
            "teamData":df_tm,"teamStats":df_tmstats,"matchData":df_match}

def procesar_fichero(ruta,fichero,output, export=1):
    try:
        print("\nParseando fichero de partido: {}".format(fichero))
        result = lectura_json(ruta, fichero)
        try:
            game_data = get_events(result)
            if export:
                for k in game_data:
                    game_data[k].to_csv("{}/{}_{}.csv".format(os.path.join(ruta,output),
                                                              get_plains(result,'matchId'),
                                                              k
                                                              ),decimal=',',sep=';',index=False)
            #old_dir = os.path.join(ruta, "old")
            #os.makedirs(old_dir, exist_ok=True)  # Crear la carpeta si no existe
            #shutil.move(os.path.join(ruta, fichero), os.path.join(old_dir, fichero))
            print("Proceso Completado")
        except Exception as e:
            print(e)
            pass
    except Exception as e:
        print("ERROR - {}".format(e))

def procesar_ficheros_lista(ruta,subr):
    json_list=[f for f in os.listdir(ruta) if f.endswith(".json") and f.split("_")[-1].replace(".json",".csv").replace(".","_eventData.") not in os.listdir(os.path.join(ruta,subr))]
    counter=0
    counter_ok=0
    for json_file in json_list:
        counter+=1
        print("\n({}/{})".format(counter,len(json_list)))
        procesar_fichero(ruta,json_file,subr)
        counter_ok+=1
    print("\n\nFicheros Leidos: {}".format(counter))
    print("Ficheros Parseados: {}".format(counter_ok))