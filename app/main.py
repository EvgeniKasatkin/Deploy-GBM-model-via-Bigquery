from flask import Flask, send_file, jsonify, request
import feature_engeneering
import pandas as pd
import json
import traceback
import time
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
app = Flask(__name__)

@app.route("/")
def download_scoring_data():
    try:
        initial_time = time.strftime("%d %m %Y, %H:%M:%S", time.localtime())

        scoring_initiation = feature_engeneering.DownloaderDataFrame(int(os.getenv('days_of_scoring')))
        df_to_score = scoring_initiation.df_to_scoring()

        feature_estimate = feature_engeneering.FeatureTargetEncoding(int(os.getenv('days_of_scoring')), os.getenv('list_of_features').strip(',').split(', '), df_to_score)
        df_of_features, list_of_TE_features = feature_estimate.feature_evaluation()

        downloader = feature_engeneering.DownloadToBQ(df_of_features, str(os.getenv('dataset_id')), str(os.getenv('table_id')), list_of_TE_features, destination = 'WRITE_APPEND')
        downloader_status = downloader.bq_table_update()

        end_time = time.strftime("%d %m %Y, %H:%M:%S", time.localtime())

        feature_engeneering.MessageToTg('Sucsessfully Daily traffic evalute ' + str(str(time.ctime()))).message_alarm()

        return jsonify({'status': 'Ok', 'time_start': str(initial_time), 'time_end': str(end_time)})

    except:
        feature_engeneering.MessageToTg('Scoring Daily mistake (1 hour) ' + str(time.ctime()) + ' ' + str(traceback.format_exc())).message_alarm()
        return jsonify({'status': 'Mistake'})










