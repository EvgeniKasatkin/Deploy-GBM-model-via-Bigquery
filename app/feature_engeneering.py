from google.cloud import bigquery
from google.oauth2 import service_account
import json
import pandas as pd
import numpy as np
import pickle
import joblib
import lightgbm
import telebot
from telebot import types
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())


class DownloaderDataFrame:
    def __init__(self, days):
        self.days = days

    def df_to_scoring(self):
        self.file_of_creds = open(str(os.getenv('path')), "r")
        json_account_info = json.loads(self.file_of_creds.read())
        self.file_of_creds.close()
        self.cred = service_account.Credentials.from_service_account_info(json_account_info)
        self.client = bigquery.Client(credentials=self.cred)
        sql = "select * from `%s` where date(timestamp_micros(event_tmstp)) > DATE_SUB(current_date(), INTERVAL %s DAY) " %(os.getenv('daily_table_for_scoring_evaluate'), self.days)
        self.df_to_insert = self.client.query(sql).result().to_dataframe(create_bqstorage_client=True)
        return self.df_to_insert

class FeatureTargetEncoding:
    def __init__(self, days, feature_name_list, df_to_insert):
        self.days = days
        self.feature_name_list = feature_name_list
        self.df_to_insert = df_to_insert
        self.list_of_category_parameters = list(os.getenv('list_of_category_features'))
        self.list_of_behavioral_parameters = list(os.getenv('list_of_behavioral_parameters'))
        self.list_of_all_features = self.list_of_category_parameters + self.list_of_behavioral_parameters
        self.list_of_TE_features = []

    def feature_evaluation(self):
        self.file_of_creds = open(str(os.getenv('path')), "r")
        json_account_info = json.loads(self.file_of_creds.read())
        self.file_of_creds.close()
        self.cred = service_account.Credentials.from_service_account_info(json_account_info)
        self.client = bigquery.Client(credentials=self.cred)
        df_to_insert = self.df_to_insert

        for feature_name in self.feature_name_list:
            if feature_name in self.list_of_category_parameters:
                query_string = """
                        select %s, coalesce(avg(cast(smooth_TE_value as float64) ), %s) as %s_TE 
                        from ((
                                   select distinct case when length(%s) < 2 then 'NoData' else %s end as %s 
                                   from `%s`
                                   where date(timestamp_micros(event_tmstp)) > DATE_SUB(current_date(), INTERVAL %s DAY) ) t1 
                        left join (
                            select feature_value, smooth_TE_value 
                            from `%s` 
                            where feature_name = '%s' ) t2 on t2.feature_value = t1.%s)
                        group by 1""" %(feature_name, float(os.getenv('averaging_value_parameter')), feature_name, feature_name, feature_name, feature_name, os.getenv('daily_table_for_scoring_evaluate'), os.getenv('table_list_of_feature_values'), self.days, feature_name, feature_name)
                df_feature = self.client.query(query_string).result().to_dataframe(create_bqstorage_client=True)
                df_to_insert = df_to_insert.merge(df_feature, on=[str(feature_name)], how='left')
                del df_to_insert[str(feature_name)]
                del df_feature

            elif feature_name in self.list_of_behavioral_parameters:
                query_string = """
                      select %s, coalesce(avg(cast(smooth_BH_value as float64) ), %s) as %s_BH 
                      from ((
                        select distinct case when length(cast(%s as string)) < 1 then 'NoData' else cast(%s as string) end as %s 
                        from `%s`
                        where date(timestamp_micros(event_tmstp)) > DATE_SUB(current_date(), INTERVAL %s DAY) ) t1 
                    left join ( select feature_value, smooth_BH_value 
                                from `%s` where feature_name = '%s' ) t2 on cast(t2.feature_value as string) = cast(t1.%s as string ))\
                      group by 1"""  %(feature_name, float(os.getenv('averaging_value_parameter')), feature_name, feature_name, feature_name, feature_name, os.getenv('daily_table_for_scoring_evaluate'), os.getenv('table_list_of_bh_feature_values'), self.days, feature_name, feature_name)
                df_feature = self.client.query(query_string).result().to_dataframe(create_bqstorage_client=True)
                df_to_insert[str(feature_name)] = df_to_insert[str(feature_name)].astype(str)
                df_to_insert[str(feature_name)] = df_to_insert[str(feature_name)].fillna('NoData')
                df_to_insert = df_to_insert.merge(df_feature, on=[str(feature_name)], how='left')
                del df_to_insert[str(feature_name)]
                del df_feature

        for feature_name in self.list_of_category_parameters:
            feature_ = str(feature_name) + '_TE'
            self.list_of_TE_features.append(feature_)
            df_to_insert[feature_] = df_to_insert[feature_].fillna(float(os.getenv('averaging_value_parameter')))

        for feature_name in self.list_of_behavioral_parameters:
            feature_ = str(feature_name) + '_BH'
            self.list_of_TE_features.append(feature_)
            df_to_insert[feature_] = df_to_insert[feature_].fillna(float(os.getenv('averaging_value_BH_parameter')))

        #scoring
        loaded_model = joblib.load(str(os.getenv('model_gbm_file')))
        scoring_list = loaded_model.predict(df_to_insert[self.list_of_all_features], num_iteration = loaded_model.best_iteration_)
        df_to_insert['scoring_value'] = list(scoring_list)
        df_to_insert['scoring_datitime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return df_to_insert, self.list_of_TE_features

class DownloadToBQ:
    def __init__(self, df_score, dataset_id, table_id, list_of_TE_features, destination):
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.destination = destination
        self.df_score = df_score
        self.list_of_TE_features = list_of_TE_features
        self.schema_list = []
        self.name_of_columns = []
        self.type_of_columns  = []

    def type_of_column_f(self, type_):
        if type_ == "object":
            return 'STRING'
        elif type_ == "int64":
            return 'INTEGER'
        elif type_ == "float64":
            return 'FLOAT64'
        elif type_ == "datetime64[ns, UTC]":
            return 'DATETIME'

    def bq_table_update(self):
        self.file_of_creds = open(str(os.getenv('path')), "r")
        json_account_info = json.loads(self.file_of_creds.read())
        self.file_of_creds.close()
        self.cred = service_account.Credentials.from_service_account_info(json_account_info)
        self.client = bigquery.Client(credentials=self.cred)

        df_score = self.df_score
        columns_to_download = list(os.getenv('list_of_id_and_scoring_values')) + self.list_of_TE_features
        df_score = df_score[columns_to_download]
        for column_ in list(os.getenv('list_of_id_and_scoring_values')):
            df_score[column_] = df_score[column_].astype('str')

        self.name_of_columns = list(df_score.columns.values)
        self.type_of_columns = list(df_score.dtypes)
        for ind_, _ in enumerate(self.name_of_columns):
            type_ = self.type_of_columns[ind_]
            self.schema_list.append(bigquery.SchemaField(str(self.name_of_columns[ind_]), str(self.type_of_column_f(type_)), "NULLABLE"))

        dataset = self.client.dataset(self.dataset_id)
        table = dataset.table(self.table_id)
        job_config = bigquery.LoadJobConfig()
        if self.destination == 'WRITE_TRUNCATE':
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        elif self.destination == 'WRITE_APPEND':
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        job_config.schema = self.schema_list
        job = self.client.load_table_from_dataframe(df_score, table, job_config = job_config, parquet_compression = 'snappy')
        return 'status: 200-ok'

class MessageToTg:
    def __init__(self, send_message):
        self.send_message = send_message

    def message_alarm(self):
        bot = telebot.TeleBot(str(os.getenv('bot_id')))
        for chat_id in os.getenv('list_of_chats').strip(',').split(', '):
            bot.send_message(int(chat_id), self.send_message)














