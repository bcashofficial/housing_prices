# database_helpers.py

# ##########################################################################################
# Imports.
# ##########################################################################################

import pandas as pd
import sqlalchemy
import numpy as np
from configparser import RawConfigParser

import urllib.parse as up # Python 3+

# ##########################################################################################
# Parsing configuration file. 
# ##########################################################################################
		
def get_config_file_path():		
	
	# THIS VALUE SHOULD NEVER CHANGE. CAN'T RETRIEVE DIRECTLY FROM DB as it would cause Circular Dependency.
	config_file_path = '\\opt\\api\\config\\python_config.ini' 
	
	return config_file_path
	
def remove_config_comments(value):
	try:
		comment_delimeter = '|'
		comment_delimeter_index = value.index(comment_delimeter)
		result = value[:comment_delimeter_index].strip()
		
		return result
	except:
		result = value.strip()
		return result
	
def read_config_file(section, key):
	config_parser = RawConfigParser()   
	config_file_path = get_config_file_path()
	config_parser.read(config_file_path)
	result = config_parser.get(section, key)
	result = remove_config_comments(result)
		
	return result
	
# ##########################################################################################
# Connecting & closing the database.
# ##########################################################################################

# Prerequisite Variables.

mysql_connection_string_prefix 	= read_config_file('MysqlConnectionValues', 'MysqlConnectionStringPrefix')
mysql_port_number 				= read_config_file('MysqlConnectionValues', 'MysqlPortNumber')
colon_value						= ':'
at_value 						= '@'
slash_value 					= '/'

# mysql_connection_string variable can be created from this method and passed to connect_to_database(connection_string)

def get_mysql_connection_string_credentials(server, username, password):
	result = mysql_connection_string_prefix + username + colon_value + up.quote(password, safe = '')
	result = result + at_value + server + mysql_port_number + slash_value
	
	return result
	
def connect_to_environment():
	
	run_local = int(read_config_file('Instance', 'Local'))
	run_dev = int(read_config_file('Instance', 'DEV'))
	run_qa = int(read_config_file('Instance', 'QA'))
	run_stage = int(read_config_file('Instance', 'STAGE'))
	run_prod = int(read_config_file('Instance', 'PROD'))
	
	if (run_local == 1):
		server_param = read_config_file('DBServers', 'Local')
		username_param = read_config_file('DBServerUsernames', 'Local')
		password_param = read_config_file('DBServerPasswords', 'Local')
	
	elif (run_dev == 1):
		server_param = read_config_file('DBServers', 'DEV')
		username_param = read_config_file('DBServerUsernames', 'DEV')
		password_param = read_config_file('DBServerPasswords', 'DEV')
	
	elif (run_qa == 1):
		server_param = read_config_file('DBServers', 'QA')
		username_param = read_config_file('DBServerUsernames', 'QA')
		password_param = read_config_file('DBServerPasswords', 'QA')
	
	elif (run_stage == 1):
		server_param = read_config_file('DBServers', 'STAGE')
		username_param = read_config_file('DBServerUsernames', 'STAGE')
		password_param = read_config_file('DBServerPasswords', 'STAGE')
		
	elif (run_prod == 1):
		server_param = read_config_file('DBServers', 'PROD')
		username_param = read_config_file('DBServerUsernames', 'PROD')
		password_param = read_config_file('DBServerPasswords', 'PROD')
	
	mysql_connection_string = get_mysql_connection_string_credentials(server_param, username_param, password_param)
	
	return mysql_connection_string

def connect_to_database(connection_string):
	database_engine = sqlalchemy.create_engine(connection_string)
	
	try:
		database_connection = database_engine.connect()
	except Exception as error:
		print(repr(error))
			
	return database_engine
	
def close_database(database_connection):
	try:
		database_connection.close()
	except Exception as error:
		print(repr(error))
		
# ##########################################################################################
# Dataframe operations.
# ##########################################################################################

def query_to_dataframe(database_connection, query_to_execute):
	try:
		data_frame = (pd.read_sql(query_to_execute, con=database_connection).sort_index(ascending=False))
	except Exception as error:
		print(repr(error))
	
	return data_frame
	
def dataframe_to_pivot(data_frame, displayed_values, index_list, pivot_columns):
	try:
		pivot_table = pd.pivot_table(data_frame, values = displayed_values, index = index_list, columns = pivot_columns)
	except Exception as error:
		print(repr(error))
		
	return pivot_table
	
def dataframe_to_pivot_with_summary(data_frame, displayed_values, index_list, pivot_columns):
	try:
		pivot_table = pd.pivot_table(data_frame, values = displayed_values, index = index_list, columns = pivot_columns, aggfunc = sum, margins=True)
	except Exception as error:
		print(repr(error))
		
	return pivot_table
	
def pivot_to_dataframe(python_pivot_table):
	try:
		data_frame = (pd.DataFrame(python_pivot_table.to_records()))
	except Exception as error:
		print(repr(error))
	
	return data_frame
	
def write_df_to_csv(data_frame, csv_name):
	try:
		result = data_frame.to_csv(csv_name, sep=',', index = False, encoding = 'utf-8')
	except Exception as error:
		print(repr(error))
	
	return result
	
def write_df_to_excel(data_frame, file_path, spreadsheet_name):
	try:
		writer = pd.ExcelWriter(file_path, engine = 'xlsxwriter')
		data_frame.to_excel(writer, index = False, sheet_name=spreadsheet_name)
		writer.save()
	except Exception as error:
		print(repr(error))

def write_df_to_text(data_frame, text_name):
	try:
		data_frame = data_frame[::-1]
		result = data_frame.to_csv(text_name, index = False, sep='\t', mode = 'a', encoding = 'utf-8', header = False)
	except Exception as error:
		print(repr(error))
	
	return result

def write_df_to_sql(data_frame, sql_table_name, sql_engine, sql_schema, append_type):
	try:
		result = data_frame.to_sql(sql_table_name, sql_engine, schema=sql_schema, if_exists=append_type, index = False)
	except Exception as error:
		print(repr(error))
		
def combine_data_frames(data_frame_one, data_frame_two):
	try:
		result = pd.concat([data_frame_one, data_frame_two], axis = 0, ignore_index = True)
	except Exception as error:
		print(repr(error))
		
	return result
		
# ##########################################################################################
# Database operations.
# ##########################################################################################

def retrieve_result(database_cursor, query):
	try:
		database_cursor.execute(query)
		
		result = database_cursor.fetchone()
		
		if result is not None:
			return result[0]
	except Exception as error:
		print(repr(error))
	
# ##########################################################################################
# ##########################################################################################