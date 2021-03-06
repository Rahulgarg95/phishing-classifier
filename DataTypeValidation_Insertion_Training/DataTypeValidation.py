import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
from application_logging.logger import App_Logger
from AzureBlobStorage.azureBlobStorage import AzureBlobStorage
from MongoDB.mongoDbDatabase import mongoDBOperation


class dBOperation:
    """
      This class shall be used for handling all the SQL operations.
    """

    def __init__(self):
        self.path = 'Training_Database'
        self.badFilePath = "Training_Bad_Raw_Files_Validated"
        self.goodFilePath = "Training_Good_Raw_Files_Validated"
        self.logger = App_Logger()
        self.dbObj = mongoDBOperation()
        self.azureObj = AzureBlobStorage()

    def createTableDb(self, DatabaseName, column_names):
        """
            Method Name: createTableDb
            Description: This method creates a table in the given database which will be used to insert the Good data after raw data validation.
            Output: None
            On Failure: Raise Exception
        """

        try:
            self.dbObj.createOrGetCollection(DatabaseName, 'Good_Raw_Data')
            file = 'DbTableCreateLog'
            self.logger.log(file, "Tables created successfully!!")
            file = 'DataBaseConnectionLog'
            self.logger.log(file, "Closed %s database successfully" % DatabaseName)

        except Exception as e:
            file = 'DbTableCreateLog'
            self.logger.log(file, "Error while creating table: %s " % e)
            file = 'DataBaseConnectionLog'
            self.logger.log(file, "Closed %s database successfully" % DatabaseName)
            raise e

    def insertIntoTableGoodData(self, Database):

        """
           Method Name: insertIntoTableGoodData
           Description: This method inserts the Good data files from the Good_Raw folder into the
                        above created table.
           Output: None
           On Failure: Raise Exception
        """
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = self.azureObj.listDirFiles(goodFilePath)
        log_file = 'DbInsertLog'
        self.dbObj.dropCollection(Database, 'Good_Raw_Data')
        for file in onlyfiles:
            try:
                df_csv = self.azureObj.csvToDataframe(self.goodFilePath, file)
                self.dbObj.dataframeToRecords(Database, 'Good_Raw_Data', df_csv)

            except Exception as e:
                self.logger.log(log_file, "Error while creating table: %s " % e)
                self.azureObj.moveFileToFolder(goodFilePath, badFilePath, file)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
        print('Data pushed to mongodb...')

    def selectingDatafromtableintocsv(self, Database):

        """
           Method Name: selectingDatafromtableintocsv
           Description: This method exports the data in GoodData table as a CSV file. in a given location.
                        above created.
           Output: None
           On Failure: Raise Exception
        """

        self.fileFromDb = 'Training_FileFromDB'
        self.fileName = 'InputFile.csv'
        log_file = 'ExportToCsv'
        try:
            tmp_csv = self.dbObj.recordsToDataFrame(Database, 'Good_Raw_Data')
            print(tmp_csv.head())
            self.azureObj.saveDataframeToCsv(self.fileFromDb, self.fileName, tmp_csv)
            print('Saving data to final csv')
            print(tmp_csv.shape)
            self.logger.log(log_file, "File exported successfully!!!")

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)