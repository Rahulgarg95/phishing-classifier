"""
This is the Entry point for Training the Machine Learning Model.
"""


# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
from MongoDB.mongoDbDatabase import mongoDBOperation
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from Email_Trigger.send_email import email
from datetime import datetime
#Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = 'ModelTrainingLog'
        self.dbObj = mongoDBOperation()
        self.performance_list = []
        self.emailObj = email()

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()


            """doing the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            #data=preprocessor.remove_columns(data,['Wafer']) # remove the unnamed column as it doesn't contribute to prediction.

            #removing unwanted columns as discussed in the EDA part in ipynb file
            #data = preprocessor.dropUnnecessaryColumns(data,['veiltype'])

            #repalcing '?' values with np.nan as discussed in the EDA part

            data = preprocessor.replaceInvalidValuesWithNull(data)



            # check if missing values are present in the dataset
            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                data=preprocessor.impute_missing_values(data,cols_with_missing_values) # missing value imputation


            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='Result')

            """ Applying the clustering approach"""

            kmeans=clustering.KMeansClustering(self.file_object,self.log_writer) # object initialization.
            number_of_clusters=kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels']=Y

            # getting the unique clusters from our dataset
            list_of_clusters=X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data=X[X['Cluster']==i] # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=36, stratify=cluster_label)

                model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

                #getting the best model for each of the clusters
                best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test,i)

                #saving the best model to the directory.
                file_op = file_methods.File_Operation(self.file_object,self.log_writer)
                save_model=file_op.save_model(best_model,best_model_name+str(i))
                self.performance_list.extend(model_finder.perf_data)

            # logging the successful Training
            print(self.performance_list)
            print(type(self.performance_list))
            print('Inserting Performance Metrics to MongoDB')
            for dict_l in self.performance_list:
                self.dbObj.insertOneRecord('mushroomClassifierDB', 'performance_metrics', dict_l)
            self.log_writer.log(self.file_object, 'Successful End of Training')
            print('Successfully end training')

            # Triggering Email
            msg = MIMEMultipart()
            msg['Subject'] = 'Phishing Classifier - Model Train | ' + str(datetime.now())
            body = 'Model Training Done Successfully. Please find the models in models/ directory... <br><br> Thanks and Regards, <br> Rahul Garg'
            msg.attach(MIMEText(body, 'html'))
            to_addr = ['rahulgarg366@gmail.com']
            self.emailObj.trigger_mail(to_addr, [], msg)

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training: ' + e)
            raise Exception