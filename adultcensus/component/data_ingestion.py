import sys

from numpy.lib.shape_base import split
from adultcensus.entity.config_entity import DataIngestionConfig
import sys,os
from adultcensus.exception import AdultCensusException
from adultcensus.logger import logging
from adultcensus.entity.artifact_entity import DataIngestionArtifact
import tarfile
import numpy as np
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:

    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AdultCensusException(e,sys)

        
    def download_adultcensus_data(self,) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

             #folder location to download file
            download_dir = self.data_ingestion_config.download_dir

            if os.path.exists(download_dir):
                os.remove(download_dir)

            os.makedirs(download_dir,exist_ok =True)

            adultcensus_file_name = os.path.basename(download_url)

            file_path = os.path.join(download_dir,adultcensus_file_name)
            
            logging.info(f"Downloading file from :[{download_url}] into :[{file_path}]")
            urllib.request.urlretrieve(download_url,file_path)
            logging.info(f"file : [{file_path}] has been downloaded successfully.")
            return file_path

        except Exception as e:
            raise AdultCensusException(e,sys) from e


    
    def extract_file(self,file_path:str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok =True)

            logging.info(f"Extracting  file : [{file_path}] into dir: [{raw_data_dir}]")
            with tarfile.open(file_path) as adultcensus_file_obj:

                adultcensus_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")

        except Exception as e:
            raise AdultCensusException(e,sys) from e

    
    def split_data_as_train_test(self)-> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            adultcensus_file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading csv file: [{adultcensus_file_path}]")
            adultcensus_data_frame = pd.read_csv(adultcensus_file_path)

            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, 
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                            file_name)     

            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index =False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)    
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index = False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                 )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")                     
            return data_ingestion_artifact


        except Exception as e:
            raise AdultCensusException(e,sys) from e

        
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            file_path = self.download_adultcensus_data()

            self.extract_file(file_path=file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise AdultCensusException(e,sys) from e



    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")

