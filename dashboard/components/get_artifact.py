import mlflow
import json
import pandas as pd
import shutil

class DownloadArtifact:
    
    def __init__(self,helper):

        self.helper = helper

        for i in range(len(helper.art)):
            if(self.helper.art[i].path.startswith('confusion_matrix')):
                self.confmat_artifact_index = i
            if(self.helper.art[i].path.startswith('roc_curve')):
                self.roc_artifact_index = i
            if self.helper.art[i].path in ['XGBClassifier', 'LogisticRegression', 'DtClassifier','GBTClassifier',
                                           'model_XGBRegressor','model_LinearRegression','model_GBTRegressor','model_RandomForestRegressor','model_DecisionTreeRegressor']:
                self.model_artifact_index = i
            if self.helper.art[i].path.startswith('feature_importance'):
                self.feat_importance_artifact_index = i
            if self.helper.art[i].path.startswith('actual_vs_predicted'):
                self.actual_vs_predicted_index = i
    
    def get_confusion_matrix(self):
        artifact_path = self.helper.client.download_artifacts(self.helper.run_id,self.helper.art[self.confmat_artifact_index].path,dst_path='artifacts')
        return artifact_path
    
    def get_auc(self):
        artifact_path = self.helper.client.download_artifacts(self.helper.run_id,self.helper.art[self.roc_artifact_index].path,dst_path='artifacts')
        return artifact_path
    
    def get_feat_importances(self):
        artifact_path = self.helper.client.download_artifacts(self.helper.run_id,self.helper.art[self.feat_importance_artifact_index].path,dst_path='artifacts')
        return artifact_path
    
    def get_model(self):
        artifact_path = self.helper.client.download_artifacts(self.helper.run_id,self.helper.art[self.model_artifact_index].path,dst_path='artifacts')
        return artifact_path

    def get_actual_vs_predicted(self):
        artifact_path = self.helper.client.download_artifacts(self.helper.run_id,self.helper.art[self.actual_vs_predicted_index].path,dst_path='artifacts')
        return artifact_path

