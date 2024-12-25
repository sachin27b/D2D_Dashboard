from mlflow import MlflowClient
import yaml

class Helper:

    def __init__(self,name=None):
        self.client = MlflowClient()
        # self.run_id = None
        with open('../config/config.yaml', 'r') as f:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
            
        if name == None:
            self.run_id = conf['xgboost']
        else:
            self.run_id = conf[name]
            
        self.run = self.client.get_run(self.run_id)
        self.art = self.client.list_artifacts(self.run_id)

    def get_run(self):

        return self.run
    
    def get_art(self):

        return self.art
    
    def get_metrics(self):

        return self.run.data.metrics

        
if __name__ == '__main__':

    obj = Helper()
    print(obj.run_id)

