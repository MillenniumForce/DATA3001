from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class MultiClassRFClassifier():
    def __init__(self, multilabel=False):
        self.multilabel = multilabel

    def fit(self,X,Y):
        self.models = {colname: RandomForestClassifier() for colname in Y.columns}

        for colname in Y.columns:
            self.models[colname].fit(X,Y[colname])
        
        return self
    
    def predict_proba(self, X):
        predictions = {k:pd.Series(self.models[k].predict_proba(X)[:,1]) for k in self.models.keys()}
        final = pd.DataFrame(predictions)
        # print(final)
        return final

    def predict(self, X, override_multilabel=None):
        multilabel = self.multilabel
        if override_multilabel is not None:
            multilabel = override_multilabel

        if multilabel:
            predictions = {k:pd.Series(self.models[k].predict(X)) for k in self.models.keys()}
            return pd.DataFrame(predictions)
        else:
            probs = self.predict_proba(X)
            maxes = self.predict_proba(X).max(axis=1)
            final = pd.DataFrame({col:(probs[col]==maxes).astype(int) for col in probs.columns})*self.predict(X,override_multilabel=True)
            return final
            

