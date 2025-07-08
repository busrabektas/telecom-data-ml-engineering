import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from imblearn.over_sampling import BorderlineSMOTE, SMOTE,ADASYN
from xgboost import XGBClassifier  # pip install xgboost
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

class SupervisedLearningPipeline:
    def __init__(self, data_path="clustering/segmented_customers_enhanced.csv"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train_reg = None
        self.y_val_reg = None
        self.y_test_reg = None
        self.y_train_clf = None
        self.y_val_clf = None
        self.y_test_clf = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.regression_models = {}
        self.classification_models = {}
        self.feature_names = None
        
    def load_and_prepare_data(self):
        print("Data is loading...")
        self.df = pd.read_csv(self.data_path)
        
        # general infos
        print(f"Data shape: {self.df.shape}")
        print(f"[INFO] Target variables:")
        print(f"  - DataUsageGB: {self.df['DataUsageGB'].describe()}")
        print(f"  - PlanType: {self.df['PlanType'].value_counts()}")
        
        return self.df
        
    def feature_engineering(self):

        print("Feature engineering")

        # One-hot eencode
        one_hot_cols = ['Gender', 'ServiceType', 'PaymentMethod']

        # Label encode
        label_cols = ['Location']

      
        for col in one_hot_cols:
            if col in self.df.columns:
                one_hot = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, one_hot], axis=1)
        for col in label_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))

        base_feature_cols = [
            'Age', 'Tenure', 'Location_encoded',
            'VoiceCallMinutes', 'SMSsent', 'IntlCallMinutes',
            'DataSpeed', 'Latency', 'NetworkIssues', 'DroppedCalls', 'SupportCalls',
            'SatisfactionScore', 'PaymentHistory',
            'CallsPerMonth', 'SMSPerMonth', 'SupportCallsPerMonth', 'SupportToIssueRatio',
            'IssueRate', 'LatencyRatio', 'AvgCallDuration',
            'Cluster', 'PCA1', 'PCA2'
        ]

        # add one hot features
        one_hot_generated_cols = [col for col in self.df.columns if any(col.startswith(prefix + "_") for prefix in one_hot_cols)]
        
        # concat all features
        self.feature_names = [col for col in base_feature_cols if col in self.df.columns] + one_hot_generated_cols

        print(f"Feature number that will be used: {len(self.feature_names)}")

        return self.feature_names

        

    
    def prepare_targets_and_split(self, test_size=0.2, val_size=0.2):
        
        features = self.feature_engineering()
        X = self.df[features].copy()
        X = X.fillna(X.median())
        
        # Regression target: DataUsageGB
        y_reg = self.df['DataUsageGB'].copy()
        
        # Classification target:PlanType
        y_clf = self.label_encoder.fit_transform(self.df['PlanType'].astype(str))
        
        X_train, X_temp, y_train_reg, y_temp_reg, y_train_clf, y_temp_clf = train_test_split(
            X, y_reg, y_clf, test_size=test_size+val_size, random_state=42, stratify=y_clf
        )
        
        X_val, X_test, y_val_reg, y_test_reg, y_val_clf, y_test_clf = train_test_split(
            X_temp, y_temp_reg, y_temp_clf, test_size=test_size/(test_size+val_size), 
            random_state=42, stratify=y_temp_clf
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Class attributes
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train_reg = y_train_reg
        self.y_val_reg = y_val_reg
        self.y_test_reg = y_test_reg
        self.y_train_clf = y_train_clf
        self.y_val_clf = y_val_clf
        self.y_test_clf = y_test_clf
        
        print(f"[Train size: {X_train.shape[0]}")
        print(f"Validation size: {X_val.shape[0]}")
        print(f"est size: {X_test.shape[0]}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_regression_models(self):
        print("\ntraining regression models")
        
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n[INFO] {name} eğitiliyor...")
            
            # train model
            model.fit(self.X_train, self.y_train_reg)
            
            # Validation predictions
            y_val_pred = model.predict(self.X_val)
            
            # Metrics
            mse = mean_squared_error(self.y_val_reg, y_val_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_val_reg, y_val_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_val_pred
            }
            
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - R²: {r2:.4f}")
            

        self.regression_models = results
        return results
    


    
    
    def train_classification_models(self):

        print("\nClassification models training with different oversampling techniques")

        samplers = {
            "SMOTE": SMOTE(random_state=42),
            "ADASYN": ADASYN(random_state=42),
            "BorderlineSMOTE": BorderlineSMOTE(random_state=42)
        }

        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        }

        all_results = {}

        for sampler_name, sampler in samplers.items():
            print(f"\n-- {sampler_name}--")
            
            # Resample
            print(f"{sampler_name} applied...")
            X_resampled, y_resampled = sampler.fit_resample(self.X_train, self.y_train_clf)
            print(f"[INFO] {sampler_name} sonrası X: {X_resampled.shape}, y: {np.bincount(y_resampled)}")

            sampler_results = {}

            for model_name, model in models.items():
                print(f"\ntraining {model_name} ({sampler_name})")
                model.fit(X_resampled, y_resampled)

                y_val_pred = model.predict(self.X_val)
                y_val_pred_proba = model.predict_proba(self.X_val)

                accuracy = model.score(self.X_val, self.y_val_clf)
                try:
                    auc = roc_auc_score(self.y_val_clf, y_val_pred_proba, multi_class='ovr')
                except:
                    auc = None

                print(f"  - Accuracy: {accuracy:.4f}")
                print(f"  - AUC: {auc:.4f}" if auc else "  - AUC hesaplanamadı")

                sampler_results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_val_pred,
                    'probabilities': y_val_pred_proba
                }

            all_results[sampler_name] = sampler_results

        # determine best model (max accuracy)
        best_model_info = None
        best_accuracy = -1
        for sampler_name, models_dict in all_results.items():
            for model_name, info in models_dict.items():
                if info['accuracy'] > best_accuracy:
                    best_model_info = (sampler_name, model_name, info)
                    best_accuracy = info['accuracy']

        best_sampler, best_model_name, best_info = best_model_info
        print(f"\n[INFO] En iyi model: {best_model_name} ({best_sampler}) - Accuracy: {best_accuracy:.4f}")

        self.classification_models = {
            f"{best_model_name}_{best_sampler}": best_info
        }
        return self.classification_models
        
    def feature_importance_analysis(self):
        """Feature importance analizi"""
        print("\n[INFO] Feature importance analizi yapılıyor...")
        
        # best regresion model
        best_reg_model = max(self.regression_models.items(), key=lambda x: x[1]['r2'])
        print(f"[INFO] En iyi regression model: {best_reg_model[0]}")
        
        # best classification model
        best_clf_model = max(self.classification_models.items(), key=lambda x: x[1]['accuracy'])
        print(f"[INFO] En iyi classification model: {best_clf_model[0]}")
        
        # feature importance plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Regression feature importance
        if hasattr(best_reg_model[1]['model'], 'feature_importances_'):
            reg_importance = best_reg_model[1]['model'].feature_importances_
            reg_indices = np.argsort(reg_importance)[::-1][:15]  
            
            axes[0].bar(range(len(reg_indices)), reg_importance[reg_indices])
            axes[0].set_title(f'Top 15 Features - {best_reg_model[0]} (Regression)')
            axes[0].set_xticks(range(len(reg_indices)))
            axes[0].set_xticklabels([self.feature_names[i] for i in reg_indices], rotation=45)
        
        # Classification feature importance
        if hasattr(best_clf_model[1]['model'], 'feature_importances_'):
            clf_importance = best_clf_model[1]['model'].feature_importances_
            clf_indices = np.argsort(clf_importance)[::-1][:15] 
            
            axes[1].bar(range(len(clf_indices)), clf_importance[clf_indices])
            axes[1].set_title(f'Top 15 Features - {best_clf_model[0]} (Classification)')
            axes[1].set_xticks(range(len(clf_indices)))
            axes[1].set_xticklabels([self.feature_names[i] for i in clf_indices], rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_reg_model, best_clf_model
        
    def evaluate_on_test_set(self, best_reg_model, best_clf_model):
        print("\n[INFO] Test setinde final evaluation yapılıyor...")

        # regression test
        reg_model = best_reg_model[1]['model']
        y_test_pred_reg = reg_model.predict(self.X_test)

        test_mse = mean_squared_error(self.y_test_reg, y_test_pred_reg)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test_reg, y_test_pred_reg)

        print(f"\n[REGRESSION RESULTS - {best_reg_model[0]}]")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}")

        # classification test
        clf_model = best_clf_model[1]['model']
        y_test_pred_clf = clf_model.predict(self.X_test)
        y_test_pred_proba_clf = clf_model.predict_proba(self.X_test)

        test_accuracy = clf_model.score(self.X_test, self.y_test_clf)
        try:
            test_auc = roc_auc_score(self.y_test_clf, y_test_pred_proba_clf, multi_class='ovr')
        except:
            test_auc = None

        print(f"\n[CLASSIFICATION RESULTS - {best_clf_model[0]}]")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        if test_auc:
            print(f"Test AUC: {test_auc:.4f}")

        #classification report
        print("\nClassification Report:")
        plan_types = self.label_encoder.classes_
        print(classification_report(self.y_test_clf, y_test_pred_clf, target_names=plan_types))

        # roc curve

        n_classes = len(np.unique(self.y_test_clf))
        y_test_binarized = label_binarize(self.y_test_clf, classes=range(n_classes))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_test_pred_proba_clf[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Multi-Class)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=300)
        plt.show()

        #confusion matrix

        cm = confusion_matrix(self.y_test_clf, y_test_pred_clf)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=plan_types, yticklabels=plan_types)
        plt.title("Confusion Matrix Heatmap")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.show()

        # regression actual vs predict
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test_reg, y_test_pred_reg, alpha=0.5)
        plt.plot([self.y_test_reg.min(), self.y_test_reg.max()],
                [self.y_test_reg.min(), self.y_test_reg.max()], 'r--')
        plt.xlabel("Actual Data Usage (GB)")
        plt.ylabel("Predicted Data Usage (GB)")
        plt.title("Regression: Actual vs Predicted")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("regression_actual_vs_predicted.png", dpi=300)
        plt.show()

        return {
            'regression': {
                'model': reg_model,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'predictions': y_test_pred_reg
            },
            'classification': {
                'model': clf_model,
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'predictions': y_test_pred_clf,
                'probabilities': y_test_pred_proba_clf
            }
        }
        
    def save_models(self, results):
        """Modelları kaydet"""
        print("\n[INFO] Modeller kaydediliyor...")
        
        # Best models
        reg_model = results['regression']['model']
        clf_model = results['classification']['model']
        
        # Save models
        joblib.dump(reg_model, 'best_regression_model.pkl')
        joblib.dump(clf_model, 'best_classification_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        # Save feature names
        with open('feature_names.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        print("Models are save:")
        print("  - best_regression_model.pkl")
        print("  - best_classification_model.pkl")
        print("  - feature_scaler.pkl")
        print("  - label_encoder.pkl")
        print("  - feature_names.txt")
    
    def run_full_pipeline(self):
        print("SUPERVISED LEARNING PIPELINE STARTED")
        
        self.load_and_prepare_data()
        self.prepare_targets_and_split()
        
        reg_results = self.train_regression_models()
        clf_results = self.train_classification_models()
        
        best_reg, best_clf = self.feature_importance_analysis()
        final_results = self.evaluate_on_test_set(best_reg, best_clf)        
        self.save_models(final_results)
        
        
        print("SUPERVISED LEARNING PIPELINE COMPLETED")
        
        return final_results


if __name__ == "__main__":
    pipeline = SupervisedLearningPipeline()
    pipeline.run_full_pipeline()