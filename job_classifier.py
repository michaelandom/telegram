import pandas as pd
import numpy as np
import time
import logging
import os
import sys
import multiprocessing as mp
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score,
    adjusted_rand_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
from custom_bert import train_bert_model
from bert_trainer import BertTrainer
warnings.filterwarnings('ignore')


class BertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='bert-base-uncased', num_labels=2, device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, X, y=None):
        # Placeholder: Typically BERT is fine-tuned with PyTorch training loops
        print("BERT fine-tuning requires a custom training loop. Skipping fit for now.")
        return self

    def predict(self, X):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(pred[0])
        return all_preds

    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs[0])
        return all_probs


def setup_logging():
    """Setup comprehensive logging configuration"""
    log_filename = f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def timer_decorator(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(
            f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


logger = setup_logging()
logger.info("=== Enhanced ML Pipeline Starting ===")



@timer_decorator
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    logger.info("Loading dataset...")
    try:
        data = pd.read_csv("training_dataset.csv")
        logger.info(
            f"Dataset loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")

        missing_data = data.isnull().sum()
        if missing_data.any():
            logger.warning(
                f"Missing values found:\n{missing_data[missing_data > 0]}")
            data = data.dropna(subset=["combined_description_text", "title"])
            logger.info(f"After removing missing values: {data.shape[0]} rows")

        X = data["combined_description_text"]
        y = data["title"]

        if len(X) == 0 or len(y) == 0:
            raise ValueError("Dataset is empty after preprocessing")

        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Unique job titles: {y.nunique()}")
        logger.info(
            f"Average description length: {X.str.len().mean():.1f} characters")

        return X, y
    except FileNotFoundError:
        logger.error("Dataset file 'huhu_output.csv' not found!")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


@timer_decorator
def prepare_features_enhanced(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Enhanced feature preparation with Train-Validation-Test split
    Default ratios: 70% train, 15% validation, 15% test
    """
    logger.info(
        f"Enhanced data preparation with ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    class_counts = y.value_counts()
    logger.info(f"Original classes: {len(class_counts)}")
    logger.info(f"Class distribution (top 10):\n{class_counts.head(10)}")

    rare_classes = class_counts[class_counts == 1]
    if len(rare_classes) > 0:
        logger.warning(
            f"Found {len(rare_classes)} classes with only 1 instance")
        logger.info("Removing rare classes to enable stratified splitting...")

        mask = y.isin(class_counts[class_counts > 1].index)
        X_filtered = X[mask].reset_index(drop=True)
        y_filtered = y[mask].reset_index(drop=True)

        logger.info(
            f"After removing rare classes: {len(X_filtered)} samples, {y_filtered.nunique()} classes")
        X, y = X_filtered, y_filtered

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info(f"Final number of classes: {len(label_encoder.classes_)}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded,
        test_size=test_ratio,
        random_state=42,
        stratify=y_encoded
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=42,
        stratify=y_temp
    )

    logger.info(f"Data split sizes:")
    logger.info(
        f"  Training set: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(
        f"  Validation set: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"  Test set: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    for split_name, split_data in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique_classes = len(np.unique(split_data))
        logger.info(f"  {split_name} set classes: {unique_classes}")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, X, y


@timer_decorator
def vectorize_text_enhanced(X_train, X_val, X_test, X_all):
    """Enhanced TF-IDF vectorization"""
    logger.info("Creating enhanced TF-IDF vectors...")

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=2000,  
        ngram_range=(1, 3),  
        min_df=2,
        max_df=0.95,
        sublinear_tf=True  
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    X_all_tfidf = tfidf.transform(X_all)

    logger.info(f"TF-IDF matrix shapes:")
    logger.info(f"  Train: {X_train_tfidf.shape}")
    logger.info(f"  Validation: {X_val_tfidf.shape}")
    logger.info(f"  Test: {X_test_tfidf.shape}")
    logger.info(
        f"  Total vocabulary size: {len(tfidf.get_feature_names_out())}")

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, X_all_tfidf, tfidf


def calculate_comprehensive_metrics(y_true, y_pred, average_types=['macro', 'micro', 'weighted']):
    """Calculate comprehensive classification metrics"""
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    for avg in average_types:
        metrics[f'precision_{avg}'] = precision_score(
            y_true, y_pred, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(
            y_true, y_pred, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(
            y_true, y_pred, average=avg, zero_division=0)

    return metrics


@timer_decorator
def perform_cross_validation(models, X_train_tfidf, y_train, cv_folds=5):
    """Perform cross-validation for all models"""
    logger.info(f"Performing {cv_folds}-fold cross-validation...")

    cv_results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        logger.info(f"Cross-validating {name}...")
        try:
            cv_scores = cross_val_score(model, X_train_tfidf, y_train,
                                        cv=skf, scoring='accuracy', n_jobs=-1)

            f1_scores = cross_val_score(model, X_train_tfidf, y_train,
                                        cv=skf, scoring='f1_macro', n_jobs=-1)

            cv_results[name] = {
                'accuracy_mean': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'f1_macro_mean': f1_scores.mean(),
                'f1_macro_std': f1_scores.std()
            }

            logger.info(f"{name} CV Results:")
            logger.info(
                f"  Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info(
                f"  F1-Macro: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")

        except Exception as e:
            logger.error(f"Error in cross-validation for {name}: {str(e)}")
            cv_results[name] = {
                'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                'f1_macro_mean': 0.0, 'f1_macro_std': 0.0
            }

    return cv_results


@timer_decorator
def hyperparameter_tuning(X_train_tfidf, y_train, X_val_tfidf, y_val):
    """Perform hyperparameter tuning for selected models"""
    logger.info("Performing hyperparameter tuning...")

    tuned_models = {}

    # Logistic Regression hyperparameter tuning
    logger.info("Tuning Logistic Regression...")
    lr_params = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    }

    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42),
        lr_params,
        cv=3,  
        scoring='f1_macro',
        n_jobs=-1
    )
    lr_grid.fit(X_train_tfidf, y_train)
    tuned_models['Logistic Regression'] = lr_grid.best_estimator_
    logger.info(f"Best LR params: {lr_grid.best_params_}")
    logger.info(f"Best LR CV score: {lr_grid.best_score_:.4f}")

    logger.info("Tuning Random Forest...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_random = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        n_iter=20,  
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )
    rf_random.fit(X_train_tfidf, y_train)
    tuned_models['Random Forest'] = rf_random.best_estimator_
    logger.info(f"Best RF params: {rf_random.best_params_}")
    logger.info(f"Best RF CV score: {rf_random.best_score_:.4f}")

    # XGBoost hyperparameter tuning
    logger.info("Tuning XGBoost...")
    xgb_params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_random = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0),
        xgb_params,
        n_iter=15,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )
    xgb_random.fit(X_train_tfidf, y_train)
    tuned_models['XGBoost'] = xgb_random.best_estimator_
    logger.info(f"Best XGB params: {xgb_random.best_params_}")
    logger.info(f"Best XGB CV score: {xgb_random.best_score_:.4f}")

    return tuned_models


@timer_decorator
def train_and_evaluate_models(X_train_tfidf, X_val_tfidf, X_test_tfidf,
                              y_train, y_val, y_test, tuned_models=None):
    """Train and comprehensively evaluate models"""
    logger.info("Training and evaluating models with comprehensive metrics...")

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB(),
        # "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    if tuned_models:
        for name in tuned_models:
            base_models[f"{name}_Tuned"] = tuned_models[name]

    results = {}
    trained_models = {}

    for name, model in base_models.items():
        logger.info(f"Training and evaluating {name}...")
        start_time = time.time()

        try:
            model.fit(X_train_tfidf, y_train)

            val_pred = model.predict(X_val_tfidf)
            val_metrics = calculate_comprehensive_metrics(y_val, val_pred)

            test_pred = model.predict(X_test_tfidf)
            test_metrics = calculate_comprehensive_metrics(y_test, test_pred)

            results[name] = {
                'validation': val_metrics,
                'test': test_metrics,
                'training_time': time.time() - start_time
            }
            trained_models[name] = model

            logger.info(f"{name} Results:")
            logger.info(
                f"  Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1-Macro: {val_metrics['f1_macro']:.4f}")
            logger.info(
                f"  Test - Accuracy: {test_metrics['accuracy']:.4f}, F1-Macro: {test_metrics['f1_macro']:.4f}")
            logger.info(
                f"  Training time: {results[name]['training_time']:.2f}s")

        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            results[name] = {
                'validation': {'accuracy': 0.0, 'f1_macro': 0.0},
                'test': {'accuracy': 0.0, 'f1_macro': 0.0},
                'training_time': 0.0
            }

    return results, trained_models


@timer_decorator
def train_deep_learning_model_enhanced(X_train_tfidf, X_val_tfidf, X_test_tfidf,
                                       y_train, y_val, y_test, num_classes):
    """Enhanced deep learning model with validation and early stopping"""
    logger.info("Training enhanced deep learning model...")

    try:
        # Convert to dense arrays
        X_train_dl = X_train_tfidf.toarray()
        X_val_dl = X_val_tfidf.toarray()
        X_test_dl = X_test_tfidf.toarray()

        # One-hot encode targets
        y_train_dl = to_categorical(y_train, num_classes=num_classes)
        y_val_dl = to_categorical(y_val, num_classes=num_classes)
        y_test_dl = to_categorical(y_test, num_classes=num_classes)

        # Build enhanced model
        dl_model = Sequential([
            Dense(256, input_shape=(X_train_dl.shape[1],), activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        dl_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )

        logger.info("Training neural network with validation...")
        history = dl_model.fit(
            X_train_dl, y_train_dl,
            validation_data=(X_val_dl, y_val_dl),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        # Evaluate on test set
        test_loss, test_accuracy = dl_model.evaluate(
            X_test_dl, y_test_dl, verbose=0)

        # Get predictions for detailed metrics
        test_pred_proba = dl_model.predict(X_test_dl, verbose=0)
        test_pred = np.argmax(test_pred_proba, axis=1)

        # Calculate comprehensive metrics
        dl_metrics = calculate_comprehensive_metrics(y_test, test_pred)

        logger.info(f"Deep Learning Results:")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Test F1-Macro: {dl_metrics['f1_macro']:.4f}")
        logger.info(
            f"  Training stopped at epoch: {len(history.history['loss'])}")

        return dl_metrics, dl_model, history

    except Exception as e:
        logger.error(f"Error training deep learning model: {str(e)}")
        return {'accuracy': 0.0, 'f1_macro': 0.0}, None, None


@timer_decorator
def perform_clustering_enhanced(X_all_tfidf, y_encoded):
    """Enhanced clustering analysis"""
    logger.info("Performing enhanced clustering analysis...")

    clustering_results = {}

    # KMeans clustering with multiple initializations
    try:
        # Cap at 20 for computational efficiency
        n_clusters = min(len(np.unique(y_encoded)), 20)
        logger.info(f"Running KMeans with {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        kmeans_labels = kmeans.fit_predict(X_all_tfidf)

        kmeans_ari = adjusted_rand_score(y_encoded, kmeans_labels)
        kmeans_silhouette = silhouette_score(X_all_tfidf, kmeans_labels)

        clustering_results["KMeans_ARI"] = kmeans_ari
        clustering_results["KMeans_Silhouette"] = kmeans_silhouette

        logger.info(
            f"KMeans - ARI: {kmeans_ari:.4f}, Silhouette: {kmeans_silhouette:.4f}")

    except Exception as e:
        logger.error(f"Error in KMeans clustering: {str(e)}")
        clustering_results["KMeans_ARI"] = 0
        clustering_results["KMeans_Silhouette"] = -1
        kmeans_labels = None

    # DBSCAN clustering with parameter optimization
    try:
        logger.info("Running DBSCAN clustering...")

        # Try different eps values
        best_dbscan_score = -1
        best_dbscan_labels = None
        best_eps = 0.5

        for eps in [0.3, 0.5, 0.7, 1.0]:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_all_tfidf)

            unique_labels = set(dbscan_labels)
            n_clusters_dbscan = len(unique_labels) - \
                (1 if -1 in dbscan_labels else 0)

            if n_clusters_dbscan > 1:
                valid_mask = dbscan_labels != -1
                if np.sum(valid_mask) > 1:
                    try:
                        silhouette = silhouette_score(
                            X_all_tfidf[valid_mask], dbscan_labels[valid_mask])
                        if silhouette > best_dbscan_score:
                            best_dbscan_score = silhouette
                            best_dbscan_labels = dbscan_labels
                            best_eps = eps
                    except:
                        continue

        if best_dbscan_labels is not None:
            valid_mask = best_dbscan_labels != -1
            dbscan_ari = adjusted_rand_score(
                y_encoded[valid_mask], best_dbscan_labels[valid_mask])

            clustering_results["DBSCAN_ARI"] = dbscan_ari
            clustering_results["DBSCAN_Silhouette"] = best_dbscan_score

            n_clusters = len(set(best_dbscan_labels)) - \
                (1 if -1 in best_dbscan_labels else 0)
            n_noise = list(best_dbscan_labels).count(-1)

            logger.info(
                f"DBSCAN (eps={best_eps}) - Clusters: {n_clusters}, Noise: {n_noise}")
            logger.info(
                f"DBSCAN - ARI: {dbscan_ari:.4f}, Silhouette: {best_dbscan_score:.4f}")
        else:
            clustering_results["DBSCAN_ARI"] = 0
            clustering_results["DBSCAN_Silhouette"] = -1
            logger.warning("DBSCAN: No meaningful clusters found")

    except Exception as e:
        logger.error(f"Error in DBSCAN clustering: {str(e)}")
        clustering_results["DBSCAN_ARI"] = 0
        clustering_results["DBSCAN_Silhouette"] = -1

    return clustering_results, kmeans_labels


def create_comprehensive_visualization(results, cv_results, history=None):
    """Create comprehensive visualizations of results"""
    logger.info("Creating comprehensive visualizations...")

    try:
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))

        # 1. Model Comparison - Test Accuracy
        plt.subplot(2, 3, 1)
        model_names = []
        test_accuracies = []

        for name, result in results.items():
            if 'test' in result:
                model_names.append(name.replace('_', ' '))
                test_accuracies.append(result['test']['accuracy'])

        bars = plt.bar(range(len(model_names)), test_accuracies)
        plt.title('Model Comparison - Test Accuracy',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(model_names)),
                   model_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # Color bars by performance
        for i, bar in enumerate(bars):
            if test_accuracies[i] == max(test_accuracies):
                bar.set_color('gold')
            elif test_accuracies[i] >= np.percentile(test_accuracies, 75):
                bar.set_color('lightgreen')
            else:
                bar.set_color('lightcoral')

        # 2. Cross-Validation Results
        plt.subplot(2, 3, 2)
        cv_names = list(cv_results.keys())
        cv_means = [cv_results[name]['accuracy_mean'] for name in cv_names]
        cv_stds = [cv_results[name]['accuracy_std'] for name in cv_names]

        plt.errorbar(range(len(cv_names)), cv_means, yerr=cv_stds,
                     fmt='o-', capsize=5, capthick=2)
        plt.title('Cross-Validation Results', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('CV Accuracy')
        plt.xticks(range(len(cv_names)), cv_names, rotation=45, ha='right')
        plt.grid(alpha=0.3)

        # 3. Precision-Recall-F1 Heatmap (for best model)
        plt.subplot(2, 3, 3)
        best_model_name = max(results.keys(),
                              key=lambda x: results[x]['test']['accuracy'] if 'test' in results[x] else 0)

        if 'test' in results[best_model_name]:
            metrics_data = []
            metric_names = ['Precision', 'Recall', 'F1-Score']
            avg_types = ['macro', 'micro', 'weighted']

            for avg in avg_types:
                row = []
                for metric in ['precision', 'recall', 'f1']:
                    key = f"{metric}_{avg}"
                    row.append(results[best_model_name]['test'].get(key, 0))
                metrics_data.append(row)

            sns.heatmap(metrics_data, annot=True, fmt='.3f',
                        xticklabels=metric_names, yticklabels=avg_types,
                        cmap='Blues', cbar_kws={'shrink': 0.8})
            plt.title(
                f'Detailed Metrics - {best_model_name}', fontsize=14, fontweight='bold')

        # 4. Training Time vs Performance
        plt.subplot(2, 3, 4)
        times = []
        accuracies = []
        names = []

        for name, result in results.items():
            if 'test' in result and 'training_time' in result:
                times.append(result['training_time'])
                accuracies.append(result['test']['accuracy'])
                names.append(name)

        scatter = plt.scatter(times, accuracies, s=100, alpha=0.7)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Test Accuracy')
        plt.title('Performance vs Training Time',
                  fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)

        # Add labels for each point
        for i, name in enumerate(names):
            plt.annotate(name.replace('_', ' '), (times[i], accuracies[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 5. Deep Learning Training History (if available)
        plt.subplot(2, 3, 5)
        if history is not None:
            epochs = range(1, len(history.history['accuracy']) + 1)
            plt.plot(
                epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
            plt.plot(
                epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.title('Deep Learning Training History',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No Deep Learning\nHistory Available',
                     transform=plt.gca().transAxes, ha='center', va='center',
                     fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            plt.title('Deep Learning Training History',
                      fontsize=14, fontweight='bold')

        # 6. Model Complexity vs Performance
        plt.subplot(2, 3, 6)
        complexity_scores = {
            'Naive Bayes': 1, 'Logistic Regression': 2, 'KNN': 2,
            'SVM': 3, 'Random Forest': 4, 'XGBoost': 5, 'Deep Learning': 6
        }

        x_complexity = []
        y_performance = []
        model_labels = []

        for name, result in results.items():
            if 'test' in result:
                base_name = name.replace('_Tuned', '').replace('_', ' ')
                if base_name in complexity_scores:
                    x_complexity.append(complexity_scores[base_name])
                    y_performance.append(result['test']['accuracy'])
                    model_labels.append(name.replace('_', ' '))

        if x_complexity:
            plt.scatter(x_complexity, y_performance, s=100, alpha=0.7)
            plt.xlabel('Model Complexity')
            plt.ylabel('Test Accuracy')
            plt.title('Model Complexity vs Performance',
                      fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)

            for i, label in enumerate(model_labels):
                plt.annotate(label, (x_complexity[i], y_performance[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()

        # Save the comprehensive plot
        plot_filename = f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Comprehensive visualization saved as {plot_filename}")

    except Exception as e:
        logger.error(f"Error creating comprehensive visualization: {str(e)}")


def create_confusion_matrix_plot(y_true, y_pred, label_encoder, model_name):
    """Create confusion matrix visualization for best model"""
    try:
        # Limit to top classes to keep visualization readable
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 15:
            # Show only most common classes
            class_counts = pd.Series(y_true).value_counts()
            top_classes = class_counts.head(15).index.tolist()
            mask = np.isin(y_true, top_classes)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred
            top_classes = unique_classes

        cm = confusion_matrix(
            y_true_filtered, y_pred_filtered, labels=top_classes)

        plt.figure(figsize=(12, 10))
        class_names = [label_encoder.inverse_transform(
            [cls])[0][:20] for cls in top_classes]  # Truncate long names

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_filename = f"confusion_matrix_{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Confusion matrix saved as {cm_filename}")

    except Exception as e:
        logger.error(f"Error creating confusion matrix: {str(e)}")


def generate_detailed_report(results, cv_results, clustering_results, label_encoder):
    """Generate a detailed performance report"""
    logger.info("Generating detailed performance report...")

    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE ML PIPELINE PERFORMANCE REPORT")
    report.append("="*80)
    report.append(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total number of classes: {len(label_encoder.classes_)}")
    report.append("")

    # Cross-validation summary
    report.append("CROSS-VALIDATION RESULTS")
    report.append("-" * 40)
    for model, cv_result in cv_results.items():
        report.append(f"{model}:")
        report.append(
            f"  Accuracy: {cv_result['accuracy_mean']:.4f} (+/- {cv_result['accuracy_std']*2:.4f})")
        report.append(
            f"  F1-Macro: {cv_result['f1_macro_mean']:.4f} (+/- {cv_result['f1_macro_std']*2:.4f})")
        report.append("")

    # Test set performance
    report.append("TEST SET PERFORMANCE")
    report.append("-" * 40)

    # Sort models by test accuracy
    sorted_models = sorted(results.items(),
                           key=lambda x: x[1]['test']['accuracy'] if 'test' in x[1] else 0,
                           reverse=True)

    for model_name, result in sorted_models:
        if 'test' in result:
            test_metrics = result['test']
            report.append(f"{model_name}:")
            report.append(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            report.append(
                f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
            report.append(
                f"  Recall (macro): {test_metrics['recall_macro']:.4f}")
            report.append(
                f"  F1-Score (macro): {test_metrics['f1_macro']:.4f}")
            report.append(
                f"  F1-Score (weighted): {test_metrics['f1_weighted']:.4f}")
            report.append(f"  Training Time: {result['training_time']:.2f}s")
            report.append("")

    # Clustering results
    report.append("CLUSTERING ANALYSIS")
    report.append("-" * 40)
    for metric, score in clustering_results.items():
        report.append(f"{metric}: {score:.4f}")
    report.append("")

    # Best model recommendation
    best_model = sorted_models[0][0] if sorted_models else "None"
    best_accuracy = sorted_models[0][1]['test']['accuracy'] if sorted_models else 0

    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    report.append(f"Best performing model: {best_model}")
    report.append(f"Best test accuracy: {best_accuracy:.4f}")

    # Performance insights
    if best_accuracy > 0.9:
        report.append("✅ Excellent performance achieved!")
    elif best_accuracy > 0.8:
        report.append("✅ Good performance achieved.")
    elif best_accuracy > 0.7:
        report.append(
            "Moderate performance. Consider feature engineering or more data.")
    else:
        report.append("Low performance. Significant improvements needed.")

    report.append("")
    report.append("="*80)

    # Save report
    report_text = "\n".join(report)
    report_filename = f"ml_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_filename, 'w') as f:
        f.write(report_text)

    # Also log to console
    for line in report:
        logger.info(line)

    logger.info(f"Detailed report saved as {report_filename}")

    return report_text


def predict_new_job_enhanced(best_model_name, trained_models, dl_model, dl_metrics,
                             tfidf, label_encoder, new_job_text):
    """Enhanced prediction with confidence scores and alternatives"""
    logger.info("Making enhanced prediction for new job description...")

    try:
        new_vec = tfidf.transform([new_job_text])

        if best_model_name == "Deep Learning" and dl_model is not None:
            pred_proba = dl_model.predict(new_vec.toarray(), verbose=0)
            predicted_class = np.argmax(pred_proba)
            predicted_label = label_encoder.inverse_transform([predicted_class])[
                0]
            confidence = np.max(pred_proba)

            top_indices = np.argsort(pred_proba[0])[-3:][::-1]
            top_predictions = [(label_encoder.inverse_transform([idx])[0], pred_proba[0][idx])
                               for idx in top_indices]

        else:
            predicted_class = trained_models[best_model_name].predict(new_vec)[
                0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[
                0]

            if hasattr(trained_models[best_model_name], 'predict_proba'):
                proba = trained_models[best_model_name].predict_proba(new_vec)[
                    0]
                confidence = np.max(proba)

                # Get top 3 predictions
                top_indices = np.argsort(proba)[-3:][::-1]
                top_predictions = [(label_encoder.inverse_transform([idx])[0], proba[idx])
                                   for idx in top_indices]
            else:
                confidence = "N/A"
                top_predictions = [(predicted_label, "N/A")]

        logger.info(f"Top predictions:")
        for i, (job_title, conf) in enumerate(top_predictions, 1):
            if isinstance(conf, float):
                logger.info(f"  {i}. {job_title} (confidence: {conf:.4f})")
            else:
                logger.info(f"  {i}. {job_title} (confidence: {conf})")

        return predicted_label, confidence, top_predictions

    except Exception as e:
        logger.error(f"Error in enhanced prediction: {str(e)}")
        return "Error in prediction", 0.0, []

def setup_environment():
    """Set up the environment to avoid segmentation faults."""
    try:
        mp.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        logger.info("Multiprocessing start method already set")
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  
    os.environ['OMP_NUM_THREADS'] = '1' 
    
    if sys.platform == 'darwin':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def main():
    """Enhanced main execution function"""
    total_start_time = time.time()

    try:
        setup_environment()
        logger.info("🚀 Starting Enhanced ML Pipeline")
        bertTrainer = BertTrainer(csv_path="job_output.csv")
        bertTrainer.train_model()
        logger.info("BertTrainer Train data")

        X, y = load_and_preprocess_data()

        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, X_filtered, y_filtered = prepare_features_enhanced(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # bert_result = train_bert_model(
        #     X_train, y_train,
        #     X_val, y_val,
        #     num_classes=11,
        #     epochs=2,
        #     batch_size=2
        # )
        X_train_tfidf, X_val_tfidf, X_test_tfidf, X_all_tfidf, tfidf = vectorize_text_enhanced(
            X_train, X_val, X_test, X_filtered
        )

        base_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Naive Bayes": MultinomialNB(),
            # "SVM": SVC(probability=True, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                   random_state=42, verbosity=0),
            "KNN": KNeighborsClassifier(n_neighbors=7)
        }

        cv_results = perform_cross_validation(
            base_models, X_train_tfidf, y_train)

        tuned_models = hyperparameter_tuning(
            X_train_tfidf, y_train, X_val_tfidf, y_val)

        results, trained_models = train_and_evaluate_models(
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            y_train, y_val, y_test, tuned_models
        )

        dl_metrics, dl_model, dl_history = train_deep_learning_model_enhanced(
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            y_train, y_val, y_test, len(label_encoder.classes_)
        )

        results["Deep Learning"] = {
            'test': dl_metrics,
            'training_time': 0.0  
        }

        clustering_results, kmeans_labels = perform_clustering_enhanced(
            X_all_tfidf, label_encoder.transform(y_filtered)
        )

        best_model_name = max(results.keys(),
                              key=lambda x: results[x]['test']['accuracy'] if 'test' in results[x] else 0)

        logger.info(f"\n🏆 Best Model: {best_model_name}")
        logger.info(
            f"   Test Accuracy: {results[best_model_name]['test']['accuracy']:.4f}")
        logger.info(
            f"   Test F1-Macro: {results[best_model_name]['test']['f1_macro']:.4f}")

        create_comprehensive_visualization(results, cv_results, dl_history)

        if best_model_name != "Deep Learning":
            best_model = trained_models[best_model_name]
            test_pred = best_model.predict(X_test_tfidf)
            create_confusion_matrix_plot(
                y_test, test_pred, label_encoder, best_model_name)

        generate_detailed_report(
            results, cv_results, clustering_results, label_encoder)

        new_job_example = """
        Yegna Trading PLC is seeking a motivated and experienced Commercial Manager to join our team.
        The successful candidate will be responsible for developing and implementing commercial strategies,
        managing client relationships, and driving business growth. Requirements include a bachelor's degree
        in business or related field, 5+ years of commercial experience, and strong analytical skills.
        """

        predicted_title, confidence, top_predictions = predict_new_job_enhanced(
            best_model_name, trained_models, dl_model, dl_metrics,
            tfidf, label_encoder, new_job_example
        )

        total_time = time.time() - total_start_time
        logger.info(
            f"\n⏱️  Total Enhanced Pipeline Execution Time: {total_time:.2f} seconds")
        logger.info("🎉 Enhanced ML Pipeline Completed Successfully!")
        logger.info(
            "📊 Check generated files for detailed results and visualizations")

        return {
            'results': results,
            'cv_results': cv_results,
            'clustering_results': clustering_results,
            'best_model': best_model_name,
            'prediction_example': {
                'predicted_title': predicted_title,
                'confidence': confidence,
                'top_predictions': top_predictions
            }
        }

    except Exception as e:
        logger.error(f"Enhanced pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    pipeline_results = main()
