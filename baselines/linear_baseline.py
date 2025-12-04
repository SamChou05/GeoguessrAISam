"""
Linear classifier baseline on raw pixels.

Baseline: Linear classifier on raw pixels
- Flatten 224x224x3 images
- Apply PCA to reduce to 256 dimensions
- Train logistic regression

This shows "how far a simple linear method goes" as a sanity check.
"""

import numpy as np
import cv2
from typing import List
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


class LinearPixelClassifier:
    """
    Linear classifier on raw pixel features with PCA dimensionality reduction.
    """
    
    def __init__(
        self,
        input_size: int = 224,
        n_components: int = 256,
        C: float = 1.0
    ):
        """
        Args:
            input_size: Size to resize images to
            n_components: Number of PCA components
            C: Regularization strength for logistic regression
        """
        self.input_size = input_size
        self.n_components = n_components
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.classifier = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        self.class_names = None
    
    def _load_and_flatten(self, image_path: str) -> np.ndarray:
        """Load image, resize, and flatten to 1D vector."""
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(self.input_size * self.input_size * 3)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Flatten
        return image.flatten()
    
    def _extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract flattened features for a batch of images."""
        features = []
        for path in tqdm(image_paths, desc="Extracting pixel features"):
            features.append(self._load_and_flatten(path))
        return np.array(features)
    
    def fit(
        self,
        train_paths: List[str],
        train_labels: List[int],
        class_names: List[str]
    ):
        """
        Fit the linear classifier pipeline.
        
        Args:
            train_paths: List of training image paths
            train_labels: List of training labels
            class_names: List of class names
        """
        self.class_names = class_names
        
        print("Extracting pixel features...")
        X_train = self._extract_features(train_paths)
        
        print("Fitting scaler...")
        X_train = self.scaler.fit_transform(X_train)
        
        print(f"Fitting PCA ({self.n_components} components)...")
        X_train = self.pca.fit_transform(X_train)
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        print("Training logistic regression...")
        self.classifier.fit(X_train, train_labels)
        print("Classifier trained!")
    
    def predict(self, image_paths: List[str]) -> np.ndarray:
        """Predict labels for a list of images."""
        X = self._extract_features(image_paths)
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return self.classifier.predict(X)
    
    def predict_proba(self, image_paths: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        X = self._extract_features(image_paths)
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return self.classifier.predict_proba(X)
    
    def evaluate(
        self,
        test_paths: List[str],
        test_labels: List[int]
    ) -> dict:
        """
        Evaluate the classifier.
        
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(test_paths)
        
        top1_acc = accuracy_score(test_labels, predictions)
        macro_f1 = f1_score(test_labels, predictions, average='macro')
        
        # Top-5 accuracy (using probabilities)
        probas = self.predict_proba(test_paths)
        top5_preds = np.argsort(probas, axis=1)[:, -5:]
        top5_acc = np.mean([test_labels[i] in top5_preds[i] for i in range(len(test_labels))])
        
        results = {
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'macro_f1': macro_f1,
            'predictions': predictions,
            'report': classification_report(
                test_labels, predictions,
                target_names=self.class_names,
                output_dict=True
            )
        }
        
        return results
    
    def save(self, path: str):
        """Save the pipeline."""
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'classifier': self.classifier,
            'class_names': self.class_names,
            'input_size': self.input_size,
            'n_components': self.n_components
        }, path)
        print(f"Linear classifier saved to {path}")
    
    def load(self, path: str):
        """Load a saved pipeline."""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.classifier = data['classifier']
        self.class_names = data['class_names']
        self.input_size = data['input_size']
        self.n_components = data['n_components']
        print(f"Linear classifier loaded from {path}")


if __name__ == "__main__":
    print("LinearPixelClassifier module loaded successfully.")
    print("Usage:")
    print("  classifier = LinearPixelClassifier(n_components=256)")
    print("  classifier.fit(train_paths, train_labels, class_names)")
    print("  results = classifier.evaluate(test_paths, test_labels)")

