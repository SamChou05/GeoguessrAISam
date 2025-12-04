"""
Bag of Visual Words (BoVW) baseline using ORB/SIFT descriptors.

Side Experiment B: ORB/SIFT Bag-of-Visual-Words baseline
Pipeline:
1. Detect keypoints (FAST or Harris)
2. Extract ORB descriptors (32-D binary) or SIFT (128-D)
3. Sample descriptors from training images; run k-means to build codebook
4. For each image, quantize descriptors to nearest codeword
5. Build TF-IDF-weighted histogram; L2 normalize
6. Train a linear SVM or multinomial logistic regression

This ties together keypoints, descriptors, clustering, and linear classifiers.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional
from collections import Counter
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


class BagOfVisualWords:
    """
    Bag of Visual Words feature extractor.
    
    Uses ORB or SIFT descriptors with k-means clustering to create
    a visual vocabulary and encode images as histogram vectors.
    """
    
    def __init__(
        self,
        descriptor_type: str = 'orb',  # 'orb' or 'sift'
        n_clusters: int = 256,
        max_descriptors_per_image: int = 500,
        use_tfidf: bool = True
    ):
        """
        Args:
            descriptor_type: Type of descriptor ('orb' or 'sift')
            n_clusters: Number of visual words (codebook size)
            max_descriptors_per_image: Max descriptors to extract per image
            use_tfidf: Whether to apply TF-IDF weighting
        """
        self.descriptor_type = descriptor_type
        self.n_clusters = n_clusters
        self.max_descriptors_per_image = max_descriptors_per_image
        self.use_tfidf = use_tfidf
        
        # Initialize detector and descriptor
        if descriptor_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=max_descriptors_per_image)
        else:  # sift
            self.detector = cv2.SIFT_create(nfeatures=max_descriptors_per_image)
        
        self.kmeans = None
        self.idf_weights = None
        self.scaler = StandardScaler()
    
    def _extract_descriptors(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract descriptors from a single image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            return None
        
        # For ORB, convert binary descriptors to float
        if self.descriptor_type == 'orb':
            descriptors = descriptors.astype(np.float32)
        
        return descriptors
    
    def fit_codebook(
        self,
        image_paths: List[str],
        max_images: int = 5000,
        sample_per_image: int = 100
    ):
        """
        Build visual vocabulary using k-means on sampled descriptors.
        
        Args:
            image_paths: List of training image paths
            max_images: Maximum images to use for codebook
            sample_per_image: Descriptors to sample per image
        """
        print(f"Building codebook with {self.n_clusters} visual words...")
        
        all_descriptors = []
        
        # Sample images
        if len(image_paths) > max_images:
            indices = np.random.choice(len(image_paths), max_images, replace=False)
            sampled_paths = [image_paths[i] for i in indices]
        else:
            sampled_paths = image_paths
        
        # Extract descriptors
        for path in tqdm(sampled_paths, desc="Extracting descriptors"):
            image = cv2.imread(path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            descriptors = self._extract_descriptors(image)
            if descriptors is None:
                continue
            
            # Sample descriptors
            if len(descriptors) > sample_per_image:
                indices = np.random.choice(len(descriptors), sample_per_image, replace=False)
                descriptors = descriptors[indices]
            
            all_descriptors.append(descriptors)
        
        # Stack all descriptors
        all_descriptors = np.vstack(all_descriptors)
        print(f"Total descriptors for clustering: {len(all_descriptors)}")
        
        # Run k-means
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            n_init=3
        )
        self.kmeans.fit(all_descriptors)
        print("Codebook built successfully!")
    
    def _compute_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Compute BoVW histogram for a set of descriptors."""
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters, dtype=np.float32)
        
        # Assign descriptors to nearest visual words
        assignments = self.kmeans.predict(descriptors)
        
        # Build histogram
        histogram = np.zeros(self.n_clusters, dtype=np.float32)
        for word_id in assignments:
            histogram[word_id] += 1
        
        # Normalize (TF: term frequency)
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def fit_idf(self, image_paths: List[str], max_images: int = 5000):
        """
        Compute IDF weights from training set.
        
        Args:
            image_paths: Training image paths
            max_images: Maximum images to use
        """
        if not self.use_tfidf:
            return
        
        print("Computing IDF weights...")
        
        # Count document frequency for each visual word
        document_freq = np.zeros(self.n_clusters, dtype=np.float32)
        n_docs = 0
        
        if len(image_paths) > max_images:
            indices = np.random.choice(len(image_paths), max_images, replace=False)
            sampled_paths = [image_paths[i] for i in indices]
        else:
            sampled_paths = image_paths
        
        for path in tqdm(sampled_paths, desc="Computing IDF"):
            image = cv2.imread(path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            descriptors = self._extract_descriptors(image)
            if descriptors is None:
                continue
            
            assignments = self.kmeans.predict(descriptors)
            unique_words = np.unique(assignments)
            document_freq[unique_words] += 1
            n_docs += 1
        
        # Compute IDF: log(N / (df + 1))
        self.idf_weights = np.log(n_docs / (document_freq + 1))
        print("IDF weights computed!")
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Transform a single image to BoVW feature vector.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            feature: L2-normalized BoVW histogram
        """
        descriptors = self._extract_descriptors(image)
        histogram = self._compute_histogram(descriptors)
        
        # Apply IDF weighting
        if self.use_tfidf and self.idf_weights is not None:
            histogram = histogram * self.idf_weights
        
        # L2 normalize
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
        
        return histogram
    
    def transform_batch(self, image_paths: List[str]) -> np.ndarray:
        """Transform a batch of images to feature vectors."""
        features = []
        
        for path in tqdm(image_paths, desc="Extracting BoVW features"):
            image = cv2.imread(path)
            if image is None:
                features.append(np.zeros(self.n_clusters, dtype=np.float32))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features.append(self.transform(image))
        
        return np.array(features)
    
    def save(self, path: str):
        """Save the BoVW model."""
        joblib.dump({
            'kmeans': self.kmeans,
            'idf_weights': self.idf_weights,
            'descriptor_type': self.descriptor_type,
            'n_clusters': self.n_clusters,
            'use_tfidf': self.use_tfidf
        }, path)
        print(f"BoVW model saved to {path}")
    
    def load(self, path: str):
        """Load a saved BoVW model."""
        data = joblib.load(path)
        self.kmeans = data['kmeans']
        self.idf_weights = data['idf_weights']
        self.descriptor_type = data['descriptor_type']
        self.n_clusters = data['n_clusters']
        self.use_tfidf = data['use_tfidf']
        print(f"BoVW model loaded from {path}")


class BOVWClassifier:
    """
    Complete BoVW classification pipeline with SVM or Logistic Regression.
    """
    
    def __init__(
        self,
        descriptor_type: str = 'orb',
        n_clusters: int = 256,
        classifier_type: str = 'svm',  # 'svm' or 'logistic'
        use_tfidf: bool = True
    ):
        """
        Args:
            descriptor_type: 'orb' or 'sift'
            n_clusters: Number of visual words
            classifier_type: 'svm' or 'logistic'
            use_tfidf: Whether to use TF-IDF weighting
        """
        self.bovw = BagOfVisualWords(
            descriptor_type=descriptor_type,
            n_clusters=n_clusters,
            use_tfidf=use_tfidf
        )
        
        self.classifier_type = classifier_type
        if classifier_type == 'svm':
            self.classifier = LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=10000,
                random_state=42
            )
        else:
            self.classifier = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.class_names = None
    
    def fit(
        self,
        train_paths: List[str],
        train_labels: List[int],
        class_names: List[str]
    ):
        """
        Fit the complete BoVW pipeline.
        
        Args:
            train_paths: List of training image paths
            train_labels: List of training labels
            class_names: List of class names
        """
        self.class_names = class_names
        
        # Build codebook
        self.bovw.fit_codebook(train_paths)
        
        # Compute IDF
        self.bovw.fit_idf(train_paths)
        
        # Extract features
        print("Extracting training features...")
        X_train = self.bovw.transform_batch(train_paths)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        
        # Train classifier
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train, train_labels)
        print("Classifier trained!")
    
    def predict(self, image_paths: List[str]) -> np.ndarray:
        """Predict labels for a list of images."""
        X = self.bovw.transform_batch(image_paths)
        X = self.scaler.transform(X)
        return self.classifier.predict(X)
    
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
        
        results = {
            'top1_accuracy': top1_acc,
            'macro_f1': macro_f1,
            'predictions': predictions,
            'report': classification_report(
                test_labels, predictions,
                target_names=self.class_names,
                output_dict=True
            )
        }
        
        return results
    
    def save(self, dir_path: str):
        """Save the complete pipeline."""
        os.makedirs(dir_path, exist_ok=True)
        self.bovw.save(os.path.join(dir_path, 'bovw.pkl'))
        joblib.dump(self.classifier, os.path.join(dir_path, 'classifier.pkl'))
        joblib.dump(self.scaler, os.path.join(dir_path, 'scaler.pkl'))
        joblib.dump(self.class_names, os.path.join(dir_path, 'class_names.pkl'))
        print(f"Pipeline saved to {dir_path}")
    
    def load(self, dir_path: str):
        """Load a saved pipeline."""
        self.bovw.load(os.path.join(dir_path, 'bovw.pkl'))
        self.classifier = joblib.load(os.path.join(dir_path, 'classifier.pkl'))
        self.scaler = joblib.load(os.path.join(dir_path, 'scaler.pkl'))
        self.class_names = joblib.load(os.path.join(dir_path, 'class_names.pkl'))
        print(f"Pipeline loaded from {dir_path}")


if __name__ == "__main__":
    print("BoVW module loaded successfully.")
    print("Usage:")
    print("  classifier = BOVWClassifier(descriptor_type='orb', n_clusters=256)")
    print("  classifier.fit(train_paths, train_labels, class_names)")
    print("  results = classifier.evaluate(test_paths, test_labels)")

