# Heart Disease Detection: Predictive Modeling & Visualization Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements an interactive machine learning system for heart disease detection, featuring comprehensive data analysis, predictive modeling, and visualization capabilities. Built upon the research paper "Performance Analysis of Heart Disease Detection using Different Machine Learning Approaches," this tool provides healthcare professionals and researchers with insights into cardiovascular risk factors and their relationships.

## Key Features

- **Multi-Algorithm Approach**: Implements both Logistic Regression and K-Nearest Neighbors (KNN) for robust predictions
- **Comprehensive EDA**: Interactive visualizations including correlation heatmaps, distribution plots, and feature analysis
- **LIME Interpretability**: Explainable AI features to understand model decisions
- **Performance Metrics**: Complete evaluation suite with ROC curves, confusion matrices, and classification reports
- **Data Quality Assurance**: Automated missing value handling and outlier detection
- **Interactive Visualizations**: Rich plotting capabilities for data exploration and model interpretation

## Technology Stack

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **LIME**: Local interpretable model-agnostic explanations
- **Jupyter Notebook**: Interactive development environment

## Installation & Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AliAhmed015/Heart-Disease-Detection.git
cd heart-disease-detection
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

The project uses the Heart Disease dataset from the UCI Machine Learning Repository. Run the setup script to download and prepare the data:

```bash
python setup.py
```

### Step 5: Run the Analysis

```bash
# Start Jupyter Notebook
jupyter notebook

# Or run the complete pipeline
python src/main.py
```

## Usage Examples

### Basic Model Training

```python
from src.models.logistic_regression import LogisticRegressionModel
from src.models.knn import KNNModel
from src.data.preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train Logistic Regression
lr_model = LogisticRegressionModel()
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.evaluate(X_test, y_test)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Train KNN
knn_model = KNNModel(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.evaluate(X_test, y_test)
print(f"KNN Accuracy: {knn_accuracy:.4f}")
```

### Generating Visualizations

```python
from src.data.visualization import create_correlation_heatmap, plot_age_distribution

# Create correlation heatmap
create_correlation_heatmap(data)

# Plot age distribution
plot_age_distribution(data)
```

## Model Performance

### Logistic Regression Results
- **Accuracy**: 87.78%
- **Precision**: High precision for both classes
- **Recall**: Balanced recall across classes

### K-Nearest Neighbors Results
- **Accuracy**: 92.00%
- **Precision**: Excellent precision scores
- **Recall**: Superior recall performance

## Visualizations

The project generates comprehensive visualizations for data exploration and model interpretation:

### 1. Feature Correlation Heatmap
![Correlation Heatmap](images/DiagonalCorrelationMatrix.png)
*Reveals relationships between features, highlighting Thalassemia and MaxHeartRateAchieved as strong predictors*

### 2. Age Distribution
![Age Distribution](images/AgeDistribution.png)
*Shows the age profile of the dataset with majority around 60 years*

### 3. Cholesterol Levels by Target
![Cholesterol Boxplot](images/BoxplotofCholesterolLevelsbyTargetVariable.png)
*Demonstrates outlier patterns in cholesterol levels for heart disease patients*

### 4. Chest Pain Type Distribution
![Chest Pain Distribution](images/DistributionofChestPainTypes.png)
*Distribution of chest pain types across the dataset*

### 5. Age vs Maximum Heart Rate Scatter Plot
![Age vs Heart Rate](images/Agevs.MaximumHeartRatebyTargetVariable.png)
*Relationship between age and maximum heart rate by heart disease status*

### 6. ROC Curves Comparison
![ROC Curves](images/ROCCurve.png)
*Performance comparison between Logistic Regression and KNN models*

### 7. LIME Feature Importance
![LIME Analysis](images/LimeExplanation.png)
*Local interpretable explanations for model predictions*

### 8. Logistic Regression Confusion Matrix
![LR Classification Report](images/ConfusionMatrixforLogisticRegressionModel.png)
*Confusion matrix analysis for Logistic Regression*

### 11. KNN Confusion Matrix
![KNN Confusion Matrix](images/ConfusionMatrixKNN.png)
*Confusion matrix analysis for K-Nearest Neighbors*

## Contributing

We welcome contributions to improve this project! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Important Notes & Limitations

### Current Limitations
- **Dataset Size**: Limited to UCI Heart Disease dataset (303 samples)
- **Feature Set**: Based on specific clinical parameters available in the dataset
- **Generalizability**: Model performance may vary with different populations
- **Real-time Processing**: Not optimized for real-time clinical deployment

### Medical Disclaimer
**This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult healthcare professionals for medical decisions.**

### Known Issues
- Requires manual handling of categorical variables in some cases
- LIME explanations may be computationally intensive for large datasets
- Model interpretability features work best with the current feature set

## Future Improvements

### Research Directions
- Integration with electronic health records (EHR)
- Multi-modal data fusion (imaging + clinical data)
- Federated learning for privacy-preserving model training
- Explainable AI improvements for clinical decision support

## References

1. "Performance Analysis of Heart Disease Detection using Different Machine Learning Approaches" - Primary research paper
2. UCI Machine Learning Repository - Heart Disease Dataset
3. Scikit-learn Documentation
4. LIME: Local Interpretable Model-agnostic Explanations

## Support & Contact

- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Join our GitHub Discussions for questions and community support
- **Email**: [m.ali.ahmed015@gmail.com] for direct contact

---

**⭐ If you find this project helpful, please give it a star on GitHub!**
