---
title: "Implementing Machine Learning Pipelines on Real Data | Episode 3"
seoTitle: "Implementing Machine Learning Pipelines on Real Data"
seoDescription: "Join us in Episode 3 to learn how to create effective machine learning models and predict laptop prices."
datePublished: Sat Nov 18 2023 02:17:48 GMT+0000 (Coordinated Universal Time)
cuid: clp3f5tot000808i7g8ui48oa
slug: implementing-machine-learning-pipelines-episode-3
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1700159423624/5cc4309e-d657-4a29-bc14-3b3459a7493b.jpeg
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1700159427856/2acf41ed-737a-42f8-8a92-9081f37ad7a4.jpeg
tags: artificial-intelligence, data-science, machine-learning, data-analysis, deep-learning

---

Hey there, welcome back to the third episode of our series "From Theory to Action: Implementing Machine Learning Pipelines on Real Data". In the previous episodes, we covered all the theoretical aspects of machine learning pipelines. We explored the different components, examined the complexities of preprocessing, and visualized the journey from raw data to actionable insights.

In Episode 3, we're shifting gears and diving into the practical implementation of machine learning pipelines. Our goal is to take a raw dataset and guide it through the different stages of a machine-learning pipeline.

Join us as we tackle some real-world challenges, uncover the nuances of data transformation, and witness the power of applying theoretical concepts to actual data. In this episode, we're not just talking about concepts, we're rolling up our sleeves and getting our hands dirty with the core of machine learning, where algorithms meet data and predictions turn into tangible outcomes.

---

### **Introduction to the Real-world Dataset**

Hi! Today, we're exploring a laptop dataset from GitHub. It contains brand, specifications, condition, and prices. We'll tidy up the data, build a pipeline for predicting prices, and develop a machine-learning model. Our goal is to empower consumers and retailers with actionable insights.

**Dataset**

This dataset, focusing on laptops, holds key details such as brand, specifications, and prices

Dataset-Link: [https://github.com/everydaycodings/Dataset/blob/master/article/data/laptop\_data.csv](https://github.com/everydaycodings/Dataset/blob/master/article/data/laptop_data.csv)

```python
import pandas as pd
import numpy as np
```

```python
data = pd.read_csv("laptop_data.csv)
data.head()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1699907498646/4278eb54-8a23-4923-bd75-8a8bcca431a3.png align="center")

```python
data.info()
```

Output

```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1303 entries, 0 to 1302
Data columns (total 12 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Unnamed: 0        1303 non-null   int64  
 1   Company           1303 non-null   object 
 2   TypeName          1303 non-null   object 
 3   Inches            1303 non-null   float64
 4   ScreenResolution  1303 non-null   object 
 5   Cpu               1303 non-null   object 
 6   Ram               1303 non-null   object 
 7   Memory            1303 non-null   object 
 8   Gpu               1303 non-null   object 
 9   OpSys             1303 non-null   object 
 10  Weight            1303 non-null   object 
 11  Price             1303 non-null   float64
dtypes: float64(2), int64(1), object(9)
memory usage: 122.3+ KB
```

The Pandas command `data.info()` is a concise tool that provides a summary of your DataFrame. It presents important information, like the number of non-null values per column, data types (such as integers, floats, and datetimes), and estimated memory usage. This summary is useful in quickly identifying missing values, understanding data types that are essential for analysis, and assessing the memory usage of your dataset. Overall, `data.info()` is a powerful tool that helps you to assess the cleanliness and structure of your data, making data exploration and preparation more efficient.

```python
print("Duplicated Values: ", data.duplicated().sum())
print("Null Values: ", data.isnull().sum())
```

When you run this code, you will see that there are no null or duplicated values. This is beneficial because it simplifies our work. However, if you're working with a different dataset that has duplicates or null values, you can use the following commands to remove them `data = data.drop_duplicates()` and `data = data.dropna()` respectively.

```python
data = data.drop(columns=['Unnamed: 0'])
```

We remove the `Unnamed: 0` Column because it is useless for our Assignment.

```python
data['Ram'] = data['Ram'].str.replace('GB','')
data['Weight'] = data['Weight'].str.replace('kg','')
```

As you can see in the "Ram" column, the values are in the format of 8GB, 16GB, or something similar. However, since RAM is measured in GB, we don't need to include "GB" in the values. To remove "GB" from the values, we can use the command `data['Ram'].str.replace('GB','')`. This command will remove "GB" from the values and leave only the numerical part, such as 8 or 16. The same applies to the "Weight" column, where values such as 8kg can be replaced with just 8 by using this command `data['Weight'].str.replace('kg','')`.

```python
data['Ram'] = data['Ram'].astype('int32')
data['Weight'] = data['Weight'].astype('float32')
```

We can now convert the remaining values in the Ram and Weight columns to a numerical data type. In our case, we will use int32 for Ram and float32 for Weight. We are using float32 for Weight because it contains decimal points (e.g. 4.6).

---

### **Exploratory Data Analysis**

Exploratory Data Analysis is an approach to analyzing datasets to summarize their main characteristics, often with the help of statistical graphics and other data visualization methods. The primary goal of EDA is to understand the data's structure, patterns, and relationships between variables, aiding in the formulation of hypotheses and guiding further analysis.

**EDA Result**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1699909847752/3d7411cb-1301-4bcb-bea5-c62e15857276.png align="center")

* General info:
    
    * The majority of laptops are from Dell, Lenovo, and HP.
        
    * The majority of laptops are Notebooks, which make up 55.84% of the total laptops
        
    * Most laptops have 15.6 inches, which makes 51.08% of the total laptops
        
* Price:
    
    * There are laptops with prices over 3000:
        
        * Most of them are Gaming PCs or Workstations with Intel CPUs.
            
        * There is also one Notebook with a price close to 5000 euros and Gaming laptops, with a price close to 5500 euros and 6000 euros
            
    * The most expensive notebook is the Lenovo Thinkpad P51, with an Intel XEON CPU and Nvidia Quadro GPU!
        
* Brand:
    
    * Laptops with Intel CPUs are more expensive.
        
    * Laptops with AMD CPUs also have AMD GPUs
        
    * Laptops with Nvidia GPUs are more expensive.
        
* RAM:
    
    * According to the slope of the linear regression between price and RAM, every GB of RAM added to the PC adds roughly 107$ to the laptop's value.
        
    * Most laptops have 8 GB RAM, which makes 47.54% of the total laptops
        
* GPU:
    
    * The 2 most common GPUs are integrated Intel GPUs HD Graphics 620 and 520, while the third one is the Nvidia GTX1050.
        
* CPU:
    
    * All the TOP 15 most common CPUs are from Intel.
        
    * The most common CPU is the Intel i5 7200U, the second is the i7 7700HQ and the third is the i7 7500U.
        
    * Out of the 15 CPUs, 10 are series 'U' (low voltage), 3 are series 'HQ' (high performance) 10 and 2 are Celerons (low-end).
        
    * Most laptops have a 2.5 GHz CPU, which makes up 22.5% of the total laptops
        
* Hard drives:
    
    * Most PCs have 256 GB of storage, which is for the most part SSD. Moreover, for storage of 1 TB or higher, most of them are HDD.
        
    * Most second hard drive storages are 1 TB HDD disks
        
* Correlation Matrix:
    
    * RAM has a high positive correlation with price (+0.75): more expensive laptops tend to have a higher price
        
    * CPU Freq has a quite moderate positive correlation with the price (+0.45)
        
    * Inches and Weight have a high positive correlation (+0.82) since laptops with bigger screens tend to be heavier.
        

I haven't covered the EDA Part in depth since I believe that the EDA I have done is quite basic. In the future, I plan to create a comprehensive article on EDA. If you are interested in learning more about EDA for the time being, you can check out this resource: [https://www.youtube.com/watch?v=fHFOANOHwh8](https://www.youtube.com/watch?v=fHFOANOHwh8).

---

### **Data Preprocessing**

Data preprocessing is a vital step in the data analysis and machine learning pipeline. It consists of cleaning and transforming raw data into a format that is suitable for analysis or model training. The main goals of data preprocessing are to handle missing or inconsistent data, address outliers, and format the data in a way that boosts the performance of machine learning models.

> **A bad model with good data is better than a good model with bad data**

#### **Preprocessing Step For** `ScreenResolution` **Column**

1. **Creating Binary Features:**
    
    ```python
    data['Touchscreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    data['Ips'] = data['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    ```
    
    * We have created two new binary features, 'Touchscreen' and 'Ips,' indicating whether each laptop has a touchscreen or an IPS (In-Plane Switching) panel.
        
2. **Extracting Screen Resolution Components:**
    
    ```python
    new = data['ScreenResolution'].str.split('x', n=1, expand=True)
    data['X_res'] = new[0]
    data['Y_res'] = new[1]
    ```
    
    * The 'ScreenResolution' values have been split into two components using 'x' as the delimiter. This was done using the `str.split()` method on the 'ScreenResolution' column and the two components were stored in two new columns, **X\_res,** and **Y\_res** to store the horizontal and vertical resolutions, respectively.
        
3. **Cleaning and Converting Resolution Components:**
    
    ```python
    data['X_res'] = data['X_res'].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
    data['X_res'] = data['X_res'].astype('int')
    data['Y_res'] = data['Y_res'].astype('int')
    ```
    
    * We have removed commas from the 'X\_res' column and converted the 'X\_res' and 'Y\_res' columns to integer data types. This step ensures that the resolution components are in a numeric format for further calculations.
        
4. **Calculating Pixels Per Inch (PPI):**
    
    ```python
    data['ppi'] = (((data['X_res']**2) + (data['Y_res']**2))**0.5 / data['Inches']).astype('float')
    ```
    
    * We have calculated the Pixels Per Inch (PPI) by applying the formula for diagonal resolution on each laptop. This metric represents the pixel density and is adjusted based on the screen size ('Inches').
        
5. **Dropping Unnecessary Columns:**
    
    ```python
    data.drop(columns=['ScreenResolution'], inplace=True)
    data.drop(columns=['Inches', 'X_res', 'Y_res'], inplace=True)
    ```
    
    * You've removed the original 'ScreenResolution' column, the 'Inches' column, and the intermediate 'X\_res' and 'Y\_res' columns, as they are no longer needed after extracting the relevant information.
        

In summary, these preprocessing steps have transformed the 'ScreenResolutions' column into more informative features such as 'Touchscreen,' 'Ips,' and 'ppi,' providing a clearer representation of the screen characteristics for each laptop in your dataset.

#### **Preprocessing step for** `CPU` **column**

1. **Extracting CPU Name**
    

```python
# Extracting the first three words from the 'CPU' column and creating a new 'Cpu Name' column
data['Cpu Name'] = data['CPU'].apply(lambda x: " ".join(x.split()[0:3]))
```

**Explanation:**

* This step creates a new column called 'Cpu Name' by splitting each entry in the 'CPU' column into words and joining the first three words together.
    
* The goal is to capture a concise representation of the CPU type and model.
    

1. **Categorizing Processor Type**
    

```python
# Defining a function to categorize processors into specific types
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

# Applying the function to create a new 'Cpu brand' column
data['Cpu brand'] = data['Cpu Name'].apply(fetch_processor)
```

**Explanation:**

* The function `fetch_processor` categorizes processors into specific types: 'Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Other Intel Processor', and 'AMD Processor'.
    
* The 'Cpu brand' column is created by applying this function to the 'Cpu Name' column.
    

1. **Dropping Unnecessary Columns**
    

```python
# Dropping the original 'CPU' column and the intermediate 'Cpu Name' column
data.drop(columns=['CPU', 'Cpu Name'], inplace=True)
```

**Explanation:**

* This step removes the original 'CPU' column, as its information has been extracted into the 'Cpu Name' column.
    
* It also drops the 'Cpu Name' column as it was an intermediate step and is no longer needed.
    

In summary, these steps transform the 'CPU' column by creating new features ('Cpu Name' and 'Cpu brand') that provide structured information about the processor type and brand. The original and intermediate columns are then dropped to streamline the dataset.

#### **Preprocessing step for** `Memory` **column**

1. **Standardizing and Cleaning Memory Values**
    

```python
# Convert 'Memory' column to string type and remove decimal points
data['Memory'] = data['Memory'].astype(str).replace('\.0', '', regex=True)

# Remove 'GB' and replace 'TB' with '000' for uniform representation
data["Memory"] = data["Memory"].str.replace('GB', '')
data["Memory"] = data["Memory"].str.replace('TB', '000')
```

**Explanation:**

* The 'Memory' column is converted to a string type to ensure consistent handling.
    
* Decimal points (`.0`) are removed for clarity.
    
* 'GB' is removed, and 'TB' is replaced with '000' to standardize the representation.
    

1. **Extracting Layers of Memory**
    

```python
# Split the 'Memory' column into two parts using the '+' as a separator
new = data["Memory"].str.split("+", n=1, expand=True)

# Create new columns 'first' and 'second' to store the two parts
data["first"] = new[0]
data["first"] = data["first"].str.strip()
data["second"] = new[1]
```

**Explanation:**

* The 'Memory' column is split into two parts using the '+' as a separator.
    
* New columns ('first' and 'second') are created to store these two parts.
    

1. **Processing Layers of Memory**
    

```python
# Extract features for the first layer
data["Layer1HDD"] = data["first"].apply(lambda x: 1 if "HDD" in x else 0)
data["Layer1SSD"] = data["first"].apply(lambda x: 1 if "SSD" in x else 0)
data["Layer1Hybrid"] = data["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
data["Layer1Flash_Storage"] = data["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
data['first'] = data['first'].str.replace(r'\D', '')

# Fill missing values in 'second' column with '0'
data["second"].fillna("0", inplace=True)

# Extract features for the second layer
data["Layer2HDD"] = data["second"].apply(lambda x: 1 if "HDD" in x else 0)
data["Layer2SSD"] = data["second"].apply(lambda x: 1 if "SSD" in x else 0)
data["Layer2Hybrid"] = data["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
data["Layer2Flash_Storage"] = data["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)
data['second'] = data['second'].str.replace(r'\D', '')
```

**Explanation:**

* Features are extracted for each layer, indicating the presence of 'HDD', 'SSD', 'Hybrid', and 'Flash Storage'.
    
* Non-numeric characters are removed from the 'first' and 'second' columns.
    

1. **Calculating Memory Sizes**
    

```python
# Convert 'first' and 'second' columns to integer type
data["first"] = data["first"].astype(int)
data["second"] = data["second"].astype(int)

# Calculate the total memory sizes for each type
data["HDD"] = (data["first"] * data["Layer1HDD"] + data["second"] * data["Layer2HDD"])
data["SSD"] = (data["first"] * data["Layer1SSD"] + data["second"] * data["Layer2SSD"])
data["Hybrid"] = (data["first"] * data["Layer1Hybrid"] + data["second"] * data["Layer2Hybrid"])
data["Flash_Storage"] = (data["first"] * data["Layer1Flash_Storage"] + data["second"] * data["Layer2Flash_Storage"])
```

**Explanation:**

* 'first' and 'second' columns are converted to integer types for numerical calculations.
    
* Total memory sizes for 'HDD', 'SSD', 'Hybrid', and 'Flash Storage' are calculated based on the extracted features.
    

1. **Dropping Unnecessary Columns**
    

```python
# Drop unnecessary columns used for intermediate steps
data.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
                   'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'], inplace=True)

data.drop(columns=['Memory'],inplace=True)
```

**Explanation:**

* Columns used for intermediate steps are dropped to keep the dataset clean and concise.
    

In summary, these steps preprocess the 'Memory' column, extracting information about different memory types and calculating total memory sizes for 'HDD', 'SSD', 'Hybrid', and 'Flash Storage'. The resulting dataset is more structured and suitable for analysis and modeling.

```python
data.drop(columns=['Hybrid','Flash_Storage'],inplace=True)
```

I dropped the 'Hybrid' and 'Flash\_Storage' columns it has almost 0 correlation to the price column so no use for this column.

#### Preprocessing for `Gpu` column

```python
# Create a new column 'Gpu brand' by extracting the first word from each entry in the 'Gpu' column
data['Gpu brand'] = data['Gpu'].apply(lambda x: x.split()[0])
```

**Explanation:**

* The 'Gpu brand' column is created by applying a lambda function to the 'Gpu' column.
    
* The lambda function splits each entry in the 'Gpu' column into words and extracts the first word, representing the GPU brand.
    

In summary, this preprocessing step extracts the GPU brand information from the 'Gpu' column and stores it in a new column called 'Gpu brand'. This allows for better analysis and understanding of the different GPU brands present in the dataset.

#### Preprocessing on `OpSys` column

```python
# Define a function 'cat_os' to categorize operating systems
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

# Apply the 'cat_os' function to create a new column 'os'
data['os'] = data['OpSys'].apply(cat_os)
data.drop(columns=['OpSys'],inplace=True)
```

**Explanation:**

* The function `cat_os` categorizes operating systems into three main types: 'Windows', 'Mac', and 'Others/No OS/Linux'.
    
* The 'os' column is created by applying this function to the 'OpSys' column.
    

In summary, this preprocessing step categorizes different operating systems into broader categories and stores the information in a new column called 'os'. This simplifies the representation of operating systems for analysis and modeling.

**Preprocessing Result:**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1699912261111/c737a552-e4ed-4b56-85f8-1ecc0ff299c6.png align="center")

<mark>Note: More features can be extracted, but I did not include them as this article covers basic preprocessing. An in-depth preprocessing article will come soon.</mark>

---

### Model Creation

#### Preparing Features and Target Variable

```python
X = data.drop(columns=['Price'])
y = np.log(data['Price'])
```

**Explanation:**

* `X` is assigned the features (independent variables) of the dataset, excluding the 'Price' column.
    
* `y` is assigned the target variable (dependent variable), which is the natural logarithm of the 'Price' column. This transformation is commonly done to handle skewed or non-normally distributed target variables.
    

#### Splitting the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
```

**Explanation:**

* The `train_test_split` function is used to split the dataset into training and testing sets.
    
* `X_train` and `y_train` represent the features and target variable of the training set, respectively.
    
* `X_test` and `y_test` represent the features and target variable of the testing set, respectively.
    
* The `test_size` parameter is set to 0.15, meaning that 15% of the data will be used for testing, and the rest for training.
    
* The `random_state` parameter is set to 2 for reproducibility, ensuring that the split is the same every time the code is run.
    

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
```

#### Defining Column Transformer

```python
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')
```

A ColumnTransformer is defined to apply OneHotEncoder to categorical columns specified in the transformer list (\[0, 1, 7, 10, 11\]). The drop='first' parameter drops the first level of each categorical feature to avoid the "dummy variable trap." The remainder='passthrough' parameter keeps the non-categorical columns unchanged.

1. **Define Models:**
    
    * A list of tuples is created, where each tuple contains the name of a regression model and an instance of that model with specific hyperparameters.
        
    * Models include Ridge, Lasso, KNeighborsRegressor, DecisionTreeRegressor, SVR, and RandomForestRegressor.
        

```python
models = [
    ('Ridge', Ridge(alpha=10)),
    ('Lasso', Lasso(alpha=0.001)),
    ('KNeighbors', KNeighborsRegressor(n_neighbors=3)),
    ('DecisionTree', DecisionTreeRegressor(max_depth=8)),
    ('SVR', SVR(kernel='rbf', C=10000, epsilon=0.1)),
    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15))
]
```

1. **Create Empty DataFrame:**
    
    * An empty DataFrame `results_df` is created with columns 'Model', 'R2 Score', and 'MAE' to store the results.
        

```python
results_df = pd.DataFrame(columns=['Model', 'R2 Score', 'MAE'])
```

1. **Loop through Models and Fit Pipelines:**
    
    * A loop iterates through each model.
        
    * Inside the loop, a pipeline (`pipe`) is created for each model. The pipeline consists of the previously defined `step1` (column transformer) and the current model.
        
    * The pipeline is trained on the training data (`X_train`, `y_train`).
        

```python
for model_name, model_instance in models:
    step2 = model_instance
    pipe = Pipeline([
        ('step1', step1),
        ('step2', step2)
    ])
    pipe.fit(X_train, y_train)
```

1. **Make Predictions and Evaluate:**
    
    * After training, the pipeline is used to make predictions on the test data (`X_test`).
        
    * R2 score and MAE are calculated for each model.
        

```python
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
```

1. **Append Results to DataFrame:**
    
    * The results (model name, R2 score, and MAE) are appended as a new row to the `results_df` DataFrame.
        

```python
results_df = results_df.append({'Model': model_name, 'R2 Score': r2, 'MAE': mae}, ignore_index=True)
```

1. **Display Results DataFrame:**
    
    * Finally, the `results_df` DataFrame, which now contains the evaluation results for each model, is printed.
        

```python
print(results_df)
```

In summary, this code loops through different regression models, fits pipelines for each model, evaluates the model performance, and stores the results in a DataFrame. The final DataFrame provides a comparison of R2 scores and MAE for different regression models on the test data.

**Sample Output:**

```plaintext
           Model  R2 Score       MAE
0          Ridge  0.818    0.155
1          Lasso  0.820    0.153
2     KNeighbors  0.805    0.160
3   DecisionTree  0.828    0.148
4            SVR   0.812    0.158
5  RandomForest   0.830    0.145
```

This is just sample data so show how will it look not the actual result

#### Export Your Model

```python
import pickle

pickle.dump(data,open('data.pkl','wb'))
```

<mark>Just to clarify, this was a basic version of machine learning. We haven't gone into any specific methods yet, as we are just starting in this series. As we proceed and dive deeper, we will discuss every aspect of machine learning in greater detail.</mark>

---

### Resources For Further Research

**Online Courses:**

1. **Coursera - Machine Learning by Andrew Ng:**
    
    * [Machine Learning](https://www.coursera.org/learn/machine-learning)
        
    * A highly regarded course that covers the fundamentals of machine learning. Taught by Andrew Ng, a prominent figure in the field.
        
2. **Udacity - Intro to Machine Learning with PyTorch:**
    
    * [Intro to Machine Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188)
        
    * Focuses on practical aspects of machine learning using PyTorch.
        

**Books:**

1. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron:**
    
    * A practical guide covering various machine learning concepts and implementations.
        
2. **"Python Machine Learning" by Sebastian Raschka and Vahid Mirjalili:**
    
    * A comprehensive introduction to machine learning using Python.
        

**Websites and Documentation:**

1. [**Scikit-Learn Documentation**](https://scikit-learn.org/stable/documentation.html)**:**
    
    * The official documentation for the popular machine learning library, Scikit-Learn.
        
2. [**TensorFlow Tutorials**](https://www.tensorflow.org/tutorials)**:**
    
    * TensorFlow is an open-source machine learning library. The tutorials cover a wide range of topics.
        
3. [**PyTorch Tutorials**](https://pytorch.org/tutorials/)**:**
    
    * PyTorch is another popular deep learning library. The tutorials cover a variety of topics, from basics to advanced concepts.
        

**Online Platforms for Practice:**

1. [**Kaggle**](https://www.kaggle.com/)**:**
    
    * Kaggle is a platform for predictive modeling and analytics competitions. You can find datasets, participate in competitions, and collaborate with other data scientists.
        
2. [**DataCamp**](https://www.datacamp.com/)**:**
    
    * Offers interactive courses on various data science and machine learning topics.
        

**YouTube Channels:**

1. [**3Blue1Brown**](https://www.youtube.com/c/3blue1brown)**:**
    
    * Provides visually appealing explanations of various mathematical concepts, including those related to machine learning.
        
2. [**sentdex**](https://www.youtube.com/user/sentdex)**:**
    
    * Covers a wide range of topics related to machine learning and artificial intelligence using Python.
        

**Community and Forums:**

1. [**Stack Overflow**](https://stackoverflow.com/)**:**
    
    * A community of developers where you can ask and answer questions related to machine learning.
        
2. [**Towards Data Science on Medium**](https://towardsdatascience.com/)**:**
    
    * Medium publication with a wealth of articles on data science and machine learning.
        

---

### 🎙️ **Message for Next Episode:**

Join us for our upcoming episode on data science where we'll explore the difference between Supervised and Unsupervised Learning in machine learning. We'll demystify their distinctions, show you their real-world applications, and highlight when to use each. Stay tuned for 'Supervised vs. Unsupervised Learning | Episode 4.' 🎧✨

---

## **By the way…**

#### Call to action

*Hi, Everydaycodings— I’m building a newsletter that covers deep topics in the space of engineering. If that sounds interesting,* [***subscribe***](https://neuralrealm.hashnode.dev/newsletter) *and don’t miss anything. If you have some thoughts you’d like to share or a topic suggestion, reach out to me via* [***LinkedIn***](https://www.linkedin.com/in/kumar-saksham1891/) *or* [***X***](https://twitter.com/everydaycodings).

#### References

*And if you’re interested in diving deeper into these concepts, here are some great starting points:*

* [**Kaggle Stories**](https://neuralrealm.hashnode.dev/series/kaggle-stories) *\-* Each episode of Kaggle Stories takes you on a journey behind the scenes of a Kaggle notebook project, breaking down tech stuff into simple stories.
    
* [**Machine Learning**](https://neuralrealm.hashnode.dev/series/machine-learning) *\-* This series covers ML fundamentals & techniques to apply ML to solve real-world problems using Python & real datasets while highlighting best practices & limits.