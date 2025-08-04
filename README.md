# Project(Problem Statement): Fake News Detection Model using TensorFlow

**• Objective:**
Develop a deep learning model using TensorFlow to classify news
articles as FAKE or REAL based on their text content.

**• Dataset: Fake News Dataset**

# Load required Python libraries:


```python
# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


# Load the dataset using Pandas and explore its structure.


```python
# Load dataset
df = pd.read_csv("news.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
```

# Data Insights


```python
df.head()
```





  <div id="df-bf87214c-f611-4283-b244-689c782ce673" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bf87214c-f611-4283-b244-689c782ce673')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bf87214c-f611-4283-b244-689c782ce673 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bf87214c-f611-4283-b244-689c782ce673');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-65d5839d-78b2-47b7-9bec-289422ea9f6c">
      <button class="colab-df-quickchart" onclick="quickchart('df-65d5839d-78b2-47b7-9bec-289422ea9f6c')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-65d5839d-78b2-47b7-9bec-289422ea9f6c button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




# **Convert text data into numerical format using Tokenization.**


```python
# Clean text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
df['main_text'] = df['title'] + ' ' + df['clean_text']
```

# Convert labels (FAKE, REAL) into binary values (0,1).


```python
# Drop unnecessary columns
processed_data = df.drop(['title', 'text', 'clean_text'], axis=1)
```


```python
processed_data.head()
```





  <div id="df-4575beaf-b366-4f94-940b-4c3d047a1c5f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>main_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>You Can Smell Hillary’s Fear daniel greenfield...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Kerry to go to Paris in gesture of sympathy u ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>The Battle of New York: Why This Primary Matte...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4575beaf-b366-4f94-940b-4c3d047a1c5f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4575beaf-b366-4f94-940b-4c3d047a1c5f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4575beaf-b366-4f94-940b-4c3d047a1c5f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-f71ed444-eb30-45bc-abd1-60e1f2c35605">
      <button class="colab-df-quickchart" onclick="quickchart('df-f71ed444-eb30-45bc-abd1-60e1f2c35605')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-f71ed444-eb30-45bc-abd1-60e1f2c35605 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




# Split dataset into training and testing sets (80%-20%).


```python
# Prepare training data
X = processed_data['main_text']
y = processed_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```


```python
# Tokenization and Padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 20000
max_length = 300

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

```


```python
print(processed_data['label'].value_counts())
```

    label
    1    3171
    0    3164
    Name: count, dtype: int64



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=processed_data)
plt.title('Distribution of Target Variable (label)')
plt.xlabel('Label (0: FAKE, 1: REAL)')
plt.ylabel('Count')
plt.xticks([0, 1], ['FAKE', 'REAL'])
plt.show()
```


    
![png](output_16_0.png)
    



```python

```


```python
%pip install gensim
```

    Requirement already satisfied: gensim in /usr/local/lib/python3.11/dist-packages (4.3.3)
    Requirement already satisfied: numpy<2.0,>=1.18.5 in /usr/local/lib/python3.11/dist-packages (from gensim) (1.26.4)
    Requirement already satisfied: scipy<1.14.0,>=1.7.0 in /usr/local/lib/python3.11/dist-packages (from gensim) (1.13.1)
    Requirement already satisfied: smart-open>=1.8.1 in /usr/local/lib/python3.11/dist-packages (from gensim) (7.3.0.post1)
    Requirement already satisfied: wrapt in /usr/local/lib/python3.11/dist-packages (from smart-open>=1.8.1->gensim) (1.17.2)


# Create the Embedding Matrix


```python
import numpy as np
import pandas as pd # Import pandas
import re # Import re
import string # Import string
import nltk # Import nltk
from nltk.corpus import stopwords # Import stopwords
from nltk.stem import WordNetLemmatizer # Import WordNetLemmatizer
from sklearn.model_selection import train_test_split # Import train_test_split
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === 0. Data Loading and Preprocessing ===
# Load dataset
df = pd.read_csv("news.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Clean text data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
df['main_text'] = df['title'] + ' ' + df['clean_text']

# Drop unnecessary columns
processed_data = df.drop(['title', 'text', 'clean_text'], axis=1)

# Prepare training data
X = processed_data['main_text']
y = processed_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# === 1. Load Word2Vec pretrained model ===
import gensim.downloader as api
print("Loading Word2Vec model...")
word2vec_model = api.load("word2vec-google-news-300")  # ~1.6GB
print("Word2Vec model loaded.")

# === 2. Prepare tokenizer and sequences ===
vocab_size = 20000
max_length = 300

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)  # Make sure X_train is preprocessed
word_index = tokenizer.word_index

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# === 3. Create the Embedding Matrix ===
embedding_dim = 300  # Word2Vec Google News vectors are 300-dimensional
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    if i >= vocab_size:
        continue
    if word in word2vec_model:
        embedding_matrix[i] = word2vec_model[word]
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


    Loading Word2Vec model...
    [==================================================] 100.0% 1662.8/1662.8MB downloaded
    Word2Vec model loaded.


# Build the Model with Word2Vec Embedding

# Model Architecture:

• Define a Sequential Deep Learning Model using TensorFlow:

o Input Layer (Embedding Layer).

o LSTM (Long Short-Term Memory) Layer for text processing.

o Dense Layers for classification.

o Activation function: Sigmoid for binary classification.


```python
#  Build the Model with Word2Vec Embedding ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=False,  # Set to True if you want to fine-tune Word2Vec
        mask_zero=False
    ),
    Bidirectional(LSTM(
        units=64,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        unit_forget_bias=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        return_sequences=False
    )),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros')
])
```

    /usr/local/lib/python3.11/dist-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.
      warnings.warn(


# Compile the model using Binary Crossentropy Loss and Adam Optimizer.


```python
# === 5. Compile the model ===
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)           │ ?                      │     <span style="color: #00af00; text-decoration-color: #00af00">6,000,000</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional (<span style="color: #0087ff; text-decoration-color: #0087ff">Bidirectional</span>)   │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ ?                      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ ?                      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ ?                      │   <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,000,000</span> (22.89 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,000,000</span> (22.89 MB)
</pre>




```python
# Ensure your labels are NumPy arrays or Pandas Series
y_train = np.array(y_train)
y_test = np.array(y_test)
print(y_train)
print(y_test)
```

    [1 0 0 ... 0 0 0]
    [1 0 1 ... 1 0 1]



```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

```


```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

```

# Train the model on training data.


```python
model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

```

    Epoch 1/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m124s[0m 2s/step - accuracy: 0.5790 - loss: 0.6626 - val_accuracy: 0.8077 - val_loss: 0.4583
    Epoch 2/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m112s[0m 2s/step - accuracy: 0.8019 - loss: 0.4643 - val_accuracy: 0.8314 - val_loss: 0.3952
    Epoch 3/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m144s[0m 2s/step - accuracy: 0.8195 - loss: 0.4164 - val_accuracy: 0.8531 - val_loss: 0.3554
    Epoch 4/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m142s[0m 2s/step - accuracy: 0.8520 - loss: 0.3484 - val_accuracy: 0.8373 - val_loss: 0.3836
    Epoch 5/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m140s[0m 2s/step - accuracy: 0.8557 - loss: 0.3497 - val_accuracy: 0.8600 - val_loss: 0.3343
    Epoch 6/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m141s[0m 2s/step - accuracy: 0.8753 - loss: 0.3039 - val_accuracy: 0.8402 - val_loss: 0.3587
    Epoch 7/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m136s[0m 2s/step - accuracy: 0.8766 - loss: 0.3016 - val_accuracy: 0.8491 - val_loss: 0.3528
    Epoch 8/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m143s[0m 2s/step - accuracy: 0.8806 - loss: 0.2894 - val_accuracy: 0.8540 - val_loss: 0.3297
    Epoch 9/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m145s[0m 2s/step - accuracy: 0.8882 - loss: 0.2642 - val_accuracy: 0.8639 - val_loss: 0.3310
    Epoch 10/10
    [1m64/64[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m151s[0m 2s/step - accuracy: 0.9007 - loss: 0.2602 - val_accuracy: 0.8679 - val_loss: 0.3249





    <keras.src.callbacks.history.History at 0x784575428590>



# Visualizing the results


```python
# Display Training Results
import matplotlib.pyplot as plt

history = model.history

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


    
![png](output_32_0.png)
    



    
![png](output_32_1.png)
    



```python
#  Calculate Test Accuracy
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f}")
```

    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m14s[0m 344ms/step - accuracy: 0.8740 - loss: 0.2834
    Test Accuracy: 0.8674



```python
# Compare Training, Test, and Validation Results

# Get metrics from training history
train_accuracy = history.history['accuracy'][-1]
train_loss = history.history['loss'][-1]
val_accuracy = history.history['val_accuracy'][-1]
val_loss = history.history['val_loss'][-1]

# Get test metrics
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)

print("--- Model Performance Comparison ---")
print(f"Training Accuracy:   {train_accuracy:.4f}")
print(f"Training Loss:       {train_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss:     {val_loss:.4f}")
print(f"Test Accuracy:       {test_accuracy:.4f}")
print(f"Test Loss:           {test_loss:.4f}")
```

    --- Model Performance Comparison ---
    Training Accuracy:   0.8986
    Training Loss:       0.2570
    Validation Accuracy: 0.8679
    Validation Loss:     0.3249
    Test Accuracy:       0.8674
    Test Loss:           0.2852


# Model Evaluation and Prediction:


```python
# Evaluate the model
y_pred = model.predict(X_test_pad)
y_pred_classes = (y_pred > 0.5).astype(int) # Convert probabilities to binary predictions

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))
```

    [1m40/40[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m15s[0m 346ms/step
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.85      0.89      0.87       633
               1       0.88      0.85      0.86       634
    
        accuracy                           0.87      1267
       macro avg       0.87      0.87      0.87      1267
    weighted avg       0.87      0.87      0.87      1267
    
    
    Confusion Matrix:
    [[563  70]
     [ 98 536]]


# RESULT INTERPRETATION

Precision: The model is slightly better at predicting malignant tumors (Class 1).

Recall: It detects benign tumors (Class 0) slightly more effectively.

F1-Score: Balanced performance between both classes.

# Confusion Matrix:

True Positives (Class 1 correctly identified): 536

True Negatives (Class 0 correctly identified): 563

False Positives (Class 0 predicted as 1): 70

False Negatives (Class 1 predicted as 0): 98

# Test the model on new unseen news articles


```python

# Define some new news articles
new_articles = [
    "Headline: Scientists Discover New Planet Text: Astronomers have found a new exoplanet in the Kepler-186 system, raising hopes for finding life beyond Earth.",
    "Headline: Local Politician Caught in Scandal Text: Reports indicate that a local politician is under investigation for misuse of public funds."
]

# Preprocess the new articles (using the same clean_text function, tokenizer, and max_length)
cleaned_new_articles = [clean_text(article) for article in new_articles]
new_articles_seq = tokenizer.texts_to_sequences(cleaned_new_articles)
new_articles_pad = pad_sequences(new_articles_seq, maxlen=max_length, padding='post', truncating='post')

# Predict the labels for the new articles
predictions = model.predict(new_articles_pad)
predicted_classes = (predictions > 0.5).astype(int)

# Display the predictions
print("Predictions for new articles (0: FAKE, 1: REAL):")
for i, article in enumerate(new_articles):
    print(f"Article {i+1}:")
    print(f"  Text: {article}")
    print(f"  Predicted Label: {predicted_classes[i][0]}")
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 439ms/step
    Predictions for new articles (0: FAKE, 1: REAL):
    Article 1:
      Text: Headline: Scientists Discover New Planet Text: Astronomers have found a new exoplanet in the Kepler-186 system, raising hopes for finding life beyond Earth.
      Predicted Label: 0
    Article 2:
      Text: Headline: Local Politician Caught in Scandal Text: Reports indicate that a local politician is under investigation for misuse of public funds.
      Predicted Label: 0


# Conclusion & Limitation:

The model performs well with a balanced and high classification score for both tumor types.

Slight misclassification exists, especially in detecting malignant cases (98 false negatives), which is critical in medical diagnosis.

Overall, the model is reliable but could be improved further, especially to reduce false negatives.
