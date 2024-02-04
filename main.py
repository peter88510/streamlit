import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

### Confusion_Matrix


iris = datasets.load_iris()

X = iris.data[:, :4] # 取前两列特征
y = iris.target

# 随机抽取33%的数据作为测试集,其余作为训练集
np.random.seed(0)
indices = np.random.permutation(len(X))
cut = int(len(X) * 0.67)

X_train = X[indices[:cut]]
y_train = y[indices[:cut]]

X_test = X[indices[cut:]]
y_test = y[indices[cut:]]

st.title('Iris Flower Classification App')

with st.sidebar:
    st.header('Input Features')

    sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.5)
    sepal_width = st.slider('Sepal Width', 2.0, 5.0, 3.0)
    petal_length = st.slider('Petal Length', 0.5, 7.0, 1.5)
    petal_width = st.slider('Petal Width', 0.1, 3.0, 0.3)

    input_data = {'sepal_length': sepal_length,
                  'sepal_width': sepal_width,
                  'petal_length': petal_length,
                  'petal_width': petal_width}

    features = pd.DataFrame(input_data, index=[0])

    st.dataframe(features, column_config={'sepal_length': st.column_config.NumberColumn("sepal length", format="%f cm"),
                                      'sepal_width': st.column_config.NumberColumn("sepal width", format="%f cm"),
                                      'petal_length': st.column_config.NumberColumn("petal length", format="%f cm"),
                                      'petal_width': st.column_config.NumberColumn("sepal width", format="%f cm")}, hide_index=True)

# model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_prediction = clf.predict(X_test)
confunsion_mat = confusion_matrix(y_true=y_test, y_pred=y_prediction)
precision = precision_score(y_test, y_prediction, average='weighted')
recall = recall_score(y_test, y_prediction, average='weighted')
f1 = f1_score(y_test, y_prediction, average='weighted')
Acc = accuracy_score(y_test, y_prediction)

st.subheader('Model Evaluation')
st.markdown('(*RandomForestClassifier*)')
st.caption("#comfusion matrix")
df_comfusion = pd.DataFrame(confunsion_mat, columns=["setosa-pred", "versicolor-pred", "virginica-pred"], index=["setosa-true", "versicolor-true", "virginica-true"])
st.dataframe(df_comfusion.style.highlight_max(axis=0), use_container_width=True)
st.caption("#validation index")
st.write("precision score =  %3f" % precision)
st.write("recall score =  %3f" % recall)
st.write("f1 score =  %3f" % f1)
st.write("accuracy score =  %3f" % Acc)


st.subheader('Prediction Probability')

st.dataframe(clf.predict_proba(features),
             column_config={
                 "0": st.column_config.NumberColumn("setosa(0)", help="山鳶尾", format="%f %%"),
                 "1": st.column_config.NumberColumn("versicolor(1)", help="變色鳶尾", format="%f %%"),
                 "2": st.column_config.NumberColumn("virginica(2)", help="維吉尼亞鳶尾", format="%f %%"),
}, use_container_width=True)



prediction = clf.predict(features)
st.subheader('Prediction')
df = st.dataframe(prediction, column_config={"value": "label"})
if prediction[0] == 0:
    a = "**setosa**"
elif prediction[0] == 1:
    a = "**versicolor**"
elif prediction[0] == 2:
    a = "**virginica**"
st.write(f"the prediction is :red[{a}] !")


