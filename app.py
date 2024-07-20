import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import plotly.express as px
import pickle as pk
from PIL import Image

image=Image.open('hearti.jpg')





model=pk.load(open('disease.pkl','rb'))


st.set_page_config(page_title='Heart_Disease_Prediction',layout='wide')

df=pd.read_csv('heart.csv')
with st.sidebar:
    selected=option_menu(
        menu_title='Heart Disease',
        options=['Dashboard','Predict_Disease'],
        icons=['info-circle-fill','person-circle'],
        menu_icon='cast',
        styles={
        # "container": {"padding": "5!important","background-color":'white'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"black","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#c970d8"},
        "nav-link-selected": {"background-color": "#e9232f"},
            } 
#         # orientation='horizontal'
    )
# horizontal menu :
# selected=option_menu(
#         menu_title='Heart Disease',
#         options=['About','Dashboard','Predict_Disease'],
#         icons=['info-circle-fill','person-circle'],
#         menu_icon='cast',
#         orientation='horizontal',
#         default_index=1,
#         styles={
#         # "container": {"padding": "5!important","background-color":'white'},
#         "icon": {"color": "white", "font-size": "23px"}, 
#         "nav-link": {"color":"black","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#c970d8"},
#         "nav-link-selected": {"background-color": "#e9232f"},
#             }                 
#     )


if selected=='Dashboard':
    h1,h2=st.columns([0.1,0.9])
    with h1:
        st.image(image,width=100)
    htm_title="""
        <style>
        .title-test{
        font-weight:bold;
        padding:5px;
        border-radius:6px
        }
        </style>
        <center><h1 class="title-test">Heart Disease Data_visualization</h1></center>"""
    with h2:
        st.markdown(htm_title,unsafe_allow_html=True)

    st.markdown('________________________________________________')
    c1,c2,c3=st.columns(3)
    
    with c1:
        st.subheader('Disease Accourding to Gender')
        fig,ax=plt.subplots()
        pd.crosstab(df.sex,df.target).plot(kind='bar',ax=ax)
        plt.xlabel('Gender')
        plt.legend(['Don"t have Disease','Have Disease'])
        st.pyplot(fig,use_container_width=True)

    _,view1,down1,view2,down2,view3,_=st.columns([0.05,0.30,0.15,0.35,0.10,0.30,0.10])
    st.markdown('________________________________________________')
    with view1:
        da=df[['sex','target']].groupby(by=['sex'])['target'].value_counts()
        expendes=st.expander('Disease Accourding to Gender📥')
        expendes.write(da)
        expendes.write('0 = male, 1 = female')
        expendes.write('0=no disease,1= presence of disease')
    # with down1:
    #     st.download_button('Get_data',data=df['sex'].to_csv().encode('utf-8'),
    #                        file_name='Gender_unique_Data.csv',
    #                        mime=('text/csv'))
    with c2:
        st.subheader('Disease Accourding to CP')
        fig,ax=plt.subplots()
        sns.countplot(x='cp',hue='target',data=df,ax=ax)
        plt.legend(['Don"t have Disease','Have Disease'])
        plt.xlabel('Chest Pain')
        st.pyplot(fig,use_container_width=True)
    with view2:
        da=df[['cp','target']].groupby(by=['cp'])['target'].value_counts()
        expendes=st.expander('View CP Distribution As Per Target📥')
        expendes.write(da)
        expendes.write('0 = Have Disease, 1 = Don"t have Disease')
        expendes.write('0=Typical angina,\n\n1=Atypical angina,2=Non-anginal pain,   3: Asymptomatic')

    with c3:
        st.subheader('Data Balance')
        fig=plt.figure()
        sns.countplot(x='target',data=df)
        st.pyplot(fig,use_container_width=True)
    with view3:
        tar=df['target'].value_counts()
        exp=st.expander('View Imbalance target Data	📥')
        exp.write(tar)
        exp.write('0=Don"t have Disease ,1=Have Disease')


    
    c6,c7,c8=st.columns(3)
    with c6:
        st.subheader('Disease Accourding to fbs')
        fig,ax=plt.subplots()
        pd.crosstab(df.fbs,df.target).plot(kind='bar',ax=ax)
        plt.xlabel('Fasting blood sugar level')
        plt.legend(['Don"t have Disease','Have Disease'])
        st.pyplot(fig,use_container_width=True)
    _,view4,down4,view5,down5,view6,_=st.columns([0.05,0.30,0.15,0.35,0.10,0.30,0.10])
    with view4:
        da=df[['fbs','target']].groupby(by=['fbs'])['target'].value_counts()
        expendes=st.expander('View Disease Accourding to fbs📥')
        expendes.write(da)
        expendes.write('0 = Have Disease, 1 = Don"t have Disease')
        expendes.write('1 = true, 0 = false')

    with c7:
        st.subheader('Disease Accourding to Slope')
        fig,ax=plt.subplots()
        pd.crosstab(df.slope,df.target).plot(kind='bar',ax=ax)
        plt.xlabel('Slope of the peak exercise')
        plt.legend(['Don"t have Disease','Have Disease'])
        st.pyplot(fig,use_container_width=True)
    with view5:
        da=df[['slope','target']].groupby(by=['slope'])['target'].value_counts()
        expendes=st.expander('View Disease Accourding to Slope📥')
        expendes.write(da)
        expendes.write('0 = Have Disease, 1 = Don"t have Disease')
        expendes.write('0: Upsloping,1: Flat,2: Downsloping')
    with c8:
        st.subheader('Disease Accourding to BP')
        fig,ax=plt.subplots()
        pd.crosstab(df.restecg,df.target).plot(kind='bar',ax=ax)
        plt.xlabel('Resting blood pressure in mm Hg')
        plt.legend(['Don"t have Disease','Have Disease'])
        st.pyplot(fig,use_container_width=True)
    with view6:
        da=df[['restecg','target']].groupby(by=['restecg'])['target'].value_counts()
        expendes=st.expander('View Disease Accourding to BP📥')
        expendes.write(da)
        expendes.write('0 = Have Disease, 1 = Don"t have Disease')

    st.markdown('________________________________________________')
    c4,c5=st.columns(2)
    st.markdown('________________________________________________')
    with c4:
        st.subheader('Heart disease accourding to Age')
        fig,ax=plt.subplots()
        data1=pd.crosstab(df.age,df.target)
        data1.plot(kind='bar',ax=ax)
        plt.legend(['Don"t have Disease','Have Disease'])
        plt.xlabel('Age')
        st.pyplot(fig,use_container_width=True)
    with c5:
        cor=df.corr()
        st.subheader('Correlation between feature ')
        fig=plt.figure(figsize=(10,5))
        sns.heatmap(cor,annot=True,fmt='.2f',cmap='coolwarm',vmin=-1,vmax=1,center=0)
        # plt.rcParams['fo']
        st.pyplot(fig,use_container_width=True)




####################################################################################################
if selected=='Predict_Disease':
    attribution="""
    - Age of the patient in years
    - Gender of the patient (0 = male, 1 = female)
    - Chest pain type:(0: Typical angina,1: Atypical angina,2: Non-anginal pain,3: Asymptomatic)
    - Resting blood pressure in mm Hg
    - Serum cholesterol in mg/dl
    - Fasting blood sugar level, categorized as above 120 mg/dl (1 = true, 0 = false)
    - Resting electrocardiographic results:(0: Normal,1: Having ST-T wave abnormality,2: Showing probable or definite left ventricular hypertrophy)
    - Maximum heart rate achieved during a stress test
    - Exercise-induced angina (1 = yes, 0 = no)
    - ST depression induced by exercise relative to rest
    - Slope of the peak exercise ST segment:(0: Upsloping,1: Flat,2: Downsloping)
    - Number of major vessels (0-4) colored by fluoroscopy
    - Thalium stress test result:(0: Normal,1: Fixed defect,2: Reversible defect,3: Not described)
    - Heart disease status (0 = no disease, 1 = presence of disease)
"""
    st.title('Hear Disease Prediction ')
    # ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    #    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    with st.expander('Attribute Info :'):
        st.markdown(attribution)
    a1,a2,a3=st.columns((3))
    with a1:
        
        age=st.number_input('Age',28,100)
        sex=st.radio('Gender of the patient :',('Male','Female'),horizontal=True)
        cp=st.selectbox('Chest pain type',('Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'))
        trestbps=st.slider('Resting blood pressure in mm Hg',94,200)
        chol=st.slider('Serum cholesterol in mg/dl',126,564)

    with a2 :
        fbs=st.radio('Fasting blood sugar level, categorized as above 120 mg/dl ',('YES','No'),horizontal=True)
        restecg=st.selectbox('Resting electrocardiographic results:',('Normal','Having ST-T wave abnormality','Showing probable or definite left ventricular hypertrophy'))
        thalach=st.slider('Maximum heart rate achieved during a stress test',71,202)
        exang=st.radio('Exercise-induced angina:', ('YES','NO'))
    with a3:
        oldpeak=st.slider('ST depression induced by exercise relative to rest',0,7)
        slope=st.selectbox('Slope of the peak exercise ST segment:',('Upsloping','Flat','Downsloping'))
        ca=st.slider('Number of major vessels (0-4) colored by fluoroscopy',0,4)
        thal=st.selectbox('Thalium stress test result:',('Normal','Fixed defect','Reversible defect','Not described'))
    
    input_option=pd.DataFrame(
        [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
    )
    with st.expander('You selected options'):
        st.dataframe(input_option)
    input_option['sex'].replace(['Male','Female'],[0,1],inplace=True)
    input_option['cp'].replace(['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'],[0,1,2,3],inplace=True)
    input_option['fbs'].replace(['YES','No'],[1,0],inplace=True)
    input_option['restecg'].replace(['Normal','Having ST-T wave abnormality','Showing probable or definite left ventricular hypertrophy'],[0,1,2],inplace=True)
    input_option['exang'].replace(['YES','NO'],[1,0],inplace=True)
    input_option['slope'].replace(['Upsloping','Flat','Downsloping'],[0,1,2],inplace=True)
    input_option['thal'].replace(['Normal','Fixed defect','Reversible defect','Not described'],[0,1,2,3],inplace=True)
    b1,b2,b3=st.columns(3)
    if b2.button('predict'):
        disease=model.predict(input_option)
        if disease==0:
            st.success('Hey you Don"t have Heart disease😊')
        else:
            st.error('Hey you have heart disease😞')






if selected=='About':
    st.title(f'your selected is {selected}')
        







