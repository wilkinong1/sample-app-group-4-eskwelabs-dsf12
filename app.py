import streamlit as st
import os
from dotenv import load_dotenv
from skllm.config import SKLLMConfig
from skllm import MultiLabelZeroShotGPTClassifier
import openai
import data_files as data
import matplotlib.pyplot as plt
from model_files import run_model

load_dotenv()
openai.api_key = st.secrets['API_KEY']
SKLLMConfig.set_openai_key(openai.api_key)
mlzs_classifier = MultiLabelZeroShotGPTClassifier(openai_model="gpt-3.5-turbo", max_labels=3)

st.set_page_config(page_title='Sprint 4 App', page_icon=None, layout="wide", initial_sidebar_state='collapsed')

candidate_labels = [
    "Quality",
    "Price",
    "Delivery",
    "Service",
    "Customer Support",
    "Packaging",
    "User Experience",
    "Return Policy",
    "Product Information",
]

mlzs_classifier.fit(None, [candidate_labels])

if 'card_mean' not in st.session_state:
    st.session_state['card_mean'] = data.check_mean(data.df['submission_time'].min().date(), data.df['submission_time'].max().date(), list(set(data.df['brand_name'])))

if 'line_average' not in st.session_state:
    st.session_state['line_average'] = data.load_line(data.df['submission_time'].min().date(), data.df['submission_time'].max().date(), list(set(data.df['brand_name'])))
if 'combo_chart' not in st.session_state:
    st.session_state['combo_chart'] = data.load_combo(data.df['submission_time'].min().date(), data.df['submission_time'].max().date(), list(set(data.df['brand_name'])))
if 'combo_sentiment' not in st.session_state:
    st.session_state['combo_sentiment'] = data.load_sentiment(data.df['submission_time'].min().date(), data.df['submission_time'].max().date(), list(set(data.df['brand_name'])))
if 'topic_chart' not in st.session_state:
    st.session_state['topic_chart'] = data.load_topics(str(data.df['submission_time'].min().date()), str(data.df['submission_time'].max().date()), list(set(data.df['brand_name'])))


if 'reviews' not in st.session_state:
    st.session_state['reviews'] = []
if 'review_labels' not in st.session_state:
    st.session_state['review_labels'] = []
if 'review_class' not in st.session_state:
    st.session_state['review_class'] = []
if 'review_count' not in st.session_state:
    st.session_state['review_count'] = 0


def write_review():
    for i in range(st.session_state["review_count"]):
        st.subheader(body=f'Review {i+1}')
        st.write(f'Review: {st.session_state["reviews"][i]}')
        st.write(f'Review Label/s: {st.session_state["review_labels"][i]}')
        st.write(f'Rating Prediction: {st.session_state["review_class"][i][0]}')

def color_sentiment(sentiment):
    if sentiment == 'Positive':
        return 'background-color: #097969'
    elif sentiment == 'Negative':
        return 'background-color: #880808'
    else:
        return 'background-color: #36454F'

reviews_tab, dashboard_tab = st.tabs(['Reviews', 'Dashboard'])

def main():
    
    with reviews_tab:
        st.title("Sprint 4 App")
        with st.form(key='review-form', clear_on_submit=True):
            review_text = st.text_area(label='Product Review:')
            review_submit = st.form_submit_button(label='Post')
            if review_submit:
                st.session_state['reviews'].append(review_text)
                labels = mlzs_classifier.predict([review_text])
                review_class = run_model(review_text)
                st.session_state['review_class'].append(review_class)
                st.session_state['review_labels'].append(labels[0])
                st.session_state['review_count'] = st.session_state['review_count'] + 1

        if st.session_state['review_count'] > 0:
            write_review()
    
    with dashboard_tab:
        st.title("Reviews and Sentiment Dashboard")

        with st.sidebar:
            date_select = st.date_input(
                label="Select date range: ",
                value=(data.df['submission_time'].min(), data.df['submission_time'].max()),
                min_value=data.df['submission_time'].min(),
                max_value=data.df['submission_time'].max(),
            )

            brand_select = st.multiselect('Select brand: ', list(set(data.df['brand_name'])))
            
            def update_charts():
                try:
                    min_date = date_select[0]
                    max_date = date_select[1]
                except IndexError:
                    min_date = date_select[0]
                    max_date = date_select[0]

                st.session_state['line_average'] = data.load_line(min_date, max_date, list(brand_select))
                st.session_state['combo_chart'] = data.load_combo(min_date, max_date, list(brand_select))
                st.session_state['combo_sentiment'] = data.load_sentiment(min_date, max_date, list(brand_select))
                st.session_state['card_mean'] = data.check_mean(min_date, max_date, list(brand_select))
                st.session_state['topic_chart'] = data.load_topics(str(min_date), str(max_date), list(brand_select))

            st.button('Filter', on_click=update_charts)
        
        main_col1, main_col2 = st.columns(spec=[1, 3], gap='small')
        with main_col1.container():
            sub_col1, sub_col2 = st.columns(spec=[1, 1], gap='medium')

            sub_col1.metric(label=' ', value='', label_visibility='hidden')
            sub_col1.metric("Average Rating", f"{st.session_state['card_mean'][0]}")
            sub_col1.metric("Date Range", f"{st.session_state['card_mean'][1]} - {st.session_state['card_mean'][2]}")
            sub_col1.metric(label=' ', value='', label_visibility='hidden')

            
            sub_col2.metric(label=' ', value='', label_visibility='hidden')
            sub_col2.metric("Review Count", f"{st.session_state['card_mean'][3]}")

        # with main_col1.container():

        with main_col2.container():
            sub_col3, sub_col4, = st.columns(spec=[1, 1], gap='medium')
            sub_col3.plotly_chart(st.session_state['line_average'])
            sub_col3.plotly_chart(st.session_state['combo_sentiment'])

            sub_col4.plotly_chart(st.session_state['combo_chart'])
            sub_col4.plotly_chart(st.session_state['topic_chart'])

            

        with st.container():
            st.header('Recent Reviews')
            st.dataframe(data.df_most_recent.style.applymap(color_sentiment, subset=['rule_based_sentiment']), hide_index=True, use_container_width=True, 
                         column_config={
                             'rating': 'Rating',
                             'review_text': 'Review',
                             'product_name': 'Product',
                             'brand_name': 'Brand',
                             'rule_based_sentiment': 'Sentiment'
                         })
        
        with st.container():
            st.header('Topics Recommendations')
            rec_col1, rec_col2 = st.columns(spec=[1, 1], gap='small')

            select_topic = rec_col1.selectbox('Select Topic to view Recommendations for: ', ('Brand Reputation', 'Marketing and Promotions', 'Pricing and Value', 'Product Quality', 'Product Variety'))

            rec_col2.subheader(select_topic)
            rec_col2.write(data.topic_recs[select_topic.lower()])



if __name__ == "__main__":
    main()
