import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from wordcloud import WordCloud

# import matplotlib.pyplot as plt

df = pd.read_csv('dataset_filtered_sentiment.csv')
df_topics = pd.read_csv('with_topics.csv')

df_summary = df.copy(deep=True)
# df_summary.groupby('sentiment')

df['submission_time'] = pd.to_datetime(df['submission_time'])

def check_mean(start, end, brands):
    if len(brands) == 0:
        brands = list(set(df['brand_name']))

    df_filtered = df.copy(deep=True)
    df_filtered['date_index'] = df_filtered['submission_time']
    df_filtered.set_index('date_index', inplace=True)
    df_filtered = df_filtered[df_filtered['brand_name'].isin(brands)]
    df_filtered = df_filtered.sort_values(by='submission_time', ascending=True)
    df_filtered = df_filtered[start:end]

    mean_rating = round(df_filtered['rating'].mean(), 2)

    min_date = df_filtered['submission_time'].min().date().strftime('%m/%d')
    max_date = df_filtered['submission_time'].max().date().strftime('%m/%d')

    review_count = df_filtered['rating'].count()

    return [mean_rating, min_date, max_date, review_count]



# Line Chart
def load_line(start, end, brands):
    df_line_ave = df.groupby(['submission_time', 'brand_name']).agg({'rating' :'mean'}).reset_index()

    if len(brands) == 0:
        brands = list(set(df['brand_name']))

    df_line_ave_filtered = df_line_ave[df_line_ave['brand_name'].isin(brands)].groupby('submission_time').agg({'rating':'mean'}).reset_index()
    df_line_ave_filtered['date_index'] = df_line_ave_filtered['submission_time']
    df_line_ave_filtered.set_index('date_index', inplace=True)
    
    daily_average_plot = px.line(data_frame=df_line_ave_filtered[start:end], x='submission_time', y='rating', 
            labels={'submission_time': 'Date', 'rating': 'Average Rating'}, 
            title='Daily Average Rating')
    return daily_average_plot

# Combo Chart
def load_combo(start, end, brands):
    df_most_reviewed = df.groupby(['brand_name', 'submission_time']).agg({'rating' :'mean', 'product_name':'count'}).reset_index().sort_values(by=['product_name', 'rating'], ascending=[False, False])

    if len(brands) == 0:
        brands = list(set(df['brand_name']))

    df_most_reviewed_filtered = df_most_reviewed.copy(deep=True)
    df_most_reviewed_filtered['date_index'] = df_most_reviewed_filtered['submission_time']
    df_most_reviewed_filtered = df_most_reviewed_filtered.sort_values(by='date_index', ascending=True)
    df_most_reviewed_filtered.set_index('date_index', inplace=True)
    df_most_reviewed_filtered = df_most_reviewed_filtered[start:end]
    df_most_reviewed_filtered = df_most_reviewed_filtered.groupby('brand_name').agg({'rating':'mean', 'product_name':'sum'}).reset_index().sort_values(by=['product_name', 'rating'], ascending=[False, False])
    df_most_reviewed_filtered = df_most_reviewed_filtered[df_most_reviewed_filtered['brand_name'].isin(brands)]
    

    combo_chart = make_subplots(specs=[[{"secondary_y": True}]])
    combo_chart.add_trace(
        go.Scatter(
            x=df_most_reviewed_filtered.head(10)['brand_name'],
            y=df_most_reviewed_filtered.head(10)['rating'],
            name="Average Rating",
            yaxis='y2'
        ))
    combo_chart.add_trace(
        go.Bar(
            x=df_most_reviewed_filtered.head(10)['brand_name'],
            y=df_most_reviewed_filtered.head(10)['product_name'],
            name="Review Count",
            text = df_most_reviewed_filtered.head(10)['product_name'],
            textposition='outside',
            textfont=dict(
            size=13,
            color='#1f77b4'),
            yaxis='y1'
        ))
    combo_chart.update_traces(texttemplate='%{text:.2s}')
    combo_chart.update_layout(legend_title_text='Legend',
                    title_text='Average Rating of Most Reviewed Products')

    return combo_chart


# df sorted by date most recent
df_most_recent = df.sort_values(by='submission_time', ascending=False)[['rating', 'review_text', 'product_name', 'brand_name', 'rule_based_sentiment']].head(50)


# sentiment over time
def load_sentiment(start, end, brands):
    if len(brands) == 0:
        brands = list(set(df['brand_name']))

    df_filtered = df.copy(deep=True)
    df_filtered['date_index'] = df_filtered['submission_time']
    df_filtered.set_index('date_index', inplace=True)
    df_filtered = df_filtered[df_filtered['brand_name'].isin(brands)]
    df_filtered = df_filtered.sort_values(by='submission_time', ascending=True)
    df_filtered = df_filtered[start:end]

    df_sentiment = df_filtered.groupby(['submission_time', 'rule_based_sentiment']).agg({'brand_name': 'count'}).reset_index(names=['submission_time', 'rules_based_sentiment']).sort_values(by='submission_time', ascending=True)

    combo_sentiment = make_subplots()
    combo_sentiment.add_trace(
        go.Scatter(
            x=df_sentiment[df_sentiment['rules_based_sentiment']=='Positive']['submission_time'],
            y=df_sentiment[df_sentiment['rules_based_sentiment']=='Positive']['brand_name'],
            name="Positive Sentiment",
            stackgroup='stack'
        ))
    combo_sentiment.add_trace(
        go.Scatter(
            x=df_sentiment[df_sentiment['rules_based_sentiment']=='Neutral']['submission_time'],
            y=df_sentiment[df_sentiment['rules_based_sentiment']=='Neutral']['brand_name'],
            name="Neutral Sentiment",
            stackgroup='stack'
        ))
    combo_sentiment.add_trace(
        go.Scatter(
            x=df_sentiment[df_sentiment['rules_based_sentiment']=='Negative']['submission_time'],
            y=df_sentiment[df_sentiment['rules_based_sentiment']=='Negative']['brand_name'],
            name="Negative Sentiment",
            stackgroup='stack'
        ))

    combo_sentiment.update_layout(legend_title_text='Legend',
                    title_text='Sentiment Over Time')
    
    return combo_sentiment


# topics

def load_topics(start, end, brands):
    if len(brands) == 0:
        brands = list(set(df['brand_name']))

    df_topics_filtered = df_topics.copy(deep=True)
    df_topics_filtered.dropna(inplace=True)
    df_topics_filtered['date_index'] = df_topics_filtered['submission_time']
    df_topics_filtered.set_index('date_index', inplace=True)
    df_topics_filtered = df_topics_filtered.sort_values(by='submission_time', ascending=True)
    df_topics_filtered = df_topics_filtered[df_topics_filtered['brand_name'].isin(brands)]
    df_topics_filtered = df_topics_filtered[str(start):str(end)]


    df_topics_grouped = df_topics_filtered.groupby(['dominant_topic', 'submission_time']).aggregate({'brand_name': 'count'}).reset_index()

    # print(df_topics_grouped)

    # print(list(df_topics_grouped[df_topics_grouped['brand_name']==0]))
    # stacked_100 = go.Figure()
    # stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==0]['brand_name']), name='Topic 0')
    # stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==1]['brand_name']), name='Topic 1')
    # stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==2]['brand_name']), name='Topic 2')
    # stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==3]['brand_name']), name='Topic 3')
    # stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==4]['brand_name']), name='Topic 4')
    # stacked_100.update_layout(barmode='relative', title_text='Topics Per Day')
    # return stacked_100

    stacked100 = px.histogram(df_topics_grouped, x='submission_time', y='brand_name', color='dominant_topic', title='Topics Per Day', barnorm='percent', nbins=25, labels={"submission_time":"Date", 'brand_name': "Topic Count", "dominant_topic": "Topic"})
    return stacked100






topic_recs = {'brand reputation': """
                           Based on the feedback from customers, the main takeaways for our business stakeholders are:

                        1. The product is highly effective in hydrating lips and providing long-lasting moisture.
                        2. Customers appreciate the variety of scents available.
                        3. The product is easy to use and comes with a convenient applicator.
                        4. Some customers have concerns about the safety of certain ingredients and suggest conducting research on the ingredients.
                        5. There is a demand for more subtle and gentle scents.

                           """,
              
              'marketing and promotions': """
                           Based on the feedback provided, the main takeaways are:

                            1. Customers value products that effectively address specific skin concerns such as dark circles, uneven skin texture, and dullness.
                            2. Products with vitamin C and E are highly sought after for their skin-hydrating and brightening properties.
                            3. Customers appreciate products that provide noticeable results in a short period of time.
                            4. Packaging and convenience of use are important factors for customer satisfaction.
                            5. Some customers have experienced negative reactions, such as skin irritation and breakouts, with certain products.
                            6. Customers appreciate when products are gifted to them for review and are more likely to recommend those that they received as gifts.

                            Based on these takeaways, the following problems should be focused on for your business stakeholders:

                            1. Developing effective solutions for reducing dark circles.
                            2. Creating products with vitamin C and E that provide hydration and illumination to the skin.
                            3. Exploring ways to improve skin texture and appearance.
                            4. Enhancing product packaging and convenience of use.
                            5. Addressing any negative reactions or irritations caused by certain products.
                            6. Implementing strategies to increase customer satisfaction and engagement through gifting and product trials.
                           
                           """,

                'pricing and value': """
                            Based on the customer feedback, here are the main takeaways:

                            1. Hydration is a key benefit of the product.
                            2. The product is effective for dry and flaky lips.
                            3. The product is especially helpful during the winter months.
                            4. Some customers experienced negative reactions, such as breakouts or skin irritation.
                            5. The product is praised for its soothing properties and ability to treat eczema and sensitive skin.
                            6. The product is versatile and can be used on different parts of the body.
                            7. The size and value of the product are appreciated by customers.

                            Based on these takeaways, the following problems can be identified for business stakeholders:

                            1. Improve long-term effectiveness: Address the concerns of customers who did not see improvement in the long run by enhancing the product's formula or developing additional products to provide sustained hydration.
                            2. Reduce negative reactions: Investigate and address the reports of breakouts and skin irritation to ensure the product is suitable for all skin types.
                            3. Enhance versatility: Explore different formulations or product variations to meet the specific needs of different body parts and use cases.
                            4. Increase size options: Consider offering different sizes or packaging options to cater to different customer preferences and needs.
                            5. Improve value proposition: Assess the pricing strategy and consider offering more competitive prices or value-added features to enhance the perception of value for customers.
                           """,

                'product quality': """
                            Main Takeaways:
                            1. Customers love the effectiveness of the product in terms of cleansing and hydrating the skin.
                            2. The scent of the product is a point of concern for some customers, with differing opinions on whether it is pleasant or not.
                            3. The price of the product is a common criticism, with some customers feeling that it is expensive for what it offers.
                            4. Some customers have experienced dryness or irritation after using the product.
                            5. Packaging and application are mentioned positively by some customers.

                            Problems to focus on for business stakeholders:
                            1. Improve the scent of the product to cater to a wider range of customer preferences.
                            2. Evaluate the pricing strategy to optimize value for customers and address concerns about the product being expensive.
                            3. Address issues with dryness and irritation by reviewing the product formulation and making necessary improvements.
                            4. Consider customer feedback on packaging and application and determine if any modifications are needed to enhance the user experience.
                            """,

                'product variety': """
                        Based on the customer reviews, here are the main takeaways and potential problems that should be focused on for your business stakeholders:

                        1. Effectiveness: Some customers found the lip mask to be effective in hydrating and softening their lips, especially for dry and cracked lips. However, other customers did not find it moisturizing enough or experienced chapped lips after using it. This indicates a need to improve the formula or address potential allergies.

                        2. Pricing: Although the request was not to provide pricing-related recommendations, many customers mentioned the high price of the product and questioned its value for money. This suggests the need to address price perception and potentially offer more affordable options.

                        3. Packaging and scent: Customers appreciated the pleasant scent and packaging of the product, particularly mentioning the gummy bear and fruity scents. This presents an opportunity to continue innovating with enjoyable scents and appealing packaging.

                        4. Longevity: Some customers mentioned that the lip mask lasted for a long time, while others felt that it dissipated quickly once applied. This discrepancy in longevity could be addressed to ensure consistency and prolong the product's effectiveness.
                           """
              
              
              }
