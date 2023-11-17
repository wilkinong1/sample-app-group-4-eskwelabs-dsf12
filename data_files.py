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
df_most_recent = df.sort_values(by='submission_time', ascending=False)[['rating', 'review_text', 'product_name', 'brand_name', 'rule_patterns_sentiment']].head(50)


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

    df_sentiment = df_filtered.groupby(['submission_time', 'rule_patterns_sentiment']).agg({'brand_name': 'count'}).reset_index(names=['submission_time', 'rules_patterns_sentiment']).sort_values(by='submission_time', ascending=True)

    combo_sentiment = make_subplots()
    combo_sentiment.add_trace(
        go.Scatter(
            x=df_sentiment[df_sentiment['rules_patterns_sentiment']=='Positive']['submission_time'],
            y=df_sentiment[df_sentiment['rules_patterns_sentiment']=='Positive']['brand_name'],
            name="Positive Sentiment",
            stackgroup='stack'
        ))
    combo_sentiment.add_trace(
        go.Scatter(
            x=df_sentiment[df_sentiment['rules_patterns_sentiment']=='Neutral']['submission_time'],
            y=df_sentiment[df_sentiment['rules_patterns_sentiment']=='Neutral']['brand_name'],
            name="Neutral Sentiment",
            stackgroup='stack'
        ))
    combo_sentiment.add_trace(
        go.Scatter(
            x=df_sentiment[df_sentiment['rules_patterns_sentiment']=='Negative']['submission_time'],
            y=df_sentiment[df_sentiment['rules_patterns_sentiment']=='Negative']['brand_name'],
            name="Negative Sentiment",
            stackgroup='stack'
        ))

    combo_sentiment.update_layout(legend_title_text='Legend',
                    title_text='Sentiment Over Time')
    
    return combo_sentiment


# topics

def load_topics(start, end, brands):
    df_topics_filtered = df_topics.copy(deep=True)
    df_topics_filtered.dropna(inplace=True)
    df_topics_filtered['date_index'] = df_topics_filtered['submission_time']
    df_topics_filtered.set_index('date_index', inplace=True)
    df_topics_filtered = df_topics_filtered.sort_values(by='submission_time', ascending=True)
    df_topics_filtered = df_topics_filtered[df_topics_filtered['brand_name'].isin(brands)]
    df_topics_filtered = df_topics_filtered[start:end]

    df_topics_grouped = df_topics_filtered.groupby(['dominant_topic', 'submission_time']).aggregate({'brand_name': 'count'}).reset_index()

    print(list(df_topics_grouped[df_topics_grouped['brand_name']==0]))
    stacked_100 = go.Figure()
    stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==0]['brand_name']), name='Topic 0')
    stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==1]['brand_name']), name='Topic 1')
    stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==2]['brand_name']), name='Topic 2')
    stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==3]['brand_name']), name='Topic 3')
    stacked_100.add_bar(x=list(set(df_topics_grouped['submission_time'])), y=list(df_topics_grouped[df_topics_grouped['dominant_topic']==4]['brand_name']), name='Topic 4')
    stacked_100.update_layout(barmode='relative', title_text='Topics Per Day')
    return stacked_100




topic_recs = {'topic 0': """
                           Based on the customer feedback, here are the main takeaways and the corresponding problems that should be focused on for your business stakeholders:

                            1. Lip product: The product made customers' lips dry, cracked, and bleeding.
                           - Problem: The product formulation may be too drying and needs improvement to provide better lip hydration.

                            2. Skincare product for sensitive skin: The product caused skin irritation, breakouts, and redness.
                            - Problem: The product may contain ingredients that are not suitable for sensitive skin or may need to be reformulated to be gentler on sensitive skin.

                            3. Moisturizer for dry, sensitive skin: The product caused burning and redness, and was not effective for dry, sensitive skin.
                            - Problem: The product formulation may be too harsh and irritating for sensitive skin, or it may not provide enough moisture for dry
                           """,
              
              'topic 1': """
                           Based on the feedback provided, some key takeaways and problems that should be focused on for the business stakeholders are:

                            1. Lip Mask:
                            - Problem: Inconsistent effectiveness in hydrating lips overnight
                            - Solution: Improve the hydrating properties of the lip mask to deliver consistent results

                            2. Face Cleanser:
                            - Problem: Leaves residue on the skin, causes breakouts, and feels oily
                            - Solution: Reformulate the cleanser to effectively remove makeup and leave the skin feeling clean without any residue or oiliness

                            3. Moisturizer:
                            - Problem: Dries out the skin and causes breakouts, not suitable for oily skin, and leaves a white cast
                            - Solution: Develop a moisturizer that effectively hydrates the skin without causing dryness or breakouts, and that is suitable for various skin types and tones

                            4. Face Cream:
                            - Problem: Feels oily, causes breakouts, and doesn't provide enough hydration
                            - Solution: Create a face cream that provides adequate hydration without feeling heavy or causing breakouts

                            5. Sunscreen:
                            - Problem: Leaves a white residue, doesn't blend well with makeup, breaks out the skin, and has an unpleasant scent
                            - Solution: Develop a sunscreen that is lightweight, blends well with makeup, doesn't leave a white residue, and is suitable for sensitive skin

                            6. Exfoliating Treatment:
                            - Problem: Feels sandy and stings the eyes
                            - Solution: Refine the exfoliating treatment to provide a smoother texture and eliminate any potential irritation or discomfort

                            7. Cleansing Balm:
                            - Problem: Leaves an oil residue on the skin, doesn't effectively remove makeup,
                           
                           """,

                'topic 3': """
                            Based on the feedback provided, the following problems can be identified for the business stakeholders:

                            1. Inconsistent product experience: Some customers have reported varying experiences with the same product. This could indicate a problem with product quality or formulation that needs to be addressed.

                            2. Skin irritation and breakouts: Several customers have reported experiencing skin irritation, breakouts, and even cystic acne after using certain products. This suggests a potential issue with product ingredients or formulation that may not be suitable for all skin types.

                            3. Ineffective results: Some customers have expressed disappointment with the lack of noticeable results from using certain products. This indicates a need to improve product effectiveness and ensure that customers see the benefits they expect.

                            4. Packaging and product usability: Customers have mentioned difficulties with product packaging, including leaks and separations. Additionally, some customers have found products to be hard to use, flaky, or with a pilling effect. These comments highlight the importance of addressing packaging and usability issues to enhance the overall customer experience.

                            5. Price perception: A few customers have mentioned that certain products are pricey and may not be worth the cost. This feedback suggests a need to evaluate pricing strategies and ensure that products are perceived as a good value for the price.

                            Overall, the business stakeholders should focus on addressing these problems related to product experience, skin irritation, effectiveness, packaging usability, and price perception in order to improve customer satisfaction and loyalty.
                           """,
                'topic 2': """
                            Based on the feedback provided, the main problems that should be focused on for the business stakeholders are:

                            1. Product effectiveness: Customers have expressed mixed opinions on the effectiveness of the products. Some have seen positive results, while others have experienced negative effects, such as breakouts or skin irritation. The business should explore ways to improve the formulation and performance of their products to ensure consistent and satisfactory results for customers.

                            2. Allergic reactions and skin sensitivity: Several customers have reported experiencing allergic reactions, redness, and irritation after using the products. This highlights the need for the business to review and potentially modify their ingredients to minimize the risk of adverse reactions and cater to customers with sensitive skin.

                            3. Texture and application issues: Many customers have commented on the texture and application experience of the products, including greasiness, pilling, and difficulty in mixing the product. These issues can impact the usability and
                            """,

                'topic 4': """
                        Based on the feedback provided, the main problems that should be focused on for business stakeholders are:

                        1. Product effectiveness: Some customers are reporting positive results and improvements in their skin and appearance after using the product, while others have expressed disappointment and did not see any difference. It is important for stakeholders to understand the factors that contribute to the effectiveness of the product and identify any potential issues that may be affecting its performance.

                        2. Skin reactions: A few customers have mentioned experiencing skin reactions, such as breakouts, blackheads, or dryness, after using the product. Stakeholders should investigate these reactions further and determine whether there are any potential allergens or irritants in the product formulation that may be causing these issues.

                        3. Container size and value for money: Several customers have expressed dissatisfaction with the amount of product they received in relation to the price. Stakeholders should assess the packaging and pricing strategy to ensure customers feel they are getting a fair value for their purchase.

                        4. Texture and fragrance: Some customers have commented on the texture and fragrance of the product, with mixed opinions. Stakeholders should consider customer preferences and determine if any improvements can be made to enhance the overall sensory experience.

                        5. Targeted marketing and pricing options: Customers have mentioned being interested in trying other products or brands that offer similar benefits at a different price point. Stakeholders should explore targeted marketing strategies to retain these customers and consider offering different pricing options to cater to varying customer budgets.

                        6. Customer education and understanding: The feedback highlights a lack of understanding of certain product concepts, such as double cleansing. Stakeholders should consider providing educational resources or information to customers to address any confusion and ensure they can fully benefit from the product.

                        By addressing these problems, stakeholders can enhance the product's performance, customer satisfaction, and overall business success.
                           """
              
              
              }