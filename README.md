# Product_Matching_Using_CNN

### Overview
Product matching in e-commerce has recently become a frequently discussed topic among retailers. They use product matching to ensure a flawless buying experience and take advantage of selling online. Usually, products offered on different platforms are described by a title, attributes, and image. A product title is usually a brief text identifying the key information about a product. It can be a product name and its characteristics. Some commonly used technique of Product Matching are stated below.

- **Title Similarity** - The title similarity approach to product matching in eCommerce allows ML to compare the same offers by quantifying the similarity of the titles. This can be achieved either by using Distance Algorithm and for more sophiscated approach use BERT for finding similar titlte.

- **Image Matching** - Image similarity is based on the similar principle as title similarity. To detect the same products, the process of quantifying image similarity is involved. The image matching can be achieved by using one of the pre-trained CNN model as they are very efficient.

### Dataset
This data was collected from one of the competitions hosted at Kaggle.com. It contains data of a leading ecommerce site namely Shopee, based from Southeast Asia and Taiwan. In Shopee's case, everyday users can upload their own images and write their own product descriptions, which creates a problem of identifying near duplicates in a pool of data. Below is how the data looks like - 

- posting_ID - the ID code for the posting.
- image - the image classification 
- image_phash - a perceptual hash of the image.
- title - the product description for the posting.
- label_group - ID code for all postings that map to the same product.

### EDA

1. Muliple rows, each containing details about product, can have same label_group. Below are a few images of same label_group -

- Sample Input 1 having label_group = **297977**
<p float="left">
<img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/eg_1_a.jpg" width="300" height="300">
<img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/eg_1_b.jpg" width="300" height="300">
</p>

- Sample Input 2 having label_group - **33999540**
<p float="left">
  <img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/eg_2_a.jpg" width="300" height="300">
  <img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/eg_2_b.jpg" width="300" height="300">
  <img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/eg_3_b.jpg" width="300" height="300">
</p>

2. All the products have same label_group and image_phase are duplicates. Below is a sample of same image_phash and label_group -

- Sample Input 1 having label_group value as **1261987196** and image_phash - **fff24181c2a2d5e4** with different Titles

<p float="left">
  <img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/img_phash_1.JPG" width="300" height="300">
  <img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/img_phash_2.JPG" width="300" height="300">
  <img src="https://github.com/Ruparna25/Product_Matching_Using_CNN/blob/main/Images/img_phash_3.JPG" width="300" height="300">
</p>

### Model -
2 separate models were built and the results from them combined to produce the final output. 
- Title Similarity - Finding title which are close to each other, for this 2 options were tried one - first one was a simple distance measurement between products of same label_group. This is simple approach, where the cosine similarity was used to measure the distance between the text embeddings.


- Image Matching


BERT Model - https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1?tf-hub-format=compressed
