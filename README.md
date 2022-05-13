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


BERT Model - https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1?tf-hub-format=compressed
