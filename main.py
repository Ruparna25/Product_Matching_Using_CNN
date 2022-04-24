import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import keras as K
import pandas as pd
import os
from tensorflow.keras.applications.resnet import ResNet50
from keras.models import Sequential
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB3
#from bert import tokenization
from cuml.feature_extraction.text import TfidfVectorizer
import tensorflow_hub as hub
import cupy
import cudf
#from transformers import BertTokenizer, TFBertModel

copyfile(src = "../input/tokenize/tokenization.py", dst = "../working/tokenization.py")

train_data


BATCH_SIZE=8
IMAGE_SIZE=[512,512]
SEED=42
VERBOSE=1
N_CLASSES=train_data['label_group'].nunique()
path='Product_Matching_Resnet/train_images'

##### For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

def decode_image(image_data):
    image=tf.image.decode_jpeg(image_data,channels=3)
    image=tf.image.resize(image,IMAGE_SIZE)
    image=tf.cast(image,tf.float32)/255.0
    return image

def read_image(image):
    print('read')
    image=tf.io.read_file(image)
    image=decode_image(image)
    return image

def get_dataset(image):
    dataset = tf.data.Dataset.from_tensor_slices(image)
    dataset = dataset.map(read_image, num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset
	
def image_embeddings(imgpath):
    print("Using Regular Tensorflow Model For Predictions \n")
    embeds=[]
    
    start_time = time.time()
    margin=ArcMarginProduct(
        n_classes=N_CLASSES,
        s=30,
        m=0.7,
        name='head/arc_margin',
        dtype='float_32'
    )
    
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE,3),name='inp1')
    label=tf.keras.layers.Input(shape=(),name='inp2')
    x= EfficientNetB3(weights = None, include_top = False)(inp)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=margin([x,label])

    output=tf.keras.softmax(dtype='float32')(x)
    model=tf.keras.Model(inputs=[inp,label],outputs=[output])
    
    #loading saved weights
    model.load_weights('EfficientNet_b3_15_0.0001_512_42_final.h5')
    model=tf.keras.models.Model(inputs=model.input[0],outputs=model.layers[-4].output)
    
    chunk = 5000
    iterator = np.arange(np.ceil(len(train_data)/chunk))
    for j in terator:
        a=int(j*chunk)
        b=int((j+1)*chunk)
        image_dataset=get_dataset(imagepath[a:b])
        image_embeddings=model.predict(image_dataset)
        embeds.append(image_embeddings)
        
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    return image_embeddings

"""
    def bert_encode(texts,tokenizer,max_len=512):
    all_tokens=[]
    all_masks=[]
    all_segments=[]
    
    for txt in texts:
        print(txt)
        text=tokenizer.tokenize(txt)
        print(max_len)
        text=text[:max_len-2]
        print(text)
        input_seq=["[CLS]"]+text+["[SEP]"]
        pad_len=max_len-len(input_seq)
"""

"""
    def bert_get_text_embeddings(train_data,max_len=70):
        embeds=[]
    module_url = '../input/bert-en-uncased-l24-h1024-a16-1'
    bert_layer = hub.KerasLayer(module_url,trainable=True)
    vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
    #print(do_lower_case)
    tokenizer=tokenization.FullTokenizer(vocab_file,do_lower_case) #='../input/tokenize/tokenization.py',do_lower_case=True)
    #text=bert_encode(train_data['title'].values,tokenizer,max_len=max_len)
    print(tokenizer.tokenize('Hi! how are you'))
"""

def get_text_embeddings(train_cudf):
    model=TfidfVectorizer(stop_words='english',binary=True,max_features=25_000)
    text_embeddings=model.fit_transform(train_cudf.title).toarray()
    print(text_embeddings.shape)
    return text_embeddings

def title_match(text_embeddings):
    preds=[]
    CHUNK=5000
    
    print('Finding similar titles')
    iterator = np.arange(np.ceil(len(train_data)/CHUNK))
    print(iterator)
    for i in iterator:
        a=int(i*CHUNK)
        b=int((i+1)*CHUNK)
        b = min(b,len(train_data))
        
        dist = cupy.matmul(text_embeddings,text_embeddings[a:b].T).T
        
        for j in range(b-a):
            idx=cupy.where(dist[j,]>0.7)[0]
            o=train_data.iloc[cupy.asnumpy(idx)].posting_id.values
            preds.append(o)
            
    return preds

train_cudf=cudf.DataFrame(train_data)
text_embeddings=get_text_embeddings(train_cudf)

text_pred = title_match(text_embeddings)

train_data['preds']=text_pred
train_data[:5]
