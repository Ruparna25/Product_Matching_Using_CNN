{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T05:34:48.618069Z\",\"iopub.execute_input\":\"2022-04-22T05:34:48.618350Z\",\"iopub.status.idle\":\"2022-04-22T05:34:59.069055Z\",\"shell.execute_reply.started\":\"2022-04-22T05:34:48.618320Z\",\"shell.execute_reply\":\"2022-04-22T05:34:59.068259Z\"}}\n!pip install bert-tensorflow\n\n# %% [markdown]\n# Importing the necessary library - RESNET is being used for image classification, so importing necessary libraries\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T06:08:45.478508Z\",\"iopub.execute_input\":\"2022-04-22T06:08:45.478785Z\",\"iopub.status.idle\":\"2022-04-22T06:08:45.485860Z\",\"shell.execute_reply.started\":\"2022-04-22T06:08:45.478754Z\",\"shell.execute_reply\":\"2022-04-22T06:08:45.484944Z\"}}\nimport numpy as np\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nimport tensorflow as tf\nimport keras as K\nimport pandas as pd\nimport os\nfrom tensorflow.keras.applications.resnet import ResNet50\nfrom keras.models import Sequential\nfrom tensorflow.keras.applications.resnet import preprocess_input\nfrom tensorflow.keras.preprocessing import image\nfrom tensorflow.keras.applications.efficientnet import EfficientNetB3\nfrom bert import tokenization\nimport tensorflow_hub as hub\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T04:22:37.948820Z\",\"iopub.execute_input\":\"2022-04-22T04:22:37.949046Z\",\"iopub.status.idle\":\"2022-04-22T04:22:38.127594Z\",\"shell.execute_reply.started\":\"2022-04-22T04:22:37.949013Z\",\"shell.execute_reply\":\"2022-04-22T04:22:38.126924Z\"}}\ntrain_data = pd.read_csv('../input/shopee-product-matching/train.csv')\ntrain_data\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T04:22:38.128846Z\",\"iopub.execute_input\":\"2022-04-22T04:22:38.130518Z\",\"iopub.status.idle\":\"2022-04-22T04:22:38.140392Z\",\"shell.execute_reply.started\":\"2022-04-22T04:22:38.130478Z\",\"shell.execute_reply\":\"2022-04-22T04:22:38.139737Z\"}}\nBATCH_SIZE=8\nIMAGE_SIZE=[512,512]\nSEED=42\nVERBOSE=1\nN_CLASSES=train_data['label_group'].nunique()\npath='Product_Matching_Resnet/train_images'\n\n##### For tf.dataset\nAUTO = tf.data.experimental.AUTOTUNE\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T04:22:38.142223Z\",\"iopub.execute_input\":\"2022-04-22T04:22:38.142527Z\",\"iopub.status.idle\":\"2022-04-22T04:22:38.151628Z\",\"shell.execute_reply.started\":\"2022-04-22T04:22:38.142492Z\",\"shell.execute_reply\":\"2022-04-22T04:22:38.150761Z\"}}\ndef decode_image(image_data):\n    image=tf.image.decode_jpeg(image_data,channels=3)\n    image=tf.image.resize(image,IMAGE_SIZE)\n    image=tf.cast(image,tf.float32)/255.0\n    return image\n\ndef read_image(image):\n    print('read')\n    image=tf.io.read_file(image)\n    image=decode_image(image)\n    return image\n\ndef get_dataset(image):\n    dataset = tf.data.Dataset.from_tensor_slices(image)\n    dataset = dataset.map(read_image, num_parallel_calls = AUTO)\n    dataset = dataset.batch(BATCH_SIZE)\n    dataset = dataset.prefetch(AUTO)\n    return dataset\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T04:22:38.152965Z\",\"iopub.execute_input\":\"2022-04-22T04:22:38.153328Z\",\"iopub.status.idle\":\"2022-04-22T04:22:38.171531Z\",\"shell.execute_reply.started\":\"2022-04-22T04:22:38.153291Z\",\"shell.execute_reply\":\"2022-04-22T04:22:38.170817Z\"}}\nclass ArcMarginProduct(tf.keras.layers.Layer):\n    def __init__(self,s=30,m=0.5,easy_margin=False,ls_eps=0.0,**kwargs):\n        super(ArcMarginProduct, self).__init__(**kwargs)\n        self.n_classes = n_classes\n        self.s=s\n        self.m=m\n        self.ls_eps=ls_eps\n        self.easy_margin=easy_margin\n        self.cos_m=tf.math.cos(m)\n        self.sin_m=tf.math.sin(m)\n        self.th=tf.math.cos(math.pi-m)\n        self.mm=tf.math.sin(math.pi-m)*m\n        \n    def get_config(self):\n        config=super().get_config().copy()\n        config.update({\n            'n_classes': self.n_classes,\n            's':self.s,\n            'm':self.m,\n            'ls_eps':self.ls_eps,\n            'easy_margin':self.easy_margin\n        })\n        return config\n    \n    def build(self,input_shape):\n        super(ArcMarginProduct,self).build(input_shape[0])\n        self.w=self.add_weight(\n            name='W',\n            shape=(int(input_shape[0][-1]),self.n_classes),\n            intializer='glorat_uniform',\n            dtype='float32',\n            trainable=True,\n            regularizer=None\n        )\n        \n    def call(self, inputs):\n        X,y=inputs\n        y=tf.cast(y,dtype=tf.int32)\n        cosine=tf.matmul(\n            tf.math.l2_normalize(X,axis=1),\n            tf.math.l2_normalize(self.w,axis=0)\n        )\n        sine=tf.math.sqrt(1.0-tf.math.pow(cosine,2))\n        phi=cosine*self.cos_m-sine*self.sine_m\n        if self.easy_margin:\n            phi = tf.where(cosine > 0, phi, cosine)\n        else:\n            phi=tf.where(cosine>self.th,phi,cosine-self.mm)\n        one_hot = tf.cast(\n            tf.one_hot(y,depth=self.n_classes),\n            dtype=cosine.dtype\n        )\n        if self.ls_eps > 0:\n            one_hot = (1-self.ls_eps)*one_hot + self.ls_eps/self.n_classes\n            \n        output=(one_hot*phi)+((1.0-one_hot)*cosine)\n        output*=s\n    \nprint(\"Ran till ArchMargin\")\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T04:22:38.173439Z\",\"iopub.execute_input\":\"2022-04-22T04:22:38.173868Z\",\"iopub.status.idle\":\"2022-04-22T04:22:38.184451Z\",\"shell.execute_reply.started\":\"2022-04-22T04:22:38.173834Z\",\"shell.execute_reply\":\"2022-04-22T04:22:38.183832Z\"}}\ndef image_embeddings(imgpath):\n    print(\"Using Regular Tensorflow Model For Predictions \\n\")\n    embeds=[]\n    \n    start_time = time.time()\n    margin=ArcMarginProduct(\n        n_classes=N_CLASSES,\n        s=15,\n        m=0.0001,\n        name='head/arc_margin',\n        dtype='float_32'\n    )\n    \n    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE,3),name='inp1')\n    label=tf.keras.layers.Input(shape=(),name='inp2')\n    x= EfficientNetB3(weights = None, include_top = False)(inp)\n    x=tf.keras.layers.GlobalAveragePooling2D()(x)\n    x=margin([x,label])\n\n    output=tf.keras.softmax(dtype='float32')(x)\n    model=tf.keras.Model(inputs=[inp,label],outputs=[output])\n    \n    #loading saved weights\n    model.load_weights('EfficientNet_b3_15_0.0001_512_42_final.h5')\n    model=tf.keras.models.Model(inputs=model.input[0],outputs=model.layers[-4].output)\n    \n    chunk = 5000\n    iterator = np.arange(np.ceil(len(train_data)/chunk))\n    for j in terator:\n        a=int(j*chunk)\n        b=int((j+1)*chunk)\n        image_dataset=get_dataset(imagepath[a:b])\n        image_embeddings=model.predict(image_dataset)\n        embeds.append(image_embeddings)\n        \n    del model\n    image_embeddings = np.concatenate(embeds)\n    print(f'Our image embeddings shape is {image_embeddings.shape}')\n    del embeds\n    return image_embeddings\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T06:48:36.144022Z\",\"iopub.execute_input\":\"2022-04-22T06:48:36.144734Z\",\"iopub.status.idle\":\"2022-04-22T06:48:36.150023Z\",\"shell.execute_reply.started\":\"2022-04-22T06:48:36.144694Z\",\"shell.execute_reply\":\"2022-04-22T06:48:36.149142Z\"}}\ndef bert_encode(texts,tokenizer,max_len=512):\n    all_tokens=[]\n    all_masks=[]\n    all_segments=[]\n    \n    for txt in texts:\n        print(txt)\n        text=tokenizer.tokenize(txt)\n        print(max_len)\n        text=text[:max_len-2]\n        print(text)\n        input_seq=[\"[CLS]\"]+text+[\"[SEP]\"]\n        pad_len=max_len-len(input_seq)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T06:44:31.041024Z\",\"iopub.execute_input\":\"2022-04-22T06:44:31.041321Z\",\"iopub.status.idle\":\"2022-04-22T06:44:31.046459Z\",\"shell.execute_reply.started\":\"2022-04-22T06:44:31.041291Z\",\"shell.execute_reply\":\"2022-04-22T06:44:31.045756Z\"}}\ndef get_text_embeddings(train_data,max_len=70):\n    embeds=[]\n    module_url = '../input/bert-en-uncased-l24-h1024-a16-1'\n    bert_layer = hub.KerasLayer(module_url,trainable=True)\n    #vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()\n    #do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()\n    #print(do_lower_case)\n    tokenizer=tokenization.FullTokenizer(vocab_file='../input/tokenize',do_lower_case=True)\n    print(tokenizer)\n    text=bert_encode(train_data['title'].values,tokenizer,max_len=max_len)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2022-04-22T06:48:38.673438Z\",\"iopub.execute_input\":\"2022-04-22T06:48:38.674030Z\",\"iopub.status.idle\":\"2022-04-22T06:48:38.679154Z\",\"shell.execute_reply.started\":\"2022-04-22T06:48:38.673988Z\",\"shell.execute_reply\":\"2022-04-22T06:48:38.678197Z\"}}\n#get_text_embeddings(train_data[:5])\n\n# %% [raw]\n# ","metadata":{"_uuid":"2c083e65-9d21-4aea-8f8d-905def549e53","_cell_guid":"b47c24bc-6947-481b-8133-b4d6b275e5f1","collapsed":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-04-22T06:50:05.642711Z","iopub.execute_input":"2022-04-22T06:50:05.642976Z","iopub.status.idle":"2022-04-22T06:50:14.629349Z","shell.execute_reply.started":"2022-04-22T06:50:05.642946Z","shell.execute_reply":"2022-04-22T06:50:14.628408Z"},"trusted":true},"execution_count":66,"outputs":[]}]}