from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_data[['title']], train_data['label_group'], shuffle = True, stratify = train_data['label_group'], random_state = 42, test_size = 0.33)

with tf.device('/GPU:0'):
    
    def bert_encode(texts,tokenizer,max_len=512):
        all_tokens=[]
        all_masks=[]
        all_segments=[]
        
        for txt in texts:
            text=tokenizer.tokenize(txt)
            text=text[:max_len-2]
            input_seq=["[CLS]"]+text+["[SEP]"]
            pad_len=max_len-len(input_seq)
            
            tokens = tokenizer.convert_tokens_to_ids(input_seq)
            tokens += [0]*pad_len
            
            pad_masks = [1]*len(input_seq)+[0]*pad_len
            segment_ids=[0]*max_len
            
            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)
        
        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    def build_bert_model(bert_layer,max_len=512):
        
        margin=ArcMarginProduct(
            n_classes=N_CLASSES,
            s=30,
            m=0.5,
            name='head/arc_margin',
            dtype='float32'
        )
    
        input_word_ids=tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name='input_word_ids')
        input_mask=tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name='input_mask')
        segment_ids=tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name='segment_ids')
        label = tf.keras.layers.Input(shape=(),name='label')
    
        _,sequence_output=bert_layer([input_word_ids,input_mask,segment_ids])
        clf_output=sequence_output[:,0,:]
        print(clf_output)
        x=margin([clf_output,label])
    
        output=tf.keras.layers.Softmax(dtype='float32')(x)
        model = tf.keras.models.Model(inputs = [input_word_ids, input_mask, segment_ids, label], outputs = [output])
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.00001),
                      loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
                      metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
        return model
    
    def bert_load_and_train(X_train,y_train,X_val,y_val):
        embeds=[]
        module_url = '../input/bert-en-uncased-l24-h1024-a16-1'
        bert_layer = hub.KerasLayer(module_url,trainable=True)
        vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
        #print(do_lower_case)
        tokenizer=tokenization.FullTokenizer(vocab_file,do_lower_case)
        X_train=bert_encode(X_train['title'].values,tokenizer,max_len=50) 
        X_val=bert_encode(X_val['title'].values,tokenizer,max_len=50)
        y_train=y_train.values
        y_val=y_val.values
    
        print(len(X_train))
        X_train=(X_train[0],X_train[1],X_train[2],y_train)
        X_val=(X_val[0],X_val[1],X_val[2],y_train)
        print(len(X_train))
        bert_model=build_bert_model(bert_layer,max_len=50)
    
        checkpoint=tf.keras.callbacks.ModelCheckpoint(f'Bert_{42}.h5',
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='min')
    
        history=bert_model.fit(X_train,y_train,
                              validation_data=(X_val,y_val),
                              epochs=25,
                              callbacks=[checkpoint],
                              batch_size=32,
                              verbose=1)
    
    bert_load_and_train(X_train,y_train,X_val,y_val)