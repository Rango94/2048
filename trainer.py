import tensorflow as tf
import numpy as np
import random as rd
import Map
import math

def generate_data(y,pro):
    size=1000
    maps=[]
    for i in range(size):
        maps.append(Map.map())

    while len(maps)>size/100:
        batch_x=[]
        for i in maps:
            batch_x.append(i.get_vector())
        y__=[i.tolist().index(np.max(i)) for i in sess.run(y, feed_dict={x:np.array(batch_x)})]
        for idx,i in enumerate(y__):
            if rd.random()<pro:
                y__[idx]=rd.randint(0,3)
        idx=0
        while idx <len(y__):
            move=maps[idx].move(y__[idx])
            if move:
                maps.pop(idx)
                y__.pop(idx)
                if len(maps)<size/100:
                    break
            else:
                idx+=1
    out=[]
    max_=0
    for map in maps:
        if map.get_max() >max_:
            max_=map.get_max()
    for map in maps:
        if map.get_max()==max_:
            out.append(map.get_history())
    return out,max_

def dropout(batchs):
    for idx in range(len(batchs)):
        v,l=batchs[idx]
        tmp=np.zeros(4)
        for kk in l:
            tmp+=kk
        for k,i in enumerate(tmp):
            if i==0:
                tmp[k]=999999
        id=tmp.tolist().index(np.min(tmp))
        idx_=0
        while idx_<len(v):
            if l[idx_].tolist().index(max(l[idx_].tolist()))!=id:
                # print(l[idx_],id)
                v=np.delete(v,idx_,0)
                l = np.delete(l, idx_, 0)
            else:
                idx_+=1
        batchs[idx]=[v,l]
    return batchs


w1=tf.get_variable('w1',[16,16],initializer=tf.random_normal_initializer())
w2=tf.get_variable('w2',[16,32],initializer=tf.random_normal_initializer())
w3=tf.get_variable('w3',[32,4],initializer=tf.random_normal_initializer())
b1=tf.get_variable('b1',[16],initializer=tf.random_normal_initializer())
b2=tf.get_variable('b2',[32],initializer=tf.random_normal_initializer())
b3=tf.get_variable('b3',[4],initializer=tf.random_normal_initializer())

x=tf.placeholder(tf.float32,[None,16],name='x-input')
y_=tf.placeholder(tf.float32,[None,4],name='y-input')

a=tf.nn.relu(tf.matmul(x,w1)+b1)
b = tf.nn.relu(tf.matmul(a, w2) + b2)
y=tf.matmul(b, w3) + b3

loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_)
train_step=tf.train.AdadeltaOptimizer(1).minimize(loss)
saver=tf.train.Saver()

# object_={}
# for i in range(1,12):
#     object_[i]=math.pow(2,i+1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pro=0.5
    last_max=0
    while True:
        batchs,max_=generate_data(y,pro)
        # batchs=dropout(batchs)
        for xs,ys in batchs:
            sess.run(train_step,feed_dict={x:xs,y_:ys})
            # if ob<=3:
            #     y__=sess.run(y,feed_dict={x:xs})
            #     for idx,i in enumerate(y__):
            #         if i.tolist().index(np.max(i))==ys[idx].tolist().index(np.max(ys[idx])):
            #             r+=1
            #         a+=1
        if max_>last_max :
            if last_max!=0:
                saver.save(sess, './model/model.ckpt')
                pro-=0.1
            last_max=max_
        try:
            print(batchs[len(batchs)-1], max_)
        except:
            print(batchs,max_)














