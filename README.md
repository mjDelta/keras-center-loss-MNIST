# keras-center-loss-MNIST
MNIST implementation of A Discriminative Feature Learning Approach for Deep Face Recognition.


### Purpose
There are two main implementation of center loss by keras online. One takes use of Embedding Layer, automatically computing the centers. Another takes use of self-defined layer, computing and updating the centers with self-defined equations strictly based on what the paper purposes in 'A Discriminative Feature Learning Approach for Deep Face Recognition'. So I implement both to compare the performance. Here are the details.

#### Joint Supervision Loss
![Alt Joint Supervision Loss](https://github.com/mjDelta/keras-center-loss-MNIST/blob/master/center_loss_imgs/joint_supervision.png)
#### Backwarding of Centers
![Alt Backwarding of Centers](https://github.com/mjDelta/keras-center-loss-MNIST/blob/master/center_loss_imgs/centers_backward.png)

## Embedding Results
```
input_=Input(shape=(1,))
centers=Embedding(10,2)(input_)
intra_loss=Lambda(lambda x:K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True))([out1,centers])
model_center_loss=Model([inputs,input_],[out2,intra_loss])
model_center_loss.compile(optimizer="sgd",
                          loss=["categorical_crossentropy",lambda y_true,y_pred:y_pred],
                          loss_weights=[1,lambda_c/2.],
                          metrics=["acc"])
```
*More details, please refer to the code in this repo.*</br>
<img src="https://github.com/mjDelta/keras-center-loss-MNIST/blob/master/center_loss_imgs/center_loss_embedding.png" width=400 height=350/>
<img src="https://github.com/mjDelta/keras-center-loss-MNIST/blob/master/center_loss_imgs/nocenter_loss.png" width=400 height=350 />

## Self-defined Results
```
class CenterLossLayer(Layer):
    def __init__(self,alpha=0.5,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
    def build(self,input_shape):
        self.centers=self.add_weight(name="centers",
                                    shape=(10,2),
                                    initializer="uniform",
                                    trainable=False)
        super().build(input_shape)
    def call(self,x,mask=None):
        #x[0]:N*2 x[1]:N*10 centers:10*2
        delta_centers=K.dot(K.transpose(x[1]),K.dot(x[1],self.centers)-x[0])
        centers_count=K.sum(K.transpose(x[1]),axis=-1,keepdims=True)+1
        delta_centers/=centers_count
        new_centers=self.centers-self.alpha*delta_centers
        self.add_update((self.centers,new_centers),x)
        
        self.result=x[0]-K.dot(x[1],self.centers)
        self.result=K.sum(self.result**2,axis=1,keepdims=True)
        return self.result
    def compute_output_shape(self,input_shape):
        return K.int_shape(self.result)
input_=Input((10,))
center_loss=CenterLossLayer()([out1,input_])

model_center_loss=Model([inputs,input_],[out2,center_loss])
model_center_loss.compile(optimizer="sgd",
                          loss=["categorical_crossentropy",lambda y_true,y_pred:y_pred],
                          loss_weights=[1,lambda_c/2.],
                          metrics=["acc"])
```
*More details, please refer to the code in this repo.*</br>
<img src="https://github.com/mjDelta/keras-center-loss-MNIST/blob/master/center_loss_imgs/center_loss_self_defined_layer.png" width=400 height=350/>
<img src="https://github.com/mjDelta/keras-center-loss-MNIST/blob/master/center_loss_imgs/nocenter_loss.png" width=400 height=350 />
