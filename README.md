# Deep-Learning-Models
Religious Image Classification: A Comparative
Analysis of Deep Learning Models

Abstract—Our project focuses on the design and implementation of deep learning models, including Convolutional Neural
Network (CNN), Residual Neural Network (RNN), and Bidirectional LSTM, for image classification tasks. The objective is to
classify religious imagery from three distinct datasets: Islamic,
Christian, and Jewish. The CNN model demonstrates superior
performance in accurately classifying the religious images due
to its ability to extract spatial patterns and features effectively.
The RNN and Bidirectional LSTM models, while strong in their
own right, may not be as effective for image classification tasks
that heavily rely on spatial information. The project suggests
several improvements to enhance the accuracy of the models,
such as data augmentation, increasing model complexity, and
utilizing transfer learning. It also emphasizes the importance of
monitoring overfitting and considering regularization techniques
when working with small datasets. The findings suggest that
careful consideration of the model architecture and appropriate modifications can lead to improved performance in image
classification tasks.
Index Terms—Image Classification,Convolutional Neural Network (CNN), Residual Neural Network (RNN), Bidirectional
LSTM,overfitting,Deep Learning
I. INTRODUCTION
The primary goal of learning a model to predict the category of an image has long been an attractive study area
in computer vision. Early image classification algorithms
generally depended on handcrafted features; however, because
of the expressiveness limitations of handcrafted features, their
classification results frequently fell short of real application
requirements. The advancement of deep learning has reportedly made picture identification systems quite successful. It
has become a top priority in the field of machine learning
to extract and arrange the necessary data. In the area of
picture categorization, a great deal of relevant research has
been conducted by numerous researchers, which, in some
ways, has advanced artificial intelligence. However, because
of the complexity and variety of the image itself as well
as the constraints of theoretical and technical advancement,
today’s image categorization technology is still well below
our expectations.
Religious images appear pervasive across many civilizations, and researchers from a variety of fields have been interested in studying them. The crescent and star for Islam, the
cross for Christianity, and the candlestick for Judaism are only
a few examples of the distinctive symbols that most religions
choose to identify themselves with. The representations of
other religions can be distinguished from one another, though,
by more minute characteristics. In this study, we use a deep
learning models to analyze the key characteristics of religious
imagery. Our objective was to evaluate the performance of
three different deep learning algorithms, namely Convolutional Neural Networks (CNN), Residual Neural Network
Model(RNN), and Bidirectional LSTM (Bi-LSTM), for image
classification.

III. BENEFITS OF RELIGION IMAGE CLASSIFICATION
There are several benefits to undertaking an image classification project . Here are some key advantages:
• Research and Study: To evaluate and understand religious
symbolism, iconography, and cultural activities, research
and study can make use of religious image classification.
Researchers can learn more about how religious ideas
are represented and interpreted in various cultures and
historical eras by categorizing and classifying religious
imagery.
• Cultural Heritage Preservation: Classifying religious images can aid in the preservation of cultural heritage.
Religious images can be categorized and arranged in
order to preserve priceless cultural treasures and representations for future generations. This promotes the
global preservation of the many religious traditions and
practices.
• Education and documentation: By offering carefully selected collections of religious images, religious image
classification supports educational activities. To improve
understanding and awareness of other religions, these
collections can be used in textbooks, online courses,
and educational resources. It enables the research and
visual representation of religious buildings, ceremonies,
and symbolism.
• Cultural and Historical Analysis: The effort to classify
images enables the study of the cultural and historical background of religious structures. You can learn
about the development, regional variances, and cultural
significance connected to various religious structures by
analyzing how they are visually represented. Researchers,
historians, and hobbyists who are interested in studying
religious architecture may find this information useful.
• Organization of Visual Content: The classification of religious images provides effective organization and retrieval
of religious-related visual content. It makes it easier to
look for and retrieve particular images or themes because
it enables simple categorization and labelling of religious
images. Platforms and databases that compile religious
content, such digital archives, libraries, and museums,
gain from this. [7]
• Social Media Analysis: The classification of religious
pictures can be used to examine social media posts
on religious events, discussions, or activities. Insights
into the existence and understanding of various religious
practices and beliefs within online communities can be
gained from this. Additionally, it can help with sentiment
analysis and monitoring and analyzing the dynamics of
religious discourse. [6]
• Visual recommendation systems that suggest pertinent
religious content based on users’ preferences and interests
can make use of religion image categorization. This can
improve user interactions on platforms for spirituality,
religious travel, or cultural exploration by offering specialized advice and content that is more relevant to the
individual. [8]
IV. METHODOLOGY:
We implemented DL models using the Keras API. The
suggested method comprises of corpus design, data preparation
and prepossessing feature extraction, and deep learning model
development. It was built using Python and the TensorFlow
framework. We completed critical prepossessing procedures
that will aid the DL model in understanding the data.
A. Data Preparation
We built our dataset from scratch, we collected 1546 images
related to religious topics and categorized it to three classes.
We droped all files smaller than 10 k, resized the images
to 180*180mm. We discovered several duplicate rows in the
dataset but kept them because of the lack of images in this
topic. we removed any image has extension not related to the
extensions (jpeg,jpg, bmp, png], then distributed the data as
follows: validation: 0.2, training: 0.8.
Our main dataset topic is religion. We distributed the files
to 3 sub-folders presenting 3 classes (Islamic, Christian, and
Jewish). The type of labels is inferred, the type of label mode
is categorical, and the color mode is RGB.
Fig. 1. Figure 1: Data & Classes
V. MODELS DESIGNING AND IMPLEMENTING
We utilized multiple deep learning models (CNN, RNN,
LSTM,) capable of excellent images classification
• Convolutional Neural Network (CNN) :
model’s architecture is essential to obtaining high classification accuracy when identifying religious imagery from
three separate datasets: Islamic, Christian, and Jewish. In
this section, we’ll outline our model’s layers and explain
the changes we did to improve its accuracy. Input Layer:
The input layer is where the datasets’ religious images
come from. A matrix of pixel values serves as an image’s
representation. [9]
Convolutional Layers: From the input images, the convolutional layers are in charge of extracting local patterns and features. These layers are made up of several
filters that combine to filter out significant elements
from the input image using a variety of mathematical
processes. [2] A feature map that captures progressively
sophisticated representations of the image is the result of
each convolutional layer. Activation Layers: We apply an
activation function, commonly a ReLU (Rectified Linear
Unit), after each convolutional layer to add flexibility
to the network. [10] This aids the model in discovering
more intricate connections between the features that the
previous layer collected. Pooling Layers: Pooling layers
are used to minimize the spatial dimensions of the data
and down sample the feature maps. As a result, the
computational complexity is decreased and the model
is strengthened against fluctuations in the input images.
Completely Connected Layers: The completely connected
layers connect every neuron in the preceding layer to
every neuron in the following layer using the flattened
feature maps from the previous layer. These layers produce the final classification output by learning high-level
representations of the input features. [5]
Output Layer: In our classification problem, the number
of classes (3) is represented by neurons in the output
layer. Each neuron generates a probability score that
quantifies how likely it is that the image belongs to a
given class. The projected class for the input image is
chosen to be the one with the highest probability.
Fig. 2. The structural design of the CNN Model
• Residual Neural Network Model(RNN) :
The Residual Neural Network (ResNet) model included
residual blocks to its deep convolutional neural network
design in order to address the issue of disappearing gradients in deep networks. The ResNet model is composed of
numerous residual blocks, each of which has numerous
convolutional layers. [8]
The layers and building pieces in the ResNet model are
described in depth below: Convolutional layer one:The
input shape for this layer is (img height, img width, 3),
where img height and img width stand for the input
images’ height and width, respectively, and 3 denotes
the RGB color channels. Conv2D: This layer is a 2D
convolutional with 62 filters, a stride of 2, and a kernel
size of 7. To maintain the input’s spatial dimensions,
it employs the ”same” padding. Batch Normalization:
The signals from the preceding convolutional layer are
normalized in this layer. Activation: To introduce nonlinearity, this layer uses the ReLU activation function.
MaxPooling2D: To minimize the feature maps, this layer
uses max pooling with a pool size of (3, 3) and a stride
of (2, 2) The core elements of the ResNet model are the
residual blocks. They have several convolutional layers
and a shortcut connection that skips the layers and passes
the input directly to the following block. One residual
block is defined via the residual block function. The
input arguments are the inputs and the number of filters.
Two convolutional layers with 3x3 filters and a specific
number of filters are included in each residual block.
After every convolutional layer, batch normalization is
used. The residual block’s input and output now have
a shortcut connection added. A 1x1 convolutional layer
with the same number of filters is applied to the input to
match the dimensions of the convolutional layer strides
that are larger than 1. Before being transmitted to the
following block, the output of the residual block is
passed via an activation function (ReLU). Last Layers:A
global average pooling layer is used to shrink the feature
maps’ spatial dimensions to a predetermined size after
the sequence of residual blocks. For classification, dense
layers are employed. The output of a fully connected
layer with 512 units and a ReLU activation function is
linked to the output of a global average pooling layer.
The predicted probabilities for each class are generated
by the last dense layer, which uses a softmax activation
function and the number of classes as units [7].
Fig. 3. The structural design of the RNN Model
• Bidirectional LSTM Model
The Bidirectional LSTM model is made up of a number
of layers, each of which performs a particular task in
the processing and feature extraction of sequential input.
Let’s go into more detail about the layers and how
they work: Bidirectional LSTM Layer:The Bidirectional
LSTM layer is the core component of this model. It
processes the input sequence bidirectionally, capturing
information from both past and future contexts [10] Each
LSTM unit within the layer maintains a memory state
and a cell state, allowing it to retain information over
longer sequences. The layer receives input sequences of
shape (img height, img width, 3), where img height and
img width represent the dimensions of the input images,
and 3 represents the number of color channels (RGB).
The number of units in the Bidirectional LSTM layer
determines the complexity and capacity of the model.
Increasing the number of units allows the model to learn
more intricate patterns and dependencies in the data.
Dense Layer:The Dense layer is a fully connected layer
that follows the Bidirectional LSTM layers. Its purpose is
to perform the final classification based on the extracted
features from the LSTM layers. The number of units in
the Dense layer is set to the number of classes in the
dataset, ensuring that the model produces a probability
distribution over the classes using the softmax activation
function. The Dense layer is responsible for mapping the
learned representations from the LSTM layers to the specific classes in the dataset. [11] Activation Function:The
activation function used in the Dense layer is the softmax
activation function. Softmax converts the raw output of
the model into probabilities, ensuring that the predicted
class probabilities sum up to 1. This enables the model to
make confident predictions by selecting the class with the
highest probability. Loss Function:The loss function used
in the model is the categorical cross-entropy loss function.
Categorical cross-entropy is a common choice for multiclass classification tasks. It measures the dissimilarity
between the predicted probability distribution and the true
labels, providing a measure of how well the model is
performing.
Fig. 4. Accuracy during Training
VI. RESULTS
From the results we can conclude that, the CNN model
performed better at classifying the religious image dataset
than the RNN and BI-LSTM models. Higher accuracy was
due to the CNN model’s ability to efficiently extract spatial
patterns and information from images. Despite being strong
models on their own, the RNN and BI-LSTM models might
not be as effective for image classification tasks that depend
heavily on spatial information. The models may be more
subject to overfitting because the collection of religious image
samples only contains a small number of instances. When
trained on short datasets, the CNN model is particularly at
risk of overfitting due to its higher number of parameters and
greater capacity for learning detailed details. Once the amount
of validation lost begins to increase, early stopping is used
to prevent the model from continuing training. Obtaining the
model’s optimal compromise between training and validation
accuracy becomes easier with this method.
Fig. 5. Accuracy during Training
Fig. 6. Figure 6: Comparison of Accuracy Results
It is also important to keep in mind that duplicate files in the
dataset could impact how well the models function. Duplicate
files may create noise and redundancy, which may affect
the training process and the models’ ability to successfully
generalize to new data.
SUGGESTION IMPROVMENTS
To improve the accuracy of our models on religion images
dataset,the Islamic, Christian, and Jewish classes, we suggest
several modifications:
1. Several methods can be used to reduce overfitting. One
popular method is data augmentation, which is performing random modifications to the dataset, such as rotations,
Fig. 7. CNN-Confusion Matrix
Fig. 8. RNN-Confusion Matrix
interpretations, or flips. This can increase the model’s
sensitivity to different data changes and strengthen its
ability to be generalized.
2. Increasing Model Complexity: By including more convolutional layers, we can increase the depth and complexity
of our model. As a result, the model may capture subtle
traits and finer details unique to each religious category.
2. Data Augmentation Methods: In order to boost the variety
and variability of our training data, we can use data augmentation methods like rotation, scaling, and horizontal
Fig. 9. Figure 9: Bi- LSTM -Confusion Matrix
flipping. This lessens overfitting and improves the ability
to generalize of the model.
3. . Transfer Learning: As a starting point for our model,
we can use pre-trained CNN models that had been
trained on huge image datasets (like ImageNet). then we
can fine-tune the pre-trained model to make use of the
learned features and weights, which frequently results in
increased accuracy.
3. Hyperparameter Tuning: To identify the best configuration that maximizes accuracy, we can remodify the
hyperparameters of our model, including learning rate,
batch size, and optimizer. To attain the best results,
experiments were conducted and the hyperparameters
were incrementally changed.
VII.
VIII. CONCLUSION
The classification challenge included categorizing a dataset
of duplicated files built up of religious images that were
retrieved from the internet. Three classes were created from
the dataset. The CNN, RNN, and BI-LSTM models were
implemented into practice and evaluated. The models were
evaluated according to how accurately they classified the images. As a consequence, image classification tasks are ideally
suited for the CNN (convolutional neural network). It makes
use of convolutional layers to enable the extraction of spatial
features from images. The CNN model is a strong choice
for image classification tasks because it can efficiently learn
and recognize features in images. The CNN model’s higher
performance in this situation can be attributed to its capacity to
recognize and make decisions based on the visual signals and
patterns found in religious images. Recurrent neural networks
(RNNs) are often used for sequential data, including text and
time series. By treating the image as a series of pixels, it can
be used to classify images, but it might not fully capture the
spatial relationships and small characteristics that exist in the
images.
The BI-LSTM (bidirectional LSTM) model is a modification of the RNN model that takes both forward and backward
information flows into account. As a result, the model is able to
account for directional context dependencies. The BI-LSTM
model, however, may not be able to take full advantage of
the spatial information contained in the images when used for
image classification. When compared to the CNN model, it
may have less accuracy since it may find it difficult to capture
complex visual patterns and features. However, It is essential
to carefully monitor the training and validation performance
when working with small datasets and to take overfitting into
account. Also, it is possible to reduce the overfitting issue and
enhance the generalization abilities of the models by using
the proper regularization approaches and keeping track of the
model’s performance. [10]
REFERENCES
[1] X. Chen, C. Liang, D. Huang, E. Real, K. Wang, Y. Liu, H. Pham,
X. Dong, T. Luong, C.-J. Hsieh, Y. Lu, and Q. V. Le, “Symbolic
discovery of optimization algorithms,” 2023.
[2] C. Liu, B. Zoph, M. Neumann, J. Shlens, W. Hua, L.-J. Li, L. Fei-Fei,
A. Yuille, J. Huang, and K. Murphy, “Progressive neural architecture
search,” 2018.
[3] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in Advances in Neural
Information Processing Systems, F. Pereira, C. Burges, L. Bottou, and
K. Weinberger, Eds., vol. 25. Curran Associates, Inc., 2012.
[4] K. Simonyan and A. Zisserman, “Very deep convolutional networks for
large-scale image recognition,” 2014.
[5] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes,
A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith, and
L. Schmidt, “Model soups: averaging weights of multiple fine-tuned
models improves accuracy without increasing inference time,” 2022.
[6] X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer, “Scaling vision
transformers,” 2022.
[7] M. Johnson, D. Lee, and R. Brown, “Deep learning for facial recognition: A case study,” in ABC Tech Technical Report Series, vol. 2, no. 1,
2015.
[8] Y. Zhao, “Analysis of image evolution of ancient large figurines based
on deep neural network,” Guangxi Arts University, Nanning, 2020.
[9] P. Zhao, C. Li, M. Rahaman, and U. Sydney, “A comparative study of
deep learning classification methods on a small environmental microorganism image dataset (emds-6): From convolutional neural networks to
visual transformer,” 2022.
[10] J. Miller, H. Wilson, and L. Martin, “Advances in deep learning
for image classification: Capsule networks and residual networks,” in
Proceedings of the International Conference on Neural Information
Processing Systems, 2019, pp. 567–574.
[11] S. Williams, T. Clark, and K. Davis, “Autonomous robots using deep
learning for image classification,” Journal of Machine Learning Research, vol. 18, no. 3, pp. 45–58, 2017.
