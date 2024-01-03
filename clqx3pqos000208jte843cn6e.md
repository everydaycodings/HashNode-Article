---
title: "Anomaly Detection Algorithm | Part 2 | Episode 15"
seoTitle: "Understanding and Explaining Anomaly Detection"
seoDescription: "Learn the secrets of Anomaly Detection Algorithms Master the latest techniques for identifying outliers and maintaining data integrity."
datePublished: Wed Jan 03 2024 01:30:10 GMT+0000 (Coordinated Universal Time)
cuid: clqx3pqos000208jte843cn6e
slug: anomaly-detection-algorithm-part-2-episode-15
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1704208392334/fb8c5cfb-046c-40e4-a53c-ed875456d661.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1704208399036/98ef49b6-9729-4d6a-8f18-432b7a5fb7a2.png
tags: algorithms, python, data-science, machine-learning, coding, unsupervised-learning, anomaly-detection

---

In the [**previous episode**](https://neuralrealm.hashnode.dev/anomaly-detection-algorithm-part-1-episode-14), we learned about Anomaly Detection and the mathematical workings of the algorithm. In this episode, we will be focusing on how to develop and evaluate an Anomaly Algorithm, the differences between Anomaly Detection and Unsupervised Learning, and how to choose features for Anomaly Detection.

### Developing and Evaluating an Anomaly Detection Algorithm

If you're developing an anomaly detection system, I can offer you some practical tips. One of the key ideas is to have a way to evaluate the system, even as you're developing it. This allows you to make decisions and changes to the system more quickly and effectively.

When developing a learning algorithm, such as choosing different features or trying different values of parameters like epsilon, it's much easier to make decisions if you can evaluate the algorithm. This is sometimes referred to as real number evaluation. Essentially, you need a way to quickly change the algorithm, such as changing a feature or parameter, and then compute a number that tells you if the algorithm improved or worsened. This is how it's often done in anomaly detection.

While we've mainly been talking about unlabeled data, I will assume that we have some labeled data, including a small number of previously observed anomalies. For example, if you've been making airplane engines for a few years, you may have seen a few anomalous engines. In this case, I'll associate a label y=1 to indicate an anomalous engine and y=0 for normal engines.

The training set for the anomaly detection algorithm will still be an unlabeled training set of x1 through xm, but we'll assume that all of these examples are normal and not anomalous, so y=0. In practice, a few anomalous examples may slip into the training set, but the algorithm will usually still perform well.

To evaluate your algorithm, create a cross-validation set, denoted x\_cv^1, y\_cv^1 through x\_cv^mcv, y\_cv^mcv. This is a similar notation as you've seen in the second course of this specialization. Have a test set of some examples where both the cross-validation and test sets hopefully include a few anomalous examples. In other words, the cross-validation and test sets will have a few examples of y=1, but also a lot of examples where y=0.

Again, in practice, the anomaly detection algorithm will work okay if some examples are anomalous but were accidentally labeled with y=0.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1704180186632/20dfc1f6-7f5f-451b-972f-75d2dc491957.png align="center")

In this example, we will use aircraft engine manufacturing data to illustrate how to apply the anomaly detection algorithm. Assuming that you have collected data from 10,000 normal engines and 20 anomalous engines over the years, we can split the dataset into a training set, a cross-validation set, and a test set. To be specific, we will put 6,000 normal engines in the training set, 2,000 normal engines and 10 anomalous engines in the cross-validation set, and the remaining 2,000 normal engines and 10 anomalous engines in the test set.

The algorithm will be trained on the training set by fitting Gaussian distributions to the normal engine data. Then, we can tune the parameter epsilon on the cross-validation set to ensure that the algorithm can detect the known anomalies while minimizing false positives. We can also add or subtract features to improve the algorithm's performance.

After tuning the algorithm, we can evaluate its performance on the test set to see how many anomalous engines it can detect, as well as how many normal engines it incorrectly flags as anomalous. It's worth noting that having a small number of anomalies to evaluate the algorithm is helpful, even though the training set has no labels.

However, when the number of flawed engines is extremely small, it might be better to skip the test set and only use the training and cross-validation sets. In this case, we can tune the algorithm on the cross-validation set and use it to evaluate the algorithm's performance. Nonetheless, there's a higher risk of overfitting to the cross-validation set, which might result in poor performance on real data.

In conclusion, splitting the dataset into a training set, a cross-validation set, and a test set is crucial for evaluating the anomaly detection algorithm's performance. We can use the cross-validation set to tune the algorithm and the test set to evaluate its performance.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1704182655547/eb4ef776-8127-4a5d-9865-d51e462ecfed.png align="center")

Remember the following instructions for anomaly detection. First, fit the model p of x on the training set, which consists of 6,000 examples of good engines. Then, compute p of x on any cross-validation or test example x, and predict y equals 1 if p of x is anomalous and less than Epsilon. If p of x is greater than or equal to Epsilon, predict y equals 0. Evaluate the accuracy of the algorithm's predictions on the cross-validation or test set by comparing the predicted labels y with the actual labels in the cross-validation or test sets.

In the third week of the second course, we learned about handling highly skewed data distributions where the number of positive examples (y equals 1) is much smaller than the number of negative examples (y equals 0). This is also the case for many anomaly detection applications where the number of anomalies in the cross-validation set is much smaller. To handle this, you can compute metrics such as true positive, false positive, false negative, and true negative rates, as well as precision, recall, and F\_1 score. These metrics can be more effective than classification accuracy when the data distribution is highly skewed.

You can apply these evaluation metrics to determine how well your learning algorithm is performing at identifying the small number of anomalies or positive examples amidst the much larger set of negative examples of normal plane engines. Use the cross-validation set to determine how many anomalies are being identified, as well as how many normal engines are being incorrectly flagged as anomalies. This will help you choose a good value for the parameter Epsilon. If you didn't watch the optional videos on handling highly skewed data distributions, don't worry about it! The intuition is simply to use the cross-validation set to evaluate the algorithm's performance and choose the best value for Epsilon.

### Anomaly Detection VS Supervised Learning

*Now, this does raise the question, if you have a few labeled examples, since you'll still be using an unsupervised learning algorithm, why not take those labeled examples and use a supervised learning algorithm instead?*

If you have a small number of positive examples (0-20) and a relatively large number of negative examples, then anomaly detection is usually the better choice. In this case, the parameters for p of x are learned only from the negative examples. The positive examples are used only in the cross-validation and test sets for parameter tuning and evaluation. However, if you have a larger number of positive and negative examples, then supervised learning might be more appropriate.

The main difference between anomaly detection and supervised learning is that if you think there are many different types of anomalies or positive examples, then anomaly detection might be more appropriate. Anomaly detection looks at the normal (y = 0) examples and tries to model what they look like. Anything that deviates a lot from normal is flagged as an anomaly, including if there's a brand new way for an aircraft engine to fail that had never been seen before in your data set. In contrast, supervised learning tries to learn from the positive examples what the positive examples look like. If there are many different ways for an aircraft engine to go wrong, and if tomorrow there may be a brand new way for an aircraft engine to have something wrong with it, then your 20 positive examples may not cover all of the ways an aircraft engine could go wrong. This makes it hard for any algorithm to learn from the small set of positive examples what the anomalies look like, and future anomalies may look nothing like any of the anomalous examples seen so far.

In supervised learning, we assume that the future positive examples are likely to be similar to the ones in the training set. This type of learning works well for finding previously observed forms of fraud, classifying email spam, predicting the weather, and identifying known and previously seen defects.

On the other hand, anomaly detection is useful when trying to detect brand new types of fraud that have never been seen before or when trying to find new previously unseen defects, such as if there are brand new ways for an aircraft engine to fail in the future that you still want to detect. In anomaly detection, the algorithm tries to find brand new positive examples that may be unlike anything seen before.

However, the choice of features is crucial when building anomaly detection systems. Many security-related applications use anomaly detection because hackers are often finding brand-new ways to hack into systems.

### Choosing What Features to Use

The text discusses the process of anomaly detection, which involves transforming data to make it more Gaussian. The aim is to ensure that normal examples have a high probability (greater than or equal to epsilon), while anomalous examples have a low probability (less than epsilon). However, sometimes the algorithm may fail to detect anomalies if the probability for both normal and anomalous examples is comparable. In such cases, one can identify new features that would help distinguish anomalous examples from normal examples and improve the algorithm's performance.

The process involves training a model, identifying anomalies that the algorithm fails to detect, and then creating new features to successfully identify those anomalies. For instance, let's consider building an anomaly detection system to monitor computers in a data center. The goal is to identify computers that might be behaving unusually, indicating a hardware failure or security breach.

To accomplish this, we need to select features that could take on high or low values in the event of an anomaly. We might start by choosing features such as memory use, disk accesses per second, CPU load, and network traffic. However, if our algorithm fails to detect some anomalies, we can create new features by combining old ones.

For example, if a computer has a high CPU load but low network traffic, we could create a new feature that calculates the ratio of CPU load to network traffic. We can also experiment with other features, such as the square of the CPU load divided by the network traffic volume. Our goal is to find the right combination of features that make the probability of X large for normal examples but small for anomalies in our cross-validation set.

The text also suggests that selecting the right features is crucial when creating an algorithm for anomaly detection. In supervised learning, the algorithm can figure out how to best use the features provided, even if they are not perfect. However, in anomaly detection, which relies on unlabeled data, it is harder for the algorithm to determine which features to use. Therefore, carefully selecting features is even more important for anomaly detection than for supervised learning.

One way to improve the anomaly detection algorithm is to ensure that the features are roughly Gaussian. If a feature's distribution is not Gaussian, it can be transformed to make it more Gaussian. For instance, taking the logarithm of the feature or raising it to an exponential power can help make it more likely that the algorithm will fit the data well.

When building an anomaly detection system, it is recommended to plot a histogram of each feature to check for non-Gaussian distributions. If there are any such distributions, various transformations can be attempted until a Gaussian-like distribution is achieved. The parameters can be adjusted quickly, and the histogram can be plotted to try to get a more Gaussian-like distribution. There are also methods in the machine learning literature to automatically measure how close these distributions are to Gaussian.

<mark>To gain a better understanding of how this process works, I recommend referring to this Notebook</mark>: [**Anomaly Detection Notebook**](https://github.com/everydaycodings/Notebooks/blob/master/extra/Machine%20Learning/Cluster%20Algorithm/C3W1A2/Anomaly_Detection.ipynb)

---

## **🎙️ Message for Next Episode**

So, you've just witnessed the process of building an anomaly detection system. In the next episode, we will take a dataset and apply anomaly detection to it, to understand the practical applications of the technique and how it can be applied to real data.

---

### **Resources For Further Research**

1. **Coursera** - [**Machine Learning Specialization**](https://www.coursera.org/learn/machine-learning/home/welcome)
    
    * This Article is heavily inspired by this Course so I will also recommend you to check this course out, there is an option to watch the `course for free` or you can buy the course to gain a `Specialization Certificate`.
        
    * Optional labs provided in this article are from this course itself.
        
2. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron:**
    
    * A practical guide covering various machine learning concepts and implementations.
        

---

## **By the way…**

#### Call to action

*Hi, Everydaycodings— I’m building a newsletter that covers deep topics in the space of engineering. If that sounds interesting,* [***subscribe***](https://neuralrealm.hashnode.dev/newsletter) *and don’t miss anything. If you have some thoughts you’d like to share or a topic suggestion, reach out to me via* [***LinkedIn***](https://www.linkedin.com/in/kumar-saksham1891/) *or* [***X(Twitter)***](https://twitter.com/everydaycodings).

#### References

*And if you’re interested in diving deeper into these concepts, here are some great starting points:*

* [**Kaggle Stories**](https://neuralrealm.hashnode.dev/series/kaggle-stories) *\-* Each episode of Kaggle Stories takes you on a journey behind the scenes of a Kaggle notebook project, breaking down tech stuff into simple stories.
    
* [**Machine Learning**](https://neuralrealm.hashnode.dev/series/machine-learning) *\-* This series covers ML fundamentals & techniques to apply ML to solve real-world problems using Python & real datasets while highlighting best practices & limits.