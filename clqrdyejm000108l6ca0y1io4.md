---
title: "Anomaly Detection Algorithm | Part 1 | Episode 14"
seoTitle: "Understanding and Explaining Anomaly Detection"
seoDescription: "Learn the secrets of Anomaly Detection Algorithms Master the latest techniques for identifying outliers and maintaining data integrity."
datePublished: Sat Dec 30 2023 01:30:13 GMT+0000 (Coordinated Universal Time)
cuid: clqrdyejm000108l6ca0y1io4
slug: anomaly-detection-algorithm-part-1-episode-14
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1703871714649/169f8a5d-d35b-4323-a8ad-dea2fc823f52.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1703871723748/cec4d4a7-0fb8-48ed-ab87-ce4408d64f9c.png
tags: code, algorithms, technology, data-science, machine-learning, data-analysis, unsupervised-learning

---

In the field of data science and machine learning, anomaly detection is a crucial aspect that helps identify irregularities or outliers within datasets. In this article, we explore the fundamentals of anomaly detection, highlighting the importance of detecting anomalies and understanding their various forms. We'll also discuss unsupervised learning and its relevance in the context of anomaly detection, before diving into the details of a specific algorithm featured in Episode 13.

Anomalies can take many forms, such as unusual financial transactions or abnormal medical data, and identifying them can provide valuable insights for decision-making in various industries. This article aims to simplify the core concepts, methodologies, and applications of anomaly detection while providing an in-depth analysis of a state-of-the-art algorithm.

### **Foundations of Anomaly Detection**

Anomaly detection serves as a critical component in data analysis, focusing on the identification of patterns or instances that deviate significantly from the expected norm within a dataset. As we embark on understanding the foundations of anomaly detection, several key concepts come to the forefront.

1. **Definition of Anomalies:**
    
    * At its core, anomalies represent data points or patterns that exhibit behavior significantly different from the majority of the dataset. These anomalies can take various forms, including outliers, novelties, or irregularities.
        
2. **Types of Anomalies:**
    
    * *Point Anomalies:* Singular instances that stand out as significantly different from the rest of the data.
        
    * *Contextual Anomalies:* Anomalies that depend on the context in which they occur, with their abnormality being context-specific.
        
    * *Collective Anomalies:* Groups of data instances that collectively exhibit abnormal behavior when analyzed together.
        
3. **Challenges in Anomaly Detection:**
    
    * ***Scarcity of Anomalies:*** True anomalies are often rare events, making it challenging to obtain sufficient labeled data for training robust models.
        
    * ***Imbalanced Datasets:*** The inherent rarity of anomalies can lead to imbalanced datasets, where the majority of instances are normal. This poses challenges in model training and evaluation.
        

It is crucial to strike a balance between sensitivity to anomalies and resilience against false positives for effective anomaly detection systems.

### **Applications of Anomaly Detection**

Anomaly detection is a valuable tool that can be used to identify unusual patterns or outliers within large data sets. This technology has many practical applications across a wide range of industries. By providing valuable insights into data, anomaly detection algorithms can help decision-makers make better decisions and ensure the reliability of systems. Let's take a closer look at some of the key applications of anomaly detection.

1. **Cybersecurity:**
    
    * Identifying unusual patterns in network traffic, login attempts, or system behaviors to detect potential security breaches and cyber threats.
        
2. **Finance:**
    
    * Monitoring financial transactions for anomalies that could indicate fraudulent activities, such as unauthorized transactions or unusual spending patterns.
        
3. **Healthcare:**
    
    * Detecting anomalies in patient data, vital signs, or medical images to identify potential health issues or irregularities in treatment outcomes.
        
4. **Manufacturing and Industry:**
    
    * Monitoring equipment and production processes to detect deviations from normal operating conditions, preventing equipment failures, and optimizing production efficiency.
        
5. **Energy Sector:**
    
    * Identifying anomalies in energy consumption patterns or equipment performance to improve the efficiency of energy production and distribution.
        
6. **Telecommunications:**
    
    * Detecting unusual patterns in call records or network performance that may indicate technical issues or fraudulent activities.
        
7. **E-commerce:**
    
    * Monitoring user behavior, such as purchasing patterns or website interactions, to detect anomalies that may signal fraudulent transactions or security breaches.
        
8. **Health Monitoring:**
    
    * Analyzing physiological data from wearables or medical devices to identify anomalies that could indicate health issues or emergencies.
        
9. **Environmental Monitoring:**
    
    * Detecting anomalies in environmental data, such as air quality or temperature, to identify unusual events or potential environmental hazards.
        
10. **Supply Chain Management:**
    
    * Monitoring inventory levels, logistics, and supply chain processes to identify anomalies that could lead to disruptions or inefficiencies.
        
11. **Fraud Detection in Banking:**
    
    * Identifying unusual patterns in account transactions, such as large withdrawals or international transactions, to detect and prevent fraudulent activities.
        

The following applications demonstrate how anomaly detection can be used in different domains to ensure the integrity, security, and efficiency of systems and processes. In the subsequent sections, we will discuss the challenges associated with anomaly detection and provide best practices for its implementation.

---

# Diving Deep in

### Anomaly Detection Example

Let's consider an example. Some of my acquaintances were working on using anomaly detection to identify potential issues with the aircraft engines that were being produced. When a company manufactures an aircraft engine, it is essential that it is dependable and performs well since an aircraft engine failure can have severe consequences. Therefore, some of my acquaintances were utilizing anomaly detection to determine if an aircraft engine seemed unusual or if there were any flaws with it after it was manufactured.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1703639760111/e26cd8bf-986d-425c-84b0-bcfe1c4cd465.png align="center")

The idea is to compute different features of an aircraft engine after it rolls off the assembly line. For instance, feature x1 measures the heat generated by the engine, while feature x2 measures the vibration intensity. However, to simplify the slide, let's focus on just the heat and vibration features.

Aircraft engine manufacturers usually produce good engines, so the easiest data to collect is from the m engines they've manufactured. By collecting data on how these m engines behave in terms of heat and vibration, we can determine if a brand new aircraft engine rolling off the assembly line is similar to the ones that have been manufactured before. This is the anomaly detection problem.

To detect an anomaly, we use an algorithm that plots the examples x1 through xm on a graph. Each data point corresponds to a specific engine with a specific amount of heat and vibration. If a new aircraft engine has a heat and vibration signature that is similar to the examples plotted on the graph, we assume it's probably okay. But if the new engine has a heat and vibration signature that is very different from the examples, we assume it's an anomaly and inspect it more carefully before installing it on an airplane.

The most common way to carry out anomaly detection is through a technique called density estimation.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1703642701129/cb1306fe-8588-4ff1-87de-b1eff3a98d58.png align="center")

When given a set of training examples, the first step is to build a model for the probability of x. This means that the learning algorithm will try to figure out which values of the features x1 and x2 are more likely to occur and which are less likely. For instance, in this example, it is very likely to see examples in the ellipse in the middle, so this region would have a high probability. On the other hand, things outside this ellipse have a lower probability. The details of how to determine which regions have higher or lower probability will be explained in the following section.

Once the model for p of x has been built, we can use it to compute the probability of a new test example Xtest. If this probability is small, meaning less than a predefined threshold epsilon, we will flag it as an anomaly. For example, if Xtest is very unlikely to occur, such as a value all the way down below, we will flag it as an anomaly. Conversely, if p of Xtest is greater than or equal to epsilon, we will say that it looks okay, and it's not an anomaly.

In conclusion, the model p of x helps us identify anomalies by computing the probability of new test examples. If these examples are unlikely to occur, we will flag them as anomalies. However, if they are likely to occur, we will say that it's not an anomaly.

### Gaussian (Normal) Distribution

To apply anomaly detection, we need to use the Gaussian distribution, which is also known as the normal distribution. If you hear me say either Gaussian or normal distribution, they mean the same thing. The bell-shaped distribution is also a reference to the same thing. If you haven't heard of the bell-shaped distribution, don't worry. Let's take a closer look at what the Gaussian or normal distribution is.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1703684145254/40014afe-ce92-4270-bdb2-ed0eb683b927.png align="center")

If we consider a number x as a random variable, it can take on various random values. When the probability of x is given by a normal or Gaussian distribution, it has a mean parameter Mu and a variance Sigma squared. This distribution can be represented by a curve that is centered at Mu and has a width that is determined by Sigma.

The bell-shaped curve that we commonly refer to is a graphical representation of this distribution. It resembles the shape of large bells found in old buildings. The probability of x, denoted by p of x, can be interpreted as the likelihood of getting a particular value of x from the Gaussian distribution. If we draw a histogram of a large number of values of x from this distribution, it will have a bell-shaped curve similar to the one shown in the graph.

The formula for p of x is given by the expression: p of x equals 1 over square root 2 Pi times Sigma times e to the negative x minus Mu squared divided by 2 Sigma squared, where Pi is approximately 22 over 7. By plotting this formula as a function of x, we can generate a bell-shaped curve that is centered at Mu and has a width determined by Sigma.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1703685722764/9f6d0e81-73fd-4519-b3fb-e547aea60a5e.png align="center")

Let's explore how the Gaussian distribution is affected by changing the values of Mu and Sigma.

Firstly, assume Mu equals 0 and Sigma equals 1. Here's a plot of the Gaussian distribution with mean 0, which is also Mu equals 0, and standard deviation Sigma equals 1. The distribution is centered at zero and the standard deviation Sigma is equal to 1.

Now, let's decrease the standard deviation Sigma to 0.5. If we plot the Gaussian distribution with Mu equals 0 and Sigma equals 0.5, it will look like this. The distribution is still centered at zero, but it has become a much thinner curve because Sigma is now 0.5. Remember that Sigma squared is also called the variance, which is equal to 0.5 squared or 0.25. Probabilities always sum up to one, therefore, the area under the curve is always equal to one. That's why when the Gaussian distribution becomes skinnier, it has to become taller as well.

Let's now increase Sigma to 2, so the standard deviation is 2 and the variance is 4. This creates a much wider distribution because Sigma is now much larger, and because it's a wider distribution, it becomes shorter as well, since the area under the curve still equals 1.

Lastly, let's change the mean parameter Mu, and leave Sigma at 0.5. In this case, the center of the distribution Mu moves to the right. However, the width of the distribution remains the same as the previous one because the standard deviation is 0.5 in both cases.

These are how different values of Mu and Sigma affect the Gaussian distribution.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1703689227978/52c706b5-6d67-45ca-8b09-2c8436d91eb7.png align="center")

When you're applying this to anomaly detection, here's what you have to do. You are given a dataset of m examples, and here x is just a number. Here, are plots of the training sets with 11 examples. What we have to do is try to estimate what a good choice is for the mean parameter Mu, as well as for the variance parameter Sigma squared. Given a dataset like this, it would seem that a Gaussian distribution maybe looking like that with a center here and a standard deviation like that. This might be a pretty good fit to the data. The way you would compute Mu and Sigma squared mathematically is our estimate for Mu will be just the average of all the training examples. It's 1 over m times sum from i equals 1 through m of the values of your training examples. The value we will use to estimate Sigma squared will be the average of the squared difference between two examples, and that Mu that you just estimated here on the left. It turns out that if you implement these two formulas in code with this value for Mu and this value for Sigma squared, then you pretty much get the Gaussian distribution that I hand-drew on top. This will give you a choice of Mu and Sigma for a Gaussian distribution so that it looks like the 11 training samples might have been drawn from this Gaussian distribution. If you've taken an advanced statistics class, you may have heard that these formulas for Mu and Sigma squared are technically called the maximum likelihood estimates for Mu and Sigma. Some statistics classes will tell you to use the formula 1 over n minus 1 instead of 1 over m. In practice, using 1 over m or 1 over n minus 1 makes very little difference. I always use 1 over m, but just some other properties of dividing by m minus 1 that some statisticians prefer. But if you don't understand what they just said, don't worry about it. All you need to know is that if you set Mu according to this formula and Sigma squared according to this formula, you'd get a pretty good estimate of Mu and Sigma and in particular, you get a Gaussian distribution that will be a possible probability distribution in terms of what's the probability distribution that the training examples had come from. You can probably guess what comes next. If you were to get an example over here, then the p of x is pretty high. Whereas if you were to get an example, we are here, then p of x is pretty low, which is why we would consider this example, okay, not really anomalous, not a lot like the other ones. Whereas an example we are here to be pretty unusual compared to the examples we've seen, and therefore more anomalous because p of x, which is the height of this curve, is much lower over here on the left compared to this point over here, closer to the middle. Now, we've done this only for when x is a number as if you had only a single feature for your anomaly detection problem. For practical anomaly detection applications, you usually have a lot of different features.

You have just learned about the Gaussian distribution and how it works. If x is a single number, it means that you have only one feature for your anomaly detection problem. However, in real-life anomaly detection scenarios, you will have multiple features. This could be two, three, or even more features, denoted by the variable n. In order to build a more advanced anomaly detection algorithm, we can use what we learned about single Gaussian and extend it to handle multiple features. This will make our algorithm more sophisticated and efficient in detecting anomalies.

### Anomaly Detection Algorithm

Now that you've seen how the Gaussian or the normal distribution works for a single number, we're ready to build our anomaly detection algorithm. Let's dive in.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1703785507515/3aaf9d96-7876-47e8-8978-3a56904529ba.png align="center")

You have a training set consisting of x1 through xm. Each example x in this set has n features, which means that x is a vector with n numbers. For instance, in the case of the airplane engine example, we had two features - heat and vibrations. Therefore, each Xi would be a two-dimensional vector, with n equal to 2. However, in practical applications, n can be much larger, with dozens or even hundreds of features.

Our aim with this training set is to carry out density estimation, which means building a model or estimating the probability for p(x). In other words, what is the probability of any given feature vector? Our model for p(x) is based on the probability of each feature. We model p(x) as the probability of x1 multiplied by the probability of x2, and so on down to xn.

If you've studied statistics before, you may recognize that this equation assumes that the features x1, x2, and so on, are statistically independent. However, this algorithm often works fine even if the features are not actually statistically independent. Don't worry if you don't understand this concept, as it's not necessary to complete this class and use the anomaly detection algorithm effectively.

To fill in this equation further, we can say that the probability of all the features of this vector x is the product of p(x1) and p(x2), and so on up through p(xn). For each feature, we estimate two parameters - mu and sigma squared. For example, to model p(x1) - heat feature in the airplane engine example, we estimate the mean and variance of the feature, which are mu1 and sigma1 squared. Similarly, for p(x2) - vibrations feature, we estimate mu2 and sigma2 squared. If we have additional features, we estimate mu3 and sigma3 squared up through mun.

To illustrate why we multiply probabilities, suppose an aircraft engine has a 1/10 chance of being unusually hot and a 1/20 chance of vibrating really hard. Then, what is the chance that it runs really hot and vibrates really hard? The chance of that happening is 1/10 multiplied by 1/20, which is 1/200. It's unlikely to get an engine that both runs really hot and vibrates really hard.

A more compact way to write this equation is to say that it is equal to the product from j =1 through n of p(xj), with parameters mu j and sigma squared j. The symbol used here is similar to the summation symbol, but instead of addition, this symbol corresponds to multiplying the terms for j =1 through n.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1703788859408/324342d4-8db3-40db-9e4b-8d31eb26b8e0.png align="center")

To build an anomaly detection system, you need to follow a few steps. Firstly, you need to choose the features that you think might be indicative of anomalous examples. Once you have identified the features, you need to fit the parameters mu1 through mun and sigma square 1 through sigma squared n, for the n features in your data set.

To compute mu j, you simply take the average of xj of the feature j of all the examples in your training set. And sigma square j is the average of the square difference between the feature and the value mu j that you just computed. If you have a vectorized implementation, you can also compute mu as the average of the training examples by using the vectorized way of computing mu 1 through mu and all at the same time.

By estimating these parameters on your unlabeled training set, you have now computed all the parameters of your model. Finally, when you are given a new example, x test, what you would do is compute p(x) and see if it's large or small.

To compute p(x), you need to calculate the product from j=1 through n of the probability of the individual features. So p(x) j with parameters mu j and sigma square j. By substituting the formula for this probability, you end up with an expression that is 1 over root 2 pi sigma j of e to the expression over here.

If you compute this formula, you get some number for p(x). The final step is to see if p(x) is less than epsilon. If it is, then you flag that it is an anomaly. One intuition behind what this algorithm is doing is that it will tend to flag an example as anomalous if one or more of the features are either very large or very small relative to what the algorithm has seen in the training set.

For each of the features xj, you're fitting a Gaussian distribution. So, if even one of the features of the new example was way out of range, then P f xJ would be very small. If just one of the terms in this product is very small, then the overall product, when you multiply it together, will tend to be very small. This is what anomaly detection is doing in this algorithm - it's a systematic way of quantifying whether or not this new example x has any features that are unusually large or small.

---

## **🎙️ Message for Next Episode**

So you've seen the process of how to build an anomaly detection system. However, you might have questions about selecting the parameter epsilon and evaluating the performance of your system. In the next episode, we will explore these topics in more detail to help you develop and assess your anomaly detection system better. Let's move on to the next episode.

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