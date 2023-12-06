---
title: "Understanding Multiple Linear Regression | Episode 7"
seoTitle: "Mastering Multiple Linear Regression: A Comprehensive Guide"
seoDescription: "Dive into the intricacies of multiple linear regression. Explore key concepts, applications, and practical insights in this comprehensive guide."
datePublished: Wed Dec 06 2023 01:30:10 GMT+0000 (Coordinated Universal Time)
cuid: clpt3dvsg000108l5asrc6ir0
slug: understanding-multiple-linear-regression-episode-7
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1701768254420/f73a64f9-6842-42ac-bab5-72abd9eb38a6.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1701768267199/80dfc2f3-114b-4eb3-a306-6e08aefc73c2.png
tags: algorithms, programming-blogs, technology, python, data-science, machine-learning, deep-learning, mathematics

---

## Introduction

Welcome back! In this episode, we will learn how to make linear regression faster and more powerful. By the end of this episode, you will completely understand linear regression from end to end.

Multiple linear regression is a statistical technique that helps in analyzing the relationship between a dependent variable and two or more independent variables. In simple linear regression, there is only one independent variable to predict the dependent variable, whereas in multiple linear regression, there are several independent variables to predict the dependent variable. This technique is useful in making predictions, identifying the strength of the relationships, and estimating the impact of different independent variables on the dependent variable.

### **Breaking Down the Basics**

| **Size (sqft)** | **Price (1000s dollars)** |
| --- | --- |
| 2104 | 460 |
| 1416 | 232 |
| 852 | 178 |

Let's start by exploring a version of linear regression that considers multiple features instead of just one. In the original linear regression model, only one feature, the size of the house (x), was used to predict the price (y) of the house with the equation `fwb(x) = wx + b`. However, in the updated version, multiple features are taken into account to make the predictions more accurate.

| **Size (sqft)** | **Number of Bedrooms** | **Number of Floors** | **Age of Home** | **Price (1000s dollars)** |
| --- | --- | --- | --- | --- |
| 2104 | 5 | 1 | 45 | 460 |
| 1416 | 3 | 2 | 40 | 232 |
| 852 | 2 | 1 | 35 | 178 |

Imagine you have more information to predict the price of a house than just its size. What if you also knew the number of bedrooms, floors, and the age of the home in years? This additional information would provide you with much more data to make a more accurate prediction of the price.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701685603978/5e9175d7-f3d8-4457-8f56-16fc66ef6e59.png align="center")

Let me explain some new notation that we will be using. Firstly, we will use four variables, X\_1, X\_2, X\_3, and X\_4, to represent the four features. To simplify things, we will use X subscript j to denote the list of features. Here, j will range from one to four since we have four features. We will also use lowercase n to represent the total number of features, which in this case is 4.

Furthermore, we will use X superscript i to denote the ith training example, which is a list of four numbers or a vector that includes all the features of the ith training example. For example, X superscript in parentheses 2 will be a vector of the features for the second training example, which is 1416, 3, 2, and 40. Technically, these numbers are written in a row, so sometimes it is called a row vector rather than a column vector.

To refer to a specific feature in the ith training example, we will write X superscript i, subscript j. For instance, X superscript 2 subscript 3 represents the value of the third feature, which is the number of floors in the second training example, and so it is equal to 2.

To emphasize that X superscript i is not a number but a list of numbers, which is a vector, we might draw an arrow on top of it just to show that it is a vector. However, this is optional, and you do not need to include it in your notation.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701687165042/bd97b73f-97a0-4a7a-961c-36b8939d172d.png align="center")

When you have n features, the model will look like this. To simplify this expression, let's introduce some notation. We'll define W as a list of parameters, W\_1 through W\_n. This is called a vector and can be designated with a little arrow on top. B is a single number and together with W, it forms the parameters of the model. X is a list that lists all of the features X\_1, X\_2, X\_3 up to X\_n. This is also a vector and can be designated with an arrow on top. Using these notations, the model can be rewritten more succinctly as f(x) = W.X + b. The dot product of two vectors W and X is computed by multiplying the corresponding pairs of numbers, summing up the products, and then adding the number b. This notation gives us the same expression as before, but with fewer characters. This type of linear regression model with multiple input features is called multiple linear regression.

### Gradient descent for multiple linear regression

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701705122546/59dbcfc8-118e-4328-8ea4-4aefb99a262c.png align="center")

Let's simplify our notation by using vector notation. Instead of considering w\_1 to w\_n as separate parameters, we can collect them into a vector w of length n. Similarly, we can represent b as a number. Using vector notation, we can write the model as f\_w, b of x equals the dot product of the vector w and the vector x, plus b.

Our cost function J can be defined as a function of the parameter vector w and the number b. We can use gradient descent to update each parameter w\_j to be w\_j minus Alpha times the derivative of the cost J. We can write this as J of vector w and number b.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701708944972/4f2d068c-b695-4cab-9f6c-e300d717dda1.png align="center")

When we have multiple features, gradient descent becomes a little different compared to when we have only one feature. In the case of univariate regression, we had only one feature, which we called xi without any subscript. We had an update rule for w and a separate update rule for b. The update rule for w involved the derivative of the cost function J with respect to the parameter w. Similarly, we had an update rule for parameter b.

Now, let's consider the case where we have n features, where n is two or more. In this case, we get a new notation for gradient descent. We update w\_1 to be w\_1 minus alpha times the derivative of the cost J with respect to w\_1. The formula for the derivative of J with respect to w\_1 looks very similar to the case of one feature. The error term still takes a prediction f of x minus the target y.

One difference is that w and x are now vectors. Just as w on the left has now become w\_1 on the right, xi here on the left is now instead xi\_1 here on the right, and this is just for J equals 1. For multiple linear regression, we have J ranging from 1 through n, so we'll update the parameters w\_1, w\_2, all the way up to w\_n, and then as before, we'll update b.

If you implement this, you get gradient descent for multiple regression. That's it for gradient descent for multiple regression.

Congratulations! You have now learned about multiple linear regression, which is considered to be the most widely used learning algorithm in the world today.

<mark>To gain a better understanding of how this process works, I recommend referring to this Notebook</mark>: [Multiple Variables Notebook](https://github.com/everydaycodings/Notebooks/blob/master/article/extra/Multiple%20LInear%20Regression/Multiple_Variable.ipynb)**,** [**Gradient Descent Normal Notebook**](https://github.com/everydaycodings/Notebooks/blob/master/article/extra/Multiple%20LInear%20Regression/Sklearn_Normal.ipynb)**,** [**Gradient Descent With Scikit-learn Notebook**](https://github.com/everydaycodings/Notebooks/blob/master/article/extra/Multiple%20LInear%20Regression/Sklearn_GD.ipynb)**.**

---

### **🎙️ Message for Next Episode:**

In the upcoming episode, we will go beyond regression and discuss our first classification algorithm, which can predict categories. Join me next episode.

---

### **Resources For Further Research**

1. **Coursera** - [**Machine Learning Specialization**](https://www.coursera.org/learn/machine-learning/home/welcome)
    
    This Article is heavily inspired by this Course so I will also recommend you to check this course out, there is an option to watch the course for free.
    
2. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron:**
    
    * A practical guide covering various machine learning concepts and implementations.
        

---

## **By the way…**

#### Call to action

*Hi, Everydaycodings— I’m building a newsletter that covers deep topics in the space of engineering. If that sounds interesting,* [***subscribe***](https://neuralrealm.hashnode.dev/newsletter) *and don’t miss anything. If you have some thoughts you’d like to share or a topic suggestion, reach out to me via* [***LinkedIn***](https://www.linkedin.com/in/kumar-saksham1891/) *or* [***X***](https://twitter.com/everydaycodings).

#### References

*And if you’re interested in diving deeper into these concepts, here are some great starting points:*

* [**Kaggle Stories**](https://neuralrealm.hashnode.dev/series/kaggle-stories) *\-* Each episode of Kaggle Stories takes you on a journey behind the scenes of a Kaggle notebook project, breaking down tech stuff into simple stories.
    
* [**Machine Learning**](https://neuralrealm.hashnode.dev/series/machine-learning) *\-* This series covers ML fundamentals & techniques to apply ML to solve real-world problems using Python & real datasets while highlighting best practices & limits.