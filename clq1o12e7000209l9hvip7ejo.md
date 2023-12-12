---
title: "Understanding Logistic Regression| Episode 9"
seoTitle: "Mastering Logistic Regression: A Deep Dive"
seoDescription: "Unlock the power of Logistic Regression. Explore S-shaped curves, avoid linear regression pitfalls, and enhance your binary decision"
datePublished: Tue Dec 12 2023 01:30:13 GMT+0000 (Coordinated Universal Time)
cuid: clq1o12e7000209l9hvip7ejo
slug: understanding-logistic-regression-episode-9
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1702307521830/95686050-4133-495f-b28a-b6eb4c3160e9.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1702307512173/1febdf8e-733a-41b5-aedf-5f349f3167ea.png
tags: algorithms, python, data-science, machine-learning, data-analysis, logistic-regression

---

Welcome back! In the [previous episodes](https://neuralrealm.hashnode.dev/understanding-linear-regression-episode-5), we learned about Linear Regression, a powerful tool for predicting continuous outcomes. We learned how to draw the best-fitting line through our data by optimizing parameters to minimize the difference between predicted and actual values.

## Introduction

### Logistic Regression: Simplifying Binary Classification

Logistic Regression is a core machine learning algorithm designed for binary classification tasks. Unlike Linear Regression, which predicts continuous outcomes, Logistic Regression focuses on situations where the goal is to categorize instances into one of two classes.

### Key Features:

1. **Binary Outcome:**
    
    * Suited for scenarios with two possible outcomes, such as spam or non-spam emails, customer churn, or disease presence.
        
2. **Sigmoid Function:**
    
    * Utilizes the Sigmoid function to transform a linear combination of features into probabilities between 0 and 1.
        
3. **Probability Threshold:**
    
    * Sets a threshold to convert probabilities into binary outcomes, determining class labels.
        
4. **Log Odds and Odds Ratio:**
    
    * Outputs log odds, offering insights into the likelihood of an event, expressed as odds ratios.
        

### Applications:

* **Medical Diagnosis:** Predicting the presence of a medical condition.
    
* **Finance:** Assessing loan default or detecting fraudulent transactions.
    
* **Marketing:** Predicting customer churn or campaign response.
    
* **Social Sciences:** Analyzing factors influencing binary outcomes in research.
    

## Basics of Logistic Regression

In this episode, we'll unravel the fundamentals of Logistic Regression, a powerful algorithm tailored for binary classification tasks. Unlike Linear Regression, which predicts continuous outcomes, Logistic Regression is adept at handling scenarios where the goal is to categorize instances into one of two classes.

1. ### Binary Outcome
    

Logistic Regression is ideal when dealing with a binary outcome, where there are only two possible classes. Examples include spam or non-spam emails, customer churn (yes or no), and medical diagnosis (presence or absence of a condition).

1. ### Sigmoid Function
    

At the core of Logistic Regression is the Sigmoid function, also known as the logistic function. This mathematical curve transforms a linear combination of input features into probabilities. It squashes the output values between 0 and 1, making them interpretable as probabilities.

1. ### Probability Threshold
    

Unlike Linear Regression, Logistic Regression doesn't directly provide class labels. Instead, it produces probabilities. A threshold is set to convert these probabilities into binary outcomes. For example, if the probability is greater than 0.5, the instance is classified as the positive class; otherwise, it's classified as the negative class.

---

## Learn From Scratch

### Introduction

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702148991491/31e46c98-2729-4b0c-ab4e-f6c4d508bc05.png align="center")

To explain how Logistic Regression works, let's take the example of determining whether a tumor is malignant. We assign the label '1' or 'yes' to positive cases (malignant tumors) and '0' or 'no' to negative cases (benign tumors). To represent the dataset in a graph, we plot the tumor size on the horizontal axis and only 0 and 1 values on the vertical axis since it is a classification problem. Using logistic regression, we can fit an S-shaped curve to the dataset.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702149345241/808a1b15-ab87-4486-bff5-1b11af32cb9e.png align="center")

For this example, if a patient comes in with a tumor of this size, which I'm showing on the x-axis, then the algorithm will output 0.7 suggesting that is closer or maybe more likely to be malignant and benign. Will say more later what 0.7 actually means in this context. But the output label y is never 0.7 is only ever 0 or 1. To build out the logistic regression algorithm, there's an important mathematical function I like to describe which is called the Sigmoid function, sometimes also referred to as the logistic function.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702150110327/a32a1967-bf15-4ee4-b695-a96cb273c936.png align="center")

The Sigmoid function has a distinct shape, which is important to understand. The graph on the left represents tumor size on the x-axis, with only positive numbers. On the other hand, the graph on the right has a horizontal axis labeled Z, ranging from -3 to + 3 including negative and positive values. The Sigmoid function outputs a value between 0 and 1.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702151516813/f027f335-66c6-4aeb-b611-d0e15a4504c0.png align="center")

If we use g of z to denote a function, then the formula of g of z is equal to 1 over 1 plus e to the negative z. Here, e is a mathematical constant that has a value of approximately 2.7, and e to the negative z is e to the power of negative z. It's worth noting that if z were to be a large number, such as 100, e to the negative z would be e to the negative 100, which is tiny. Thus, g of z is 1 over 1 plus a tiny number, making the denominator close to 1. Consequently, when z is large, g of z, which is a Sigmoid function of z, is very close to 1. Similarly, when z is a very large negative number, g of z becomes 1 over a giant number, causing g of z to be close to 0. This explains why the sigmoid function has a shape that starts close to zero and gradually increases to the value of one.

Moreover, in the Sigmoid function, when z equals 0, e to the negative z is e to the negative 0, which equals 1, making g of z 1 over 1 plus 1, which equals 0.5. Therefore, the Sigmoid function passes the vertical axis at 0.5. This information helps us build up the logistic regression algorithm.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702274007395/64317434-ac56-4693-a761-85a6c63303b0.png align="center")

Let's go through the process of building up to the logistic regression algorithm step by step. In the first step, we define a straight line function as w multiplied by x plus b, similar to a linear regression function. We'll call this value z. Then, we pass this value of z to the Sigmoid function, also known as the logistic function, g. The formula for g of z is 1 over 1 plus e to the negative z. This output value lies between 0 and 1.

Now, when we combine the two equations, we get the logistic regression model f of x, which is equal to g of wx plus b, or g of z. Essentially, this model takes in a set of features X as input and outputs a number between 0 and 1.

#### Interpretation of Logistic Regression Output

Let's review how to interpret the output of logistic regression. We'll revisit the example of tumor classification. Think of logistic regression's output as the probability that the class or label y will be 1, given input x. For instance, in this application, where x is the size of the tumor, and y is either 0 or 1, if a patient has a tumor of a certain size x, and the model outputs x plus 0.7, then that means the model is predicting a 70 percent chance that the label y will be equal to 1 for this patient. In other words, the model is saying that the patient has a 70 percent chance of having a malignant tumor.

Now, let me ask you a question. If the chance of y being 1 is 70 percent, what is the chance that it is 0? Since y can only be 0 or 1, the probability of these two numbers must add up to 1 or 100 percent. Therefore, if the chance of y being 1 is 0.7 or 70 percent, then the chance of it being 0 must be 0.3 or 30 percent.

#### **<mark>To gain a better understanding of how this process works, I recommend referring to this Notebook</mark>:** [Sigmoid Function Notebook](https://github.com/everydaycodings/Notebooks/blob/master/extra/Machine%20Learning/Logistic%20Regression/Sigmoid_function.ipynb)

### Decision Boundary

Suppose you want to learn an algorithm to predict the value of y, whether it's going to be zero or one. One way to do this is by setting a threshold above which you predict y to be one and below which you predict y to be zero. Typically, people choose a threshold of 0.5, which means if the value of f(x) is greater than or equal to 0.5, then you predict y to be one. This is denoted as y hat equals 1. On the other hand, if the value of f(x) is less than 0.5, then you predict y to be zero, which is denoted as y hat equals 0.

Now, let's dive deeper into when the model would predict one. In other words, when is f(x) greater than or equal to 0.5? We'll recall that f(x) is just equal to g(z). So, f(x) is greater than or equal to 0.5 whenever g(z) is greater than or equal to 0.5. But when is g(z) greater than or equal to 0.5?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702283591608/be192a80-8e7f-42ac-8c2d-7c1c8d0a2f06.png align="center")

Whenever the value of z is greater than or equal to 0, the value of g(z) becomes greater than or equal to 0.5. This means that z must be on the right half of the axis. To determine when z is greater than or equal to zero, we should consider the formula z = w.x + b. Therefore, z is greater than or equal to zero if w.x + b is greater than or equal to zero.

To summarize, this model predicts 1 when w.x + b is greater than or equal to 0, and it predicts 0 when w.x + b is less than zero.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702285282385/39e95483-3e30-401e-8dbc-24d45b9ce27f.png align="center")

Let's examine how the model generates predictions by visualization. To illustrate this, consider a classification problem with two features, x1 and x2, instead of just one feature. We have a training set where the positive examples are represented by little red crosses and the negative examples are represented by little blue circles. The red crosses indicate y equals 1, while the blue circles correspond to y equals 0. The logistic regression model will use the function f of x equals g of z to make predictions. In this case, z is the expression w1x1 plus w2x2 plus b since we have two features x1 and x2. Let's assume that the parameter values for this example are w1 equals 1, w2 equals 1, and b equals negative 3.

Let's analyze how logistic regression predicts by determining when wx plus b is greater than or equal to 0 and when it's less than 0.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702303766265/a01f1a06-bfbe-480e-8adf-ae5eef71ea84.png align="center")

To determine a decision boundary, we look at the line where wx plus b equals 0. This line indicates the point at which we are almost neutral about whether y is 0 or 1. In our example, the decision boundary is x1 plus x2 minus 3, which corresponds to the line x1 plus x2 equals 3. This line serves as the decision boundary for logistic regression, predicting 1 for features to the right of the line and 0 for features to the left of the line. The parameters w1, w2, and b in our example are 1, 1, and -3, respectively. If we had different parameter values, the decision boundary would be a different line.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702305690709/b0c18746-dd15-4401-a4d8-0eb5a836bee4.png align="center")

Let's consider a more complex example where the decision boundary is not a straight line. The cross symbol represents class y equals 1, and the little circle symbol denotes class y equals 0. Last week, you learned how to use polynomials in linear regression, and you can use them in logistic regression as well. We set z to be w\_1, x\_1 squared plus w\_2, x\_2 squared plus b, with this choice of features, we incorporate polynomial features into a logistic regression. The function f of x, which is equal to g of z, is now g of this expression over here. Suppose we choose w\_1 and w\_2 to be 1 and b to be negative 1. Then, z is equal to 1 times x\_1 squared plus 1 times x\_2 squared minus 1. The decision boundary, as before, corresponds to when z is equal to 0. This expression equals 0 when x\_1 squared plus x\_2 squared equals 1. If we plot the diagram on the left, the curve corresponding to x\_1 squared plus x\_2 squared equals 1 is the circle. When x\_1 squared plus x\_2 squared is greater than or equal to 1, that's the area outside the circle, and we predict y to be 1. Conversely, when x\_1 squared plus x\_2 squared is less than 1, that's the area inside the circle, and we predict y to be 0.

It is possible to create more complex decision boundaries by using higher-order polynomial terms.

Logistic regression is capable of fitting complex data by learning from it. However, if you exclude higher-order polynomials and only use features like x\_1, x\_2, x\_3, etc., the decision boundary for logistic regression will always be a straight line and remain linear.

#### **<mark>To gain a better understanding of how this process works, I recommend referring to this Notebook</mark>:** [Decision Boundary Notebook](https://github.com/everydaycodings/Notebooks/blob/master/extra/Machine%20Learning/Logistic%20Regression/Decision_Boundary.ipynb)

---

### **🎙️ Message for Next Episode:**

Now that you have an understanding of what function f(x) can potentially calculate, let's focus on how to train a logistic regression model. We will start by analyzing the <mark>cost function</mark> for logistic regression and then learn how to apply <mark>gradient descent</mark> to it. Let's proceed to the next episode.

---

### **Resources For Further Research**

1. **Coursera** - [**Machine Learning Specialization**](https://www.coursera.org/learn/machine-learning/home/welcome)
    
    This Article is heavily inspired by this Course so I will also recommend you to check this course out, there is an option to watch the course for free.
    
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