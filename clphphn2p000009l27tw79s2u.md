---
title: "Understanding Linear Regression | Episode 5"
seoTitle: "Understanding Linear Regression"
seoDescription: "Unlock the power of Linear Regression in machine learning! Dive into the heart of Cost function optimization, discovering the algorithm's significance."
datePublished: Tue Nov 28 2023 02:15:42 GMT+0000 (Coordinated Universal Time)
cuid: clphphn2p000009l27tw79s2u
slug: understanding-linear-regression-episode-5
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1701137541620/c8240d65-5873-4db3-805a-f04e6d154503.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1701137552663/10d6d6ce-d5d1-447c-b01c-4c6e0dd2afaf.png
tags: artificial-intelligence, algorithms, technology, python, data-science, machine-learning, beginners

---

## Understanding Linear Regression

Linear regression is a fundamental technique in the world of statistics and machine learning that enables us to model relationships between variables effectively. It is useful in a wide range of applications, from predicting stock prices to analyzing economic trends and evaluating marketing strategies.

At its core, linear regression seeks to establish a linear relationship between a dependent variable and one or more independent variables. Despite its simplicity, linear regression is a significant tool that is the foundation for more advanced machine learning algorithms and is widely employed in different fields.

This comprehensive guide is designed to demystify linear regression, catering to both beginners and those seeking a deeper understanding. We will explain the concepts of simple and multiple linear regression, delve into the assumptions underlying the model, explore the intricacies of model building, and equip you with practical tips for effective implementation.

As we embark on this journey, we invite you to envision linear regression not merely as a mathematical concept but as a powerful analytical tool with real-world applications. Whether you are a data science enthusiast, a student exploring statistics, or a professional seeking to enhance your modeling skills, this guide will provide you with the necessary knowledge to navigate the world of linear regression with confidence. Let's dive in.

## Basics of Linear Regression

Linear regression is a statistical method used for modeling the relationship between a dependent variable (often denoted as (Y)) and one or more independent variables (denoted as (X)). The fundamental idea is to fit a straight line to the data that minimizes the difference between the predicted and observed values.

### Key Concepts

#### 1\. **Dependent and Independent Variables:**

In the context of linear regression:

* **Dependent Variable** `Y`**:** This is the variable we want to predict or explain.
    
* **Independent Variables** `X`**:** These are the variables used to predict the dependent variable.
    

#### 2\. Linear Relationship

**Linear regression assumes a straight-line relationship:**

$$\hat{y} = mX + b$$

* `m` is the slope.
    
* `b` is the intercept.
    
* `x` is the independent variable
    

### 3\. Minimizing Residuals

The goal is to minimize the difference between the predicted `y hat`  
and observed `Y` values.

---

## **Linear Regression**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701069499917/9ab72c4c-9cb0-411a-9fb5-77a57f243826.png align="center")

Supervised learning is a process where a model is trained with a dataset that includes both input features (e.g. size of a house) and output targets (e.g. price of the house). The objective is for the model to learn a function `f`that can predict new outputs `y^` based on input features `x`.

The mathematical function `f` represents the model's hypothesis and is defined by the equation `fw,b(x) = wx + b`, where w and b are numbers learned from the training data. This simple linear function predicts y values based on the input `x`, with y being the actual target in the training set, and y^ being the model's estimate.

Using a linear function makes the process simpler and serves as a foundation for more complex models. This particular model is a univariate linear regression, meaning that predictions are based on only one input variable.

The cost function is a crucial element in constructing and training machine learning models. It evaluates how well the model's predictions match the actual targets and guides the learning algorithm in adjusting the parameters to minimize errors.

This linear regression approach is a fundamental concept in machine learning and sets the stage for more advanced models. It highlights the importance of understanding and defining the cost function to optimize model performance.

## Cost Function

#### What do Parameters `w` and `b` do?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701074307890/74c5b4a5-a4fe-4d0e-9a9e-495031ace990.png align="center")

In the context of linear regression, the parameters (w) and (b) play crucial roles in determining the behavior of the model. Let's break down their functions based on the provided transcript:

1. **w - Weight:**
    
    * `w` represents the weight or coefficient associated with the input variable `x`.
        
    * Changing the value of w alters the slope of the linear function
        
    * A larger w results in a steeper slope, influencing the rate at which the model responds to changes in the input.
        
2. **b - Bias:**
    
    * `b` is the bias or `y-intercept` term in the linear function.
        
    * It determines where the line crosses the vertical axis on a graph.
        
    * Adjusting `b` shifts the entire line up or down without affecting its slope, influencing the baseline prediction.
        

In the provided examples:

* When w = 0 and(b = 1.5, the line becomes a horizontal one, constantly predicting the value b irrespective of the input.
    
* With w = 0.5 and b = 0, the line has a slope of 0.5 and intersects the origin. The predictions are proportional to the input.
    
* For w = 0.5 and b = 1, the line has the same slope but a different y-intercept, resulting in a shifted prediction.
    

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701077494850/e8129b07-b991-4d5a-8cbe-621100398449.png align="center")

### Cost Function Construction

1. **Error Measurement:**
    
    * The cost function begins with measuring the error between the predicted `yhat` and target `y` values.
        
    * The error `yhat−y` represents how far the prediction is from the actual target.
        
2. **Squared Error:**
    
    * The squared error `(yhat−y)^2` is computed to emphasize larger errors and penalize them more.
        
    * This step ensures that both positive and negative errors contribute to the overall assessment.
        
3. **Error Summation:**
    
    * The squared error is calculated for each training example (ii) in the training set.
        
    * The sum of squared errors across is obtained
        

$$\sum_{i=1}^{m} (y_{hat} - y)^2$$

1. **Average Squared Error:**
    
    * To prevent the cost function from automatically increasing with larger training sets, the average squared error is computed by dividing the sum by the number of examples `m`.
        
    * This is expressed as
        

$$\frac{1}{2m} \sum_{i=1}^{m} (y_{hat} - y)^2$$

1. **Division by 2 (Optional):**
    
    * By convention, the cost function often includes an additional division by 2, making the expression
        

$$\frac{1}{2m} \sum_{i=1}^{m} (y_{hat} - y)^2$$

* This extra division simplifies later calculations but is not strictly necessary.
    

#### Objective

The objective is to minimize this cost function `J(w,b)` by adjusting the parameters w and b. Minimizing the cost function leads to a model that makes predictions as close as possible to the true targets in the training set.

<mark>To gain a better understanding of how this process works, I recommend referring to this Notebook</mark>: [Model Representation Notebook](https://github.com/everydaycodings/Notebooks/blob/master/article/extra/Linear%20Regression/Model_Representation.ipynb)

### Cost Function Intuition

We have just learned about the mathematical definition of the cost function. However, to understand what the cost function is doing, we need to develop some intuition. In this section, we will take one example to see how the cost function can be used to find the best parameters for your model.

So far, we have seen that we want to fit a straight line to the training data and that this model, fw, b of x, is w times x plus b. The parameters here are `w` and `b`. Depending on the values chosen for these parameters, we get different straight lines. Our goal is to find values for w and b so that the straight line fits the training data well.

To measure how well a choice of w and b fits the training data, we have a cost function `J`. This function measures the difference between the model's predictions and the actual true values for `y`. Later on, we'll see that the linear regression will try to find values for w and b that make J of w as small as possible. In math, we write it like this: we want to minimize J as a function of w and b.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701088925940/f9598364-6291-4b91-bcca-47d127d7ee9c.png align="center")

To help us better understand the cost function `J` and visualize it, we will use a simplified version of the linear regression model. In this model, we will use fw of x, which is w times x. This means that we can remove the parameter b from the original model, or set b equal to 0. Now, f is only w times x, and we have only one parameter w. Our cost function J is similar to what it was before, where we take the difference, square it, and f is now equal to w times xi. We aim to find the value of w that minimizes J of w.

With this simplified model, the goal changes because we only have one parameter, w. We need to find the value of w that minimizes J of w. We can visualize this by setting b equal to 0, which means that f defines a line that passes through the origin when x is 0.

Now, let's see how the cost function J changes as we choose different values for the parameter w. We will plot the graphs of the model fw of x and the cost function J side-by-side to see how they are related. When w is fixed, fw is only a function of x, and the estimated value of y depends on the value of the input x. On the other hand, the cost function J is a function of w, where w controls the slope of the line defined by fw. The cost defined by J depends on the parameter w.

We will start by plotting the function fw of x on the left <mark>(refer to the image above)</mark>, which is a straight line with a slope of w. We will then pick a value for w, say 1, and calculate the cost J when w is 1. For this data set, when x is 1, then y is 1, and when w is also 1, fw(x) equals y for the first training example, and the difference is 0. We can do the same for the second and third examples, and we will find that the cost J is 0 for all three examples when w equals 1.

On the right, we will plot the cost function J, which is a function of the parameter w. The horizontal axis is labeled w, and the vertical axis is J. We can see that when w equals 1, J(w) is 0.

We can then change the value of w to 0.5 and see how the graphs change for different values of w. We can take on negative values, 0, and positive values, and the graphs will show us how the cost function J changes for each value of w.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701088245718/54ae16e6-2424-4a55-b2f9-a1181c50c796.png align="center")

Let's assume we set w to 0.5. In this scenario, the function f(x) would be a line with a slope of 0.5. Now, let's calculate the cost J when w is 0.5. Recall that the cost function measures the squared error or the difference between the predicted value (y hat I) and the true value (Y^i) for each example i. We can observe that visually, the difference is equal to the height of the vertical line when x is 1. This lower line represents the difference between the actual value of y and the predicted value of f(x), which is slightly below it. For the first example where x is 1, f(x) is 0.5. The squared error for this example is 0.5 minus 1 squared. To calculate the cost function, we will sum up the squared errors for all the training examples. Moving on to the second example, when x is 2, f(x) predicts 1, but the actual value of y is 2. The error for the second example is the height of the line segment, and the squared error is the square of the length of this segment, which is 1 minus 2 squared. Similarly, for the third example, the error is 1.5 minus 3 squared. We will add up all these errors, which comes out to be 3.5, and multiply this by 1 over 2m, where m is the number of training examples. Since we have three training examples, m equals 3, and this is equal to 1 over 2 times 3. If we do the math, we get 3.5 divided by 6 which results in a cost J of about 0.58. We can plot this point on the right. Let's now try another value for w, say w = 0. What do the graphs for f and J look like when w = 0? When w = 0, f of x is simply a horizontal line that intersects the x-axis. The error for each example is a line that goes from each point down to the horizontal line that represents f of x = 0. The cost J when w = 0 is 1 over 2m times the quantity 1^2 plus 2^2 plus 3^2, which evaluates to 1 over 6 times 14, which is about 2.33. We can plot this point where w = 0 and J = 2.33 on the graph. We can do this for other values of w as well, including negative values. When w = -0.5, f is a downward-sloping line and the cost increases to around 5.25. By computing a range of values for w, we can slowly trace out what the cost function J looks like.

To recap, each value of parameter w corresponds to a different straight line fit, f of x, on the graph on the left. For the given training set, that choice for a value of w corresponds to a single point on the graph on the right because, for each value of w, we can calculate the cost J of w. For each value of w, we get a different line and its corresponding costs, J of w, and we can use these points to trace out the cost function graph on the right.

Choosing a value of w that causes J of w to be as small as possible seems like a good bet. J measures how big the squared errors are, so choosing w that minimizes these squared errors makes them as small as possible and will give us a good model. In this example, choosing the value of w that results in the smallest possible value of J of w leads to w = 1, which results in the line that fits the training data very well.

To summarize, we saw plots of both f and J and worked through how the two are related. As we vary w or vary w and b, we end up with different straight lines, and when that straight line passes across the data, the cost J is small. The goal of linear regression is to find the parameters w or w and b that results in the smallest possible value for the cost function J.

### Cost Function Visualization

In the previous section, a visualization of the cost function J of w or J of w, b was presented. In this section, we will explore richer visualizations to help you better understand the behavior of the cost function.

So far, we have seen the model, its parameters w and b, the cost function J of w and b, and the goal of linear regression, which is to minimize the cost function J of w and b over parameters w and b. In the last section, we temporarily set b to zero to simplify the visualizations. Now, we will go back to the original model with both parameters w and b without setting b to be equal to 0.

As before, we want to gain a visual understanding of the model function, f of x, shown on the left, and how it relates to the cost function J of w, b, shown on the right. Let's say you pick one possible function of x, like this one. Here, I've set w to 0.06 and b to 50.

Given these values for w and b, let's examine what the cost function J of w and b may look like. Recall what we saw last time when we had only w because we temporarily set b to zero to simplify things. We had come up with a plot of the cost function that looked like a U-shaped curve, similar to a soup bowl, as a function of w only, when we had only one parameter, w.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701134806787/8af049e2-ecc4-43ee-9d85-13b000caed5f.gif align="center")

<mark>To gain a better understanding of how this process works, I recommend referring to this Notebook</mark>: [Cost Function Notebook LInk](https://github.com/everydaycodings/Notebooks/blob/master/article/extra/Linear%20Regression/Cost_function.ipynb)

---

### **🎙️ Message for Next Episode:**

When it comes to linear regression, manually trying to read a contour plot to find the best value for w and b is not a good procedure. Moreover, it won't work for more complex machine learning models. What you need is an efficient algorithm that can automatically find the values of parameters w and b that give you the best-fit line. This is where gradient descent comes in. It is an algorithm that minimizes the cost function j and is one of the most important algorithms in machine learning. Gradient descent and its variations are used to train not only linear regression but also some of the biggest and most complex models in AI. In the next episode, we will dive into this crucial algorithm called <mark>gradient descent</mark>.

---

### **Resources For Further Research**

1. **Coursera** - [Machine Learning Specialization](https://www.coursera.org/learn/machine-learning/home/welcome)
    
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